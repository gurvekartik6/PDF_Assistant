"""
qa_chain.py - RAG Q&A chain optimised for speed.

Speed improvements:
  - k=4 chunks (was 8) — less context = faster LLM processing
  - max_tokens=800 for chat, 1000 for summary/notes (was 1500 everywhere)
  - Removed double-call fallback — one shot only
  - Shorter, tighter prompts — fewer input tokens = faster TTFT
  - Summary/notes use k=6 (was 12)
  - NEW: ask_question_stream() — true SSE streaming for chat tab
"""

import os
import re
import json
import requests
from typing import Dict, Any, List, Optional, Generator

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  Core HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _call_openrouter(prompt, api_key, model,
                     temperature=0.2, max_tokens=800, stream=False):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://docent-app.local",
        "X-Title": "Docent PDF Assistant",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if not stream:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=40)
            if resp.status_code != 200:
                return f"LLM Error ({resp.status_code}): {resp.text[:200]}"
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", None)
            )
            if content is None:
                print(f"[DEBUG] Unexpected API response: {str(data)[:500]}")
                return f"LLM returned no content. Raw: {str(data)[:300]}"
            return content.strip()
        except Exception as e:
            return f"LLM Exception: {e}"

    def _gen():
        try:
            with requests.post(url, headers=headers, json=payload,
                               timeout=40, stream=True) as resp:
                if resp.status_code != 200:
                    yield f"LLM Error ({resp.status_code})"
                    return
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(line)
                        delta = (chunk.get("choices", [{}])[0]
                                      .get("delta", {})
                                      .get("content") or "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception as e:
            yield f"Stream error: {e}"

    return _gen()


# ─────────────────────────────────────────────────────────────────────────────
#  LLM wrapper for LangChain chain
# ─────────────────────────────────────────────────────────────────────────────

class OpenRouterLLM(LLM):
    api_key: str = ""
    model: str = "meta-llama/llama-3-8b-instruct"
    temperature: float = 0.2
    max_tokens: int = 800

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            object.__setattr__(self, "api_key", os.getenv("OPENROUTER_API_KEY", ""))
        env_model = os.getenv("OPENROUTER_MODEL", "")
        if env_model:
            object.__setattr__(self, "model", env_model)

    @property
    def _llm_type(self):
        return "openrouter"

    def _call(self, prompt, stop=None):
        result = _call_openrouter(prompt, self.api_key, self.model,
                                  self.temperature, self.max_tokens, stream=False)
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Tight prompts
# ─────────────────────────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    template="""Answer using ONLY the context. Be concise and direct.
If the answer is not in the context, say so briefly.

Context:
{context}

Question: {input}

Answer:""",
    input_variables=["context", "input"],
)

SUMMARY_PROMPT = "Summarise this document concisely using bullet points.\n\nDocument:\n{context}\n\nSummary:"
NOTES_PROMPT   = "Create structured study notes.\n\nDocument:\n{context}\n\n## Key Concepts\n- ...\n\n## Important Points\n- ...\n\nNotes:"
FLASHCARD_PROMPT = "Create 6-8 flashcards. Format: Q: [question] | A: [answer]\n\nDocument:\n{context}\n\nFlashcards:"
PAGE_PROMPT    = "Explain page {page_num} and answer the question.\n\nPage {page_num}:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# ─────────────────────────────────────────────────────────────────────────────
#  Intent detection
# ─────────────────────────────────────────────────────────────────────────────

def extract_page_number(query):
    m = re.search(r"page\s*(\d+)", query.lower())
    return int(m.group(1)) - 1 if m else None

def detect_intent(query):
    q = query.lower()
    if re.search(r"page\s*\d+", q): return "page"
    if any(w in q for w in ["summary", "summarize", "summarise", "overview"]): return "summary"
    if any(w in q for w in ["notes", "study notes", "key points"]): return "notes"
    if any(w in q for w in ["flashcard", "flash card", "quiz", "test me"]): return "flashcard"
    return "semantic"


# ─────────────────────────────────────────────────────────────────────────────
#  ask_question  (blocking — for summary/notes/flashcard tabs)
# ─────────────────────────────────────────────────────────────────────────────

def ask_question(chain, vector_store, question):
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
    intent  = detect_intent(question)

    def call(prompt, max_tok=800):
        return _call_openrouter(prompt, api_key, model, 0.2, max_tok, stream=False)

    if intent == "page":
        page = extract_page_number(question)
        if page is None:
            return {"answer": "Could not determine page number.", "sources": []}
        all_docs  = list(vector_store.docstore._dict.values())
        page_docs = [d for d in all_docs if d.metadata.get("page") == page]
        if not page_docs:
            return {"answer": f"Page {page + 1} not found.", "sources": []}
        context = "\n\n".join(d.page_content for d in page_docs[:4])
        return {"answer": call(PAGE_PROMPT.format(page_num=page+1, context=context, question=question)), "sources": page_docs}

    if intent == "summary":
        docs = vector_store.similarity_search("main topics summary", k=6)
        return {"answer": call(SUMMARY_PROMPT.format(context="\n\n".join(d.page_content for d in docs)), 1000), "sources": docs}

    if intent == "notes":
        docs = vector_store.similarity_search("concepts definitions key points", k=6)
        return {"answer": call(NOTES_PROMPT.format(context="\n\n".join(d.page_content for d in docs)), 1000), "sources": docs}

    if intent == "flashcard":
        docs = vector_store.similarity_search("concepts facts definitions", k=5)
        return {"answer": call(FLASHCARD_PROMPT.format(context="\n\n".join(d.page_content for d in docs)), 900), "sources": docs, "mode": "flashcard"}

    # semantic — use chain
    result = chain.invoke({"input": question})
    return {"answer": result.get("answer", "No answer found."), "sources": result.get("context", [])}


# ─────────────────────────────────────────────────────────────────────────────
#  ask_question_stream  (streaming for chat tab)
# ─────────────────────────────────────────────────────────────────────────────

def ask_question_stream(vector_store, question):
    """Retrieve k=4 docs, build prompt, stream tokens immediately."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
    intent  = detect_intent(question)

    if intent == "page":
        page = extract_page_number(question)
        if page is None:
            yield "Could not determine page number."
            return
        all_docs  = list(vector_store.docstore._dict.values())
        page_docs = [d for d in all_docs if d.metadata.get("page") == page]
        if not page_docs:
            yield f"Page {page + 1} not found."
            return
        context = "\n\n".join(d.page_content for d in page_docs[:4])
        prompt  = PAGE_PROMPT.format(page_num=page+1, context=context, question=question)
    elif intent in ("summary", "notes", "flashcard"):
        yield "Please use the dedicated tab for summaries, notes, and flashcards."
        return
    else:
        docs    = vector_store.similarity_search(question, k=4)
        context = "\n\n".join(d.page_content for d in docs)
        prompt  = f"Answer using ONLY the context. Be concise.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    yield from _call_openrouter(prompt, api_key, model, 0.2, 800, stream=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Flashcard parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_flashcards(text):
    cards = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if "|" in line and line.lower().startswith("q:"):
            parts = line.split("|", 1)
            if len(parts) == 2:
                q = parts[0].replace("Q:", "").strip()
                a = parts[1].replace("A:", "").strip()
                if q and a:
                    cards.append({"question": q, "answer": a})
    return cards


# ─────────────────────────────────────────────────────────────────────────────
#  Chain factory
# ─────────────────────────────────────────────────────────────────────────────

def create_conversational_chain(vector_store):
    llm       = OpenRouterLLM(temperature=0.2, max_tokens=800)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    doc_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    chain     = create_retrieval_chain(retriever, doc_chain)
    return chain, vector_store