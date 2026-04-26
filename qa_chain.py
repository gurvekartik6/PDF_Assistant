import os
import re
import json
import requests
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning blocks some models emit."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _get_env():
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
    return api_key, model


# ─────────────────────────────────────────────────────────────────────────────
#  Core HTTP helper  (stream=False → str | stream=True → generator)
# ─────────────────────────────────────────────────────────────────────────────

def _call_openrouter(prompt, api_key, model,
                     temperature=0.2, max_tokens=2048, stream=False):
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

    # ── Non-streaming ──
    if not stream:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                return f"API Error {resp.status_code}: {resp.text[:300]}"

            data    = resp.json()
            choices = data.get("choices") or []
            if not choices:
                print(f"[DEBUG] No choices in response: {data}")
                return "The model returned an empty response. Please try again."

            message = choices[0].get("message") or {}
            content = message.get("content")

            # Some models return content=None with finish_reason='length'
            # In that case try to grab partial content from the raw text
            if not content:
                finish = choices[0].get("finish_reason", "")
                print(f"[DEBUG] Empty content, finish_reason={finish}, data={str(data)[:400]}")
                return "The model did not return a response. Try a shorter question or switch models."

            return _strip_thinking(content)

        except Exception as e:
            return f"Request failed: {e}"

    # ── Streaming generator ──
    def _gen():
        try:
            with requests.post(url, headers=headers, json=payload,
                               timeout=60, stream=True) as resp:
                if resp.status_code != 200:
                    yield f"API Error {resp.status_code}"
                    return

                buffer = ""
                in_think = False

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() in ("[DONE]", ""):
                        continue
                    try:
                        chunk = json.loads(line)
                        delta = (chunk.get("choices") or [{}])[0] \
                                      .get("delta", {}) \
                                      .get("content") or ""
                        if not delta:
                            continue

                        # Strip <think> blocks inline
                        buffer += delta
                        while True:
                            if in_think:
                                end = buffer.find("</think>")
                                if end == -1:
                                    buffer = ""
                                    break
                                buffer = buffer[end + 8:]
                                in_think = False
                            else:
                                start = buffer.find("<think>")
                                if start == -1:
                                    yield buffer
                                    buffer = ""
                                    break
                                yield buffer[:start]
                                buffer = buffer[start + 7:]
                                in_think = True

                    except Exception:
                        continue

                if buffer and not in_think:
                    yield buffer

        except Exception as e:
            yield f"Stream error: {e}"

    return _gen()


# ─────────────────────────────────────────────────────────────────────────────
#  LangChain LLM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class OpenRouterLLM(LLM):
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        key, mdl = _get_env()
        object.__setattr__(self, "api_key", key)
        object.__setattr__(self, "model",   mdl)

    @property
    def _llm_type(self):
        return "openrouter"

    def _call(self, prompt, stop=None):
        return _call_openrouter(
            prompt, self.api_key, self.model,
            self.temperature, self.max_tokens, stream=False
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    template="""Answer using ONLY the context below. Be thorough and direct.
If the answer is not in the context, say so briefly.

Context:
{context}

Question: {input}

Answer:""",
    input_variables=["context", "input"],
)

SUMMARY_PROMPT   = "Summarise the document below with bullet points. Be comprehensive.\n\nDocument:\n{context}\n\nSummary:"
NOTES_PROMPT     = "Create structured study notes from the document below.\n\nDocument:\n{context}\n\n## Key Concepts\n\n## Important Points\n\nNotes:"
FLASHCARD_PROMPT = "Create 6-8 flashcards. One per line: Q: [question] | A: [answer]\n\nDocument:\n{context}\n\nFlashcards:"
PAGE_PROMPT      = "Explain the content of page {page_num} and answer the question.\n\nPage {page_num}:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# ─────────────────────────────────────────────────────────────────────────────
#  Intent detection
# ─────────────────────────────────────────────────────────────────────────────

def extract_page_number(query):
    m = re.search(r"page\s*(\d+)", query.lower())
    return int(m.group(1)) - 1 if m else None

def detect_intent(query):
    q = query.lower()
    if re.search(r"page\s*\d+", q):                                      return "page"
    if any(w in q for w in ["summary","summarize","summarise","overview"]): return "summary"
    if any(w in q for w in ["notes","study notes","key points"]):          return "notes"
    if any(w in q for w in ["flashcard","flash card","quiz","test me"]):   return "flashcard"
    return "semantic"


# ─────────────────────────────────────────────────────────────────────────────
#  ask_question  (blocking — used by summary / notes / flashcard tabs)
# ─────────────────────────────────────────────────────────────────────────────

def ask_question(chain, vector_store, question):
    api_key, model = _get_env()
    intent = detect_intent(question)

    def call(prompt, max_tok=2048):
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
        docs = vector_store.similarity_search("main topics summary overview", k=6)
        return {"answer": call(SUMMARY_PROMPT.format(context="\n\n".join(d.page_content for d in docs))), "sources": docs}

    if intent == "notes":
        docs = vector_store.similarity_search("concepts definitions key points", k=6)
        return {"answer": call(NOTES_PROMPT.format(context="\n\n".join(d.page_content for d in docs))), "sources": docs}

    if intent == "flashcard":
        docs = vector_store.similarity_search("concepts facts definitions", k=5)
        return {"answer": call(FLASHCARD_PROMPT.format(context="\n\n".join(d.page_content for d in docs))), "sources": docs, "mode": "flashcard"}

    # semantic — LangChain chain
    result = chain.invoke({"input": question})
    return {"answer": result.get("answer", "No answer found."), "sources": result.get("context", [])}


# ─────────────────────────────────────────────────────────────────────────────
#  ask_question_stream  (streaming chat)
# ─────────────────────────────────────────────────────────────────────────────

def ask_question_stream(vector_store, question):
    """Stream tokens directly — minimum latency for chat tab."""
    api_key, model = _get_env()
    intent = detect_intent(question)

    if intent == "page":
        page = extract_page_number(question)
        if page is None:
            yield "Could not determine page number."; return
        all_docs  = list(vector_store.docstore._dict.values())
        page_docs = [d for d in all_docs if d.metadata.get("page") == page]
        if not page_docs:
            yield f"Page {page + 1} not found."; return
        context = "\n\n".join(d.page_content for d in page_docs[:4])
        prompt  = PAGE_PROMPT.format(page_num=page+1, context=context, question=question)

    elif intent in ("summary", "notes", "flashcard"):
        yield "💡 Use the **Summary / Notes / Flashcards** tab above for this."; return

    else:
        docs    = vector_store.similarity_search(question, k=4)
        context = "\n\n".join(d.page_content for d in docs)
        prompt  = (
            "Answer using ONLY the context below. Be thorough and direct.\n"
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

    yield from _call_openrouter(prompt, api_key, model, 0.2, 2048, stream=True)


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
    llm       = OpenRouterLLM(temperature=0.2, max_tokens=2048)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    doc_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    chain     = create_retrieval_chain(retriever, doc_chain)
    return chain, vector_store