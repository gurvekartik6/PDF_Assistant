import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_api_config():
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model   = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
    return api_key, model


def _call(messages: list, stream: bool = False):
    api_key, model = get_api_config()
    if not api_key or "your_" in api_key:
        msg = "⚠️ Add OPENROUTER_API_KEY to your .env file."
        if stream:
            def _e():
                yield msg
            return _e()
        return msg

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1500,
        "stream": stream,
    }

    if not stream:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=60,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            return f"API error {r.status_code}: {r.text[:300]}"
        except Exception as e:
            return f"Request failed: {e}"

    def _gen():
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, stream=True, timeout=60,
            )
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if line.startswith("data: "):
                    line = line[6:]
                if line in ("", "[DONE]"):
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk["choices"][0].get("delta", {}).get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception as e:
            yield f"\n\n[Stream error: {e}]"
    return _gen()


def _get_context(vector_store, question: str, k: int = 5):
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(d.page_content[:700] for d in docs)
    return context, docs


def ask_question(chain, vector_store, question: str) -> dict:
    if not vector_store:
        return {"answer": "No documents loaded. Please upload and process PDFs first.", "sources": []}

    context, docs = _get_context(vector_store, question)
    q_lower = question.lower()

    if any(w in q_lower for w in ["summary", "summarize", "summarise", "overview"]):
        system = (
            "You are a professional document summariser. "
            "Write a clear, well-structured summary with:\n"
            "## Overview\n## Key Points (bullet list)\n## Conclusion\n"
            "Base everything strictly on the provided context."
        )
        user = f"Document context:\n{context}\n\nWrite a comprehensive summary."

    elif any(w in q_lower for w in ["note", "notes", "study"]):
        system = (
            "You are an expert study-notes creator. "
            "Format your output with these exact sections:\n"
            "## Key Concepts\n## Important Facts\n## Definitions\n## Summary\n"
            "Use bullet points inside each section. Be thorough and educational."
        )
        user = f"Document context:\n{context}\n\nCreate detailed study notes from this content."

    elif "flashcard" in q_lower:
        system = (
            "You are a flashcard generator. "
            "Output ONLY flashcards in this EXACT format with no intro or extra text:\n\n"
            "Q: <question here>\nA: <answer here>\n\n"
            "Q: <question here>\nA: <answer here>\n\n"
            "Generate at least 8 high-quality flashcards covering the main facts and concepts."
        )
        user = f"Document context:\n{context}\n\nGenerate flashcards from this content."

    else:
        system = (
            "You are a helpful document assistant. "
            "Answer questions accurately using ONLY the provided context. "
            "If the answer is not in the context, say so clearly."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"

    answer = _call([
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], stream=False)
    return {"answer": answer, "sources": docs}


def ask_question_stream(vector_store, question: str):
    if not vector_store:
        yield "No documents loaded. Please upload and process PDFs first."
        return

    context, _ = _get_context(vector_store, question)
    messages = [
        {"role": "system", "content": "You are a helpful document assistant. Answer using ONLY the provided context."},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    for token in _call(messages, stream=True):
        yield token


def parse_flashcards(text: str) -> list:
    cards = []

    # Primary: Q:/A: blocks
    pattern = re.compile(
        r"Q(?:uestion)?[:\.\)]\s*(.+?)\s*\nA(?:nswer)?[:\.\)]\s*(.+?)(?=\n\s*Q(?:uestion)?[:\.\)]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(text):
        q = m.group(1).strip()
        a = m.group(2).strip()
        if q and a and len(q) > 5:
            cards.append({"question": q, "answer": a})

    # Fallback: pipe-separated
    if not cards:
        for line in text.splitlines():
            if "|" in line:
                parts = [p.strip() for p in line.split("|", 1)]
                if len(parts) == 2:
                    q = re.sub(r"^[\d\.\-\*]*\s*Q(?:uestion)?[:\.\)]?\s*", "", parts[0], flags=re.I).strip()
                    a = re.sub(r"^A(?:nswer)?[:\.\)]?\s*", "", parts[1], flags=re.I).strip()
                    if q and a:
                        cards.append({"question": q, "answer": a})

    # Last resort: alternating lines
    if not cards:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for i in range(0, len(lines) - 1, 2):
            cards.append({"question": lines[i], "answer": lines[i + 1]})

    return cards[:15]


def create_conversational_chain(vector_store):
    class SimpleChain:
        def __init__(self, vs):
            self.vector_store = vs
        def invoke(self, inputs: dict) -> dict:
            question = inputs.get("input", inputs.get("question", ""))
            return ask_question(self, self.vector_store, question)
    return SimpleChain(vector_store), vector_store