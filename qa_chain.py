import os
import re
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()



class OpenRouterLLM(LLM):
    api_key: str = os.getenv("OPENROUTER_API_KEY")
    model: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if not self.api_key:
            return "Error: Missing API key"

        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code != 200:
                return f"LLM Error: {response.text}"

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"LLM Exception: {str(e)}"



QA_PROMPT = PromptTemplate(
    template="""
Answer using ONLY the given context.

Context:
{context}

Question:
{input}

Answer:
""",
    input_variables=["context", "input"]
)



def extract_page_number(query: str):
    match = re.search(r'page\s*(\d+)', query.lower())
    if match:
        return int(match.group(1)) - 1
    return None


def detect_intent(query: str):
    query = query.lower()
    if "page" in query:
        return "page"
    elif "summary" in query or "summarize" in query:
        return "summary"
    return "semantic"



def ask_question(chain, vector_store, question: str) -> Dict[str, Any]:
    intent = detect_intent(question)

    # ---------------- PAGE MODE ----------------
    if intent == "page":
        page = extract_page_number(question)

        # 🔥 FIX: get ALL chunks directly (no similarity search)
        all_docs = vector_store.docstore._dict.values()

        page_docs = [
            d for d in all_docs if d.metadata.get("page") == page
        ]

        if not page_docs:
            return {"answer": f"Page {page+1} not found", "sources": []}

        context = "\n\n".join([d.page_content for d in page_docs])

        llm = OpenRouterLLM()

        prompt = f"""
You are explaining content from a PDF.

Context (Page {page+1}):
{context}

Question:
{question}

Give a clear and structured explanation.
"""

        answer = llm._call(prompt)

        return {"answer": answer, "sources": page_docs}

    # ---------------- SUMMARY MODE ----------------
    elif intent == "summary":
        docs = vector_store.similarity_search(question, k=10)

        context = "\n\n".join([d.page_content for d in docs])

        llm = OpenRouterLLM()

        answer = llm._call(
            f"Summarize clearly:\n\n{context}"
        )

        return {"answer": answer, "sources": docs}

    # ---------------- NORMAL MODE ----------------
    else:
        result = chain.invoke({"input": question})

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("context", [])
        }



def create_conversational_chain(vector_store):
    llm = OpenRouterLLM(temperature=0.5)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    doc_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    chain = create_retrieval_chain(retriever, doc_chain)

    return chain, vector_store