import os
import gc
from typing import List

# Suppress tokenizer parallelism warnings that can cause fork issues
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_community.embeddings import HuggingFaceEmbeddings

_embedding_model = None


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:

    global _embedding_model
    if _embedding_model is None:
        print("[embeddings] Loading model (first time)…")
        gc.collect()  # free memory before the heavy load
        try:
            _embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("[embeddings] Model loaded OK")
        except Exception as e:
            print(f"[embeddings] ERROR loading model: {e}")
            raise
    return _embedding_model


def embed_text(text: str) -> List[float]:
    return get_embedding_model().embed_query(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedding_model().embed_documents(texts)