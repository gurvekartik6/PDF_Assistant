# ============================================================
# embeddings.py - Text Embedding Model (Stable Version)
# ============================================================

"""
This module provides a reusable embedding model for converting
text into numerical vectors using HuggingFace.

This version avoids:
- langchain_huggingface dependency issues
- version conflicts
- runtime import errors

Works with: langchain-community
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings

# Global singleton model
_embedding_model = None


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Load embedding model once and reuse it

    Args:
        model_name: HuggingFace model name

    Returns:
        HuggingFaceEmbeddings instance
    """
    global _embedding_model

    if _embedding_model is None:
        print("Loading embedding model... (first time only)")

        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("Embedding model loaded successfully")

    return _embedding_model


# ============================================================
# EMBED SINGLE TEXT
# ============================================================

def embed_text(text: str) -> List[float]:
    """
    Convert single text into embedding vector
    """
    model = get_embedding_model()
    return model.embed_query(text)


# ============================================================
# EMBED MULTIPLE TEXTS
# ============================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert multiple texts into embeddings
    """
    model = get_embedding_model()
    return model.embed_documents(texts)