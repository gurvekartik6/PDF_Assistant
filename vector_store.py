# ============================================================
# vector_store.py - FAISS Vector Database Management
# ============================================================

"""
This module handles:
- Creating vector store
- Saving vector store
- Loading vector store
- Searching documents

Stable version:
✔ No import issues
✔ No deprecated APIs
✔ Compatible with current LangChain
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embeddings import get_embedding_model


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Create FAISS vector store from document chunks

    Args:
        chunks: List of document chunks

    Returns:
        FAISS vector store
    """
    if not chunks:
        raise ValueError("No chunks provided for vector store")

    print(f"Creating vector store from {len(chunks)} chunks...")

    embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    print("Vector store created successfully")

    return vector_store


# ============================================================
# SAVE VECTOR STORE
# ============================================================

def save_vector_store(vector_store: FAISS, path: str = "./vector_store") -> None:
    """
    Save vector store to disk

    Args:
        vector_store: FAISS instance
        path: directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    vector_store.save_local(path)

    print(f"Vector store saved at {path}")


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store(path: str = "./vector_store") -> Optional[FAISS]:
    """
    Load vector store from disk

    Returns:
        FAISS instance or None
    """
    index_file = Path(path) / "index.faiss"
    meta_file = Path(path) / "index.pkl"

    if not index_file.exists() or not meta_file.exists():
        print("No existing vector store found")
        return None

    print("Loading vector store...")

    embedding_model = get_embedding_model()

    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    print("Vector store loaded successfully")

    return vector_store


# ============================================================
# CHECK EXISTENCE
# ============================================================

def vector_store_exists(path: str = "./vector_store") -> bool:
    """
    Check if vector store exists
    """
    return (
        (Path(path) / "index.faiss").exists()
        and (Path(path) / "index.pkl").exists()
    )


# ============================================================
# DELETE VECTOR STORE
# ============================================================

def delete_vector_store(path: str = "./vector_store") -> None:
    """
    Delete stored vector database
    """
    if Path(path).exists():
        shutil.rmtree(path)
        print("Vector store deleted")
    else:
        print("No vector store to delete")


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_similar_chunks(vector_store: FAISS, query: str, k: int = 4):
    """
    Retrieve similar chunks from vector store
    """
    return vector_store.similarity_search(query, k=k)