import gc
import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embeddings import get_embedding_model

# ── chunk size for batch indexing ─────────────────────────────────────────────
# Embedding all chunks at once can exhaust RAM on large PDFs.
# We build the index in batches and merge them.
_BATCH_SIZE = int(os.getenv("FAISS_BATCH_SIZE", "64"))


def create_vector_store(chunks: List[Document]) -> FAISS:
    
    if not chunks:
        raise ValueError("No chunks provided — cannot build vector store.")

    embedding_model = get_embedding_model()
    print(f"[vector_store] Indexing {len(chunks)} chunks in batches of {_BATCH_SIZE}…")

    vector_store: Optional[FAISS] = None

    for start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[start : start + _BATCH_SIZE]
        batch_store = FAISS.from_documents(documents=batch, embedding=embedding_model)

        if vector_store is None:
            vector_store = batch_store
        else:
            vector_store.merge_from(batch_store)
            del batch_store

        gc.collect()
        end = min(start + _BATCH_SIZE, len(chunks))
        print(f"[vector_store]   indexed {end}/{len(chunks)}")

    print("[vector_store] Done.")
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = "./vector_store") -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(path)
    print(f"[vector_store] Saved to {path}")


def load_vector_store(path: str = "./vector_store") -> Optional[FAISS]:
    index_file = Path(path) / "index.faiss"
    meta_file  = Path(path) / "index.pkl"

    if not index_file.exists() or not meta_file.exists():
        print("[vector_store] No saved index found.")
        return None

    print("[vector_store] Loading saved index…")
    embedding_model = get_embedding_model()
    vs = FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    print("[vector_store] Loaded OK")
    return vs


def vector_store_exists(path: str = "./vector_store") -> bool:
    return (
        (Path(path) / "index.faiss").exists()
        and (Path(path) / "index.pkl").exists()
    )


def delete_vector_store(path: str = "./vector_store") -> None:
    if Path(path).exists():
        shutil.rmtree(path)
        print("[vector_store] Deleted.")


def search_similar_chunks(vector_store: FAISS, query: str, k: int = 6):
    return vector_store.similarity_search(query, k=k)