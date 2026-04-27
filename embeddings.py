from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings

_embedding_model = None

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
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

def embed_text(text: str) -> List[float]:
    model = get_embedding_model()
    return model.embed_query(text)

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.embed_documents(texts)