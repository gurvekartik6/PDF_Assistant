import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_single_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return [doc for doc in documents if doc.page_content.strip()]


def load_pdfs_from_uploads(uploaded_files) -> List[Document]:
    all_documents: List[Document] = []

    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue

        # FIX: reset pointer — Streamlit may have already read the file
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()

        if not file_bytes:
            print(f"[pdf_loader] Skipping empty file: {uploaded_file.name}")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            documents = load_single_pdf(tmp_path)
            if not documents:
                print(f"[pdf_loader] No text extracted from {uploaded_file.name}")
                continue

            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
                # PyPDFLoader uses 0-based page index; keep as-is and +1 at display time
                doc.metadata.setdefault("page", 0)

            all_documents.extend(documents)
            print(f"[pdf_loader] {uploaded_file.name}: {len(documents)} pages loaded")

        except Exception as e:
            print(f"[pdf_loader] Error loading {uploaded_file.name}: {e}")

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return all_documents


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"[pdf_loader] Split into {len(chunks)} chunks")
    return chunks