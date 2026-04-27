import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_single_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip()]
    return documents


def load_pdfs_from_uploads(uploaded_files) -> List[Document]:
    all_documents = []

    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            documents = load_single_pdf(tmp_path)
            if not documents:
                continue

            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["page"] = doc.metadata.get("page", 0)

            all_documents.extend(documents)

        except Exception as e:
            print(f"Error loading {uploaded_file.name}: {e}")
            continue

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return all_documents


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    return chunks