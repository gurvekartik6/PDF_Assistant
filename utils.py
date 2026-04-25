"""
utils.py - Utility functions for the Docent app.
"""

import os
import re
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path


def initialize_session_state(st) -> None:
    """Initialize all session state variables."""
    defaults = {
        "vector_store": None,
        "chat_history": [],       # [{"role": "user"|"assistant", "content": "..."}]
        "chain": None,
        "pdf_files": {},          # {name: bytes}
        "loaded_pdfs": [],
        "pdfs_processed": False,
        "flashcards": [],         # parsed flashcard list
        "active_tab": "chat",     # "chat" | "summary" | "notes" | "flashcards"
        "last_summary": None,
        "last_notes": None,
        "status_message": "",
        "error_message": "",
        "highlight": None,
        "current_page": 1,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_session_data(st) -> None:
    """Clear all session data and reinitialize."""
    keys = [
        "vector_store", "chat_history", "chain",
        "pdf_files", "loaded_pdfs", "pdfs_processed",
        "flashcards", "last_summary", "last_notes",
        "highlight", "current_page"
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    initialize_session_state(st)


def format_source_documents(sources) -> str:
    """Format source document references for display."""
    if not sources:
        return ""
    formatted = "\n\n---\n📚 **Sources:**\n"
    for i, source in enumerate(sources[:3], 1):
        if hasattr(source, "metadata"):
            meta = source.metadata
            file_name = Path(meta.get("source", "Unknown")).name
            page = meta.get("page", "N/A")
            if isinstance(page, int):
                page += 1
            formatted += f"\n**[{i}] {file_name}** — Page {page}\n"
            preview = source.page_content[:150].strip()
            if preview:
                formatted += f"> {preview}...\n"
    return formatted


def validate_pdf_file(uploaded_file) -> tuple:
    """Validate uploaded PDF file. Returns (is_valid, error_message)."""
    if not uploaded_file.name.lower().endswith(".pdf"):
        return False, f"'{uploaded_file.name}' is not a PDF file."
    max_bytes = 50 * 1024 * 1024
    if uploaded_file.size > max_bytes:
        mb = uploaded_file.size / (1024 * 1024)
        return False, f"'{uploaded_file.name}' is {mb:.1f}MB — max allowed is 50MB."
    if uploaded_file.size < 100:
        return False, f"'{uploaded_file.name}' appears to be empty or corrupted."
    return True, ""


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_file_hash(file_content: bytes) -> str:
    """Generate MD5 hash for a file."""
    return hashlib.md5(file_content).hexdigest()


def clean_llm_response(text: str) -> str:
    """Strip common LLM preamble artifacts."""
    for prefix in ["Answer:", "Response:", "AI:", "Assistant:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length, breaking at word boundary."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def format_chat_history_for_export(chat_history: List[Dict]) -> str:
    """Format chat history as plain text."""
    if not chat_history:
        return "No chat history available."
    lines = ["=" * 60, "DOCENT - CHAT HISTORY", "=" * 60, ""]
    for msg in chat_history:
        role = "You" if msg.get("role") == "user" else "AI"
        lines.append(f"{role}: {msg.get('content', '')}")
        lines.append("")
    return "\n".join(lines)