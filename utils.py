# ============================================================
# utils.py - Utility Helper Functions
# ============================================================
#
# PURPOSE:
#   This file contains small helper functions used throughout
#   the application. Think of it as a "toolbox" of reusable
#   functions that don't belong to any specific module.
#
# FUNCTIONS IN THIS FILE:
#   - Session state management (Streamlit session)
#   - Text formatting helpers
#   - Source document formatting
#   - Flashcard parsing and display helpers
#   - File validation
#   - Error handling utilities
# ============================================================

import os
import re
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path


# ============================================================
# STREAMLIT SESSION STATE MANAGEMENT
# ============================================================
# Streamlit re-runs the entire Python script on every user
# interaction. "Session state" is how we persist data between
# these re-runs (like a dictionary that survives page reloads).

def initialize_session_state(st) -> None:
    """
    Initialize all session state variables for the app.
    
    Call this at the start of app.py to set up all the
    variables we need to persist between Streamlit reruns.
    
    WHAT IS SESSION STATE?
    Every time a user clicks a button in Streamlit, the whole
    Python script reruns from top to bottom. Session state
    variables persist across these reruns so we don't lose data.
    
    Args:
        st: Streamlit module
    """
    # Default values for each session variable
    defaults = {
        # Vector store (our searchable knowledge base)
        "vector_store": None,

        # Chat history: list of {"role": "user/assistant", "content": "..."}
        "chat_history": [],

        # The conversational chain (remembers context between messages)
        "conv_chain": None,

        # Memory object for the conversational chain
        "conv_memory": None,

        # Names of currently loaded PDF files
        "loaded_pdfs": [],

        # Whether voice mode is enabled
        "voice_mode": False,

        # Selected TTS language
        "tts_language": "en",

        # Track if PDFs have been processed (to avoid reprocessing)
        "pdfs_processed": False,

        # Store generated content for display
        "last_summary": None,
        "last_notes": None,
        "last_flashcards": None,

        # Processing status messages
        "status_message": "",
        "error_message": ""
    }

    # Only set the default if the variable doesn't already exist
    # (we don't want to overwrite existing values on rerun)
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_session_data(st) -> None:
    """
    Clear all session data and start fresh.
    
    Called when user wants to process new PDFs.
    
    Args:
        st: Streamlit module
    """
    keys_to_clear = [
        "vector_store", "chat_history", "conv_chain", "conv_memory",
        "loaded_pdfs", "pdfs_processed", "last_summary",
        "last_notes", "last_flashcards"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Reinitialize with defaults
    initialize_session_state(st)


# ============================================================
# TEXT FORMATTING FUNCTIONS
# ============================================================

def format_source_documents(sources: List[Dict]) -> str:
    """
    Format source document references for display.
    
    Creates a nicely formatted string showing where the AI's
    answer came from (which PDF file and page number).
    
    Args:
        sources: List of source dicts with 'source', 'page', 'content'
        
    Returns:
        Formatted string of source references
    """
    if not sources:
        return ""

    formatted = "\n\n---\n📚 **Sources:**\n"

    for i, source in enumerate(sources, 1):
        file_name = Path(source.get("source", "Unknown")).name
        page = source.get("page", "N/A")

        # Page numbers from PyPDF are 0-indexed, so we add 1
        if isinstance(page, int):
            page = page + 1

        formatted += f"\n**[{i}] {file_name}** — Page {page}\n"

        # Show a preview of the relevant content
        content_preview = source.get("content", "")
        if content_preview:
            # Truncate long previews
            if len(content_preview) > 200:
                content_preview = content_preview[:200] + "..."
            formatted += f"> {content_preview}\n"

    return formatted


def format_flashcards_for_display(flashcards: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Clean and format flashcards for display.
    
    Args:
        flashcards: Raw flashcard list from LLM
        
    Returns:
        Cleaned flashcard list
    """
    formatted = []
    for card in flashcards:
        # Clean up any extra whitespace or formatting
        question = card.get("question", "").strip()
        answer = card.get("answer", "").strip()

        # Skip empty cards
        if question and answer:
            formatted.append({
                "question": question,
                "answer": answer
            })

    return formatted


def truncate_text(text: str, max_length: int = 500, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum character length
        add_ellipsis: Whether to add "..." at the end
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]

    # Try to break at a word boundary (don't cut mid-word)
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # Only if space is reasonably close to end
        truncated = truncated[:last_space]

    if add_ellipsis:
        truncated += "..."

    return truncated


def clean_llm_response(text: str) -> str:
    """
    Clean up common LLM response artifacts.
    
    LLMs sometimes include unwanted formatting like "Answer:" prefix
    or extra newlines. This function cleans those up.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Cleaned text
    """
    # Remove common prefixes LLMs sometimes add
    prefixes_to_remove = ["Answer:", "Response:", "AI:", "Assistant:"]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_pdf_file(uploaded_file):
    """
    Validate that an uploaded file is a valid PDF.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    # Check file extension
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, f"'{uploaded_file.name}' is not a PDF file. Please upload PDF files only."

    # Check file size (max 50MB)
    max_size_mb = 50
    max_size_bytes = max_size_mb * 1024 * 1024
    if uploaded_file.size > max_size_bytes:
        size_mb = uploaded_file.size / (1024 * 1024)
        return False, f"'{uploaded_file.name}' is {size_mb:.1f}MB. Maximum allowed size is {max_size_mb}MB."

    # Check minimum file size (very small PDFs are likely corrupted)
    if uploaded_file.size < 100:  # Less than 100 bytes is almost certainly invalid
        return False, f"'{uploaded_file.name}' appears to be empty or corrupted."

    return True, ""


def validate_multiple_pdfs(uploaded_files) -> tuple:
    """
    Validate multiple uploaded PDF files.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        
    Returns:
        Tuple of (valid_files, error_messages)
    """
    valid_files = []
    errors = []

    for file in uploaded_files:
        is_valid, error_msg = validate_pdf_file(file)
        if is_valid:
            valid_files.append(file)
        else:
            errors.append(error_msg)

    return valid_files, errors


# ============================================================
# STATISTICS AND DISPLAY HELPERS
# ============================================================

def format_file_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable file size string.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Human-readable size string (e.g., "2.3 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_file_hash(file_content: bytes) -> str:
    """
    Generate a hash to uniquely identify a file.
    
    Used to check if a file has already been processed.
    Two identical files will have the same hash.
    
    Args:
        file_content: Raw bytes of the file
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(file_content).hexdigest()


def format_chat_history_for_export(chat_history: List[Dict]) -> str:
    """
    Format chat history as plain text for export/copying.
    
    Args:
        chat_history: List of message dicts with 'role' and 'content'
        
    Returns:
        Formatted text conversation
    """
    if not chat_history:
        return "No chat history available."

    lines = ["=" * 60, "AI PDF ASSISTANT - CHAT HISTORY", "=" * 60, ""]

    for msg in chat_history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        if role == "USER":
            lines.append(f"You: {content}")
        elif role == "ASSISTANT":
            lines.append(f"AI: {content}")

        lines.append("")  # Blank line between messages

    return "\n".join(lines)


def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count for a text.
    
    This is a simple approximation. Real tokenizers are more complex.
    Rule of thumb: ~4 characters per token for English text.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def create_processing_summary(
    pdf_names: List[str],
    total_pages: int,
    total_chunks: int,
    processing_time: float
) -> str:
    """
    Create a human-readable summary of PDF processing results.
    
    Args:
        pdf_names: Names of processed PDF files
        total_pages: Total pages processed
        total_chunks: Total chunks created
        processing_time: Time taken in seconds
        
    Returns:
        Formatted summary string
    """
    summary = f"""✅ **Processing Complete!**

📄 **PDFs Processed:** {len(pdf_names)}
{chr(10).join(f'   • {name}' for name in pdf_names)}

📊 **Statistics:**
   • Total Pages: {total_pages}
   • Text Chunks: {total_chunks}
   • Processing Time: {processing_time:.1f} seconds

💡 **Ready to use!** You can now:
   • Ask questions about your documents
   • Generate summaries
   • Create study notes
   • Generate flashcards"""

    return summary
