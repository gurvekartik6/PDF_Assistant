import os
import streamlit as st
import time
import tempfile
import warnings
import fitz

from dotenv import load_dotenv
from pdf_loader import load_pdfs_from_uploads, split_documents_into_chunks
from vector_store import create_vector_store
from qa_chain import create_conversational_chain, ask_question

warnings.filterwarnings("ignore")
load_dotenv()

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI PDF Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CLEAN SAAS UI ----------------
st.markdown("""
<style>

/* BASE */
html, body {
    background: #f5f7fb;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #111827;
}

/* MAIN */
.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* HEADER */
.header {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* CARD */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* CHAT */
.chat-user {
    background: #2563eb;
    color: white;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: right;
}

.chat-ai {
    background: #f1f5f9;
    color: #111827;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px 0;
}

/* FIX FILE UPLOADER */
[data-testid="stFileUploader"] * {
    color: #111827 !important;
}

/* INPUT */
input, textarea {
    background: white !important;
    color: #111827 !important;
}

/* BUTTON */
button {
    border-radius: 8px !important;
}

/* SLIDER */
[data-testid="stSlider"] {
    padding-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- STATE ----------------
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# ---------------- HEADER ----------------
st.markdown('<div class="header">AI PDF Assistant</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.subheader("Documents")

    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process"):
        if not files:
            st.warning("Upload PDFs first")
            st.stop()

        docs = load_pdfs_from_uploads(files)
        chunks = split_documents_into_chunks(docs)
        vs = create_vector_store(chunks)

        st.session_state.vector_store = vs
        st.session_state.chain, _ = create_conversational_chain(vs)

        for f in files:
            file_bytes = f.getvalue()
            if file_bytes:
                st.session_state.pdf_files[f.name] = file_bytes

        st.session_state.current_file = list(st.session_state.pdf_files.keys())[0]

        st.success("Documents ready")

# stop if not ready
if st.session_state.chain is None:
    st.info("Upload and process PDFs to begin")
    st.stop()

# ---------------- LAYOUT ----------------
chat_col, viewer_col = st.columns([1.2, 1])

# ================= CHAT =================
with chat_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Chat")

    for msg in st.session_state.history:
        if msg["role"] == "User":
            st.markdown(f'<div class="chat-user">{msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">{msg["text"]}</div>', unsafe_allow_html=True)

    query = st.text_input("Ask your document")

    if st.button("Send"):
        if query:
            st.session_state.history.append({"role": "User", "text": query})

            result = ask_question(
                st.session_state.chain,
                st.session_state.vector_store,
                query
            )

            answer = result["answer"]
            docs = result["sources"]

            # STREAM
            display = st.empty()
            text = ""

            for word in answer.split():
                text += word + " "
                display.markdown(f'<div class="chat-ai">{text}</div>', unsafe_allow_html=True)
                time.sleep(0.01)

            st.session_state.history.append({"role": "AI", "text": text})

            # SOURCE NAV
            st.markdown("### Sources")
            for i, d in enumerate(docs):
                fname = d.metadata.get("source")
                page = d.metadata.get("page", 0)

                if st.button(f"{fname} | Page {page+1}", key=f"src_{i}"):
                    st.session_state.current_file = fname
                    st.session_state.current_page = page + 1

            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ================= VIEWER =================
with viewer_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Viewer")

    file_names = list(st.session_state.pdf_files.keys())

    selected = st.selectbox("File", file_names)

    pdf_bytes = st.session_state.pdf_files[selected]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    total_pages = len(doc)

    # NAVIGATION
    c1, c2, c3, c4 = st.columns(4)

    if c1.button("⏮"): st.session_state.current_page = 1
    if c2.button("◀"): st.session_state.current_page -= 1
    if c3.button("▶"): st.session_state.current_page += 1
    if c4.button("⏭"): st.session_state.current_page = total_pages

    st.session_state.current_page = max(1, min(total_pages, st.session_state.current_page))

    page_num = st.slider("Page", 1, total_pages, st.session_state.current_page)
    st.session_state.current_page = page_num

    zoom = st.slider("Zoom", 50, 150, 100)

    page = doc.load_page(page_num - 1)

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom/100, zoom/100))
    img = tmp_path + ".png"
    pix.save(img)

    st.image(img, use_column_width=True)
    st.caption(f"Page {page_num} / {total_pages}")

    st.markdown('</div>', unsafe_allow_html=True)