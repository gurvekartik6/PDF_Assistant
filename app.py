import os
import time
import tempfile
import warnings
import streamlit as st
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

# ---------------- STABLE DARK UI ----------------
st.markdown("""
<style>

/* ROOT */
html, body, [data-testid="stAppViewContainer"] {
    background: #0f172a;
    color: #e2e8f0;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

/* HEADER */
.header {
    font-size: 26px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 10px;
}

/* CARD */
.card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 16px;
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
    background: #020617;
    border: 1px solid #1e293b;
    color: #e2e8f0;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px 0;
}

/* INPUT */
input, textarea {
    background: #020617 !important;
    color: #e2e8f0 !important;
}

/* FILE UPLOADER FIX */
[data-testid="stFileUploader"] * {
    color: #e2e8f0 !important;
}

/* BUTTON */
button {
    border-radius: 8px !important;
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
if "highlight_text" not in st.session_state:
    st.session_state.highlight_text = ""

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

    if st.button("Process", key="process_btn"):
        if not files:
            st.warning("Upload PDFs first")
            st.stop()

        docs = load_pdfs_from_uploads(files)

        if not docs:
            st.error("No readable content found")
            st.stop()

        chunks = split_documents_into_chunks(docs)

        if not chunks:
            st.error("No text extracted from PDF")
            st.stop()

        vs = create_vector_store(chunks)

        st.session_state.vector_store = vs
        st.session_state.chain, _ = create_conversational_chain(vs)

        # SAFE FILE STORAGE
        for f in files:
            file_bytes = f.getvalue()
            if file_bytes:
                st.session_state.pdf_files[f.name] = file_bytes

        st.session_state.current_file = list(st.session_state.pdf_files.keys())[0]

        st.success("Documents ready")

# ---------------- EMPTY STATE ----------------
if st.session_state.chain is None:
    st.markdown("""
    <div style="display:flex; height:70vh; align-items:center; justify-content:center; flex-direction:column; gap:10px;">
        <div style="font-size:40px; opacity:0.6">📂</div>
        <div style="font-size:16px; color:#e2e8f0">Upload and process PDFs to begin</div>
        <div style="font-size:13px; color:#94a3b8">Use the sidebar</div>
    </div>
    """, unsafe_allow_html=True)
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

    query = st.text_input("Ask your document", key="query")

    if st.button("Send", key="send_btn"):
        if query:
            st.session_state.history.append({"role": "User", "text": query})

            try:
                result = ask_question(
                    st.session_state.chain,
                    st.session_state.vector_store,
                    query
                )

                answer = result.get("answer", "")
                docs = result.get("sources", [])

            except Exception:
                st.error("Error generating response")
                st.stop()

            # STREAMING EFFECT
            placeholder = st.empty()
            text = ""

            for word in answer.split():
                text += word + " "
                placeholder.markdown(f'<div class="chat-ai">{text}</div>', unsafe_allow_html=True)
                time.sleep(0.01)

            st.session_state.history.append({"role": "AI", "text": text})

            # SOURCE NAVIGATION
            if docs:
                st.markdown("### Sources")
                for i, d in enumerate(docs):
                    fname = d.metadata.get("source")
                    page = d.metadata.get("page", 0)

                    if st.button(f"{fname} | Page {page+1}", key=f"src_{i}"):
                        st.session_state.current_file = fname
                        st.session_state.current_page = page + 1
                        st.session_state.highlight_text = d.page_content[:120]

            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ================= VIEWER =================
with viewer_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Viewer")

    file_names = list(st.session_state.pdf_files.keys())

    selected = st.selectbox("File", file_names)

    pdf_bytes = st.session_state.pdf_files.get(selected)

    if not pdf_bytes:
        st.error("Invalid PDF")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    total_pages = len(doc)

    # NAVIGATION
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("⏮", key="first"): st.session_state.current_page = 1
    if c2.button("◀", key="prev"): st.session_state.current_page -= 1
    if c3.button("▶", key="next"): st.session_state.current_page += 1
    if c4.button("⏭", key="last"): st.session_state.current_page = total_pages

    st.session_state.current_page = max(1, min(total_pages, st.session_state.current_page))

    page_num = st.slider("Page", 1, total_pages, st.session_state.current_page)
    st.session_state.current_page = page_num

    zoom = st.slider("Zoom", 50, 150, 100)

    page = doc.load_page(page_num - 1)

    # highlight
    if st.session_state.highlight_text:
        try:
            areas = page.search_for(st.session_state.highlight_text[:60])
            for a in areas:
                page.add_highlight_annot(a)
        except:
            pass

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom/100, zoom/100))
    img_path = tmp_path + ".png"
    pix.save(img_path)

    st.image(img_path, use_column_width=True)
    st.caption(f"Page {page_num} / {total_pages}")

    st.markdown('</div>', unsafe_allow_html=True)