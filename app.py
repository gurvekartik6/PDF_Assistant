import os
import time
import tempfile
import warnings
import streamlit as st
import fitz
from gtts import gTTS
import speech_recognition as sr
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from dotenv import load_dotenv
from pdf_loader import load_pdfs_from_uploads, split_documents_into_chunks
from vector_store import create_vector_store
from qa_chain import create_conversational_chain, ask_question

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(
    page_title="Docent · AI PDF Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Docent — AI-powered document intelligence"}
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family: 'DM Sans', sans-serif;
    background: #08090c !important;
    color: #d4d8e2;
    margin: 0; padding: 0;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────────── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d0f14 !important;
    border-right: 1px solid #1c2030 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* ── Sidebar header strip ──────────────────────────────────────────────── */
.sidebar-header {
    background: linear-gradient(135deg, #0f1824 0%, #0a1020 100%);
    border-bottom: 1px solid #1c2030;
    padding: 28px 24px 20px;
    margin-bottom: 8px;
}
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.sidebar-logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #3b6cf8 0%, #7c3aed 100%);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
}
.sidebar-logo-text {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.3px;
    color: #f0f2f7;
}
.sidebar-tagline {
    font-size: 11px;
    color: #4a5068;
    font-weight: 400;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

/* ── Sidebar inner padding ─────────────────────────────────────────────── */
.sidebar-body { padding: 0 20px 24px; }

.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #3a4060;
    margin: 20px 0 10px;
}

/* ── File uploader ─────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0d0f14 !important;
    border: 1.5px dashed #1e2438 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b6cf8 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: 12px !important;
    color: #4a5068 !important;
}

/* ── Process button ────────────────────────────────────────────────────── */
.process-btn > button {
    background: linear-gradient(135deg, #3b6cf8 0%, #7c3aed 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    letter-spacing: 0.2px !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 20px rgba(59,108,248,0.25) !important;
}
.process-btn > button:hover { opacity: 0.88 !important; }

/* ── Document badges ───────────────────────────────────────────────────── */
.doc-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #0f1218;
    border: 1px solid #1c2030;
    border-radius: 8px;
    padding: 9px 12px;
    margin: 5px 0;
    font-size: 12px;
    color: #7080a0;
    transition: background 0.15s;
}
.doc-badge:hover { background: #131720; }
.doc-dot {
    width: 6px; height: 6px;
    background: #22c55e;
    border-radius: 50%;
    flex-shrink: 0;
}
.doc-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Main columns ──────────────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] > div:first-child {
    padding-right: 12px;
}
[data-testid="stHorizontalBlock"] > div:last-child {
    padding-left: 12px;
}

/* ── Panel card ────────────────────────────────────────────────────────── */
.panel {
    background: #0d0f14;
    border: 1px solid #1a1f2e;
    border-radius: 14px;
    overflow: hidden;
}
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid #1a1f2e;
    background: #0a0c10;
}
.panel-title {
    font-size: 13px;
    font-weight: 600;
    color: #8090b8;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.panel-badge {
    font-size: 11px;
    background: #12172a;
    border: 1px solid #1e2844;
    color: #3b6cf8;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
}
.panel-body { padding: 18px 20px; }

/* ── Chat messages ─────────────────────────────────────────────────────── */
.msg-row { display: flex; margin: 8px 0; }
.msg-row.user { justify-content: flex-end; }
.msg-row.ai   { justify-content: flex-start; }

.bubble {
    max-width: 82%;
    padding: 11px 15px;
    border-radius: 12px;
    font-size: 13.5px;
    line-height: 1.6;
    position: relative;
}
.bubble.user {
    background: linear-gradient(135deg, #1d3a8a 0%, #2d2080 100%);
    color: #c8d8ff;
    border-bottom-right-radius: 3px;
}
.bubble.ai {
    background: #111520;
    border: 1px solid #1a2035;
    color: #c8cede;
    border-bottom-left-radius: 3px;
}
.bubble.ai::before {
    content: 'AI';
    position: absolute;
    top: -18px; left: 0;
    font-size: 10px;
    color: #3b6cf8;
    font-weight: 600;
    letter-spacing: 0.5px;
    font-family: 'DM Mono', monospace;
}

.chat-area {
    min-height: 340px;
    max-height: 420px;
    overflow-y: auto;
    padding: 20px 0 8px;
    scrollbar-width: thin;
    scrollbar-color: #1e2438 transparent;
}
.chat-area::-webkit-scrollbar { width: 4px; }
.chat-area::-webkit-scrollbar-thumb { background: #1e2438; border-radius: 4px; }

.chat-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 280px;
    gap: 10px;
    color: #2a3050;
}
.chat-empty-icon { font-size: 36px; opacity: 0.4; }
.chat-empty-text { font-size: 13px; }

/* ── Divider ───────────────────────────────────────────────────────────── */
.divider {
    height: 1px;
    background: #1a1f2e;
    margin: 14px 0;
}

/* ── Text input override ───────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #0a0c12 !important;
    border: 1.5px solid #1a2035 !important;
    border-radius: 8px !important;
    color: #c8d4f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13.5px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3b6cf8 !important;
    box-shadow: 0 0 0 3px rgba(59,108,248,0.12) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label { display: none !important; }

/* ── Action buttons row ────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 7px 10px !important;
    border: 1.5px solid #1c2338 !important;
    background: #0d1018 !important;
    color: #7080a0 !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #131926 !important;
    border-color: #3b6cf8 !important;
    color: #90aaf0 !important;
}

/* ── Send button override ──────────────────────────────────────────────── */
.send-btn > button {
    background: #1a3470 !important;
    border-color: #2a4898 !important;
    color: #90b8ff !important;
}
.send-btn > button:hover {
    background: #1e3d85 !important;
    color: #b8d0ff !important;
}

/* ── Mic button ────────────────────────────────────────────────────────── */
.mic-btn > button {
    background: #14130f !important;
    border-color: #2a2010 !important;
    color: #a08040 !important;
}
.mic-btn > button:hover {
    background: #1a180c !important;
    border-color: #c07020 !important;
    color: #f0a040 !important;
}

/* ── Sliders ───────────────────────────────────────────────────────────── */
[data-testid="stSlider"] {
    padding: 6px 0 !important;
}
[data-testid="stSlider"] .stSlider > div > div {
    background: #1a1f2e !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #3b6cf8 !important;
    border-color: #3b6cf8 !important;
}
[data-testid="stSlider"] label {
    font-size: 11px !important;
    color: #4a5570 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── Selectbox ─────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] label {
    font-size: 11px !important;
    color: #4a5570 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
[data-testid="stSelectbox"] > div > div {
    background: #0a0c12 !important;
    border: 1.5px solid #1a2035 !important;
    border-radius: 8px !important;
    color: #c8d4f0 !important;
    font-size: 13px !important;
}

/* ── Image viewer ──────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #1a2035 !important;
}

/* ── Audio player ──────────────────────────────────────────────────────── */
[data-testid="stAudio"] audio {
    border-radius: 8px !important;
    width: 100% !important;
    background: #0a0c12 !important;
}

/* ── Download button ───────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: #0f1a12 !important;
    border: 1.5px solid #1a3020 !important;
    color: #50c878 !important;
    font-size: 12px !important;
    border-radius: 7px !important;
    width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #132018 !important;
    border-color: #50c878 !important;
}

/* ── Status / success / error ──────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 12.5px !important;
    padding: 10px 14px !important;
}

/* ── Caption ───────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] {
    font-size: 11px !important;
    color: #3a4060 !important;
    font-family: 'DM Mono', monospace !important;
    text-align: center;
    margin-top: 6px;
}

/* ── Info banner ───────────────────────────────────────────────────────── */
.welcome-banner {
    background: linear-gradient(135deg, #0d1528 0%, #0a0e1e 100%);
    border: 1px solid #1a2440;
    border-left: 3px solid #3b6cf8;
    border-radius: 10px;
    padding: 24px 28px;
    margin: 40px auto;
    max-width: 520px;
    text-align: center;
}
.welcome-banner h3 {
    font-size: 17px;
    font-weight: 600;
    color: #d0d8f0;
    margin: 0 0 8px;
}
.welcome-banner p {
    font-size: 13px;
    color: #4a5575;
    margin: 0;
    line-height: 1.6;
}

/* ── Viewer controls strip ─────────────────────────────────────────────── */
.viewer-controls {
    display: flex;
    gap: 12px;
    align-items: flex-end;
    flex-wrap: wrap;
    margin-bottom: 14px;
}

/* ── Misc ──────────────────────────────────────────────────────────────── */
.spacer { height: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key in ["chain", "vector_store", "pdf_files", "history", "current_page", "highlight"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key == "pdf_files" else [] if key == "history" else None


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">📄</div>
            <div class="sidebar-logo-text">Docent</div>
        </div>
        <div class="sidebar-tagline">AI Document Intelligence</div>
    </div>
    <div class="sidebar-body">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Upload Documents</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    st.markdown('<div class="process-btn">', unsafe_allow_html=True)
    process_clicked = st.button("⚡  Process Documents", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if process_clicked and files:
        with st.spinner("Indexing…"):
            docs   = load_pdfs_from_uploads(files)
            chunks = split_documents_into_chunks(docs)
            vs     = create_vector_store(chunks)
            st.session_state.vector_store = vs
            st.session_state.chain, _     = create_conversational_chain(vs)
            for f in files:
                st.session_state.pdf_files[f.name] = f.getvalue()
        st.success("Ready to query")

    if st.session_state.pdf_files:
        st.markdown('<div class="sidebar-section-label">Loaded Files</div>', unsafe_allow_html=True)
        for name in st.session_state.pdf_files:
            short = name if len(name) <= 28 else name[:25] + "…"
            st.markdown(f"""
            <div class="doc-badge">
                <div class="doc-dot"></div>
                <div class="doc-name">{short}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # close sidebar-body


# ─────────────────────────────────────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.chain:
    st.markdown("""
    <div class="welcome-banner">
        <h3>Welcome to Docent</h3>
        <p>Upload one or more PDF documents in the sidebar, then click
        <strong>Process Documents</strong> to start querying your files
        with AI-powered Q&amp;A, summaries, notes, and flashcards.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  QUERY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def process_query(q, mode="normal"):
    res = ask_question(st.session_state.chain, st.session_state.vector_store, q)
    ans = res["answer"]
    if mode == "summary":
        ans = "**Summary**\n\n" + ans[:500]
    elif mode == "notes":
        ans = "**Notes**\n\n• " + ans.replace(". ", "\n• ")
    elif mode == "flash":
        ans = f"**Flashcard**\n\n**Q:** {q}\n\n**A:** {ans}"
    return ans, res["sources"]


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 1])


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT — CHAT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Conversation</span>
            <span class="panel-badge">GPT-RAG</span>
        </div>
        <div class="panel-body">
    """, unsafe_allow_html=True)

    # ── Message list ──────────────────────────────────────────────────────────
    if not st.session_state.history:
        st.markdown("""
        <div class="chat-empty">
            <div class="chat-empty-icon">💬</div>
            <div class="chat-empty-text">Ask anything about your documents</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        for m in st.session_state.history:
            role_cls = "user" if m["role"] == "User" else "ai"
            st.markdown(f"""
            <div class="msg-row {role_cls}">
                <div class="bubble {role_cls}">{m["text"]}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    query = st.text_input("Ask", placeholder="Ask a question about your documents…", label_visibility="collapsed")

    # ── Action buttons ────────────────────────────────────────────────────────
    col_send, col_sum, col_notes, col_flash = st.columns(4)
    with col_send:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send = st.button("↑ Send", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_sum:
        summary_btn = st.button("⊞ Summary", use_container_width=True)
    with col_notes:
        notes_btn = st.button("≡ Notes", use_container_width=True)
    with col_flash:
        flash_btn = st.button("⚡ Cards", use_container_width=True)

    # ── Mic ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="mic-btn">', unsafe_allow_html=True)
    mic = st.button("🎤  Voice Input", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if mic:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                query = r.recognize_google(audio)
                st.success(f"Heard: {query}")
            except Exception:
                st.error("Could not recognise speech — please try again.")

    # ── Process ───────────────────────────────────────────────────────────────
    if send or summary_btn or notes_btn or flash_btn:
        if query:
            st.session_state.history.append({"role": "User", "text": query})

            mode = "normal"
            if summary_btn: mode = "summary"
            if notes_btn:   mode = "notes"
            if flash_btn:   mode = "flash"

            answer, source_docs = process_query(query, mode)

            # Streaming output
            placeholder = st.empty()
            streamed = ""
            for word in answer.split():
                streamed += word + " "
                placeholder.markdown(
                    f'<div class="msg-row ai"><div class="bubble ai">{streamed}</div></div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.01)

            st.session_state.history.append({"role": "AI", "text": streamed.strip()})

            # TTS
            with st.spinner("Generating audio…"):
                tts = gTTS(streamed)
                audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(audio_file.name)
            st.audio(audio_file.name)

            # PDF download
            pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            doc_pdf   = SimpleDocTemplate(pdf_path)
            styles    = getSampleStyleSheet()
            doc_pdf.build([Paragraph(streamed, styles["Normal"])])
            with open(pdf_path, "rb") as fh:
                st.download_button("⬇  Download Response as PDF", fh, "response.pdf", use_container_width=True)

            # Highlight
            if source_docs:
                st.session_state.highlight = source_docs[0].page_content[:100]

            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)   # close panel-body + panel


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT — PDF VIEWER PANEL
# ══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Document Viewer</span>
            <span class="panel-badge">PDF</span>
        </div>
        <div class="panel-body">
    """, unsafe_allow_html=True)

    names = list(st.session_state.pdf_files.keys())
    selected_file = st.selectbox("Select file", names, label_visibility="collapsed")

    pdf_bytes = st.session_state.pdf_files[selected_file]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    total_pages = len(doc)

    col_page, col_zoom = st.columns(2)
    with col_page:
        page_num = st.slider("Page", 1, total_pages, 1)
    with col_zoom:
        zoom_level = st.slider("Zoom", 1, 3, 1)

    pg = doc.load_page(page_num - 1)

    if st.session_state.highlight:
        try:
            areas = pg.search_for(st.session_state.highlight[:50])
            for area in areas:
                pg.add_highlight_annot(area)
        except Exception:
            pass

    pix     = pg.get_pixmap(matrix=fitz.Matrix(zoom_level, zoom_level))
    img_out = tmp_path + ".png"
    pix.save(img_out)

    st.image(img_out, use_column_width=True)
    st.caption(f"Page {page_num} of {total_pages}  ·  {selected_file}")

    st.markdown('</div></div>', unsafe_allow_html=True)   # close panel-body + panel