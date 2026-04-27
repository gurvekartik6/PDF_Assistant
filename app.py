import os
import tempfile
import warnings
import streamlit as st
import fitz

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from pdf_loader import load_pdfs_from_uploads, split_documents_into_chunks
from vector_store import create_vector_store
from qa_chain import (create_conversational_chain, ask_question,
                      ask_question_stream, parse_flashcards, detect_intent)
from utils import initialize_session_state, format_file_size
from voice import text_to_speech, speech_to_text_from_microphone

load_dotenv()

st.set_page_config(
    page_title="DocuMentor · AI PDF Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ═══ RESET ═══════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: #f9fafb !important;
    color: #111827 !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.main .block-container {
    padding: 1.25rem 1.5rem 1.5rem !important;
    max-width: 100% !important;
}

/* ═══ SIDEBAR ══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    min-width: 240px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

.sb-brand {
    padding: 18px 16px 14px;
    border-bottom: 1px solid #f3f4f6;
    display: flex; align-items: center; gap: 10px;
}
.sb-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.sb-name { font-size: 15px; font-weight: 700; color: #111827; letter-spacing: -.3px; }
.sb-tag  { font-size: 10px; color: #9ca3af; margin-top: 1px; }

.sb-inner { padding: 10px 14px 20px; }
.sb-label {
    font-size: 10px; font-weight: 600; letter-spacing: .9px;
    text-transform: uppercase; color: #9ca3af; margin: 16px 0 7px 2px;
}

/* uploader */
[data-testid="stFileUploader"] {
    background: #f9fafb !important;
    border: 1.5px dashed #d1d5db !important;
    border-radius: 10px !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #2563eb !important; }
section[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* process button */
.proc-btn > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: #fff !important; border: none !important;
    border-radius: 9px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    padding: 10px 0 !important; width: 100% !important;
    box-shadow: 0 2px 10px rgba(37,99,235,.3) !important;
    transition: opacity .2s !important; cursor: pointer !important;
}
.proc-btn > button:hover { opacity: .88 !important; }

/* file badge */
.file-badge {
    display: flex; align-items: center; gap: 8px;
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 7px 10px; margin: 4px 0;
    font-size: 12px; color: #374151;
}
.file-dot { width: 7px; height: 7px; background: #10b981; border-radius: 50%; flex-shrink: 0; }
.file-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-size { font-size: 10px; color: #9ca3af; }

/* ═══ TABS ══════════════════════════════════════════════════ */
.tab-row {
    display: flex; gap: 4px;
    background: #f3f4f6;
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 16px;
}

/* override streamlit button inside tab-row */
div[data-tab-active="true"] > button,
.tab-active > button {
    background: #ffffff !important;
    color: #1d4ed8 !important;
    border-color: transparent !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.08) !important;
    border-radius: 7px !important;
}

.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 7px 12px !important;
    border: 1px solid #e5e7eb !important;
    background: #ffffff !important;
    color: #374151 !important;
    transition: all .15s !important;
    width: 100% !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #eff6ff !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
}

/* ═══ CHAT ══════════════════════════════════════════════════ */
.chat-wrap {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    overflow: hidden;
    display: flex; flex-direction: column;
}
.chat-header {
    padding: 13px 18px;
    border-bottom: 1px solid #f3f4f6;
    display: flex; align-items: center; justify-content: space-between;
    background: #fafafa;
}
.chat-header-title { font-size: 13px; font-weight: 600; color: #374151; }
.chat-header-badge {
    font-size: 11px; background: #eff6ff;
    border: 1px solid #bfdbfe; color: #1d4ed8;
    border-radius: 20px; padding: 2px 10px;
    font-weight: 500;
}

.chat-body {
    min-height: 360px;
    max-height: 420px;
    overflow-y: auto;
    padding: 20px 18px 10px;
    scrollbar-width: thin;
    scrollbar-color: #e5e7eb transparent;
}
.chat-body::-webkit-scrollbar { width: 4px; }
.chat-body::-webkit-scrollbar-thumb { background: #e5e7eb; border-radius: 4px; }

/* empty */
.chat-empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 320px; gap: 10px; color: #9ca3af;
}
.chat-empty-icon { font-size: 40px; opacity: .5; }
.chat-empty-text { font-size: 14px; font-weight: 500; }
.chat-empty-sub  { font-size: 12px; color: #d1d5db; }

/* user message */
.msg-user {
    display: flex; justify-content: flex-end; margin: 8px 0;
}
.bubble-user {
    background: #2563eb;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 15px;
    font-size: 14px; line-height: 1.6;
    max-width: 78%; word-wrap: break-word;
}

/* AI message */
.msg-ai {
    display: flex; align-items: flex-start;
    gap: 10px; margin: 8px 0;
}
.ai-avatar {
    width: 28px; height: 28px; flex-shrink: 0;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; color: #fff; margin-top: 2px;
}
.bubble-ai {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 4px 16px 16px 16px;
    padding: 10px 15px;
    font-size: 14px; line-height: 1.7;
    color: #111827; max-width: 85%;
    word-wrap: break-word; white-space: pre-wrap;
}

/* ═══ INPUT BAR ═════════════════════════════════════════════ */
.chat-input-area {
    padding: 12px 14px;
    border-top: 1px solid #f3f4f6;
    background: #fafafa;
}
.input-row {
    display: flex; align-items: center; gap: 8px;
    background: #ffffff;
    border: 1.5px solid #d1d5db;
    border-radius: 12px;
    padding: 6px 8px 6px 14px;
    transition: border-color .2s, box-shadow .2s;
}
.input-row:focus-within {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37,99,235,.1);
}

/* force text input transparent inside the row */
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 4px 0 !important;
    caret-color: #2563eb !important;
}
[data-testid="stTextInput"] [data-baseweb="input"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] label { display: none !important; }

/* send button */
.send-btn > button {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 7px 14px !important;
    font-size: 13px !important;
    width: auto !important;
    flex-shrink: 0;
}
.send-btn > button:hover { background: #1d4ed8 !important; border-color: #1d4ed8 !important; }

.icon-btn > button {
    background: #f9fafb !important;
    border-color: #e5e7eb !important;
    color: #6b7280 !important;
    border-radius: 8px !important;
    padding: 7px !important;
    font-size: 14px !important;
    width: 36px !important; min-width: 36px !important;
}
.icon-btn > button:hover { background: #eff6ff !important; border-color: #2563eb !important; color: #2563eb !important; }

/* ═══ CONTENT PANELS (summary/notes/flashcards) ═════════════ */
.content-panel {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    overflow: hidden;
}
.content-panel-header {
    padding: 13px 18px;
    border-bottom: 1px solid #f3f4f6;
    background: #fafafa;
    display: flex; align-items: center; justify-content: space-between;
}
.content-panel-title { font-size: 13px; font-weight: 600; color: #374151; }
.content-panel-body { padding: 20px 22px; }

.gen-btn > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: #fff !important; border: none !important;
    border-radius: 9px !important;
    font-size: 13px !important; font-weight: 600 !important;
    padding: 10px 24px !important;
    width: auto !important;
    box-shadow: 0 2px 10px rgba(37,99,235,.25) !important;
}
.gen-btn > button:hover { opacity: .88 !important; }

.regen-btn > button {
    background: #f9fafb !important;
    border-color: #e5e7eb !important;
    color: #374151 !important;
    width: auto !important;
}

.content-text {
    font-size: 14px; line-height: 1.8;
    color: #1f2937;
    white-space: pre-wrap; word-wrap: break-word;
}

/* flashcards */
.fc-card {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: border-color .15s, box-shadow .15s;
}
.fc-card:hover { border-color: #2563eb; box-shadow: 0 2px 12px rgba(37,99,235,.1); }
.fc-num { font-size: 10px; color: #9ca3af; font-weight: 600; margin-bottom: 5px; letter-spacing: .5px; }
.fc-q { font-size: 14px; font-weight: 600; color: #111827; }
.fc-a { font-size: 13px; color: #2563eb; margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb; }

/* ═══ PDF VIEWER ════════════════════════════════════════════ */
.viewer-wrap {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    overflow: hidden;
}
.viewer-head {
    padding: 13px 18px;
    border-bottom: 1px solid #f3f4f6;
    background: #fafafa;
    display: flex; align-items: center; justify-content: space-between;
}
.viewer-head-title { font-size: 13px; font-weight: 600; color: #374151; }
.viewer-badge {
    font-size: 11px; background: #fef3c7;
    border: 1px solid #fde68a; color: #92400e;
    border-radius: 20px; padding: 2px 10px; font-weight: 500;
}
.viewer-body { padding: 14px 16px; }

/* sliders */
[data-testid="stSlider"] label {
    font-size: 11px !important; font-weight: 600 !important;
    color: #9ca3af !important; text-transform: uppercase; letter-spacing: .7px;
}
[data-testid="stSlider"] > div > div > div {
    background: #e5e7eb !important;
}

/* selectbox */
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stSelectbox"] > div > div {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #111827 !important; font-size: 13px !important;
}

/* image */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 1px 6px rgba(0,0,0,.06) !important;
}

/* caption */
[data-testid="stCaptionContainer"] {
    font-size: 11px !important; color: #9ca3af !important; text-align: center; margin-top: 5px;
}

/* download button */
[data-testid="stDownloadButton"] > button {
    background: #f9fafb !important; border: 1px solid #e5e7eb !important;
    color: #374151 !important; font-size: 12px !important;
    border-radius: 8px !important; width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #2563eb !important; color: #2563eb !important; background: #eff6ff !important;
}

/* audio */
[data-testid="stAudio"] audio { border-radius: 8px !important; width: 100% !important; }

/* alerts */
[data-testid="stAlert"] { border-radius: 9px !important; font-size: 13px !important; }

/* ═══ WELCOME ════════════════════════════════════════════════ */
.welcome-outer {
    display: flex; align-items: center; justify-content: center;
    min-height: 75vh;
}
.welcome-card {
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 18px; padding: 44px 52px;
    text-align: center; max-width: 480px; width: 100%;
    box-shadow: 0 4px 28px rgba(0,0,0,.07);
}
.welcome-logo {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    border-radius: 14px; margin: 0 auto 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
}
.welcome-title { font-size: 22px; font-weight: 700; color: #111827; margin-bottom: 10px; }
.welcome-desc  { font-size: 14px; color: #6b7280; line-height: 1.7; margin-bottom: 24px; }
.welcome-steps { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.step-chip {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 20px; padding: 5px 14px;
    font-size: 12px; color: #1d4ed8; font-weight: 500;
}

/* toggle */
[data-testid="stToggle"] label { font-size: 13px !important; color: #374151 !important; }
</style>
""", unsafe_allow_html=True)

# ── session state ──────────────────────────────────────────────────────────
initialize_session_state(st)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo">📘</div>
        <div>
            <div class="sb-name">DocuMentor</div>
            <div class="sb-tag">AI DOCUMENT INTELLIGENCE</div>
        </div>
    </div>
    <div class="sb-inner">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Upload Documents</div>', unsafe_allow_html=True)
    files = st.file_uploader("PDFs", type=["pdf"],
                              accept_multiple_files=True, label_visibility="collapsed")

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="proc-btn">', unsafe_allow_html=True)
    process_clicked = st.button("⚡  Process Documents", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if process_clicked and files:
        with st.spinner("Indexing…"):
            docs = load_pdfs_from_uploads(files)
            if docs:
                chunks = split_documents_into_chunks(docs)
                vs = create_vector_store(chunks)
                st.session_state.vector_store = vs
                st.session_state.chain, _ = create_conversational_chain(vs)
                for f in files:
                    f.seek(0)
                    st.session_state.pdf_files[f.name] = f.read()
                st.session_state.pdfs_processed = True
                st.session_state.chat_history   = []
                st.success(f"✅ Ready — {len(docs)} pages indexed")
            else:
                st.error("Could not extract text from PDFs.")
    elif process_clicked:
        st.warning("Upload at least one PDF first.")

    if st.session_state.pdf_files:
        st.markdown('<div class="sb-label">Loaded Files</div>', unsafe_allow_html=True)
        for name, data in st.session_state.pdf_files.items():
            short = name if len(name) <= 25 else name[:22] + "…"
            st.markdown(f"""
            <div class="file-badge">
                <div class="file-dot"></div>
                <div class="file-name">{short}</div>
                <span class="file-size">{format_file_size(len(data))}</span>
            </div>""", unsafe_allow_html=True)

    if st.session_state.pdfs_processed:
        st.markdown('<div class="sb-label">Voice</div>', unsafe_allow_html=True)
        voice_on = st.toggle("Enable Text-to-Speech",
                             value=st.session_state.get("voice_mode", False))
        st.session_state.voice_mode = voice_on
        if voice_on:
            from voice import get_supported_languages
            langs = get_supported_languages()
            sel_lang = st.selectbox("Language", list(langs.keys()),
                                     format_func=lambda x: langs[x])
            st.session_state.tts_language = sel_lang

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.chain:
    st.markdown("""
    <div class="welcome-outer">
        <div class="welcome-card">
            <div class="welcome-logo">📘</div>
            <div class="welcome-title">Welcome to DocuMentor</div>
            <div class="welcome-desc">
                Upload your PDF documents and have an intelligent conversation.
                Get answers, summaries, study notes, and flashcards instantly.
            </div>
            <div class="welcome-steps">
                <span class="step-chip">① Upload PDFs</span>
                <span class="step-chip">② Process</span>
                <span class="step-chip">③ Ask Away</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1.15, 0.85], gap="large")

# ──────────────────────────────────────────────────────────────────────────────
#  LEFT — Tabs + Content
# ──────────────────────────────────────────────────────────────────────────────
with left_col:

    # ── Tab selector ──────────────────────────────────────────────────────────
    TABS = [
        ("chat",       "💬  Chat"),
        ("summary",    "📋  Summary"),
        ("notes",      "📝  Notes"),
        ("flashcards", "⚡  Flashcards"),
    ]
    tab_cols = st.columns(len(TABS))
    for (key, label), col in zip(TABS, tab_cols):
        with col:
            is_active = st.session_state.active_tab == key
            wrap_cls  = "tab-active" if is_active else ""
            st.markdown(f'<div class="{wrap_cls}">', unsafe_allow_html=True)
            if st.button(label, key=f"tab_{key}", use_container_width=True):
                st.session_state.active_tab = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    active = st.session_state.active_tab

    # ══════════════════════════════════════════════════════════════════════════
    #  CHAT TAB
    # ══════════════════════════════════════════════════════════════════════════
    if active == "chat":
        # header
        st.markdown("""
        <div class="chat-wrap">
          <div class="chat-header">
            <span class="chat-header-title">💬 Conversation</span>
            <span class="chat-header-badge">RAG · Streaming</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # messages
        history = st.session_state.chat_history
        if not history:
            st.markdown("""
            <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
                        border-radius:0 0 14px 14px;padding:0">
              <div class="chat-empty">
                <div class="chat-empty-icon">💬</div>
                <div class="chat-empty-text">Ask anything about your documents</div>
                <div class="chat-empty-sub">Powered by RAG · Streaming responses</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            html_msgs = ""
            for msg in history:
                if msg["role"] == "user":
                    html_msgs += f"""
                    <div class="msg-user">
                        <div class="bubble-user">{msg["content"]}</div>
                    </div>"""
                else:
                    content = msg["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
                    html_msgs += f"""
                    <div class="msg-ai">
                        <div class="ai-avatar">✦</div>
                        <div class="bubble-ai">{content}</div>
                    </div>"""
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
                        border-radius:0 0 14px 14px;">
              <div class="chat-body">{html_msgs}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        # ── Input row ─────────────────────────────────────────────────────────
        i1, i2, i3, i4 = st.columns([7, 1.4, 0.9, 0.9])
        with i1:
            query = st.text_input("msg", placeholder="Message DocuMentor…",
                                   key="chat_input", label_visibility="collapsed")
        with i2:
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            send_btn = st.button("Send ↑", key="send", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with i3:
            st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
            mic_btn = st.button("🎤", key="mic", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with i4:
            st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
            clr_btn = st.button("🗑️", key="clr", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if mic_btn:
            with st.spinner("🎤 Listening…"):
                txt, msg_str = speech_to_text_from_microphone()
            if txt:
                st.success(f"Heard: {txt}")
                query = txt
            else:
                st.warning(msg_str)

        if clr_btn:
            st.session_state.chat_history = []
            st.rerun()

        # ── SEND — streaming ─────────────────────────────────────────────────
        if send_btn and query:
            st.session_state.chat_history.append({"role": "user", "content": query})

            # show user bubble right away
            user_safe = query.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            st.markdown(f"""
            <div class="msg-user" style="margin-top:8px">
                <div class="bubble-user">{user_safe}</div>
            </div>""", unsafe_allow_html=True)

            # stream AI answer
            st.markdown("""
            <div class="msg-ai" style="margin-top:6px">
                <div class="ai-avatar">✦</div>""", unsafe_allow_html=True)

            placeholder = st.empty()
            streamed = ""
            for token in ask_question_stream(st.session_state.vector_store, query):
                streamed += token
                safe = streamed.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
                placeholder.markdown(
                    f'<div class="bubble-ai">{safe}▌</div>',
                    unsafe_allow_html=True
                )
            safe_final = streamed.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            placeholder.markdown(
                f'<div class="bubble-ai">{safe_final}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": streamed})

            # update viewer page
            try:
                docs = st.session_state.vector_store.similarity_search(query, k=1)
                if docs:
                    st.session_state.highlight    = docs[0].page_content[:80]
                    st.session_state.current_page = docs[0].metadata.get("page", 0) + 1
            except Exception:
                pass

            if st.session_state.get("voice_mode") and streamed:
                ab = text_to_speech(streamed, language=st.session_state.get("tts_language","en"))
                if ab:
                    st.audio(ab, format="audio/mp3")

            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY TAB
    # ══════════════════════════════════════════════════════════════════════════
    elif active == "summary":
        st.markdown("""
        <div class="content-panel">
          <div class="content-panel-header">
            <span class="content-panel-title">📋 Document Summary</span>
          </div>
          <div class="content-panel-body">
        """, unsafe_allow_html=True)

        if st.session_state.last_summary is None:
            st.markdown("Generate an AI-powered summary of your entire document.")
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("📋 Generate Summary", key="gen_sum"):
                with st.spinner("Analysing document…"):
                    res = ask_question(st.session_state.chain, st.session_state.vector_store,
                                       "Please provide a comprehensive summary of this document")
                    st.session_state.last_summary = res.get("answer","")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="content-text">{st.session_state.last_summary}</div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="regen-btn">', unsafe_allow_html=True)
                if st.button("🔄 Regenerate", use_container_width=True, key="regen_sum"):
                    st.session_state.last_summary = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.download_button("⬇ Download .txt",
                                   st.session_state.last_summary.encode(),
                                   "summary.txt", use_container_width=True)
            if st.session_state.get("voice_mode"):
                ab = text_to_speech(st.session_state.last_summary,
                                    st.session_state.get("tts_language","en"))
                if ab: st.audio(ab, format="audio/mp3")

        st.markdown('</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  NOTES TAB
    # ══════════════════════════════════════════════════════════════════════════
    elif active == "notes":
        st.markdown("""
        <div class="content-panel">
          <div class="content-panel-header">
            <span class="content-panel-title">📝 Study Notes</span>
          </div>
          <div class="content-panel-body">
        """, unsafe_allow_html=True)

        if st.session_state.last_notes is None:
            st.markdown("Generate structured study notes with key concepts and important points.")
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("📝 Generate Notes", key="gen_notes"):
                with st.spinner("Creating study notes…"):
                    res = ask_question(st.session_state.chain, st.session_state.vector_store,
                                       "Create detailed study notes from this document")
                    st.session_state.last_notes = res.get("answer","")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="content-text">{st.session_state.last_notes}</div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="regen-btn">', unsafe_allow_html=True)
                if st.button("🔄 Regenerate", use_container_width=True, key="regen_notes"):
                    st.session_state.last_notes = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.download_button("⬇ Download .txt",
                                   st.session_state.last_notes.encode(),
                                   "notes.txt", use_container_width=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  FLASHCARDS TAB
    # ══════════════════════════════════════════════════════════════════════════
    elif active == "flashcards":
        st.markdown("""
        <div class="content-panel">
          <div class="content-panel-header">
            <span class="content-panel-title">⚡ Flashcards</span>
          </div>
          <div class="content-panel-body">
        """, unsafe_allow_html=True)

        cards = st.session_state.flashcards
        if not cards:
            st.markdown("Generate Q&A flashcards for studying your document.")
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("⚡ Generate Flashcards", key="gen_fc"):
                with st.spinner("Creating flashcards…"):
                    res    = ask_question(st.session_state.chain, st.session_state.vector_store,
                                         "Generate flashcards from this document")
                    raw    = res.get("answer","")
                    parsed = parse_flashcards(raw)
                    st.session_state.flashcards = parsed or [{"question":"Output","answer":raw}]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(cards)} flashcard(s)** — expand each to see the answer")
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            for i, card in enumerate(cards, 1):
                with st.expander(f"Card {i}  ·  {card['question'][:65]}…"):
                    st.markdown(f"**Q:** {card['question']}")
                    st.divider()
                    st.markdown(f"**A:** {card['answer']}")
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="regen-btn">', unsafe_allow_html=True)
            if st.button("🔄 Regenerate Flashcards", key="regen_fc"):
                st.session_state.flashcards = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  RIGHT — PDF Viewer
# ──────────────────────────────────────────────────────────────────────────────
with right_col:
    st.markdown("""
    <div class="viewer-wrap">
      <div class="viewer-head">
        <span class="viewer-head-title">📄 Document Viewer</span>
        <span class="viewer-badge">PDF</span>
      </div>
    </div>""", unsafe_allow_html=True)

    names = list(st.session_state.pdf_files.keys())
    if not names:
        st.info("No documents loaded.")
    else:
        st.markdown('<div class="viewer-body">', unsafe_allow_html=True)

        selected  = st.selectbox("File", names, label_visibility="collapsed")
        pdf_bytes = st.session_state.pdf_files[selected]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        doc         = fitz.open(tmp_path)
        total_pages = len(doc)

        pg_col, zm_col = st.columns(2)
        with pg_col:
            default_pg = min(int(st.session_state.current_page or 1), total_pages)
            page_num   = st.slider("Page", 1, total_pages, default_pg)
        with zm_col:
            zoom = st.slider("Zoom", 1, 3, 2)

        pg = doc.load_page(page_num - 1)

        if st.session_state.highlight:
            try:
                for area in pg.search_for(st.session_state.highlight[:50]):
                    pg.add_highlight_annot(area)
            except Exception:
                pass

        pix      = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_path = tmp_path + f"_p{page_num}z{zoom}.png"
        pix.save(img_path)

        st.image(img_path, use_column_width=True)
        st.caption(f"Page {page_num} / {total_pages}  ·  {selected}")

        # download single page
        sp = tmp_path + "_sp.pdf"
        sd = fitz.open()
        sd.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
        sd.save(sp); sd.close()
        with open(sp,"rb") as fh:
            st.download_button("⬇ Download page", fh,
                               f"{selected}_p{page_num}.pdf",
                               mime="application/pdf",
                               use_container_width=True)
        doc.close()
        st.markdown('</div>', unsafe_allow_html=True)