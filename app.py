"""
app.py - Docent AI PDF Assistant
Light theme, fully working: chat, summary, notes, flashcards, PDF viewer, voice, zoom.
"""

import os
import tempfile
import warnings
import streamlit as st
import fitz  # PyMuPDF

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from pdf_loader import load_pdfs_from_uploads, split_documents_into_chunks
from vector_store import create_vector_store
from qa_chain import create_conversational_chain, ask_question, ask_question_stream, parse_flashcards, detect_intent
from utils import initialize_session_state, format_file_size
from voice import text_to_speech, speech_to_text_from_microphone, check_voice_availability

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMentor · PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  LIGHT THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background: #f5f6fa !important;
    color: #1a1d27 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Main content padding ── */
[data-testid="stAppViewContainer"] > section[data-testid="stSidebar"] + div {
    padding-top: 0 !important;
}
.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e8eaf0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── Sidebar brand header ── */
.sb-header {
    background: linear-gradient(135deg, #4f6ef7 0%, #7c4dff 100%);
    padding: 24px 20px 18px;
    margin-bottom: 4px;
}
.sb-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.sb-icon {
    width: 34px; height: 34px;
    background: rgba(255,255,255,0.2);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
}
.sb-title {
    font-size: 19px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.3px;
}
.sb-subtitle {
    font-size: 11px;
    color: rgba(255,255,255,0.65);
    letter-spacing: 0.6px;
    text-transform: uppercase;
}
.sb-body { padding: 8px 16px 20px; }
.sb-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #9ea5bc;
    margin: 18px 0 8px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #f8f9fc !important;
    border: 1.5px dashed #c8cde0 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4f6ef7 !important;
    background: #f0f3ff !important;
}

/* ── Process button ── */
.proc-btn > button {
    background: linear-gradient(135deg, #4f6ef7 0%, #7c4dff 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    width: 100% !important;
    box-shadow: 0 3px 14px rgba(79,110,247,0.3) !important;
    transition: opacity 0.2s !important;
}
.proc-btn > button:hover { opacity: 0.88 !important; }

/* ── Doc badge ── */
.doc-badge {
    display: flex; align-items: center; gap: 8px;
    background: #f0f3ff;
    border: 1px solid #dce1f5;
    border-radius: 8px;
    padding: 8px 11px;
    margin: 4px 0;
    font-size: 12px;
    color: #4a5280;
}
.doc-dot { width: 7px; height: 7px; background: #22c55e; border-radius: 50%; flex-shrink: 0; }
.doc-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── PANEL CARD ── */
.panel {
    background: #ffffff;
    border: 1.5px solid #e8eaf0;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 16px;
}
.panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1.5px solid #f0f2f8;
    background: #fafbff;
}
.panel-title {
    font-size: 12px; font-weight: 700;
    color: #7880a0;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.panel-badge {
    font-size: 11px;
    background: #eef1ff;
    border: 1px solid #d0d6f5;
    color: #4f6ef7;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
}
.panel-body { padding: 16px 18px; }

/* ── CHAT MESSAGES ── */
.chat-scroll {
    max-height: 400px;
    overflow-y: auto;
    padding: 4px 0 10px;
    scrollbar-width: thin;
    scrollbar-color: #d0d5e8 transparent;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-thumb { background: #d0d5e8; border-radius: 4px; }

.msg-row { display: flex; margin: 7px 0; gap: 8px; }
.msg-row.user { justify-content: flex-end; }
.msg-row.ai { justify-content: flex-start; }

.bubble {
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 13.5px;
    line-height: 1.65;
    position: relative;
}
.bubble.user {
    background: linear-gradient(135deg, #4f6ef7 0%, #6b55f0 100%);
    color: #fff;
    border-bottom-right-radius: 3px;
}
.bubble.ai {
    background: #f4f5f9;
    border: 1px solid #e2e5f0;
    color: #2a2e42;
    border-bottom-left-radius: 3px;
    white-space: pre-wrap;
}
.ai-label {
    font-size: 10px;
    font-weight: 700;
    color: #4f6ef7;
    letter-spacing: 0.5px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}

.chat-empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 240px; gap: 10px;
    color: #b0b8d0;
}
.chat-empty-icon { font-size: 38px; opacity: 0.5; }
.chat-empty-text { font-size: 13px; }

/* ── DIVIDER ── */
.divider { height: 1px; background: #eef0f8; margin: 12px 0; }

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] input {
    background: #f8f9fc !important;
    border: 1.5px solid #d8dce8 !important;
    border-radius: 9px !important;
    color: #1a1d27 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #4f6ef7 !important;
    box-shadow: 0 0 0 3px rgba(79,110,247,0.1) !important;
    outline: none !important;
    background: #fff !important;
}
[data-testid="stTextInput"] label { display: none !important; }

/* ── BUTTONS (general) ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 7px 10px !important;
    border: 1.5px solid #dde0ef !important;
    background: #f4f5fb !important;
    color: #5060a0 !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #eef1ff !important;
    border-color: #4f6ef7 !important;
    color: #4f6ef7 !important;
}

/* ── Send button ── */
.send-btn > button {
    background: #4f6ef7 !important;
    border-color: #4f6ef7 !important;
    color: #fff !important;
    font-weight: 600 !important;
}
.send-btn > button:hover {
    background: #3d5de0 !important;
    border-color: #3d5de0 !important;
}

/* ── Tab buttons (active) ── */
.tab-active > button {
    background: #eef1ff !important;
    border-color: #4f6ef7 !important;
    color: #4f6ef7 !important;
    font-weight: 600 !important;
}

/* ── Mic button ── */
.mic-btn > button {
    background: #fff8f0 !important;
    border-color: #f0d8a8 !important;
    color: #c07020 !important;
}
.mic-btn > button:hover {
    background: #fff3e0 !important;
    border-color: #e09030 !important;
    color: #e07010 !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] label {
    font-size: 11px !important;
    color: #8090b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label {
    font-size: 11px !important;
    color: #8090b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600 !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #f8f9fc !important;
    border: 1.5px solid #d8dce8 !important;
    border-radius: 9px !important;
    color: #2a2e42 !important;
    font-size: 13px !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #e0e3ef !important;
}

/* ── Audio ── */
[data-testid="stAudio"] audio {
    border-radius: 8px !important;
    width: 100% !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #f0faf3 !important;
    border: 1.5px solid #b8e8c8 !important;
    color: #1a8040 !important;
    font-size: 12px !important;
    border-radius: 8px !important;
    width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #e4f7eb !important;
    border-color: #22c55e !important;
}

/* ── Status messages ── */
[data-testid="stAlert"] {
    border-radius: 9px !important;
    font-size: 12.5px !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
    font-size: 11px !important;
    color: #8090b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-align: center;
    margin-top: 6px;
}

/* ── FLASHCARD ── */
.fc-card {
    background: #fafbff;
    border: 1.5px solid #dde2f5;
    border-radius: 12px;
    padding: 16px 18px;
    margin: 8px 0;
    cursor: pointer;
    transition: all 0.2s;
}
.fc-card:hover { border-color: #4f6ef7; background: #f4f6ff; }
.fc-q {
    font-size: 13.5px; font-weight: 600;
    color: #2a2e42; margin-bottom: 6px;
}
.fc-a {
    font-size: 13px; color: #4f6ef7;
    font-style: italic;
    border-top: 1px solid #e4e8f5;
    padding-top: 8px; margin-top: 4px;
}
.fc-num {
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: #9ea5bc; margin-bottom: 6px;
}

/* ── Welcome banner ── */
.welcome-banner {
    background: linear-gradient(135deg, #f0f3ff 0%, #faf0ff 100%);
    border: 1.5px solid #d8dff5;
    border-left: 4px solid #4f6ef7;
    border-radius: 12px;
    padding: 28px 32px;
    margin: 60px auto;
    max-width: 540px;
    text-align: center;
}
.welcome-banner h3 {
    font-size: 20px; font-weight: 700;
    color: #1a1d27; margin-bottom: 10px;
}
.welcome-banner p {
    font-size: 14px; color: #6070a0; line-height: 1.65;
}
.welcome-steps {
    display: flex; gap: 12px; justify-content: center;
    flex-wrap: wrap; margin-top: 18px;
}
.step-pill {
    background: #fff;
    border: 1.5px solid #d0d6f0;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #5060a0;
    font-weight: 500;
}
.spacer { height: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
initialize_session_state(st)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-header">
        <div class="sb-brand">
            <div class="sb-title">DocuMentor</div>
        </div>
        <div class="sb-subtitle">Document Intelligence</div>
    </div>
    <div class="sb-body">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Upload Documents</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="proc-btn">', unsafe_allow_html=True)
    process_clicked = st.button("Process Documents", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if process_clicked and files:
        with st.spinner("Indexing documents..."):
            docs   = load_pdfs_from_uploads(files)
            if docs:
                chunks = split_documents_into_chunks(docs)
                vs     = create_vector_store(chunks)
                st.session_state.vector_store = vs
                st.session_state.chain, _     = create_conversational_chain(vs)
                for f in files:
                    f.seek(0)
                    st.session_state.pdf_files[f.name] = f.read()
                st.session_state.pdfs_processed = True
                st.session_state.chat_history = []
                st.success(f"✅ Ready! Indexed {len(docs)} pages.")
            else:
                st.error("Could not extract text from PDFs.")
    elif process_clicked and not files:
        st.warning("Please upload at least one PDF first.")

    if st.session_state.pdf_files:
        st.markdown('<div class="sb-label">Loaded Files</div>', unsafe_allow_html=True)
        for name, data in st.session_state.pdf_files.items():
            short = name if len(name) <= 26 else name[:23] + "…"
            size  = format_file_size(len(data))
            st.markdown(f"""
            <div class="doc-badge">
                <div class="doc-dot"></div>
                <div class="doc-name">{short}</div>
                <span style="font-size:10px;color:#9ea5bc">{size}</span>
            </div>""", unsafe_allow_html=True)

    # Voice settings
    if st.session_state.pdfs_processed:
        st.markdown('<div class="sb-label">Voice Settings</div>', unsafe_allow_html=True)
        voice_on = st.toggle("Enable TTS", value=False)
        if voice_on:
            from voice import get_supported_languages
            langs = get_supported_languages()
            tts_lang = st.selectbox("Language", list(langs.keys()),
                                     format_func=lambda x: langs[x])
            st.session_state.tts_language = tts_lang
        st.session_state.voice_mode = voice_on

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.chain:
    st.markdown("""
    <div class="welcome-banner">
        <h3>Welcome to DocuMentor</h3>
        <p>Upload your PDF documents and chat with them using AI.
        Get instant answers, summaries, study notes, and flashcards.</p>
        <div class="welcome-steps">
            <span class="step-pill">1 · Upload PDFs</span>
            <span class="step-pill">2 · Process Documents</span>
            <span class="step-pill">3 · Ask Questions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LAYOUT  (left = chat/content | right = PDF viewer)
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1])


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — Chat + Features
# ══════════════════════════════════════════════════════════════════════════════
with left:

    # ── TAB NAV ──────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.columns(4)
    tabs = {
        "chat":       ("Chat",      t1),
        "summary":    ("Summary",   t2),
        "notes":      ("Notes",     t3),
        "flashcards": ("Flashcards", t4),
    }
    for key, (label, col) in tabs.items():
        with col:
            is_active = st.session_state.active_tab == key
            div_cls = "tab-active" if is_active else ""
            st.markdown(f'<div class="{div_cls}">', unsafe_allow_html=True)
            if st.button(label, key=f"tab_{key}", use_container_width=True):
                st.session_state.active_tab = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    active = st.session_state.active_tab

    # ─────────────────────────────────────────────────────────────────────────
    #  CHAT TAB
    # ─────────────────────────────────────────────────────────────────────────
    if active == "chat":
        st.markdown("""<div class="panel">
            <div class="panel-header">
                <span class="panel-title">Conversation</span>
                <span class="panel-badge">RAG · AI</span>
            </div><div class="panel-body">""", unsafe_allow_html=True)

        # Messages
        history = st.session_state.chat_history
        if not history:
            st.markdown("""<div class="chat-empty">
                <div class="chat-empty-icon">💬</div>
                <div class="chat-empty-text">Ask anything about your documents</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(f"""<div class="msg-row user">
                        <div class="bubble user">{msg["content"]}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    content = msg["content"].replace("\n", "<br>")
                    st.markdown(f"""<div class="msg-row ai">
                        <div style="display:flex;flex-direction:column;max-width:82%;">
                            <div class="ai-label">AI · DOCENT</div>
                            <div class="bubble ai">{content}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Input
        query = st.text_input("Ask", placeholder="Ask a question about your documents…",
                               key="chat_input", label_visibility="collapsed")

        col_send, col_clr, col_mic = st.columns([2, 1, 1])
        with col_send:
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            send_btn = st.button("↑ Send", use_container_width=True, key="send")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_clr:
            clear_btn = st.button("🗑 Clear", use_container_width=True, key="clear")
        with col_mic:
            st.markdown('<div class="mic-btn">', unsafe_allow_html=True)
            mic_btn = st.button("🎤 Voice", use_container_width=True, key="mic")
            st.markdown('</div>', unsafe_allow_html=True)

        # Voice input
        if mic_btn:
            with st.spinner("🎤 Listening…"):
                text, msg = speech_to_text_from_microphone()
            if text:
                st.success(f"Heard: {text}")
                query = text
            else:
                st.warning(msg)

        # Clear history
        if clear_btn:
            st.session_state.chat_history = []
            st.rerun()

        # Send message — streamed for instant first-token response
        if send_btn and query:
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Render user bubble immediately
            st.markdown(f"""<div class="msg-row user">
                <div class="bubble user">{query}</div>
            </div>""", unsafe_allow_html=True)

            # Stream AI response token-by-token
            st.markdown('<div class="msg-row ai"><div style="display:flex;flex-direction:column;max-width:82%;"><div class="ai-label">AI · DOCENT</div>', unsafe_allow_html=True)
            placeholder = st.empty()
            streamed = ""
            with st.spinner(""):
                for token in ask_question_stream(st.session_state.vector_store, query):
                    streamed += token
                    placeholder.markdown(
                        f'<div class="bubble ai">{streamed}▌</div>',
                        unsafe_allow_html=True
                    )
            # Final render without cursor
            placeholder.markdown(
                f'<div class="bubble ai">{streamed}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div></div>', unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": streamed})

            # Update highlight / page for viewer
            docs = st.session_state.vector_store.similarity_search(query, k=1)
            if docs:
                st.session_state.highlight = docs[0].page_content[:80]
                st.session_state.current_page = docs[0].metadata.get("page", 0) + 1

            # TTS
            if st.session_state.get("voice_mode") and streamed:
                lang = st.session_state.get("tts_language", "en")
                audio_bytes = text_to_speech(streamed, language=lang)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

            st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  SUMMARY TAB
    # ─────────────────────────────────────────────────────────────────────────
    elif active == "summary":
        st.markdown("""<div class="panel">
            <div class="panel-header">
                <span class="panel-title">Document Summary</span>
                <span class="panel-badge">AI · Generated</span>
            </div><div class="panel-body">""", unsafe_allow_html=True)

        if st.session_state.last_summary is None:
            if st.button("📋 Generate Summary", use_container_width=True):
                with st.spinner("Generating summary…"):
                    res = ask_question(st.session_state.chain, st.session_state.vector_store,
                                       "Please provide a comprehensive summary of this document")
                    st.session_state.last_summary = res.get("answer", "")
                st.rerun()
        else:
            st.markdown(st.session_state.last_summary)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.session_state.last_summary = None
                    st.rerun()
            with col_b:
                summary_bytes = st.session_state.last_summary.encode("utf-8")
                st.download_button("⬇ Download (.txt)", summary_bytes,
                                   "summary.txt", use_container_width=True)

            if st.session_state.get("voice_mode"):
                audio_bytes = text_to_speech(st.session_state.last_summary,
                                              st.session_state.get("tts_language", "en"))
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

        st.markdown('</div></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  NOTES TAB
    # ─────────────────────────────────────────────────────────────────────────
    elif active == "notes":
        st.markdown("""<div class="panel">
            <div class="panel-header">
                <span class="panel-title">Study Notes</span>
                <span class="panel-badge">AI · Generated</span>
            </div><div class="panel-body">""", unsafe_allow_html=True)

        if st.session_state.last_notes is None:
            if st.button("📝 Generate Notes", use_container_width=True):
                with st.spinner("Generating study notes…"):
                    res = ask_question(st.session_state.chain, st.session_state.vector_store,
                                       "Create detailed study notes from this document")
                    st.session_state.last_notes = res.get("answer", "")
                st.rerun()
        else:
            st.markdown(st.session_state.last_notes)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.session_state.last_notes = None
                    st.rerun()
            with col_b:
                notes_bytes = st.session_state.last_notes.encode("utf-8")
                st.download_button("⬇ Download (.txt)", notes_bytes,
                                   "notes.txt", use_container_width=True)

            if st.session_state.get("voice_mode"):
                audio_bytes = text_to_speech(st.session_state.last_notes,
                                              st.session_state.get("tts_language", "en"))
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

        st.markdown('</div></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  FLASHCARDS TAB
    # ─────────────────────────────────────────────────────────────────────────
    elif active == "flashcards":
        st.markdown("""<div class="panel">
            <div class="panel-header">
                <span class="panel-title">Flashcards</span>
                <span class="panel-badge">Study Mode</span>
            </div><div class="panel-body">""", unsafe_allow_html=True)

        cards = st.session_state.flashcards

        if not cards:
            if st.button("⚡ Generate Flashcards", use_container_width=True):
                with st.spinner("Generating flashcards…"):
                    res = ask_question(st.session_state.chain, st.session_state.vector_store,
                                       "Generate flashcards from this document")
                    raw  = res.get("answer", "")
                    parsed = parse_flashcards(raw)
                    if parsed:
                        st.session_state.flashcards = parsed
                    else:
                        # Fallback: display raw text
                        st.session_state.flashcards = [
                            {"question": "Raw Output", "answer": raw}
                        ]
                st.rerun()
        else:
            st.markdown(f"**{len(cards)} flashcard(s) generated**  —  click any card to reveal the answer")
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            for i, card in enumerate(cards, 1):
                with st.expander(f"Card {i}: {card['question'][:80]}"):
                    st.markdown(f"**Q:** {card['question']}")
                    st.markdown("---")
                    st.markdown(f"**A:** {card['answer']}")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            if st.button("🔄 Regenerate Flashcards", use_container_width=True):
                st.session_state.flashcards = []
                st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — PDF Viewer
# ══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown("""<div class="panel">
        <div class="panel-header">
            <span class="panel-title">Document Viewer</span>
            <span class="panel-badge">PDF</span>
        </div><div class="panel-body">""", unsafe_allow_html=True)

    names = list(st.session_state.pdf_files.keys())

    if not names:
        st.info("No documents loaded yet.")
    else:
        selected_file = st.selectbox("Select file", names, label_visibility="collapsed")
        pdf_bytes     = st.session_state.pdf_files[selected_file]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        total_pages = len(doc)

        col_page, col_zoom = st.columns(2)
        with col_page:
            default_page = min(st.session_state.current_page or 1, total_pages)
            page_num = st.slider("Page", 1, total_pages, default_page, key="page_slider")
        with col_zoom:
            zoom_level = st.slider("Zoom", 1, 3, 2, key="zoom_slider")

        pg = doc.load_page(page_num - 1)

        # Highlight relevant text if available
        if st.session_state.highlight:
            try:
                snippet = st.session_state.highlight[:50]
                areas = pg.search_for(snippet)
                for area in areas:
                    pg.add_highlight_annot(area)
            except Exception:
                pass

        mat = fitz.Matrix(zoom_level, zoom_level)
        pix = pg.get_pixmap(matrix=mat)

        img_path = tmp_path + f"_p{page_num}_z{zoom_level}.png"
        pix.save(img_path)

        st.image(img_path, use_column_width=True)
        st.caption(f"Page {page_num} of {total_pages}  ·  {selected_file}")

        # Export current page as PDF snippet
        single_pdf_path = tmp_path + "_page.pdf"
        single_doc = fitz.open()
        single_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
        single_doc.save(single_pdf_path)
        single_doc.close()

        with open(single_pdf_path, "rb") as fh:
            st.download_button(
                "⬇ Download this page",
                fh,
                file_name=f"{selected_file}_page{page_num}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        doc.close()

    st.markdown('</div></div>', unsafe_allow_html=True)