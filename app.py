"""
app.py - Docent AI PDF Assistant
ChatGPT-style chat UI, light theme, fully working.
"""

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

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Docent · AI PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — light, ChatGPT-style
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

html,body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"]{
    font-family:'Inter',sans-serif!important;
    background:#f7f7f8!important;
    color:#1a1a1a!important;
}

/* hide streamlit chrome */
#MainMenu,footer,header,
[data-testid="stToolbar"],
[data-testid="stDecoration"]{display:none!important}

.main .block-container{padding:0!important;max-width:100%!important}

/* ── SIDEBAR ─────────────────────────────────────── */
[data-testid="stSidebar"]{
    background:#ffffff!important;
    border-right:1px solid #e5e5e5!important;
    width:260px!important;
}
[data-testid="stSidebar"]>div:first-child{padding:0!important}

.sb-header{
    padding:20px 16px 14px;
    border-bottom:1px solid #ebebeb;
    display:flex;align-items:center;gap:10px;
}
.sb-logo{
    width:32px;height:32px;
    background:linear-gradient(135deg,#10a37f,#1a7f64);
    border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    font-size:16px;color:#fff;flex-shrink:0;
}
.sb-title{font-size:16px;font-weight:700;color:#1a1a1a;letter-spacing:-.3px}
.sb-sub{font-size:10px;color:#8e8ea0;margin-top:1px}

.sb-body{padding:12px 12px 20px}
.sb-section{
    font-size:10px;font-weight:600;letter-spacing:.8px;
    text-transform:uppercase;color:#8e8ea0;
    margin:16px 0 8px 4px;
}

/* upload zone */
[data-testid="stFileUploader"]{
    background:#fafafa!important;
    border:1.5px dashed #d9d9e3!important;
    border-radius:10px!important;
}
[data-testid="stFileUploader"]:hover{
    border-color:#10a37f!important;
    background:#f0faf7!important;
}

/* process button */
.proc-btn>button{
    background:#10a37f!important;
    color:#fff!important;border:none!important;
    border-radius:8px!important;
    font-family:'Inter',sans-serif!important;
    font-size:13px!important;font-weight:600!important;
    padding:9px 0!important;width:100%!important;
    box-shadow:0 1px 8px rgba(16,163,127,.25)!important;
    transition:background .2s!important;
}
.proc-btn>button:hover{background:#0e9170!important}

/* doc badge */
.doc-badge{
    display:flex;align-items:center;gap:8px;
    background:#f7f7f8;border:1px solid #ebebeb;
    border-radius:8px;padding:7px 10px;margin:4px 0;
    font-size:12px;color:#444;cursor:default;
}
.doc-dot{width:7px;height:7px;background:#10a37f;border-radius:50%;flex-shrink:0}
.doc-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ── MAIN AREA ───────────────────────────────────── */
.main-wrap{
    display:flex;height:100vh;overflow:hidden;
    padding:0;gap:0;
}

/* ── LEFT: chat column ───────────────────────────── */
.chat-col{
    flex:1;display:flex;flex-direction:column;
    height:100vh;overflow:hidden;
    background:#ffffff;
    border-right:1px solid #e5e5e5;
}

/* tab bar */
.tab-bar{
    display:flex;gap:2px;
    padding:10px 16px 0;
    border-bottom:1px solid #ebebeb;
    background:#fff;flex-shrink:0;
}
.tab-pill{
    padding:7px 16px;border-radius:6px 6px 0 0;
    font-size:13px;font-weight:500;cursor:pointer;
    border:1px solid transparent;border-bottom:none;
    color:#6b6b80;background:transparent;
    transition:all .15s;
}
.tab-pill.active{
    background:#fff;
    border-color:#e5e5e5 #e5e5e5 #fff;
    color:#1a1a1a;font-weight:600;
    margin-bottom:-1px;
}

/* chat messages scroll area */
.chat-messages{
    flex:1;overflow-y:auto;
    padding:24px 0;
    scrollbar-width:thin;scrollbar-color:#d9d9e3 transparent;
}
.chat-messages::-webkit-scrollbar{width:5px}
.chat-messages::-webkit-scrollbar-thumb{background:#d9d9e3;border-radius:4px}

/* empty state */
.chat-empty{
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;height:100%;gap:12px;
    color:#8e8ea0;padding:40px;
}
.chat-empty-icon{font-size:48px;opacity:.4}
.chat-empty-title{font-size:20px;font-weight:600;color:#1a1a1a}
.chat-empty-sub{font-size:14px;text-align:center;line-height:1.6;max-width:360px}

/* message rows */
.msg-group{padding:12px 60px;max-width:820px;margin:0 auto;width:100%}

/* USER bubble */
.msg-user{
    display:flex;justify-content:flex-end;margin-bottom:16px;
}
.bubble-user{
    background:#f4f4f5;
    color:#1a1a1a;
    border-radius:18px 18px 4px 18px;
    padding:12px 16px;
    font-size:14px;line-height:1.6;
    max-width:75%;
    word-wrap:break-word;
}

/* AI bubble */
.msg-ai{display:flex;gap:12px;margin-bottom:20px;align-items:flex-start}
.ai-avatar{
    width:28px;height:28px;flex-shrink:0;
    background:linear-gradient(135deg,#10a37f,#1a7f64);
    border-radius:6px;
    display:flex;align-items:center;justify-content:center;
    font-size:14px;color:#fff;margin-top:2px;
}
.bubble-ai{
    flex:1;font-size:14px;line-height:1.75;
    color:#1a1a1a;
    white-space:pre-wrap;word-wrap:break-word;
}

/* chat input area */
.input-area{
    padding:16px 60px 20px;
    background:#fff;flex-shrink:0;
    border-top:1px solid #ebebeb;
    max-width:820px;margin:0 auto;width:100%;
}
.input-box-wrap{
    display:flex;align-items:flex-end;gap:8px;
    background:#f7f7f8;
    border:1.5px solid #d9d9e3;
    border-radius:12px;
    padding:10px 12px;
    transition:border-color .2s;
}
.input-box-wrap:focus-within{
    border-color:#10a37f;
    background:#fff;
    box-shadow:0 0 0 3px rgba(16,163,127,.1);
}

/* override streamlit text input inside wrap */
[data-testid="stTextInput"]{width:100%}
[data-testid="stTextInput"] input{
    background:transparent!important;
    border:none!important;
    border-radius:0!important;
    box-shadow:none!important;
    color:#1a1a1a!important;
    font-family:'Inter',sans-serif!important;
    font-size:14px!important;
    padding:0!important;
    outline:none!important;
}
[data-testid="stTextInput"] label{display:none!important}
[data-testid="stTextInput"] [data-baseweb="input"]{
    background:transparent!important;
    border:none!important;
    box-shadow:none!important;
}

/* buttons inside input area */
.stButton>button{
    font-family:'Inter',sans-serif!important;
    font-size:12px!important;font-weight:500!important;
    border-radius:8px!important;
    padding:7px 12px!important;
    border:1px solid #e5e5e5!important;
    background:#fff!important;color:#555!important;
    transition:all .15s!important;
    white-space:nowrap!important;
}
.stButton>button:hover{
    background:#f0faf7!important;
    border-color:#10a37f!important;
    color:#10a37f!important;
}

.send-btn>button{
    background:#10a37f!important;
    border-color:#10a37f!important;
    color:#fff!important;font-weight:600!important;
    padding:8px 16px!important;
    border-radius:9px!important;
}
.send-btn>button:hover{background:#0e9170!important;border-color:#0e9170!important}

.mic-btn>button{
    background:#fff8f0!important;
    border-color:#f0d8a8!important;color:#c07020!important;
}
.mic-btn>button:hover{border-color:#e09030!important;color:#e07010!important}

/* ── FEATURE TABS content ───────────────────────── */
.tab-content{
    flex:1;overflow-y:auto;padding:24px 40px;
    scrollbar-width:thin;scrollbar-color:#d9d9e3 transparent;
}
.tab-content::-webkit-scrollbar{width:5px}
.tab-content::-webkit-scrollbar-thumb{background:#d9d9e3;border-radius:4px}

.gen-btn>button{
    background:#10a37f!important;
    color:#fff!important;border:none!important;
    border-radius:9px!important;
    font-size:13px!important;font-weight:600!important;
    padding:10px 24px!important;
    box-shadow:0 1px 8px rgba(16,163,127,.25)!important;
}
.gen-btn>button:hover{background:#0e9170!important}

.content-box{
    background:#fff;
    border:1px solid #e5e5e5;
    border-radius:12px;
    padding:24px 28px;
    font-size:14px;line-height:1.75;
    color:#1a1a1a;
    white-space:pre-wrap;
}

/* flashcard */
.fc-wrap{display:flex;flex-direction:column;gap:10px;margin-top:8px}
.fc-card{
    background:#fff;border:1px solid #e5e5e5;
    border-radius:10px;padding:16px 20px;
    transition:border-color .15s,box-shadow .15s;
}
.fc-card:hover{border-color:#10a37f;box-shadow:0 2px 12px rgba(16,163,127,.1)}
.fc-num{font-size:10px;color:#8e8ea0;font-family:'JetBrains Mono',monospace;margin-bottom:6px}
.fc-q{font-size:14px;font-weight:600;color:#1a1a1a;margin-bottom:0}
.fc-a{font-size:13px;color:#10a37f;border-top:1px solid #ebebeb;padding-top:10px;margin-top:10px}

/* ── RIGHT: PDF viewer ───────────────────────────── */
.viewer-col{
    width:420px;flex-shrink:0;
    background:#f7f7f8;
    display:flex;flex-direction:column;
    height:100vh;overflow:hidden;
}
.viewer-header{
    padding:14px 16px;
    border-bottom:1px solid #e5e5e5;
    background:#fff;flex-shrink:0;
    display:flex;align-items:center;justify-content:space-between;
}
.viewer-title{font-size:13px;font-weight:600;color:#1a1a1a}
.viewer-badge{
    font-size:11px;background:#f0faf7;
    border:1px solid #b8e8d8;color:#10a37f;
    border-radius:20px;padding:2px 10px;
    font-family:'JetBrains Mono',monospace;
}
.viewer-body{flex:1;overflow-y:auto;padding:16px;
    scrollbar-width:thin;scrollbar-color:#d9d9e3 transparent;}

/* sliders */
[data-testid="stSlider"] label{
    font-size:11px!important;color:#8e8ea0!important;
    text-transform:uppercase;letter-spacing:.6px;font-weight:600!important;
}

/* selectbox */
[data-testid="stSelectbox"] label{
    font-size:11px!important;color:#8e8ea0!important;
    text-transform:uppercase;letter-spacing:.6px;font-weight:600!important;
}
[data-testid="stSelectbox"]>div>div{
    background:#fff!important;border:1px solid #e5e5e5!important;
    border-radius:8px!important;color:#1a1a1a!important;font-size:13px!important;
}

/* image */
[data-testid="stImage"] img{
    border-radius:8px!important;border:1px solid #e5e5e5!important;
    box-shadow:0 2px 8px rgba(0,0,0,.06)!important;
}

/* download */
[data-testid="stDownloadButton"]>button{
    background:#fff!important;border:1px solid #e5e5e5!important;
    color:#555!important;font-size:12px!important;border-radius:8px!important;width:100%!important;
}
[data-testid="stDownloadButton"]>button:hover{
    border-color:#10a37f!important;color:#10a37f!important;background:#f0faf7!important;
}

/* alerts */
[data-testid="stAlert"]{border-radius:9px!important;font-size:13px!important}

/* audio */
[data-testid="stAudio"] audio{border-radius:8px!important;width:100%!important}

/* caption */
[data-testid="stCaptionContainer"]{
    font-size:11px!important;color:#8e8ea0!important;
    font-family:'JetBrains Mono',monospace!important;text-align:center;margin-top:6px;
}

/* welcome */
.welcome-wrap{
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;height:100vh;padding:40px;
    background:#fff;
}
.welcome-card{
    background:#fff;border:1px solid #e5e5e5;border-radius:16px;
    padding:40px 48px;text-align:center;max-width:500px;width:100%;
    box-shadow:0 4px 24px rgba(0,0,0,.06);
}
.welcome-icon{font-size:48px;margin-bottom:16px}
.welcome-title{font-size:24px;font-weight:700;color:#1a1a1a;margin-bottom:10px}
.welcome-desc{font-size:14px;color:#6b6b80;line-height:1.7;margin-bottom:24px}
.welcome-steps{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
.step-pill{
    background:#f0faf7;border:1px solid #b8e8d8;
    border-radius:20px;padding:6px 14px;
    font-size:12px;color:#10a37f;font-weight:500;
}
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
        <div class="sb-logo">📄</div>
        <div>
            <div class="sb-title">Docent</div>
            <div class="sb-sub">AI Document Intelligence</div>
        </div>
    </div>
    <div class="sb-body">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Upload Documents</div>', unsafe_allow_html=True)
    files = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True,
                              label_visibility="collapsed")

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
                st.session_state.chat_history = []
                st.success(f"✅ Ready — {len(docs)} pages indexed")
            else:
                st.error("Could not extract text from the PDFs.")
    elif process_clicked:
        st.warning("Upload at least one PDF first.")

    if st.session_state.pdf_files:
        st.markdown('<div class="sb-section">Loaded Files</div>', unsafe_allow_html=True)
        for name, data in st.session_state.pdf_files.items():
            short = name if len(name) <= 24 else name[:21] + "…"
            size  = format_file_size(len(data))
            st.markdown(f"""
            <div class="doc-badge">
                <div class="doc-dot"></div>
                <div class="doc-name">{short}</div>
                <span style="font-size:10px;color:#8e8ea0">{size}</span>
            </div>""", unsafe_allow_html=True)

    if st.session_state.pdfs_processed:
        st.markdown('<div class="sb-section">Voice</div>', unsafe_allow_html=True)
        voice_on = st.toggle("Enable TTS", value=st.session_state.get("voice_mode", False))
        st.session_state.voice_mode = voice_on
        if voice_on:
            from voice import get_supported_languages
            langs = get_supported_languages()
            tts_lang = st.selectbox("Language", list(langs.keys()),
                                     format_func=lambda x: langs[x])
            st.session_state.tts_language = tts_lang

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  EMPTY / WELCOME STATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.chain:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-card">
            <div class="welcome-icon">📄</div>
            <div class="welcome-title">Welcome to Docent</div>
            <div class="welcome-desc">
                Upload your PDFs and have an intelligent conversation with them.
                Get instant answers, summaries, study notes, and flashcards.
            </div>
            <div class="welcome-steps">
                <span class="step-pill">1 · Upload PDFs</span>
                <span class="step-pill">2 · Process</span>
                <span class="step-pill">3 · Ask Away</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
chat_col, viewer_col = st.columns([1.1, 0.75])

# ══════════════════════════════════════════════════════════════════════════════
#  LEFT — Chat + Feature Tabs
# ══════════════════════════════════════════════════════════════════════════════
with chat_col:

    # ── TAB BAR ──
    TABS = [
        ("chat",       "💬 Chat"),
        ("summary",    "📋 Summary"),
        ("notes",      "📝 Notes"),
        ("flashcards", "⚡ Flashcards"),
    ]
    tab_cols = st.columns(len(TABS))
    for (key, label), col in zip(TABS, tab_cols):
        with col:
            active_style = "tab-active" if st.session_state.active_tab == key else ""
            st.markdown(f'<div class="{active_style}">', unsafe_allow_html=True)
            if st.button(label, key=f"tab_{key}", use_container_width=True):
                st.session_state.active_tab = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    active = st.session_state.active_tab

    # ──────────────────────────────────────────────────────────────────────────
    #  CHAT TAB
    # ──────────────────────────────────────────────────────────────────────────
    if active == "chat":

        # render history
        history = st.session_state.chat_history
        if not history:
            st.markdown("""
            <div class="chat-empty">
                <div class="chat-empty-icon">💬</div>
                <div class="chat-empty-title">Ask anything</div>
                <div class="chat-empty-sub">Type a question below to start a conversation with your documents.</div>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="msg-group">
                        <div class="msg-user">
                            <div class="bubble-user">{msg["content"]}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    content_html = msg["content"].replace("\n", "<br>")
                    st.markdown(f"""
                    <div class="msg-group">
                        <div class="msg-ai">
                            <div class="ai-avatar">✦</div>
                            <div class="bubble-ai">{content_html}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        # ── INPUT ROW ──
        inp_col, send_col, mic_col, clr_col = st.columns([6, 1, 1, 1])
        with inp_col:
            query = st.text_input("q", placeholder="Message Docent…",
                                   key="chat_input", label_visibility="collapsed")
        with send_col:
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            send_btn = st.button("Send ↑", use_container_width=True, key="send")
            st.markdown('</div>', unsafe_allow_html=True)
        with mic_col:
            st.markdown('<div class="mic-btn">', unsafe_allow_html=True)
            mic_btn = st.button("🎤", use_container_width=True, key="mic")
            st.markdown('</div>', unsafe_allow_html=True)
        with clr_col:
            clr_btn = st.button("🗑️", use_container_width=True, key="clr")

        # voice input
        if mic_btn:
            with st.spinner("🎤 Listening…"):
                text, msg_str = speech_to_text_from_microphone()
            if text:
                st.success(f"Heard: {text}")
                query = text
            else:
                st.warning(msg_str)

        if clr_btn:
            st.session_state.chat_history = []
            st.rerun()

        # ── SEND ── streaming response
        if send_btn and query:
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Show user bubble immediately
            st.markdown(f"""
            <div class="msg-group">
                <div class="msg-user">
                    <div class="bubble-user">{query}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Stream AI response
            st.markdown("""
            <div class="msg-group">
                <div class="msg-ai">
                    <div class="ai-avatar">✦</div>""", unsafe_allow_html=True)

            placeholder = st.empty()
            streamed = ""
            for token in ask_question_stream(st.session_state.vector_store, query):
                streamed += token
                placeholder.markdown(
                    f'<div class="bubble-ai">{streamed.replace(chr(10),"<br>")}▌</div>',
                    unsafe_allow_html=True
                )
            placeholder.markdown(
                f'<div class="bubble-ai">{streamed.replace(chr(10),"<br>")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div></div>', unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": streamed})

            # update viewer highlight
            try:
                docs = st.session_state.vector_store.similarity_search(query, k=1)
                if docs:
                    st.session_state.highlight    = docs[0].page_content[:80]
                    st.session_state.current_page = docs[0].metadata.get("page", 0) + 1
            except Exception:
                pass

            # TTS
            if st.session_state.get("voice_mode") and streamed:
                lang = st.session_state.get("tts_language", "en")
                audio_bytes = text_to_speech(streamed, language=lang)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

            st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    #  SUMMARY TAB
    # ──────────────────────────────────────────────────────────────────────────
    elif active == "summary":
        st.markdown('<div style="padding:24px 0 12px">', unsafe_allow_html=True)
        if st.session_state.last_summary is None:
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("📋 Generate Summary", use_container_width=False):
                with st.spinner("Generating summary…"):
                    res = ask_question(
                        st.session_state.chain,
                        st.session_state.vector_store,
                        "Please provide a comprehensive summary of this document"
                    )
                    st.session_state.last_summary = res.get("answer", "")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="content-box">{st.session_state.last_summary.replace(chr(10),"<br>")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.session_state.last_summary = None
                    st.rerun()
            with c2:
                st.download_button("⬇ Download .txt",
                                   st.session_state.last_summary.encode(),
                                   "summary.txt", use_container_width=True)
            if st.session_state.get("voice_mode"):
                ab = text_to_speech(st.session_state.last_summary,
                                    st.session_state.get("tts_language","en"))
                if ab:
                    st.audio(ab, format="audio/mp3")
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    #  NOTES TAB
    # ──────────────────────────────────────────────────────────────────────────
    elif active == "notes":
        st.markdown('<div style="padding:24px 0 12px">', unsafe_allow_html=True)
        if st.session_state.last_notes is None:
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("📝 Generate Notes", use_container_width=False):
                with st.spinner("Generating notes…"):
                    res = ask_question(
                        st.session_state.chain,
                        st.session_state.vector_store,
                        "Create detailed study notes from this document"
                    )
                    st.session_state.last_notes = res.get("answer", "")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="content-box">{st.session_state.last_notes.replace(chr(10),"<br>")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.session_state.last_notes = None
                    st.rerun()
            with c2:
                st.download_button("⬇ Download .txt",
                                   st.session_state.last_notes.encode(),
                                   "notes.txt", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    #  FLASHCARDS TAB
    # ──────────────────────────────────────────────────────────────────────────
    elif active == "flashcards":
        st.markdown('<div style="padding:24px 0 12px">', unsafe_allow_html=True)
        cards = st.session_state.flashcards
        if not cards:
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("⚡ Generate Flashcards", use_container_width=False):
                with st.spinner("Generating flashcards…"):
                    res    = ask_question(
                        st.session_state.chain,
                        st.session_state.vector_store,
                        "Generate flashcards from this document"
                    )
                    raw    = res.get("answer", "")
                    parsed = parse_flashcards(raw)
                    st.session_state.flashcards = parsed if parsed else [
                        {"question": "Output", "answer": raw}
                    ]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(cards)} flashcard(s)** — expand to reveal answer")
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            for i, card in enumerate(cards, 1):
                with st.expander(f"Card {i} — {card['question'][:70]}"):
                    st.markdown(f"**Q:** {card['question']}")
                    st.divider()
                    st.markdown(f"**A:** {card['answer']}")
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            if st.button("🔄 Regenerate Flashcards", use_container_width=False):
                st.session_state.flashcards = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT — PDF Viewer
# ══════════════════════════════════════════════════════════════════════════════
with viewer_col:
    st.markdown("""
    <div class="viewer-header">
        <span class="viewer-title">📄 Document Viewer</span>
        <span class="viewer-badge">PDF</span>
    </div>""", unsafe_allow_html=True)

    names = list(st.session_state.pdf_files.keys())
    if not names:
        st.info("No documents loaded yet.")
    else:
        selected = st.selectbox("File", names, label_visibility="collapsed")
        pdf_bytes = st.session_state.pdf_files[selected]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        doc         = fitz.open(tmp_path)
        total_pages = len(doc)

        pg_col, zm_col = st.columns(2)
        with pg_col:
            default_pg = min(st.session_state.current_page or 1, total_pages)
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

        # Download current page
        single_path = tmp_path + "_single.pdf"
        sdoc = fitz.open()
        sdoc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
        sdoc.save(single_path)
        sdoc.close()
        with open(single_path, "rb") as fh:
            st.download_button("⬇ Download this page", fh,
                               f"{selected}_p{page_num}.pdf",
                               mime="application/pdf",
                               use_container_width=True)
        doc.close()