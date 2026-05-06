import os, tempfile, warnings
import streamlit as st
import fitz

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from pdf_loader import load_pdfs_from_uploads, split_documents_into_chunks
from vector_store import create_vector_store
from qa_chain import (create_conversational_chain, ask_question,
                      ask_question_stream, parse_flashcards)
from utils import initialize_session_state, format_file_size
from voice import text_to_speech, speech_to_text_from_microphone, get_supported_languages
load_dotenv()

st.set_page_config(page_title="DocuMentor · AI PDF Assistant",
                   page_icon="📘", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ═══════════════ LIGHT 3D GLASSMORPHISM BASE ═══════════════ */
html, body {
    background: linear-gradient(135deg, #e8eaf6 0%, #ede7f6 30%, #e3f2fd 65%, #f3e5f5 100%) !important;
    min-height: 100vh !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e8eaf6 0%, #ede7f6 30%, #e3f2fd 65%, #f3e5f5 100%) !important;
}
[data-testid="stAppViewBlockContainer"],
.main, .block-container {
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    color: #1e1b4b !important;
}
.main .block-container { padding: 1.2rem 1.5rem !important; max-width: 100% !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

/* ═══════════════ SIDEBAR ═══════════════ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: rgba(255,255,255,0.65) !important;
    backdrop-filter: blur(32px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(32px) saturate(180%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
    box-shadow: 4px 0 24px rgba(99,102,241,0.08) !important;
}
[data-testid="stSidebar"] * { color: #1e1b4b !important; }

/* ═══════════════ TEXT INPUT ═══════════════ */
[data-testid="stTextInput"] > label { display: none !important; }
[data-testid="stTextInput"] input,
div[data-baseweb="input"] > div {
    background: rgba(255,255,255,0.80) !important;
    color: #1e1b4b !important;
    border: 1.5px solid rgba(99,102,241,0.25) !important;
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 11px 16px !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.08), 0 1px 0 rgba(255,255,255,0.9) inset !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus { border-color: rgba(99,102,241,0.55) !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 2px 12px rgba(99,102,241,0.08) inset !important; }
[data-testid="stTextInput"] input::placeholder { color: rgba(100,100,150,0.5) !important; }

/* ═══════════════ BUTTONS ═══════════════ */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    background: rgba(255,255,255,0.70) !important;
    color: #4338ca !important;
    border: 1.5px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 14px !important;
    backdrop-filter: blur(10px) !important;
    transition: all .18s !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.12), 0 1px 0 rgba(255,255,255,0.9) inset !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.45) !important;
    color: #3730a3 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.20), 0 1px 0 rgba(255,255,255,0.9) inset !important;
}

/* ═══════════════ SELECTBOX ═══════════════ */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.75) !important;
    border: 1.5px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #1e1b4b !important;
}
[data-testid="stSelectbox"] label { display: none !important; }

/* ═══════════════ FILE UPLOADER ═══════════════ */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.55) !important;
    border: 2px dashed rgba(99,102,241,0.30) !important;
    border-radius: 14px !important;
    color: #4338ca !important;
}

/* ═══════════════ ALERTS ═══════════════ */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 10px !important;
    color: #1e1b4b !important;
    font-size: 13px !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.08) !important;
}

/* ═══════════════ EXPANDER ═══════════════ */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.65) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.07) !important;
}
[data-testid="stExpander"] summary { color: #4338ca !important; font-weight:500 !important; }

/* ═══════════════ DOWNLOAD BUTTON ═══════════════ */
[data-testid="stDownloadButton"] > button {
    background: rgba(255,255,255,0.70) !important;
    border: 1.5px solid rgba(99,102,241,0.2) !important;
    color: #4338ca !important; font-size: 12px !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.10) !important;
}

/* ═══════════════ SIDEBAR BRAND ═══════════════ */
.sb-brand { padding:20px 16px 16px; border-bottom:1px solid rgba(99,102,241,0.12); display:flex; align-items:center; gap:12px; }
.sb-logo { width:40px; height:40px; background:linear-gradient(135deg,#6366f1,#a855f7); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0; box-shadow:0 4px 15px rgba(99,102,241,0.35); }
.sb-name { font-size:15px; font-weight:700; color:#1e1b4b !important; }
.sb-tag  { font-size:10px; color:rgba(67,56,202,0.6) !important; margin-top:1px; letter-spacing:.7px; }
.sb-inner { padding:8px 14px 24px; }
.sb-label { font-size:10px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:rgba(67,56,202,0.55) !important; margin:18px 0 6px 2px; display:block; }

/* ═══════════════ PROCESS BUTTON ═══════════════ */
.proc-btn > button {
    background: linear-gradient(135deg,#6366f1,#a855f7) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-weight:600 !important; font-size:13px !important; padding:11px 0 !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.40), 0 1px 0 rgba(255,255,255,0.2) inset !important;
}
.proc-btn > button:hover { opacity:.90 !important; transform:translateY(-1px) !important; }

/* ═══════════════ FILE BADGE ═══════════════ */
.file-badge { display:flex; align-items:center; gap:8px; background:rgba(255,255,255,0.70); border:1px solid rgba(99,102,241,0.15); border-radius:10px; padding:8px 12px; margin:4px 0; font-size:12px; color:#4338ca; box-shadow:0 2px 8px rgba(99,102,241,0.07); }
.file-dot   { width:7px; height:7px; background:#10b981; border-radius:50%; flex-shrink:0; box-shadow:0 0 6px rgba(16,185,129,0.5); }
.file-name  { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#1e1b4b; }
.file-size  { font-size:10px; color:rgba(67,56,202,0.5); }

/* ═══════════════ TAB BAR ═══════════════ */
.tab-active > button {
    background: linear-gradient(135deg,rgba(99,102,241,0.18),rgba(168,85,247,0.12)) !important;
    color:#3730a3 !important; border-color:rgba(99,102,241,0.40) !important;
    font-weight:700 !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.20), 0 1px 0 rgba(255,255,255,0.9) inset !important;
}

/* ═══════════════ CHAT PANEL ═══════════════ */
.chat-panel {
    background: rgba(255,255,255,0.60);
    backdrop-filter: blur(28px) saturate(180%);
    -webkit-backdrop-filter: blur(28px) saturate(180%);
    border: 1.5px solid rgba(99,102,241,0.18);
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(99,102,241,0.12), 0 2px 0 rgba(255,255,255,0.9) inset;
}
.panel-hdr {
    padding:14px 20px;
    border-bottom:1px solid rgba(99,102,241,0.10);
    background: rgba(255,255,255,0.50);
    display:flex; align-items:center; justify-content:space-between;
}
.panel-hdr-title { font-size:14px; font-weight:700; color:#1e1b4b; }
.panel-badge { font-size:11px; background:rgba(99,102,241,0.10); border:1px solid rgba(99,102,241,0.25); color:#4338ca; border-radius:20px; padding:3px 12px; font-weight:500; }

/* ═══════════════ CHAT BODY ═══════════════ */
.chat-body {
    min-height:400px; max-height:440px; overflow-y:auto;
    padding:20px 18px 12px;
    scrollbar-width:thin; scrollbar-color:rgba(99,102,241,0.25) transparent;
}
.chat-body::-webkit-scrollbar { width:4px; }
.chat-body::-webkit-scrollbar-thumb { background:rgba(99,102,241,0.25); border-radius:4px; }

.chat-empty { display:flex; flex-direction:column; align-items:center; justify-content:center; height:340px; gap:12px; }
.chat-empty-icon  { font-size:48px; opacity:.25; }
.chat-empty-title { font-size:16px; font-weight:600; color:rgba(67,56,202,0.60); }
.chat-empty-sub   { font-size:13px; color:rgba(67,56,202,0.40); }

/* ═══════════════ CHAT MESSAGES ═══════════════ */
.msg-row-user { display:flex; justify-content:flex-end; margin:10px 0; animation: fadeSlideUp 0.25s ease; }
.msg-row-ai   { display:flex; align-items:flex-end; gap:10px; margin:10px 0; animation: fadeSlideUp 0.25s ease; }

@keyframes fadeSlideUp {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}

.bubble-user {
    background: linear-gradient(135deg,#6366f1,#7c3aed);
    color:#fff;
    border-radius:20px 20px 6px 20px;
    padding:12px 18px;
    font-size:14px; line-height:1.65;
    max-width:72%;
    word-wrap:break-word;
    box-shadow: 0 4px 20px rgba(99,102,241,0.30), 0 1px 0 rgba(255,255,255,0.2) inset;
}

.ai-avatar {
    width:32px; height:32px; flex-shrink:0;
    background: linear-gradient(135deg,#6366f1,#a855f7);
    border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:14px; color:#fff;
    box-shadow: 0 3px 12px rgba(99,102,241,0.35);
    margin-bottom:2px;
}

.bubble-ai {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border: 1.5px solid rgba(99,102,241,0.15);
    border-radius:6px 20px 20px 20px;
    padding:12px 18px;
    font-size:14px; line-height:1.80;
    color:#1e1b4b;
    max-width:78%;
    word-wrap:break-word;
    white-space:pre-wrap;
    box-shadow: 0 4px 16px rgba(99,102,241,0.10), 0 1px 0 rgba(255,255,255,0.9) inset;
}

.bubble-typing {
    background: rgba(255,255,255,0.85);
    border: 1.5px solid rgba(99,102,241,0.15);
    border-radius:6px 20px 20px 20px;
    padding:12px 18px;
    font-size:14px; line-height:1.80;
    color:#1e1b4b;
    max-width:78%;
    word-wrap:break-word;
    white-space:pre-wrap;
    box-shadow: 0 4px 16px rgba(99,102,241,0.10);
}

.msg-time { font-size:10px; color:rgba(67,56,202,0.35); margin-top:3px; text-align:right; }
.msg-time-ai { font-size:10px; color:rgba(67,56,202,0.35); margin-top:3px; padding-left:42px; }

/* ═══════════════ INPUT ROW ═══════════════ */
.input-area {
    background: rgba(255,255,255,0.70);
    border-top: 1px solid rgba(99,102,241,0.10);
    padding: 12px 16px;
    display:flex; align-items:center; gap:8px;
}
.send-btn > button {
    background:linear-gradient(135deg,#6366f1,#7c3aed) !important;
    border:none !important; color:#fff !important;
    font-weight:600 !important; border-radius:12px !important;
    font-size:13px !important; width:auto !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
    padding: 9px 20px !important;
}
.send-btn > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }
.icon-btn > button {
    background:rgba(255,255,255,0.80) !important;
    border:1.5px solid rgba(99,102,241,0.20) !important;
    color:#4338ca !important; border-radius:10px !important;
    font-size:15px !important; width:38px !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.10) !important;
    padding: 6px !important;
}
.icon-btn > button:hover { background:rgba(99,102,241,0.10) !important; }

/* ═══════════════ CONTENT PANELS ═══════════════ */
.cp {
    background: rgba(255,255,255,0.60);
    backdrop-filter: blur(28px) saturate(180%);
    -webkit-backdrop-filter: blur(28px) saturate(180%);
    border: 1.5px solid rgba(99,102,241,0.18);
    border-radius: 20px; overflow: hidden;
    box-shadow: 0 8px 40px rgba(99,102,241,0.12), 0 2px 0 rgba(255,255,255,0.9) inset;
}
.cp-hdr { padding:14px 20px; border-bottom:1px solid rgba(99,102,241,0.10); background:rgba(255,255,255,0.50); }
.cp-hdr-title { font-size:14px; font-weight:700; color:#1e1b4b; }
.cp-body { padding:22px 24px; }
.cp-desc { font-size:13px; color:rgba(67,56,202,0.65); margin-bottom:18px; line-height:1.65; }
.ct { font-size:14px; line-height:1.9; color:#1e1b4b; white-space:pre-wrap; word-wrap:break-word; }
.ct h1,.ct h2,.ct h3 { color:#4338ca; margin:14px 0 6px; font-weight:700; }
.ct ul,.ct ol { padding-left:22px; margin:6px 0; }
.ct li { margin:4px 0; }
.ct strong { color:#3730a3; }

.gen-btn > button {
    background:linear-gradient(135deg,#6366f1,#a855f7) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-size:13px !important; font-weight:600 !important; padding:11px 28px !important;
    width:auto !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
.gen-btn > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }

/* ═══════════════ VOICE BAR ═══════════════ */
.voice-bar { background:rgba(255,255,255,0.55); border:1px solid rgba(99,102,241,0.14); border-radius:12px; padding:10px 14px; margin-top:14px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; box-shadow:0 2px 8px rgba(99,102,241,0.07); }
.voice-label { font-size:10px; font-weight:700; color:rgba(67,56,202,0.55); text-transform:uppercase; letter-spacing:.8px; }

/* ═══════════════ PDF VIEWER ═══════════════ */
.vw { background:rgba(255,255,255,0.60); backdrop-filter:blur(28px) saturate(180%); border:1.5px solid rgba(99,102,241,0.18); border-radius:20px; overflow:hidden; box-shadow:0 8px 40px rgba(99,102,241,0.12), 0 2px 0 rgba(255,255,255,0.9) inset; }
.vw-hdr { padding:14px 20px; border-bottom:1px solid rgba(99,102,241,0.10); background:rgba(255,255,255,0.50); display:flex; align-items:center; justify-content:space-between; }
.vw-title { font-size:14px; font-weight:700; color:#1e1b4b; }
.vw-badge { font-size:11px; background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.30); color:#d97706; border-radius:20px; padding:3px 12px; font-weight:500; }
.vw-body { padding:14px; }

/* ═══════════════ WELCOME ═══════════════ */
.wc-outer { display:flex; align-items:center; justify-content:center; min-height:78vh; }
.wc {
    background: rgba(255,255,255,0.70);
    backdrop-filter: blur(40px) saturate(200%);
    border: 1.5px solid rgba(99,102,241,0.18);
    border-radius: 28px; padding:52px 60px;
    text-align:center; max-width:520px; width:100%;
    box-shadow: 0 24px 80px rgba(99,102,241,0.15), 0 2px 0 rgba(255,255,255,0.95) inset;
}
.wc-logo { width:68px; height:68px; background:linear-gradient(135deg,#6366f1,#a855f7); border-radius:18px; margin:0 auto 20px; display:flex; align-items:center; justify-content:center; font-size:30px; box-shadow:0 8px 32px rgba(99,102,241,0.40); }
.wc-title { font-size:26px; font-weight:800; color:#1e1b4b; margin-bottom:14px; }
.wc-desc  { font-size:14px; color:rgba(67,56,202,0.70); line-height:1.80; margin-bottom:26px; }
.wc-steps { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }
.step-chip { background:rgba(99,102,241,0.10); border:1.5px solid rgba(99,102,241,0.22); border-radius:20px; padding:7px 18px; font-size:12px; color:#4338ca; font-weight:600; }

/* ═══════════════ FLASHCARD ═══════════════ */
.fc-card { background:rgba(255,255,255,0.80); border:1.5px solid rgba(99,102,241,0.15); border-radius:16px; padding:18px 22px; margin-bottom:10px; box-shadow:0 3px 14px rgba(99,102,241,0.08); }
.fc-q { font-size:12px; font-weight:700; color:#6366f1; margin-bottom:8px; text-transform:uppercase; letter-spacing:.6px; }
.fc-a { font-size:14px; color:#1e1b4b; line-height:1.70; }
.fc-divider { border:none; border-top:1px solid rgba(99,102,241,0.12); margin:12px 0; }
</style>
""", unsafe_allow_html=True)

initialize_session_state(st)

import datetime
def _now():
    return datetime.datetime.now().strftime("%I:%M %p")


def _voice_bar(content_text: str, section_key: str):
    if not st.session_state.get("voice_mode"):
        return
    lang = st.session_state.get("tts_language", "en")
    st.markdown('<div class="voice-bar"><span class="voice-label">🔊 Voice</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
        if st.button("🔊", key=f"tts_{section_key}", help="Read aloud"):
            if content_text:
                ab = text_to_speech(content_text, lang)
                if ab:
                    st.audio(ab, format="audio/mp3")
                else:
                    st.warning("TTS unavailable — install gtts.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
        if st.button("🎤", key=f"stt_{section_key}", help="Voice input"):
            with st.spinner("🎤 Listening…"):
                txt, msg = speech_to_text_from_microphone()
            if txt:
                st.session_state[f"stt_result_{section_key}"] = txt
                st.rerun()
            else:
                st.warning(msg)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        result = st.session_state.get(f"stt_result_{section_key}", "")
        if result:
            st.markdown(f'<span style="font-size:12px;color:#4338ca;">🎙 "{result}"</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


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

    st.markdown('<span class="sb-label">Upload Documents</span>', unsafe_allow_html=True)
    files = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True,
                              label_visibility="collapsed")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="proc-btn">', unsafe_allow_html=True)
    process_clicked = st.button("⚡  Process Documents", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if process_clicked and files:
        with st.spinner("Indexing documents…"):
            try:
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
                    st.session_state.last_summary   = None
                    st.session_state.last_notes     = None
                    st.session_state.flashcards     = []
                    st.success(f"✅ Ready — {len(docs)} pages indexed")
                    st.rerun()
                else:
                    st.error("Could not extract text from PDFs.")
            except Exception as e:
                st.error(f"Processing error: {e}")
    elif process_clicked:
        st.warning("Upload at least one PDF first.")

    if st.session_state.pdf_files:
        st.markdown('<span class="sb-label">Loaded Files</span>', unsafe_allow_html=True)
        for name, data in st.session_state.pdf_files.items():
            short = name if len(name) <= 26 else name[:23] + "…"
            st.markdown(f"""
            <div class="file-badge">
                <div class="file-dot"></div>
                <div class="file-name">{short}</div>
                <span class="file-size">{format_file_size(len(data))}</span>
            </div>""", unsafe_allow_html=True)

    if st.session_state.pdfs_processed:
        st.markdown('<span class="sb-label">Voice Settings</span>', unsafe_allow_html=True)
        voice_on = st.toggle("Enable Voice Features", value=st.session_state.get("voice_mode", False))
        st.session_state.voice_mode = voice_on
        if voice_on:
            langs = get_supported_languages()
            sel = st.selectbox("Language", list(langs.keys()),
                               format_func=lambda x: langs[x],
                               label_visibility="collapsed")
            st.session_state.tts_language = sel

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.chain:
    st.markdown("""
    <div class="wc-outer"><div class="wc">
        <div class="wc-logo">📘</div>
        <div class="wc-title">Welcome to DocuMentor</div>
        <div class="wc-desc">Upload your PDFs and have an intelligent conversation.
        Get instant answers, summaries, study notes and flashcards.</div>
        <div class="wc-steps">
            <span class="step-chip">① Upload PDFs</span>
            <span class="step-chip">② Process</span>
            <span class="step-chip">③ Ask Away</span>
        </div>
    </div></div>
    """, unsafe_allow_html=True)
else:
    left_col, right_col = st.columns([1.30, 0.70], gap="large")

    with left_col:

        # ── Tab bar ──────────────────────────────────────────────────────────
        TABS = [("chat","💬  Chat"), ("summary","📋  Summary"),
                ("notes","📝  Notes"), ("flashcards","⚡  Flashcards")]
        tcols = st.columns(len(TABS))
        for (key, lbl), col in zip(TABS, tcols):
            with col:
                cls = "tab-active" if st.session_state.active_tab == key else ""
                st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                if st.button(lbl, key=f"tab_{key}", use_container_width=True):
                    st.session_state.active_tab = key
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        active = st.session_state.active_tab
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════
        #  CHAT
        # ════════════════════════════════════════════════════════════════════
        if active == "chat":
            st.markdown("""
            <div class="chat-panel">
              <div class="panel-hdr">
                <span class="panel-hdr-title">💬 Conversation</span>
                <span class="panel-badge">RAG · Streaming</span>
              </div>""", unsafe_allow_html=True)

            history = st.session_state.chat_history

            if not history:
                st.markdown("""
              <div class="chat-body">
                <div class="chat-empty">
                  <div class="chat-empty-icon">💬</div>
                  <div class="chat-empty-title">Start a conversation</div>
                  <div class="chat-empty-sub">Ask anything about your documents</div>
                </div>
              </div>""", unsafe_allow_html=True)
            else:
                msgs_html = '<div class="chat-body">'
                for m in history:
                    safe = (m["content"]
                            .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
                    ts   = m.get("time", "")
                    if m["role"] == "user":
                        msgs_html += f'''
                        <div class="msg-row-user">
                          <div>
                            <div class="bubble-user">{safe}</div>
                            <div class="msg-time">{ts}</div>
                          </div>
                        </div>'''
                    else:
                        safe_nl = safe.replace("\n","<br>")
                        msgs_html += f'''
                        <div class="msg-row-ai">
                          <div class="ai-avatar">✦</div>
                          <div>
                            <div class="bubble-ai">{safe_nl}</div>
                            <div class="msg-time-ai">{ts}</div>
                          </div>
                        </div>'''
                msgs_html += '</div>'
                st.markdown(msgs_html, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # close chat-panel
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

            # Input row
            c1, c2, c3, c4 = st.columns([5.8, 1.4, 0.75, 0.75])
            with c1:
                stt_prefill = st.session_state.pop("stt_result_chat", "")
                query = st.text_input("msg", placeholder="Type a message…",
                                      value=stt_prefill, key="chat_input",
                                      label_visibility="collapsed")
            with c2:
                st.markdown('<div class="send-btn">', unsafe_allow_html=True)
                send_btn = st.button("Send ↑", key="send", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
                mic_btn = st.button("🎤", key="mic_chat", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
                clr_btn = st.button("🗑️", key="clr", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if mic_btn:
                with st.spinner("🎤 Listening…"):
                    txt, msg = speech_to_text_from_microphone()
                if txt:
                    st.session_state["stt_result_chat"] = txt
                    st.rerun()
                else:
                    st.warning(msg)

            if clr_btn:
                st.session_state.chat_history = []
                st.rerun()

            if st.session_state.get("voice_mode") and history:
                last_ai = next((m["content"] for m in reversed(history)
                                if m["role"] == "assistant"), None)
                if last_ai:
                    _voice_bar(last_ai, "chat")

            if (send_btn or (query and st.session_state.get("_enter_pressed"))) and query.strip():
                now = _now()
                st.session_state.chat_history.append({"role":"user","content":query,"time":now})

                # Show user bubble immediately
                user_s = query.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                st.markdown(f'''
                <div class="msg-row-user">
                  <div>
                    <div class="bubble-user">{user_s}</div>
                    <div class="msg-time">{now}</div>
                  </div>
                </div>''', unsafe_allow_html=True)

                # AI typing indicator then stream
                st.markdown('<div class="msg-row-ai"><div class="ai-avatar">✦</div><div>', unsafe_allow_html=True)
                placeholder = st.empty()
                placeholder.markdown('<div class="bubble-typing">⠋ Thinking…</div>', unsafe_allow_html=True)

                streamed = ""
                try:
                    for token in ask_question_stream(st.session_state.vector_store, query):
                        streamed += token
                        s = (streamed.replace("&","&amp;").replace("<","&lt;")
                                     .replace(">","&gt;").replace("\n","<br>"))
                        placeholder.markdown(f'<div class="bubble-typing">{s}▍</div>',
                                             unsafe_allow_html=True)
                except Exception as e:
                    streamed = f"Error: {e}"

                # Final bubble
                s = (streamed.replace("&","&amp;").replace("<","&lt;")
                             .replace(">","&gt;").replace("\n","<br>"))
                placeholder.markdown(f'<div class="bubble-ai">{s}</div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

                ai_time = _now()
                st.session_state.chat_history.append({"role":"assistant","content":streamed,"time":ai_time})

                try:
                    docs_found = st.session_state.vector_store.similarity_search(query, k=1)
                    if docs_found:
                        st.session_state.highlight    = docs_found[0].page_content[:80]
                        st.session_state.current_page = docs_found[0].metadata.get("page", 0) + 1
                except Exception:
                    pass

                if st.session_state.get("voice_mode") and streamed:
                    ab = text_to_speech(streamed, st.session_state.get("tts_language","en"))
                    if ab:
                        st.audio(ab, format="audio/mp3")
                st.rerun()

        # ════════════════════════════════════════════════════════════════════
        #  SUMMARY
        # ════════════════════════════════════════════════════════════════════
        elif active == "summary":
            st.markdown('<div class="cp"><div class="cp-hdr"><span class="cp-hdr-title">📋 Document Summary</span></div><div class="cp-body">', unsafe_allow_html=True)

            stt_q = st.session_state.pop("stt_result_summary", "")
            if stt_q:
                with st.spinner(f'Processing…'):
                    try:
                        res = ask_question(st.session_state.chain,
                                           st.session_state.vector_store, stt_q)
                        st.session_state.last_summary = res.get("answer","")
                    except Exception as e:
                        st.error(f"Error: {e}")

            if st.session_state.last_summary is None:
                st.markdown('<p class="cp-desc">Generate an AI-powered summary of your entire document with key points and insights.</p>', unsafe_allow_html=True)
                st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
                if st.button("📋 Generate Summary", key="gen_sum"):
                    with st.spinner("Analysing document…"):
                        try:
                            res = ask_question(
                                st.session_state.chain,
                                st.session_state.vector_store,
                                "Please provide a comprehensive summary of this document"
                            )
                            st.session_state.last_summary = res.get("answer","")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ct">{st.session_state.last_summary}</div>', unsafe_allow_html=True)
                st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("🔄 Regenerate", use_container_width=True, key="rs"):
                        st.session_state.last_summary = None
                        st.rerun()
                with r2:
                    st.download_button("⬇ Download .txt",
                                       st.session_state.last_summary.encode(),
                                       "summary.txt", use_container_width=True)
                _voice_bar(st.session_state.last_summary, "summary")

            st.markdown('</div></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════
        #  NOTES
        # ════════════════════════════════════════════════════════════════════
        elif active == "notes":
            st.markdown('<div class="cp"><div class="cp-hdr"><span class="cp-hdr-title">📝 Study Notes</span></div><div class="cp-body">', unsafe_allow_html=True)

            stt_q = st.session_state.pop("stt_result_notes", "")
            if stt_q:
                with st.spinner('Processing…'):
                    try:
                        res = ask_question(st.session_state.chain,
                                           st.session_state.vector_store, stt_q)
                        st.session_state.last_notes = res.get("answer","")
                    except Exception as e:
                        st.error(f"Error: {e}")

            if st.session_state.last_notes is None:
                st.markdown('<p class="cp-desc">Generate structured study notes with key concepts, definitions, and important facts organised by topic.</p>', unsafe_allow_html=True)
                st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
                if st.button("📝 Generate Notes", key="gen_notes"):
                    with st.spinner("Creating study notes…"):
                        try:
                            res = ask_question(
                                st.session_state.chain,
                                st.session_state.vector_store,
                                "Create detailed study notes from this document"
                            )
                            st.session_state.last_notes = res.get("answer","")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ct">{st.session_state.last_notes}</div>', unsafe_allow_html=True)
                st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("🔄 Regenerate", use_container_width=True, key="rn"):
                        st.session_state.last_notes = None
                        st.rerun()
                with r2:
                    st.download_button("⬇ Download .txt",
                                       st.session_state.last_notes.encode(),
                                       "notes.txt", use_container_width=True)
                _voice_bar(st.session_state.last_notes, "notes")

            st.markdown('</div></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════
        #  FLASHCARDS
        # ════════════════════════════════════════════════════════════════════
        elif active == "flashcards":
            st.markdown('<div class="cp"><div class="cp-hdr"><span class="cp-hdr-title">⚡ Flashcards</span></div><div class="cp-body">', unsafe_allow_html=True)
            cards = st.session_state.flashcards

            stt_q = st.session_state.pop("stt_result_flashcards", "")
            if stt_q:
                with st.spinner('Processing…'):
                    try:
                        res    = ask_question(st.session_state.chain,
                                              st.session_state.vector_store, stt_q)
                        raw    = res.get("answer","")
                        parsed = parse_flashcards(raw)
                        st.session_state.flashcards = parsed or [{"question":"Output","answer":raw}]
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.rerun()

            if not cards:
                st.markdown('<p class="cp-desc">Generate interactive Q&A flashcards for studying the key concepts in your document.</p>', unsafe_allow_html=True)
                st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
                if st.button("⚡ Generate Flashcards", key="gen_fc"):
                    with st.spinner("Creating flashcards…"):
                        try:
                            res    = ask_question(
                                st.session_state.chain,
                                st.session_state.vector_store,
                                "Generate flashcards from this document"
                            )
                            raw    = res.get("answer","")
                            parsed = parse_flashcards(raw)
                            st.session_state.flashcards = parsed or [{"question":"Raw Output","answer":raw}]
                        except Exception as e:
                            st.error(f"Error: {e}")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="cp-desc"><strong style="color:#6366f1">{len(cards)}</strong> flashcard(s) — expand each card to reveal the answer.</p>', unsafe_allow_html=True)
                all_text = " ".join(f"Q: {c['question']} A: {c['answer']}" for c in cards)

                for i, card in enumerate(cards, 1):
                    q_safe = card['question'].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    a_safe = card['answer'].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
                    with st.expander(f"Card {i}  ·  {card['question'][:60]}{'…' if len(card['question'])>60 else ''}"):
                        st.markdown(f"""
                        <div class="fc-card">
                            <div class="fc-q">❓ Question</div>
                            <div style="font-size:14px;color:#1e1b4b;line-height:1.65;font-weight:500;">{q_safe}</div>
                            <hr class="fc-divider">
                            <div class="fc-q" style="color:#10b981;">✅ Answer</div>
                            <div class="fc-a">{a_safe}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
                _voice_bar(all_text, "flashcards")
                if st.button("🔄 Regenerate Flashcards", key="rfc"):
                    st.session_state.flashcards = []
                    st.rerun()

            st.markdown('</div></div>', unsafe_allow_html=True)

    # ── RIGHT — PDF Viewer ────────────────────────────────────────────────────
    with right_col:
        st.markdown("""
        <div class="vw">
          <div class="vw-hdr">
            <span class="vw-title">📄 Document Viewer</span>
            <span class="vw-badge">PDF</span>
          </div>
        </div>""", unsafe_allow_html=True)

        names = list(st.session_state.pdf_files.keys())
        if not names:
            st.info("No documents loaded yet.")
        else:
            st.markdown('<div class="vw-body">', unsafe_allow_html=True)
            selected  = st.selectbox("File", names, label_visibility="collapsed")
            pdf_bytes = st.session_state.pdf_files[selected]

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name

                doc         = fitz.open(tmp_path)
                total_pages = len(doc)

                pg_col, zm_col = st.columns(2)
                with pg_col:
                    default_pg = min(int(st.session_state.current_page or 1), total_pages)
                    page_num   = st.slider("Page", 1, total_pages, default_pg, key="page_slider")
                with zm_col:
                    zoom = st.slider("Zoom", 1, 3, 2, key="zoom_slider")

                pg = doc.load_page(page_num - 1)
                if st.session_state.get("highlight"):
                    try:
                        for area in pg.search_for(st.session_state.highlight[:50]):
                            pg.add_highlight_annot(area)
                    except Exception:
                        pass

                pix      = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                img_path = tmp_path + f"_p{page_num}z{zoom}.png"
                pix.save(img_path)

                # FIX: use_container_width=True instead of width=None
                st.image(img_path, use_container_width=True)
                st.caption(f"Page {page_num} / {total_pages}  ·  {selected}")

                sp = tmp_path + "_sp.pdf"
                sd = fitz.open()
                sd.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
                sd.save(sp)
                sd.close()
                with open(sp, "rb") as fh:
                    st.download_button("⬇ Download this page", fh,
                                       f"{selected}_p{page_num}.pdf",
                                       mime="application/pdf",
                                       use_container_width=True)
                doc.close()
                try:
                    os.remove(tmp_path)
                    os.remove(img_path)
                    os.remove(sp)
                except OSError:
                    pass

            except Exception as e:
                st.error(f"PDF viewer error: {e}")

            st.markdown('</div>', unsafe_allow_html=True)