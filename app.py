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

html, body {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 35%, #24243e 65%, #0f3460 100%) !important;
    min-height: 100vh !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 35%, #24243e 65%, #0f3460 100%) !important;
}
[data-testid="stAppViewBlockContainer"],
.main, .block-container {
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    color: #e0e7ff !important;
}
.main .block-container { padding: 1.2rem 1.5rem !important; max-width: 100% !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: rgba(15,12,41,0.78) !important;
    backdrop-filter: blur(32px) saturate(160%) !important;
    border-right: 1px solid rgba(255,255,255,0.10) !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(99,102,241,0.55) rgba(255,255,255,0.04) !important;
}
[data-testid="stSidebar"] * { color: #e0e7ff !important; }

[data-testid="stTextInput"] > label { display: none !important; }
[data-testid="stTextInput"] input,
div[data-baseweb="input"] > div {
    background: rgba(255,255,255,0.07) !important;
    color: #e0e7ff !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    -webkit-text-fill-color: #e0e7ff !important;
    caret-color: #818cf8 !important;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(160,170,210,0.45) !important; }

.stButton > button {
    font-family: 'Inter', sans-serif !important;
    background: rgba(255,255,255,0.07) !important;
    color: #c7d2fe !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 14px !important;
    backdrop-filter: blur(10px) !important;
    transition: all .18s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.22) !important;
    border-color: rgba(129,140,248,0.5) !important;
    color: #e0e7ff !important;
    transform: translateY(-1px) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    color: #e0e7ff !important;
}
[data-testid="stSelectbox"] label { display: none !important; }

[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px dashed rgba(129,140,248,0.35) !important;
    border-radius: 12px !important;
    color: #a5b4fc !important;
}
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.11) !important;
    border-radius: 10px !important;
    color: #e0e7ff !important;
    font-size: 13px !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: #c7d2fe !important; }
[data-testid="stDownloadButton"] > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    color: #a5b4fc !important; font-size: 12px !important;
    border-radius: 10px !important;
}

/* ── Brand ── */
.sb-brand { padding:20px 16px 16px; border-bottom:1px solid rgba(255,255,255,0.08); display:flex; align-items:center; gap:12px; }
.sb-logo { width:38px; height:38px; background:linear-gradient(135deg,#6366f1,#a855f7); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px; flex-shrink:0; }
.sb-name { font-size:15px; font-weight:700; color:#e0e7ff; }
.sb-tag  { font-size:10px; color:rgba(160,170,210,0.55); margin-top:1px; letter-spacing:.7px; }
.sb-inner { padding:8px 14px 24px; }
.sb-label { font-size:10px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:rgba(160,170,210,0.5); margin:18px 0 6px 2px; display:block; }

.proc-btn > button {
    background: linear-gradient(135deg,#6366f1,#a855f7) !important;
    color:#fff !important; border:none !important; border-radius:11px !important;
    font-weight:600 !important; font-size:13px !important; padding:11px 0 !important;
}
.file-badge { display:flex; align-items:center; gap:8px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:9px; padding:7px 10px; margin:3px 0; font-size:12px; color:#c7d2fe; }
.file-dot   { width:7px; height:7px; background:#34d399; border-radius:50%; flex-shrink:0; box-shadow:0 0 6px rgba(52,211,153,0.6); }
.file-name  { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.file-size  { font-size:10px; color:rgba(160,170,210,0.45); }

.tab-active > button {
    background: linear-gradient(135deg,rgba(99,102,241,0.45),rgba(168,85,247,0.35)) !important;
    color:#e0e7ff !important; border-color:rgba(129,140,248,0.35) !important;
    font-weight:600 !important;
}

/* ── Chat panel ── */
.chat-panel {
    background:rgba(255,255,255,0.05);
    backdrop-filter:blur(24px) saturate(180%);
    border:1px solid rgba(255,255,255,0.12); border-radius:18px; overflow:hidden;
    box-shadow:0 8px 40px rgba(0,0,0,0.4);
}
.panel-hdr { padding:13px 18px; border-bottom:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.04); display:flex; align-items:center; justify-content:space-between; }
.panel-hdr-title { font-size:13px; font-weight:600; color:#c7d2fe; }
.panel-badge { font-size:11px; background:rgba(99,102,241,0.2); border:1px solid rgba(129,140,248,0.3); color:#a5b4fc; border-radius:20px; padding:2px 10px; }

.chat-body { min-height:380px; max-height:420px; overflow-y:auto; padding:18px 16px 10px; scrollbar-width:thin; scrollbar-color:rgba(99,102,241,0.4) rgba(255,255,255,0.04); }
.chat-body::-webkit-scrollbar { width:4px; }
.chat-body::-webkit-scrollbar-thumb { background:rgba(99,102,241,0.4); border-radius:4px; }

.chat-empty { display:flex; flex-direction:column; align-items:center; justify-content:center; height:300px; gap:10px; }
.chat-empty-icon  { font-size:44px; opacity:.3; }
.chat-empty-title { font-size:15px; font-weight:600; color:rgba(160,174,214,0.65); }
.chat-empty-sub   { font-size:12px; color:rgba(160,174,214,0.4); }

.msg-user { display:flex; justify-content:flex-end; margin:8px 0; }
.bubble-user { background:linear-gradient(135deg,#6366f1,#7c3aed); color:#fff; border-radius:18px 18px 4px 18px; padding:11px 16px; font-size:14px; line-height:1.6; max-width:78%; word-wrap:break-word; box-shadow:0 4px 16px rgba(99,102,241,0.35); }

.msg-ai { display:flex; align-items:flex-start; gap:10px; margin:8px 0; }
.ai-av  { width:30px; height:30px; flex-shrink:0; background:linear-gradient(135deg,#6366f1,#a855f7); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:14px; color:#fff; margin-top:2px; }
.bubble-ai { background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.13); border-radius:4px 18px 18px 18px; padding:11px 16px; font-size:14px; line-height:1.8; color:#e0e7ff; max-width:85%; word-wrap:break-word; white-space:pre-wrap; }

/* ── Input row ── */
.send-btn > button { background:linear-gradient(135deg,#6366f1,#7c3aed) !important; border:none !important; color:#fff !important; font-weight:600 !important; border-radius:10px !important; font-size:13px !important; width:auto !important; }
.icon-btn > button { background:rgba(255,255,255,0.08) !important; border:1px solid rgba(255,255,255,0.13) !important; color:#a5b4fc !important; border-radius:10px !important; padding:8px !important; font-size:14px !important; width:38px !important; }

/* ── Content panels ── */
.cp { background:rgba(255,255,255,0.05); backdrop-filter:blur(24px) saturate(180%); border:1px solid rgba(255,255,255,0.12); border-radius:18px; overflow:hidden; box-shadow:0 8px 40px rgba(0,0,0,0.4); }
.cp-hdr { padding:13px 18px; border-bottom:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.04); }
.cp-hdr-title { font-size:13px; font-weight:600; color:#c7d2fe; }
.cp-body { padding:20px 22px; }
.cp-desc { font-size:13px; color:rgba(160,174,214,0.65); margin-bottom:16px; line-height:1.6; }
.ct { font-size:14px; line-height:1.9; color:#e0e7ff; white-space:pre-wrap; word-wrap:break-word; }
/* Markdown inside .ct */
.ct h1,.ct h2,.ct h3 { color:#a5b4fc; margin:12px 0 6px; }
.ct ul,.ct ol { padding-left:20px; margin:6px 0; }
.ct li { margin:3px 0; }
.ct strong { color:#c7d2fe; }

.gen-btn > button { background:linear-gradient(135deg,#6366f1,#a855f7) !important; color:#fff !important; border:none !important; border-radius:11px !important; font-size:13px !important; font-weight:600 !important; padding:10px 28px !important; width:auto !important; }

/* ── Voice bar ── */
.voice-bar { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:10px 14px; margin-top:12px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.voice-label { font-size:10px; font-weight:700; color:rgba(160,174,214,0.55); text-transform:uppercase; letter-spacing:.8px; }

/* ── PDF Viewer ── */
.vw { background:rgba(255,255,255,0.05); backdrop-filter:blur(24px) saturate(180%); border:1px solid rgba(255,255,255,0.12); border-radius:18px; overflow:hidden; box-shadow:0 8px 40px rgba(0,0,0,0.4); }
.vw-hdr { padding:13px 18px; border-bottom:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.04); display:flex; align-items:center; justify-content:space-between; }
.vw-title { font-size:13px; font-weight:600; color:#c7d2fe; }
.vw-badge { font-size:11px; background:rgba(251,191,36,0.15); border:1px solid rgba(251,191,36,0.25); color:#fbbf24; border-radius:20px; padding:2px 10px; }
.vw-body { padding:12px 14px; }

/* ── Welcome ── */
.wc-outer { display:flex; align-items:center; justify-content:center; min-height:75vh; }
.wc { background:rgba(255,255,255,0.07); backdrop-filter:blur(40px) saturate(200%); border:1px solid rgba(255,255,255,0.15); border-radius:24px; padding:48px 56px; text-align:center; max-width:500px; width:100%; }
.wc-logo { width:62px; height:62px; background:linear-gradient(135deg,#6366f1,#a855f7); border-radius:16px; margin:0 auto 18px; display:flex; align-items:center; justify-content:center; font-size:28px; }
.wc-title { font-size:24px; font-weight:700; color:#e0e7ff; margin-bottom:12px; }
.wc-desc  { font-size:14px; color:rgba(160,174,214,0.7); line-height:1.75; margin-bottom:24px; }
.wc-steps { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }
.step-chip { background:rgba(99,102,241,0.18); border:1px solid rgba(129,140,248,0.3); border-radius:20px; padding:6px 16px; font-size:12px; color:#a5b4fc; font-weight:500; }

/* ── Flashcard ── */
.fc-card { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.10); border-radius:14px; padding:16px 20px; margin-bottom:10px; }
.fc-q { font-size:13px; font-weight:600; color:#a5b4fc; margin-bottom:8px; }
.fc-a { font-size:14px; color:#e0e7ff; line-height:1.65; }
.fc-divider { border:none; border-top:1px solid rgba(255,255,255,0.08); margin:10px 0; }
</style>
""", unsafe_allow_html=True)

initialize_session_state(st)


def _voice_bar(content_text: str, section_key: str):
    if not st.session_state.get("voice_mode"):
        return
    lang = st.session_state.get("tts_language", "en")
    st.markdown('<div class="voice-bar"><span class="voice-label">🔊 Voice</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 3])
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
            st.markdown(f'<span style="font-size:12px;color:#a5b4fc;">🎙 "{result}"</span>', unsafe_allow_html=True)
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
        with st.spinner("Indexing…"):
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
            short = name if len(name) <= 25 else name[:22] + "…"
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
        Get answers, summaries, study notes and flashcards instantly.</div>
        <div class="wc-steps">
            <span class="step-chip">① Upload PDFs</span>
            <span class="step-chip">② Process</span>
            <span class="step-chip">③ Ask Away</span>
        </div>
    </div></div>
    """, unsafe_allow_html=True)
else:
    left_col, right_col = st.columns([1.25, 0.75], gap="large")

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
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

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
                  <div class="chat-empty-title">Ask anything about your documents</div>
                  <div class="chat-empty-sub">Powered by RAG · Streaming responses</div>
                </div>
              </div>""", unsafe_allow_html=True)
            else:
                msgs_html = ""
                for m in history:
                    safe = (m["content"]
                            .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
                    if m["role"] == "user":
                        msgs_html += f'<div class="msg-user"><div class="bubble-user">{safe}</div></div>'
                    else:
                        safe_nl = safe.replace("\n","<br>")
                        msgs_html += (f'<div class="msg-ai"><div class="ai-av">✦</div>'
                                      f'<div class="bubble-ai">{safe_nl}</div></div>')
                st.markdown(f'<div class="chat-body">{msgs_html}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

            # Input row
            c1, c2, c3, c4 = st.columns([6, 1.4, 0.75, 0.75])
            with c1:
                stt_prefill = st.session_state.pop("stt_result_chat", "")
                query = st.text_input("msg", placeholder="Message DocuMentor…",
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

            if send_btn and query:
                st.session_state.chat_history.append({"role":"user","content":query})
                user_s = query.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                st.markdown(f'<div class="msg-user"><div class="bubble-user">{user_s}</div></div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="msg-ai"><div class="ai-av">✦</div>', unsafe_allow_html=True)
                placeholder = st.empty()
                streamed = ""
                try:
                    for token in ask_question_stream(st.session_state.vector_store, query):
                        streamed += token
                        s = (streamed.replace("&","&amp;").replace("<","&lt;")
                                     .replace(">","&gt;").replace("\n","<br>"))
                        placeholder.markdown(f'<div class="bubble-ai">{s}▌</div>',
                                             unsafe_allow_html=True)
                except Exception as e:
                    streamed = f"Error generating response: {e}"
                s = (streamed.replace("&","&amp;").replace("<","&lt;")
                             .replace(">","&gt;").replace("\n","<br>"))
                placeholder.markdown(f'<div class="bubble-ai">{s}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append({"role":"assistant","content":streamed})
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
                with st.spinner(f'Asking: "{stt_q}"…'):
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
                st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
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
                with st.spinner(f'Asking: "{stt_q}"…'):
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
                st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
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
                with st.spinner(f'Asking: "{stt_q}"…'):
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
                st.markdown(f'<p class="cp-desc"><strong style="color:#a5b4fc">{len(cards)}</strong> flashcard(s) generated — expand each card to reveal the answer.</p>', unsafe_allow_html=True)
                all_text = " ".join(f"Q: {c['question']} A: {c['answer']}" for c in cards)

                for i, card in enumerate(cards, 1):
                    q_safe = card['question'].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    a_safe = card['answer'].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
                    with st.expander(f"Card {i}  ·  {card['question'][:65]}{'…' if len(card['question'])>65 else ''}"):
                        st.markdown(f"""
                        <div class="fc-card">
                            <div class="fc-q">❓ Question</div>
                            <div style="font-size:14px;color:#e0e7ff;line-height:1.65;">{q_safe}</div>
                            <hr class="fc-divider">
                            <div class="fc-q">✅ Answer</div>
                            <div class="fc-a">{a_safe}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                _voice_bar(all_text, "flashcards")
                if st.button("🔄 Regenerate Flashcards", key="rfc"):
                    st.session_state.flashcards = []
                    st.rerun()

            st.markdown('</div></div>', unsafe_allow_html=True)

    # ── RIGHT — PDF Viewer ───────────────────────────────────────────────────
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

                # FIX: use_column_width deprecated — use width parameter instead
                st.image(img_path, width=None)
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
                except OSError:
                    pass
            except Exception as e:
                st.error(f"PDF viewer error: {e}")

            st.markdown('</div>', unsafe_allow_html=True)