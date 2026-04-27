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
from voice import text_to_speech, speech_to_text_from_microphone
load_dotenv()

st.set_page_config(page_title="DocuMentor · AI PDF Assistant",
                   page_icon="📘", layout="wide",
                   initial_sidebar_state="expanded")

# ─── FORCE LIGHT THEME – override every Streamlit dark element ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* === GLOBAL LIGHT FORCE === */
html, body { background:#ffffff !important; color:#111827 !important; }

/* every root container */
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .block-container,
section.main > div,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background: #ffffff !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
}
.main .block-container { padding: 1.2rem 1.5rem !important; max-width: 100% !important; }

/* hide chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

/* === SIDEBAR === */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] section {
    background: #f8faff !important;
    border-right: 1px solid #e5e7eb !important;
}
[data-testid="stSidebar"] * { color: #111827 !important; }

/* === TEXT INPUT – the black box fix === */
/* target every possible wrapper Streamlit generates */
[data-testid="stTextInput"],
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stTextInput"] > label { background: transparent !important; }

[data-testid="stTextInput"] input,
[data-testid="stTextInput"] textarea,
div[data-baseweb="input"],
div[data-baseweb="input"] > div,
input[class*="st-"],
.st-emotion-cache-1n76uvr,
.st-emotion-cache-ue6h4q {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #2563eb !important;
}
[data-testid="stTextInput"] input:focus,
div[data-baseweb="input"]:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label { display: none !important; }

/* placeholder text color */
[data-testid="stTextInput"] input::placeholder { color: #9ca3af !important; opacity: 1 !important; }

/* === ALL BUTTONS === */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    background: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all .15s !important; cursor: pointer !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #eff6ff !important;
    border-color: #2563eb !important; color: #1d4ed8 !important;
}

/* === SELECTBOX === */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #111827 !important;
}
[data-testid="stSelectbox"] label { display: none !important; }

/* === SLIDERS === */
[data-testid="stSlider"] { background: transparent !important; }
[data-testid="stSlider"] label {
    font-size: 11px !important; font-weight: 600 !important;
    color: #6b7280 !important; text-transform: uppercase; letter-spacing: .7px;
}

/* === FILE UPLOADER === */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background: #f9fafb !important;
    border: 1.5px dashed #d1d5db !important;
    border-radius: 10px !important;
    color: #374151 !important;
}

/* === IMAGE === */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
}

/* === DOWNLOAD BUTTON === */
[data-testid="stDownloadButton"] > button {
    background: #f9fafb !important; border: 1px solid #d1d5db !important;
    color: #374151 !important; font-size: 12px !important;
    border-radius: 8px !important; width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #eff6ff !important; border-color: #2563eb !important; color: #1d4ed8 !important;
}

/* === ALERTS / CAPTION / AUDIO === */
[data-testid="stAlert"] { border-radius: 9px !important; font-size: 13px !important; }
[data-testid="stCaptionContainer"] { font-size: 11px !important; color: #6b7280 !important; text-align: center; }
[data-testid="stAudio"] audio { border-radius: 8px !important; width: 100% !important; }
[data-testid="stToggle"] label { font-size: 13px !important; color: #374151 !important; }

/* === CUSTOM COMPONENT CLASSES === */

/* Sidebar brand */
.sb-brand { padding:16px 14px 12px; border-bottom:1px solid #e5e7eb; display:flex; align-items:center; gap:10px; }
.sb-logo { width:34px; height:34px; background:linear-gradient(135deg,#2563eb,#7c3aed); border-radius:9px;
           display:flex; align-items:center; justify-content:center; font-size:17px; flex-shrink:0; }
.sb-name { font-size:15px; font-weight:700; color:#111827; }
.sb-tag  { font-size:10px; color:#9ca3af; margin-top:1px; letter-spacing:.5px; }
.sb-inner { padding:8px 12px 20px; }
.sb-label { font-size:10px; font-weight:700; letter-spacing:.9px; text-transform:uppercase;
            color:#9ca3af; margin:16px 0 6px 2px; display:block; }

.proc-btn > button {
    background: linear-gradient(135deg,#2563eb,#7c3aed) !important;
    color:#fff !important; border:none !important;
    border-radius:9px !important; font-weight:600 !important;
    font-size:13px !important; padding:10px 0 !important;
    box-shadow:0 2px 10px rgba(37,99,235,.28) !important;
}
.proc-btn > button:hover { opacity:.88 !important; }

.file-badge { display:flex; align-items:center; gap:7px; background:#f9fafb;
              border:1px solid #e5e7eb; border-radius:8px; padding:7px 10px; margin:3px 0;
              font-size:12px; color:#374151; }
.file-dot  { width:7px; height:7px; background:#10b981; border-radius:50%; flex-shrink:0; }
.file-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.file-size { font-size:10px; color:#9ca3af; }

/* Tab bar */
.tab-wrap { display:flex; gap:3px; background:#f3f4f6; border-radius:10px; padding:4px; margin-bottom:14px; }
.tab-active > button {
    background: #ffffff !important; color: #1d4ed8 !important;
    border-color: transparent !important; font-weight: 600 !important;
    box-shadow: 0 1px 5px rgba(0,0,0,.10) !important;
}

/* Chat panel */
.chat-panel { background:#fff; border:1px solid #e5e7eb; border-radius:14px; overflow:hidden; }
.panel-hdr  { padding:12px 18px; border-bottom:1px solid #f3f4f6; background:#fafafa;
              display:flex; align-items:center; justify-content:space-between; }
.panel-hdr-title { font-size:13px; font-weight:600; color:#374151; }
.panel-badge { font-size:11px; background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8;
               border-radius:20px; padding:2px 10px; font-weight:500; }

.chat-body { min-height:340px; max-height:400px; overflow-y:auto; padding:18px 16px 10px;
             scrollbar-width:thin; scrollbar-color:#e5e7eb transparent; }
.chat-body::-webkit-scrollbar { width:4px; }
.chat-body::-webkit-scrollbar-thumb { background:#e5e7eb; border-radius:4px; }

.chat-empty { display:flex; flex-direction:column; align-items:center; justify-content:center;
              height:300px; gap:10px; color:#9ca3af; }
.chat-empty-icon { font-size:42px; opacity:.45; }
.chat-empty-title { font-size:15px; font-weight:600; color:#6b7280; }
.chat-empty-sub   { font-size:12px; }

.msg-user  { display:flex; justify-content:flex-end; margin:7px 0; }
.bubble-user { background:#2563eb; color:#fff; border-radius:16px 16px 4px 16px;
               padding:10px 14px; font-size:14px; line-height:1.6; max-width:78%; word-wrap:break-word; }

.msg-ai    { display:flex; align-items:flex-start; gap:9px; margin:7px 0; }
.ai-av     { width:28px; height:28px; flex-shrink:0; background:linear-gradient(135deg,#2563eb,#7c3aed);
             border-radius:7px; display:flex; align-items:center; justify-content:center;
             font-size:13px; color:#fff; margin-top:2px; }
.bubble-ai { background:#f9fafb; border:1px solid #e5e7eb; border-radius:4px 16px 16px 16px;
             padding:10px 14px; font-size:14px; line-height:1.75; color:#111827;
             max-width:85%; word-wrap:break-word; white-space:pre-wrap; }

.input-area { padding:10px 14px 12px; border-top:1px solid #f3f4f6; background:#fafafa; }

.send-btn > button {
    background:#2563eb !important; border-color:#2563eb !important;
    color:#fff !important; font-weight:600 !important;
    border-radius:9px !important; font-size:13px !important;
    padding:8px 16px !important; width:auto !important;
}
.send-btn > button:hover { background:#1d4ed8 !important; }
.icon-btn > button {
    background:#f3f4f6 !important; border-color:#e5e7eb !important;
    color:#6b7280 !important; border-radius:9px !important;
    padding:8px !important; font-size:14px !important; width:38px !important;
}
.icon-btn > button:hover { background:#eff6ff !important; border-color:#2563eb !important; color:#2563eb !important; }

/* Content panels */
.cp { background:#fff; border:1px solid #e5e7eb; border-radius:14px; overflow:hidden; }
.cp-hdr { padding:12px 18px; border-bottom:1px solid #f3f4f6; background:#fafafa; }
.cp-hdr-title { font-size:13px; font-weight:600; color:#374151; }
.cp-body { padding:18px 20px; }
.cp-desc { font-size:13px; color:#6b7280; margin-bottom:14px; }
.ct { font-size:14px; line-height:1.8; color:#1f2937; white-space:pre-wrap; word-wrap:break-word; }

.gen-btn > button {
    background:linear-gradient(135deg,#2563eb,#7c3aed) !important;
    color:#fff !important; border:none !important; border-radius:9px !important;
    font-size:13px !important; font-weight:600 !important; padding:10px 22px !important;
    width:auto !important; box-shadow:0 2px 10px rgba(37,99,235,.25) !important;
}
.gen-btn > button:hover { opacity:.88 !important; }

.fc-card { background:#fafafa; border:1px solid #e5e7eb; border-radius:10px;
           padding:13px 16px; margin-bottom:9px;
           transition:border-color .15s,box-shadow .15s; }
.fc-card:hover { border-color:#2563eb; box-shadow:0 2px 12px rgba(37,99,235,.10); }
.fc-num  { font-size:10px; color:#9ca3af; font-weight:600; margin-bottom:4px; letter-spacing:.5px; }
.fc-q    { font-size:14px; font-weight:600; color:#111827; }
.fc-a    { font-size:13px; color:#2563eb; margin-top:9px; padding-top:9px; border-top:1px solid #e5e7eb; }

/* Viewer */
.vw { background:#fff; border:1px solid #e5e7eb; border-radius:14px; overflow:hidden; }
.vw-hdr { padding:12px 18px; border-bottom:1px solid #f3f4f6; background:#fafafa;
          display:flex; align-items:center; justify-content:space-between; }
.vw-title  { font-size:13px; font-weight:600; color:#374151; }
.vw-badge  { font-size:11px; background:#fef9c3; border:1px solid #fde68a; color:#92400e;
             border-radius:20px; padding:2px 10px; font-weight:500; }
.vw-body   { padding:12px 14px; }

/* Welcome */
.wc-outer { display:flex; align-items:center; justify-content:center; min-height:70vh; }
.wc { background:#fff; border:1px solid #e5e7eb; border-radius:18px; padding:44px 52px;
      text-align:center; max-width:480px; width:100%;
      box-shadow:0 4px 24px rgba(0,0,0,.07); }
.wc-logo { width:56px; height:56px; background:linear-gradient(135deg,#2563eb,#7c3aed);
           border-radius:14px; margin:0 auto 16px;
           display:flex; align-items:center; justify-content:center; font-size:26px; }
.wc-title { font-size:22px; font-weight:700; color:#111827; margin-bottom:10px; }
.wc-desc  { font-size:14px; color:#6b7280; line-height:1.7; margin-bottom:22px; }
.wc-steps { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }
.step-chip { background:#eff6ff; border:1px solid #bfdbfe; border-radius:20px;
             padding:5px 14px; font-size:12px; color:#1d4ed8; font-weight:500; }
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

    st.markdown('<span class="sb-label">Upload Documents</span>', unsafe_allow_html=True)
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
                    f.seek(0); st.session_state.pdf_files[f.name] = f.read()
                st.session_state.pdfs_processed = True
                st.session_state.chat_history   = []
                st.success(f"✅ Ready — {len(docs)} pages indexed")
            else:
                st.error("Could not extract text from PDFs.")
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
        st.markdown('<span class="sb-label">Voice</span>', unsafe_allow_html=True)
        voice_on = st.toggle("Enable Text-to-Speech",
                             value=st.session_state.get("voice_mode", False))
        st.session_state.voice_mode = voice_on
        if voice_on:
            from voice import get_supported_languages
            langs = get_supported_languages()
            sel = st.selectbox("Language", list(langs.keys()),
                               format_func=lambda x: langs[x])
            st.session_state.tts_language = sel

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  WELCOME
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
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — two columns
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1.2, 0.8], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
with left_col:

    # ── Tab bar ───────────────────────────────────────────────────────────────
    TABS = [("chat","💬  Chat"),("summary","📋  Summary"),
            ("notes","📝  Notes"),("flashcards","⚡  Flashcards")]
    tcols = st.columns(len(TABS))
    for (key, lbl), col in zip(TABS, tcols):
        with col:
            cls = "tab-active" if st.session_state.active_tab == key else ""
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(lbl, key=f"tab_{key}", use_container_width=True):
                st.session_state.active_tab = key; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    active = st.session_state.active_tab

    # ══════════════════════════════════════════════════════════════════════════
    #  CHAT
    # ══════════════════════════════════════════════════════════════════════════
    if active == "chat":
        # panel header
        st.markdown("""
        <div class="chat-panel">
          <div class="panel-hdr">
            <span class="panel-hdr-title">💬 Conversation</span>
            <span class="panel-badge">RAG · Streaming</span>
          </div>""", unsafe_allow_html=True)

        # messages
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
                if m["role"] == "user":
                    safe = m["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    msgs_html += f'<div class="msg-user"><div class="bubble-user">{safe}</div></div>'
                else:
                    safe = (m["content"].replace("&","&amp;").replace("<","&lt;")
                            .replace(">","&gt;").replace("\n","<br>"))
                    msgs_html += f'<div class="msg-ai"><div class="ai-av">✦</div><div class="bubble-ai">{safe}</div></div>'
            st.markdown(f'<div class="chat-body">{msgs_html}</div>', unsafe_allow_html=True)

        # close chat-panel div
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        # ── INPUT ROW ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns([6.5, 1.5, 0.8, 0.8])
        with c1:
            query = st.text_input("msg", placeholder="Message DocuMentor…",
                                   key="chat_input", label_visibility="collapsed")
        with c2:
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            send_btn = st.button("Send ↑", key="send", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
            mic_btn = st.button("🎤", key="mic", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
            clr_btn = st.button("🗑️", key="clr", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if mic_btn:
            with st.spinner("🎤 Listening…"):
                txt, msg_str = speech_to_text_from_microphone()
            if txt: st.success(f"Heard: {txt}"); query = txt
            else:   st.warning(msg_str)

        if clr_btn:
            st.session_state.chat_history = []; st.rerun()

        # ── SEND ──────────────────────────────────────────────────────────────
        if send_btn and query:
            st.session_state.chat_history.append({"role":"user","content":query})
            user_s = query.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            st.markdown(f'<div class="msg-user"><div class="bubble-user">{user_s}</div></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="msg-ai"><div class="ai-av">✦</div>', unsafe_allow_html=True)
            placeholder = st.empty()
            streamed = ""
            for token in ask_question_stream(st.session_state.vector_store, query):
                streamed += token
                s = (streamed.replace("&","&amp;").replace("<","&lt;")
                             .replace(">","&gt;").replace("\n","<br>"))
                placeholder.markdown(f'<div class="bubble-ai">{s}▌</div>',
                                     unsafe_allow_html=True)
            s = (streamed.replace("&","&amp;").replace("<","&lt;")
                         .replace(">","&gt;").replace("\n","<br>"))
            placeholder.markdown(f'<div class="bubble-ai">{s}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role":"assistant","content":streamed})
            try:
                docs = st.session_state.vector_store.similarity_search(query, k=1)
                if docs:
                    st.session_state.highlight    = docs[0].page_content[:80]
                    st.session_state.current_page = docs[0].metadata.get("page",0)+1
            except Exception: pass
            if st.session_state.get("voice_mode") and streamed:
                ab = text_to_speech(streamed, st.session_state.get("tts_language","en"))
                if ab: st.audio(ab, format="audio/mp3")
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    elif active == "summary":
        st.markdown('<div class="cp"><div class="cp-hdr"><span class="cp-hdr-title">📋 Document Summary</span></div><div class="cp-body">', unsafe_allow_html=True)
        if st.session_state.last_summary is None:
            st.markdown('<p class="cp-desc">Generate an AI-powered summary of your entire document.</p>', unsafe_allow_html=True)
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("📋 Generate Summary", key="gen_sum"):
                with st.spinner("Analysing…"):
                    res = ask_question(st.session_state.chain, st.session_state.vector_store,
                                       "Please provide a comprehensive summary of this document")
                    st.session_state.last_summary = res.get("answer","")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ct">{st.session_state.last_summary}</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                if st.button("🔄 Regenerate", use_container_width=True, key="rs"):
                    st.session_state.last_summary = None; st.rerun()
            with r2:
                st.download_button("⬇ Download .txt", st.session_state.last_summary.encode(),
                                   "summary.txt", use_container_width=True)
            if st.session_state.get("voice_mode"):
                ab = text_to_speech(st.session_state.last_summary, st.session_state.get("tts_language","en"))
                if ab: st.audio(ab, format="audio/mp3")
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  NOTES
    # ══════════════════════════════════════════════════════════════════════════
    elif active == "notes":
        st.markdown('<div class="cp"><div class="cp-hdr"><span class="cp-hdr-title">📝 Study Notes</span></div><div class="cp-body">', unsafe_allow_html=True)
        if st.session_state.last_notes is None:
            st.markdown('<p class="cp-desc">Generate structured study notes with key concepts and important points.</p>', unsafe_allow_html=True)
            st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
            if st.button("📝 Generate Notes", key="gen_notes"):
                with st.spinner("Creating notes…"):
                    res = ask_question(st.session_state.chain, st.session_state.vector_store,
                                       "Create detailed study notes from this document")
                    st.session_state.last_notes = res.get("answer","")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ct">{st.session_state.last_notes}</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                if st.button("🔄 Regenerate", use_container_width=True, key="rn"):
                    st.session_state.last_notes = None; st.rerun()
            with r2:
                st.download_button("⬇ Download .txt", st.session_state.last_notes.encode(),
                                   "notes.txt", use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  FLASHCARDS
    # ══════════════════════════════════════════════════════════════════════════
    elif active == "flashcards":
        st.markdown('<div class="cp"><div class="cp-hdr"><span class="cp-hdr-title">⚡ Flashcards</span></div><div class="cp-body">', unsafe_allow_html=True)
        cards = st.session_state.flashcards
        if not cards:
            st.markdown('<p class="cp-desc">Generate Q&A flashcards for studying your document.</p>', unsafe_allow_html=True)
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
            st.markdown(f'<p class="cp-desc"><strong>{len(cards)}</strong> flashcard(s) — expand each card to reveal the answer.</p>', unsafe_allow_html=True)
            for i, card in enumerate(cards, 1):
                with st.expander(f"Card {i}  ·  {card['question'][:68]}"):
                    st.markdown(f"**Q:** {card['question']}")
                    st.divider()
                    st.markdown(f"**A:** {card['answer']}")
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            if st.button("🔄 Regenerate Flashcards", key="rfc"):
                st.session_state.flashcards = []; st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT — PDF Viewer
# ─────────────────────────────────────────────────────────────────────────────
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

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes); tmp_path = tmp.name

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
            except Exception: pass

        pix      = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_path = tmp_path + f"_p{page_num}z{zoom}.png"
        pix.save(img_path)
        st.image(img_path, use_column_width=True)
        st.caption(f"Page {page_num} / {total_pages}  ·  {selected}")

        sp = tmp_path + "_sp.pdf"
        sd = fitz.open()
        sd.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
        sd.save(sp); sd.close()
        with open(sp,"rb") as fh:
            st.download_button("⬇ Download this page", fh,
                               f"{selected}_p{page_num}.pdf",
                               mime="application/pdf", use_container_width=True)
        doc.close()
        st.markdown('</div>', unsafe_allow_html=True)