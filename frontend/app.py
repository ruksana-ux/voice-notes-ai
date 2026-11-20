# Users can record their voice, transcribe it, extract action items, save notes, search using AI and download notes.

import os
import socket
import threading
import time
from datetime import datetime
import requests
import streamlit as st
import uvicorn
from streamlit_mic_recorder import mic_recorder

# 1. Populate environment variables from Streamlit Cloud Secrets (if deployed)
try:
    if "DEEPGRAM_API_KEY" in st.secrets and not os.getenv("DEEPGRAM_API_KEY"):
        os.environ["DEEPGRAM_API_KEY"] = st.secrets["DEEPGRAM_API_KEY"]
    if "DEEPGRAM_MODEL" in st.secrets and not os.getenv("DEEPGRAM_MODEL"):
        os.environ["DEEPGRAM_MODEL"] = st.secrets["DEEPGRAM_MODEL"]
    if "GEMINI_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# 2. Check and auto-start FastAPI backend if not already running
def is_backend_running(host="127.0.0.1", port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

if not is_backend_running():
    def start_server():
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, log_level="warning")

    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    for _ in range(12):
        if is_backend_running():
            break
        time.sleep(0.5)

BACKEND = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# 3. Streamlit Page Configuration
st.set_page_config(
    page_title="AI Voice Assistant Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 4. Custom Modern Styling (Glassmorphism & Gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.12) 50%, rgba(236, 72, 153, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.2);
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #818CF8;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 999px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin: 0;
    }
    
    /* Action Items Card */
    .action-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
    }
    
    .action-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34D399;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Semantic search result card */
    .search-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-left: 4px solid #6366F1;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    
    .score-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 5. Helper function to fetch notes
def fetch_all_notes():
    try:
        res = requests.get(f"{BACKEND}/notes", timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception:
        return []

all_notes = fetch_all_notes()

# 6. Sidebar Navigation & Info
with st.sidebar:
    st.markdown("### 🎙️ AI Voice Studio")
    st.caption("Deepgram Nova-3 • ChromaDB • Action Extractor")
    
    st.divider()
    
    # Live Metrics
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Total Notes", len(all_notes))
    with col_m2:
        model_name = os.getenv("DEEPGRAM_MODEL", "nova-3")
        st.metric("Model", model_name.upper())
    
    st.divider()
    
    # Engine Status
    backend_ok = is_backend_running()
    if backend_ok:
        st.success("🟢 Backend API: Connected", icon="✅")
    else:
        st.warning("🟡 Backend API: Starting...", icon="⏳")
        
    deepgram_key_ok = bool(os.getenv("DEEPGRAM_API_KEY"))
    if deepgram_key_ok:
        st.success("🟢 Deepgram: Ready", icon="🔑")
    else:
        st.error("🔴 Deepgram: Key Missing", icon="⚠️")
        
    ai_llm_ok = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY"))
    if ai_llm_ok:
        st.success("🟢 LLM Intelligence: Active", icon="🧠")
    else:
        st.info("🔵 NLP Action Extractor: Active", icon="⚙️")
        
    st.divider()
    st.markdown("#### 🚀 Flagship Features")
    st.markdown("""
    - ⚡ **Deepgram Nova-3**: Sub-second transcription with smart formatting.
    - 📋 **Action Item Extractor**: Automatically parses tasks, deadlines, and reminders.
    - 🧠 **ChromaDB Vector Search**: Semantic search by intent & meaning.
    - 📥 **One-Click Export**: Save to `.txt` anytime.
    """)

# 7. Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="badge-pill">⚡ Full-Stack AI Voice Assistant & Action-Item Extractor</div>
    <h1 class="hero-title">AI Voice Notes Studio</h1>
    <p class="hero-subtitle">Convert spoken words into structured notes, automatically extract to-do action items, and query thoughts using semantic vector AI.</p>
</div>
""", unsafe_allow_html=True)

# 8. Main Tabs Layout
tab_record, tab_search, tab_library = st.tabs([
    "🎙️ Record & Assistant", 
    "🔍 AI Semantic Search", 
    f"📚 Notes & Action Items ({len(all_notes)})"
])

# Initialize session state keys
if "recorder_key" not in st.session_state:
    st.session_state["recorder_key"] = 0
if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "action_items" not in st.session_state:
    st.session_state["action_items"] = []

# ==========================================================
# TAB 1: RECORD & ASSISTANT
# ==========================================================
with tab_record:
    col_rec, col_edit = st.columns([1, 1], gap="large")
    
    # --- Left Column: Audio Capture ---
    with col_rec:
        st.markdown("### 1. Capture Your Voice")
        st.write("Speak your thoughts, lecture, or meeting notes.")
        
        mode = st.radio("Input Source", ["🎤 Microphone Recording", "📁 Upload Audio File"], horizontal=True)
        
        audio_payload = None
        audio_mimetype = "audio/wav"
        
        if mode == "🎤 Microphone Recording":
            audio = mic_recorder(
                start_prompt="🔴 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                key=f"voice_recorder_{st.session_state['recorder_key']}"
            )
            
            if audio and audio.get("bytes"):
                audio_payload = audio["bytes"]
                audio_mimetype = "audio/wav"
                st.audio(audio_payload, format="audio/wav")
        else:
            uploaded_file = st.file_uploader(
                "Upload audio file (.wav, .mp3, .m4a, .ogg)",
                type=["wav", "mp3", "m4a", "ogg", "webm"]
            )
            if uploaded_file is not None:
                audio_payload = uploaded_file.read()
                audio_mimetype = uploaded_file.type or "audio/wav"
                st.audio(audio_payload, format=audio_mimetype)
        
        # Audio Action Buttons
        if audio_payload:
            col_t_btn, col_d_btn = st.columns([2, 1])
            with col_t_btn:
                if st.button("✨ Transcribe & Extract Actions", type="primary", use_container_width=True):
                    with st.spinner("Transcribing with Deepgram Nova-3 & extracting action items..."):
                        try:
                            response = requests.post(
                                f"{BACKEND}/transcribe",
                                files={"file": ("recording.wav", audio_payload, audio_mimetype)},
                                timeout=120
                            )
                            response.raise_for_status()
                            result = response.json()
                            st.session_state["transcript"] = result.get("transcript", "")
                            st.session_state["summary"] = result.get("summary", "")
                            st.session_state["action_items"] = result.get("action_items", [])
                            st.toast("Transcription & Action Items Ready!", icon="🎉")
                            st.rerun()
                        except requests.RequestException as e:
                            st.error(f"Transcription failed: {e}")
            
            with col_d_btn:
                if st.button("🗑️ Discard", use_container_width=True):
                    st.session_state["transcript"] = ""
                    st.session_state["summary"] = ""
                    st.session_state["action_items"] = []
                    st.session_state["recorder_key"] += 1
                    st.toast("Recording discarded.", icon="🗑️")
                    st.rerun()

    # --- Right Column: Review, Edit & Action Items ---
    with col_edit:
        st.markdown("### 2. AI Summary & Action Items")
        
        word_count = len(st.session_state["transcript"].split()) if st.session_state["transcript"] else 0
        st.caption(f"📊 Stats: **{word_count} words** • **{len(st.session_state['action_items'])} action items detected**")
        
        note_title = st.text_input(
            "Note Title",
            placeholder="e.g. Sprint Planning, Chemistry Class, Weekly Goals"
        )
        
        edit_tab_text, edit_tab_ai = st.tabs(["📝 Full Transcript", "⚡ AI Action Items & Summary"])
        
        with edit_tab_text:
            edited_text = st.text_area(
                "Voice Transcript",
                value=st.session_state["transcript"],
                height=180,
                placeholder="Your voice transcription will appear here...",
                label_visibility="collapsed"
            )
            
            if edited_text.strip() and edited_text != st.session_state["transcript"]:
                if st.button("🔄 Re-analyze Actions from Edited Text", help="Extract new action items from your edits"):
                    with st.spinner("Analyzing text..."):
                        try:
                            a_res = requests.post(f"{BACKEND}/notes/analyze", json={"transcript": edited_text})
                            a_res.raise_for_status()
                            a_data = a_res.json()
                            st.session_state["transcript"] = edited_text
                            st.session_state["summary"] = a_data.get("summary", "")
                            st.session_state["action_items"] = a_data.get("action_items", [])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            
        with edit_tab_ai:
            if st.session_state["summary"]:
                st.markdown("**📌 Executive Summary:**")
                st.info(st.session_state["summary"])
                
            if st.session_state["action_items"]:
                st.markdown("**✅ Action Items & To-Dos:**")
                for item in st.session_state["action_items"]:
                    st.checkbox(item, value=False, key=f"action_chk_{hash(item)}")
            else:
                st.write("No specific to-do action items detected yet. Record a note with tasks (e.g. *'Need to finish the report by Friday'*).")
        
        if st.button("💾 Save & Vector Index Note", type="primary", use_container_width=True):
            final_text = edited_text.strip() if edited_text.strip() else st.session_state["transcript"].strip()
            if not note_title.strip() or not final_text:
                st.warning("Please provide both a title and some note content.")
            else:
                with st.spinner("Saving note and indexing vector embeddings..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND}/notes",
                            json={
                                "title": note_title.strip(),
                                "transcript": final_text,
                                "summary": st.session_state["summary"],
                                "action_items": st.session_state["action_items"]
                            },
                            timeout=60
                        )
                        resp.raise_for_status()
                        st.toast(f"Saved '{note_title}' with action items!", icon="✅")
                        st.session_state["transcript"] = ""
                        st.session_state["summary"] = ""
                        st.session_state["action_items"] = []
                        st.session_state["recorder_key"] += 1
                        time.sleep(0.5)
                        st.rerun()
                    except requests.RequestException as e:
                        st.error(f"Failed to save note: {e}")

# ==========================================================
# TAB 2: AI SEMANTIC SEARCH
# ==========================================================
with tab_search:
    st.markdown("### 🔍 Search by Concept & Meaning")
    st.markdown("Ask natural questions or search by intent. Powered by ChromaDB vector database and `all-MiniLM-L6-v2`.")
    
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g. What were my tasks for the marketing project? Or: budget deadlines",
            label_visibility="collapsed"
        )
    with col_btn:
        search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
    
    if search_query.strip() and (search_clicked or search_query):
        with st.spinner("Searching vector index..."):
            try:
                search_res = requests.get(
                    f"{BACKEND}/search",
                    params={"query": search_query.strip()},
                    timeout=60
                )
                search_res.raise_for_status()
                matches = search_res.json()
                
                if matches:
                    st.success(f"Found {len(matches)} relevant note(s):")
                    for m in matches:
                        sim_pct = max(0, min(100, int((1 - (m["distance"] / 2)) * 100)))
                        note_summary = m["metadata"].get("summary", "")
                        
                        st.markdown(f"""
                        <div class="search-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <strong style="font-size:1.15rem; color:#F8FAFC;">📝 {m['metadata']['title']}</strong>
                                <span class="score-badge">🎯 {sim_pct}% Match</span>
                            </div>
                            {f'<p style="color:#A5B4FC; font-weight:600; margin:0 0 8px 0;">💡 Summary: {note_summary}</p>' if note_summary else ''}
                            <p style="color:#CBD5E1; line-height:1.6; margin:0 0 10px 0;">{m['transcript']}</p>
                            <span class="meta-chip">📅 Saved: {m['metadata'].get('created_at', 'N/A')[:19].replace('T', ' ')}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No matching notes found. Try searching with different concept keywords!")
            except requests.RequestException as e:
                st.error(f"Search request failed: {e}")

# ==========================================================
# TAB 3: NOTES & ACTION ITEMS LIBRARY
# ==========================================================
with tab_library:
    col_head, col_action = st.columns([4, 1])
    with col_head:
        st.markdown(f"### 📚 Your Notes & Action Items ({len(all_notes)})")
    with col_action:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.rerun()
            
    if not all_notes:
        st.info("No notes saved yet! Record your first voice note in the 'Record & Assistant' tab.")
    else:
        for idx, note in enumerate(all_notes):
            created_str = note.get('created_at', '')[:19].replace('T', ' ')
            words = len(note['transcript'].split())
            items = note.get("action_items", [])
            summary = note.get("summary", "")
            
            with st.expander(f"📝 {note['title']}  •  {created_str}  ({words} words • {len(items)} tasks)"):
                if summary:
                    st.markdown("**📌 Executive Summary:**")
                    st.info(summary)
                    
                if items:
                    st.markdown("**✅ Action Items:**")
                    for i_idx, itm in enumerate(items):
                        st.checkbox(itm, value=False, key=f"saved_chk_{note['id']}_{i_idx}")
                        
                st.markdown("**Transcript:**")
                st.write(note["transcript"])
                st.caption(f"Note ID: `{note['id']}` • Created at {created_str} UTC")
                
                # Download payload includes action items and summary
                download_text = f"Title: {note['title']}\nDate: {created_str}\n\nSummary:\n{summary}\n\nAction Items:\n" + \
                                "\n".join([f"- [ ] {x}" for x in items]) + f"\n\nTranscript:\n{note['transcript']}"
                
                col_down, col_del = st.columns([1, 1])
                with col_down:
                    st.download_button(
                        "📥 Download .TXT (with Tasks)",
                        data=download_text,
                        file_name=f"{note['title'].replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"dl_note_{note['id']}",
                        use_container_width=True
                    )
                with col_del:
                    if st.button("🗑️ Delete Note", key=f"del_note_{note['id']}", type="secondary", use_container_width=True):
                        try:
                            del_resp = requests.delete(f"{BACKEND}/notes/{note['id']}", timeout=30)
                            del_resp.raise_for_status()
                            st.toast(f"Deleted '{note['title']}'", icon="🗑️")
                            st.rerun()
                        except requests.RequestException as e:
                            st.error(f"Failed to delete note: {e}")
