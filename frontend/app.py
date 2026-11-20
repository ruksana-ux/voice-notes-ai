# Users can record their voice, transcribe it, edit the transcript, save notes, search using AI and download notes.

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
    page_title="AI Voice Notes Studio",
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
        padding: 30px;
        margin-bottom: 25px;
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
    
    /* Modern Card Container */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(129, 140, 248, 0.3);
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
    
    /* Word count chip */
    .meta-chip {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 500;
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
    st.markdown("### 🎙️ Voice Studio")
    st.caption("Deepgram Nova-3 • ChromaDB • FastAPI")
    
    st.divider()
    
    # Live Metrics
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Total Notes", len(all_notes))
    with col_m2:
        model_name = os.getenv("DEEPGRAM_MODEL", "nova-3")
        st.metric("AI Model", model_name.upper())
    
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
        st.error("🔴 Deepgram: Key Missing in .env", icon="⚠️")
        
    st.divider()
    st.markdown("#### 💡 Quick Tips")
    st.markdown("""
    - **Recording**: Speak naturally, Deepgram automatically applies punctuation & casing.
    - **Search**: Search for *meanings* (e.g. *'grocery items'* finds *'buy milk and eggs'*).
    - **Export**: Download any note as `.txt` anytime.
    """)

# 7. Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="badge-pill">⚡ Powered by Deepgram Nova-3 & Semantic Vector AI</div>
    <h1 class="hero-title">AI Voice Notes Studio</h1>
    <p class="hero-subtitle">Speak freely, get crystal-clear transcriptions in seconds, and retrieve your thoughts with meaning-based AI search.</p>
</div>
""", unsafe_allow_html=True)

# 8. Main Tabs Layout
tab_record, tab_search, tab_library = st.tabs([
    "🎙️ Record & Create", 
    "🔍 AI Semantic Search", 
    f"📚 Notes Library ({len(all_notes)})"
])

# Initialize session state keys
if "recorder_key" not in st.session_state:
    st.session_state["recorder_key"] = 0
if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""

# ==========================================================
# TAB 1: RECORD & CREATE
# ==========================================================
with tab_record:
    col_rec, col_edit = st.columns([1, 1], gap="large")
    
    # --- Left Column: Audio Capture ---
    with col_rec:
        st.markdown("### 1. Capture Your Voice")
        st.write("Record directly with your microphone or upload an audio file.")
        
        mode = st.radio("Choose Input Mode", ["🎤 Microphone Recording", "📁 Upload Audio File"], horizontal=True)
        
        audio_payload = None
        audio_mimetype = "audio/wav"
        
        if mode == "🎤 Microphone Recording":
            st.markdown("Click below to record:")
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
                "Upload a voice note (.wav, .mp3, .m4a, .ogg)",
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
                if st.button("✨ Transcribe with Deepgram", type="primary", use_container_width=True):
                    with st.spinner("Transcribing with Deepgram Nova-3..."):
                        try:
                            response = requests.post(
                                f"{BACKEND}/transcribe",
                                files={"file": ("recording.wav", audio_payload, audio_mimetype)},
                                timeout=120
                            )
                            response.raise_for_status()
                            result = response.json()
                            st.session_state["transcript"] = result["transcript"]
                            st.toast("Transcription successful! Review it on the right.", icon="🎉")
                            st.rerun()
                        except requests.RequestException as e:
                            st.error(f"Transcription failed: {e}")
            
            with col_d_btn:
                if st.button("🗑️ Discard", use_container_width=True):
                    st.session_state["transcript"] = ""
                    st.session_state["recorder_key"] += 1
                    st.toast("Recording discarded.", icon="🗑️")
                    st.rerun()

    # --- Right Column: Review, Edit & Save ---
    with col_edit:
        st.markdown("### 2. Review & Save Note")
        
        word_count = len(st.session_state["transcript"].split()) if st.session_state["transcript"] else 0
        char_count = len(st.session_state["transcript"])
        
        st.caption(f"📊 Stats: **{word_count} words** • **{char_count} characters**")
        
        edited_text = st.text_area(
            "Transcription Content",
            value=st.session_state["transcript"],
            height=210,
            placeholder="Your voice transcription will appear here. You can freely edit or polish it before saving..."
        )
        
        note_title = st.text_input(
            "Note Title",
            placeholder="e.g. Ideas for Marketing Campaign, Physics Lecture Summary"
        )
        
        if st.button("💾 Save & Vector Index Note", type="primary", use_container_width=True):
            if not note_title.strip() or not edited_text.strip():
                st.warning("Please provide both a title and some note content.")
            else:
                with st.spinner("Saving note and building vector embeddings..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND}/notes",
                            json={"title": note_title.strip(), "transcript": edited_text.strip()},
                            timeout=60
                        )
                        resp.raise_for_status()
                        st.toast(f"Saved '{note_title}' successfully!", icon="✅")
                        st.session_state["transcript"] = ""
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
    st.markdown("Ask natural questions or search ideas. Powered by ChromaDB and `all-MiniLM-L6-v2` semantic vector search.")
    
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g. What did I plan for dinner? Or: meeting action items",
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
                        # Convert cosine distance to similarity percentage
                        # cosine distance range 0 (exact) to 2.
                        sim_pct = max(0, min(100, int((1 - (m["distance"] / 2)) * 100)))
                        
                        st.markdown(f"""
                        <div class="search-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <strong style="font-size:1.1rem; color:#F8FAFC;">📝 {m['metadata']['title']}</strong>
                                <span class="score-badge">🎯 {sim_pct}% Match</span>
                            </div>
                            <p style="color:#CBD5E1; line-height:1.6; margin:0 0 10px 0;">{m['transcript']}</p>
                            <span class="meta-chip">📅 Saved: {m['metadata'].get('created_at', 'N/A')[:19].replace('T', ' ')}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No matching notes found. Try phrasing your search differently or recording more notes!")
            except requests.RequestException as e:
                st.error(f"Search request failed: {e}")

# ==========================================================
# TAB 3: NOTES LIBRARY
# ==========================================================
with tab_library:
    col_head, col_action = st.columns([4, 1])
    with col_head:
        st.markdown(f"### 📚 Your Notes Collection ({len(all_notes)})")
    with col_action:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.rerun()
            
    if not all_notes:
        st.info("No notes saved yet! Record your first voice note in the 'Record & Create' tab.")
    else:
        for idx, note in enumerate(all_notes):
            created_str = note.get('created_at', '')[:19].replace('T', ' ')
            words = len(note['transcript'].split())
            
            with st.expander(f"📝 {note['title']}  •  {created_str}  ({words} words)"):
                st.write(note["transcript"])
                st.caption(f"Note ID: `{note['id']}` • Created at {created_str} UTC")
                
                col_down, col_del = st.columns([1, 1])
                with col_down:
                    st.download_button(
                        "📥 Download .TXT",
                        data=f"Title: {note['title']}\nDate: {created_str}\n\n{note['transcript']}",
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
