# Users can record their voice, transcribe it, extract action items, save notes, search using AI and download notes.

import os
import sys
import tempfile
from pathlib import Path

# 1. ROBUST SYS.PATH RESOLUTION (Streamlit Community Cloud & Local)
# Automatically ensures project root is at index 0 of sys.path
_THIS_FILE = Path(__file__).resolve()
_FRONTEND_DIR = _THIS_FILE.parent
_ROOT_DIR = _FRONTEND_DIR.parent

for _p in [
    str(_ROOT_DIR),
    str(_FRONTEND_DIR),
    os.path.abspath("."),
    os.getcwd(),
    "/mount/src/voice-notes-ai",
]:
    if _p and os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

BASE_DIR = _ROOT_DIR

# 2. IMMEDIATE STREAMLIT SECRETS SYNC BEFORE ANY OTHER IMPORT OR SUBPROCESS
# Copies st.secrets into os.environ so DEEPGRAM_API_KEY never fails with 401 Unauthorized on Streamlit Cloud
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)) and not os.getenv(k):
                os.environ[k] = str(v).strip().strip("'\"")
        if "DEEPGRAM_API_KEY" in st.secrets:
            os.environ["DEEPGRAM_API_KEY"] = str(st.secrets["DEEPGRAM_API_KEY"]).strip().strip("'\"")
        if "DEEPGRAM_MODEL" in st.secrets:
            os.environ["DEEPGRAM_MODEL"] = str(st.secrets["DEEPGRAM_MODEL"]).strip().strip("'\"")
        if "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"]).strip().strip("'\"")
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"]).strip().strip("'\"")
except Exception:
    pass

import time
from datetime import datetime, timezone
from uuid import uuid4
import requests
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# Load Environment Variables from local .env (if present)
load_dotenv(BASE_DIR / ".env")

# 3. DIRECT IMPORT OF BACKEND SERVICES (With Safe Streamlit Cloud Fallbacks)
try:
    from backend.config import AUDIO_DIR
except Exception:
    AUDIO_DIR = Path(tempfile.gettempdir()) / "voice_notes_ai_audio"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

try:
    from backend.database import init_db, save_note, get_all_notes, delete_note
    from backend.deepgram_service import transcribe_audio
    from backend.vector_store import add_note_to_vector_db, semantic_search, delete_note_from_vector_db
    from backend.ai_service import analyze_transcript
except (ImportError, ModuleNotFoundError):
    # Retry after explicitly injecting ROOT_DIR
    if str(BASE_DIR) in sys.path:
        sys.path.remove(str(BASE_DIR))
    sys.path.insert(0, str(BASE_DIR))
    from backend.database import init_db, save_note, get_all_notes, delete_note
    from backend.deepgram_service import transcribe_audio
    from backend.vector_store import add_note_to_vector_db, semantic_search, delete_note_from_vector_db
    from backend.ai_service import analyze_transcript

# Initialize database schema and migrations
try:
    init_db()
except Exception:
    pass


def format_time(seconds) -> str:
    """Formats seconds (float/int) into MM:SS format."""
    try:
        total_s = int(float(seconds))
        m = total_s // 60
        s = total_s % 60
        return f"{m:02d}:{s:02d}"
    except Exception:
        return "00:00"


# 2. Streamlit Page Configuration
st.set_page_config(
    page_title="AI Voice Assistant Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Custom Modern Styling (Glassmorphism & Gradients)
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
        padding: 26px 30px;
        margin-bottom: 22px;
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
        margin-bottom: 10px;
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Utterance Card */
    .utterance-row {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
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

# 4. Fetch all notes
try:
    all_notes = get_all_notes()
except Exception:
    all_notes = []

# 5. Sidebar Navigation & Info
with st.sidebar:
    st.markdown("### 🎙️ AI Voice Studio")
    st.caption("Deepgram Nova-3 • Diarization • Traceability")
    
    st.divider()
    
    # Check & Clean API Key
    current_key = os.getenv("DEEPGRAM_API_KEY") or st.session_state.get("custom_dg_key")
    if current_key:
        current_key = str(current_key).strip().strip("'\"")
        if current_key.lower().startswith("token "):
            current_key = current_key[6:].strip()
        elif current_key.lower().startswith("bearer "):
            current_key = current_key[7:].strip()
        os.environ["DEEPGRAM_API_KEY"] = current_key

    is_placeholder = bool(current_key and ("your_" in current_key.lower() or "placeholder" in current_key.lower()))
    is_valid_format = bool(current_key and len(current_key) > 10 and not is_placeholder)

    # Live Metrics
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Saved Notes", len(all_notes))
    with col_m2:
        model_name = os.getenv("DEEPGRAM_MODEL", "nova-3")
        st.metric("Model", model_name.upper())
    
    st.divider()
    
    # Engine Status (Direct Core Engine)
    st.success("🟢 AI Core Engine: Ready", icon="✅")
    
    # Deepgram Key Status
    if is_valid_format:
        masked_key = f"{current_key[:4]}...{current_key[-4:]}"
        st.success(f"🟢 Deepgram: Ready (`{masked_key}`)", icon="🔑")
        with st.expander("⚙️ Change / Update API Key"):
            new_k = st.text_input("New Deepgram Key", type="password", placeholder="Paste key...", key="update_key_input")
            if st.button("Apply New Key", key="btn_apply_key"):
                if new_k.strip():
                    cleaned_k = new_k.strip().strip("'\"")
                    if cleaned_k.lower().startswith("token "):
                        cleaned_k = cleaned_k[6:].strip()
                    elif cleaned_k.lower().startswith("bearer "):
                        cleaned_k = cleaned_k[7:].strip()
                    os.environ["DEEPGRAM_API_KEY"] = cleaned_k
                    st.session_state["custom_dg_key"] = cleaned_k
                    st.rerun()
    else:
        if is_placeholder:
            st.error("🔴 Placeholder API Key Detected", icon="⚠️")
            st.caption("Please paste your real key from console.deepgram.com below:")
        else:
            st.error("🔴 Deepgram: Key Missing", icon="⚠️")
            
        user_input_key = st.text_input(
            "Enter Deepgram API Key",
            type="password",
            placeholder="Paste your key here...",
            help="Get a free key at console.deepgram.com",
            key="initial_key_input"
        )
        if user_input_key:
            cleaned_k = user_input_key.strip().strip("'\"")
            if cleaned_k.lower().startswith("token "):
                cleaned_k = cleaned_k[6:].strip()
            elif cleaned_k.lower().startswith("bearer "):
                cleaned_k = cleaned_k[7:].strip()
            os.environ["DEEPGRAM_API_KEY"] = cleaned_k
            st.session_state["custom_dg_key"] = cleaned_k
            st.success("Key applied successfully!")
            st.rerun()

    ai_llm_ok = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY"))
    if ai_llm_ok:
        st.success("🟢 LLM Intelligence: Active", icon="🧠")
    else:
        st.success("🟢 Action Extractor: Ready (Built-in NLP)", icon="⚡")
        
    st.divider()
    st.markdown("#### 🎯 Traceability Features")
    st.markdown("""
    - 👥 **Speaker Diarization**: Detects multiple speakers (`Speaker 1`, `Speaker 2`).
    - ⏱️ **Timestamp Linking**: Every action item and transcript line links to exact audio seconds.
    - 🧹 **Zero-Orphan Storage**: Discard leaves zero disk traces; Deleting notes cleans up audio files.
    """)

# 6. Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="badge-pill">⚡ Task 3: Conversation Traceability & Diarization</div>
    <h1 class="hero-title">AI Voice Notes Studio</h1>
    <p class="hero-subtitle">Every decision, action item, and speaker segment is linked to its exact audio timestamp for one-click verification.</p>
</div>
""", unsafe_allow_html=True)

# 7. Initialize session state keys
if "recorder_key" not in st.session_state:
    st.session_state["recorder_key"] = 0
if "curr_audio_bytes" not in st.session_state:
    st.session_state["curr_audio_bytes"] = None
if "curr_audio_mime" not in st.session_state:
    st.session_state["curr_audio_mime"] = "audio/wav"
if "preview_seek_time" not in st.session_state:
    st.session_state["preview_seek_time"] = 0
if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "action_items" not in st.session_state:
    st.session_state["action_items"] = []
if "utterances" not in st.session_state:
    st.session_state["utterances"] = []

# 8. Main Tabs Layout
tab_record, tab_search, tab_library = st.tabs([
    "🎙️ Record & Traceability Assistant", 
    "🔍 AI Semantic Search", 
    f"📚 Notes & Audio Library ({len(all_notes)})"
])

# ==========================================================
# TAB 1: RECORD & ASSISTANT (IN-MEMORY AUDIO & TRACEABILITY)
# ==========================================================
with tab_record:
    col_rec, col_edit = st.columns([1, 1], gap="large")
    
    # --- Left Column: Audio Capture ---
    with col_rec:
        st.markdown("### 1. Ingest Audio (In-Memory)")
        st.write("Record via mic or upload. Audio remains in memory until you choose to save.")
        
        mode = st.radio("Input Source", ["🎤 Microphone Recording", "📁 Upload Audio File"], horizontal=True)
        
        if mode == "🎤 Microphone Recording":
            audio = mic_recorder(
                start_prompt="🔴 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                key=f"voice_recorder_{st.session_state['recorder_key']}"
            )
            if audio and audio.get("bytes"):
                if st.session_state["curr_audio_bytes"] != audio["bytes"]:
                    st.session_state["curr_audio_bytes"] = audio["bytes"]
                    st.session_state["curr_audio_mime"] = "audio/wav"
                    st.session_state["preview_seek_time"] = 0
        else:
            uploaded_file = st.file_uploader(
                "Upload voice note (.wav, .mp3, .m4a, .ogg)",
                type=["wav", "mp3", "m4a", "ogg", "webm"],
                key="file_uploader_input"
            )
            if uploaded_file is not None:
                new_bytes = uploaded_file.read()
                if st.session_state["curr_audio_bytes"] != new_bytes:
                    st.session_state["curr_audio_bytes"] = new_bytes
                    st.session_state["curr_audio_mime"] = uploaded_file.type or "audio/wav"
                    st.session_state["preview_seek_time"] = 0
        
        # Audio Player with Seek Support (Preview State)
        if st.session_state["curr_audio_bytes"]:
            st.markdown("#### 🎧 Audio Preview")
            seek_val = int(st.session_state.get("preview_seek_time", 0))
            if seek_val > 0:
                st.caption(f"Jumped to: **{format_time(seek_val)}**")
            
            st.audio(
                st.session_state["curr_audio_bytes"],
                format=st.session_state["curr_audio_mime"],
                start_time=seek_val
            )
            
            col_t_btn, col_d_btn = st.columns([2, 1])
            with col_t_btn:
                if st.button("✨ Transcribe & Diarize with Deepgram", type="primary", use_container_width=True):
                    active_key = os.getenv("DEEPGRAM_API_KEY") or st.session_state.get("custom_dg_key")
                    if not active_key:
                        st.error("Please enter a valid Deepgram API Key in the sidebar first.")
                    else:
                        with st.spinner("Transcribing with Deepgram Nova-3 (Diarization & Utterances enabled)..."):
                            try:
                                res = transcribe_audio(st.session_state["curr_audio_bytes"], st.session_state["curr_audio_mime"])
                                transcript_text = res.get("transcript", "")
                                dg_summary = res.get("deepgram_summary", "")
                                utterances = res.get("utterances", [])

                                if not transcript_text:
                                    st.warning("No speech detected in audio. Please try recording again.")
                                else:
                                    analysis = analyze_transcript(transcript_text, dg_summary, utterances)
                                    st.session_state["transcript"] = transcript_text
                                    st.session_state["summary"] = analysis.get("summary", "")
                                    st.session_state["action_items"] = analysis.get("action_items", [])
                                    st.session_state["utterances"] = utterances
                                    st.session_state["preview_seek_time"] = 0
                                    st.toast("Diarized Transcription & Linked Tasks Ready!", icon="🎉")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Transcription failed: {e}")
            
            with col_d_btn:
                # Discard leaves ZERO files on disk and ZERO entries in DB
                if st.button("🗑️ Discard", use_container_width=True):
                    st.session_state["curr_audio_bytes"] = None
                    st.session_state["transcript"] = ""
                    st.session_state["summary"] = ""
                    st.session_state["action_items"] = []
                    st.session_state["utterances"] = []
                    st.session_state["preview_seek_time"] = 0
                    st.session_state["recorder_key"] += 1
                    st.toast("Recording discarded. Zero files kept.", icon="🗑️")
                    st.rerun()

    # --- Right Column: Review, Edit & Timestamped Action Items ---
    with col_edit:
        st.markdown("### 2. Traceability & Action Items")
        
        word_count = len(st.session_state["transcript"].split()) if st.session_state["transcript"] else 0
        num_spks = len(set([u.get('speaker', 'Speaker 1') for u in st.session_state["utterances"]])) if st.session_state["utterances"] else 1
        st.caption(f"📊 Stats: **{word_count} words** • **{num_spks} speaker(s)** • **{len(st.session_state['action_items'])} linked action item(s)**")
        
        note_title = st.text_input(
            "Note Title",
            placeholder="e.g. Client Design Review, Team Standup, Lecture Notes"
        )
        
        edit_tab_actions, edit_tab_utterances, edit_tab_raw = st.tabs([
            "⚡ Action Items & Decisions",
            "🗣️ Timestamped Segments",
            "📝 Raw Transcript"
        ])
        
        # Sub-tab: Action Items with Clickable Playback
        with edit_tab_actions:
            if st.session_state["summary"]:
                st.markdown("**📌 Executive Summary:**")
                st.info(st.session_state["summary"])
                
            if st.session_state["action_items"]:
                st.markdown("**✅ Linked Action Items & Decisions:**")
                st.caption("Click any timestamp to play the exact moment from audio:")
                for a_idx, item in enumerate(st.session_state["action_items"]):
                    col_btn_time, col_task_text = st.columns([1, 3])
                    st_val = float(item.get("start_time", 0.0))
                    spk_val = item.get("speaker", "Speaker 1")
                    
                    with col_btn_time:
                        if st.button(f"▶ {format_time(st_val)} • {spk_val}", key=f"prev_act_seek_{a_idx}"):
                            st.session_state["preview_seek_time"] = int(st_val)
                            st.rerun()
                    with col_task_text:
                        st.checkbox(item.get("task", ""), value=False, key=f"prev_chk_{a_idx}")
            else:
                st.write("No action items detected yet. Transcribe an audio recording above.")

        # Sub-tab: Timestamped Utterances with Clickable Playback
        with edit_tab_utterances:
            if st.session_state["utterances"]:
                st.markdown("**🗣️ Diarized Speaker Segments:**")
                st.caption("Click any segment to seek audio player to that second:")
                for u_idx, u in enumerate(st.session_state["utterances"]):
                    col_u_btn, col_u_txt = st.columns([1, 3])
                    u_start = float(u.get("start", 0.0))
                    u_spk = u.get("speaker", "Speaker 1")
                    with col_u_btn:
                        if st.button(f"▶ {format_time(u_start)} • {u_spk}", key=f"prev_u_seek_{u_idx}"):
                            st.session_state["preview_seek_time"] = int(u_start)
                            st.rerun()
                    with col_u_txt:
                        st.markdown(f"**{u_spk}:** {u.get('text', '')}")
            else:
                st.write("Diarized conversation timeline will appear here after transcription.")

        # Sub-tab: Raw Transcript Text Area
        with edit_tab_raw:
            edited_text = st.text_area(
                "Voice Transcript",
                value=st.session_state["transcript"],
                height=180,
                placeholder="Full raw transcript...",
                label_visibility="collapsed"
            )
            if edited_text.strip() and edited_text != st.session_state["transcript"]:
                if st.button("🔄 Re-extract Actions from Edits"):
                    with st.spinner("Analyzing updated text..."):
                        a_data = analyze_transcript(edited_text, utterances=st.session_state.get("utterances", []))
                        st.session_state["transcript"] = edited_text
                        st.session_state["summary"] = a_data.get("summary", "")
                        st.session_state["action_items"] = a_data.get("action_items", [])
                        st.rerun()

        # Save Note with Zero-Orphan Audio Policy
        if st.button("💾 Save Note & Audio", type="primary", use_container_width=True):
            final_text = edited_text.strip() if 'edited_text' in locals() and edited_text.strip() else st.session_state["transcript"].strip()
            if not note_title.strip() or not final_text:
                st.warning("Please provide a note title and transcription content before saving.")
            else:
                with st.spinner("Saving note, persisting audio, and indexing embeddings..."):
                    try:
                        note_id = str(uuid4())
                        created_at = datetime.now(timezone.utc).isoformat()
                        audio_path = ""

                        # Persist audio to data/audio/<uuid>.<ext> ONLY upon Save
                        if st.session_state["curr_audio_bytes"]:
                            mime_t = st.session_state.get("curr_audio_mime", "audio/wav")
                            ext = "wav"
                            if "mp3" in mime_t:
                                ext = "mp3"
                            elif "ogg" in mime_t:
                                ext = "ogg"
                            elif "webm" in mime_t:
                                ext = "webm"
                            elif "m4a" in mime_t:
                                ext = "m4a"
                            
                            dest_file = AUDIO_DIR / f"{note_id}.{ext}"
                            with open(dest_file, "wb") as f:
                                f.write(st.session_state["curr_audio_bytes"])
                            audio_path = str(dest_file)

                        # Save to SQLite
                        save_note(
                            note_id,
                            note_title.strip(),
                            final_text,
                            created_at,
                            summary=st.session_state["summary"],
                            action_items=st.session_state["action_items"],
                            audio_path=audio_path,
                            utterances=st.session_state["utterances"]
                        )

                        # Index in ChromaDB
                        add_note_to_vector_db(
                            note_id,
                            note_title.strip(),
                            final_text,
                            created_at,
                            summary=st.session_state["summary"]
                        )

                        st.toast(f"Saved '{note_title}' with audio verification!", icon="✅")
                        st.session_state["curr_audio_bytes"] = None
                        st.session_state["transcript"] = ""
                        st.session_state["summary"] = ""
                        st.session_state["action_items"] = []
                        st.session_state["utterances"] = []
                        st.session_state["preview_seek_time"] = 0
                        st.session_state["recorder_key"] += 1
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
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
            placeholder="e.g. Who committed to the project deadline? Or: budget tasks",
            label_visibility="collapsed"
        )
    with col_btn:
        search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
    
    if search_query.strip() and (search_clicked or search_query):
        with st.spinner("Searching vector index..."):
            try:
                matches = semantic_search(search_query.strip())
                
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
            except Exception as e:
                st.error(f"Search request failed: {e}")

# ==========================================================
# TAB 3: NOTES & AUDIO LIBRARY (CLICKABLE VERIFICATION)
# ==========================================================
with tab_library:
    col_head, col_action = st.columns([4, 1])
    with col_head:
        st.markdown(f"### 📚 Saved Notes & Conversation Traceability ({len(all_notes)})")
    with col_action:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.rerun()
            
    if not all_notes:
        st.info("No notes saved yet! Record or ingest audio in the 'Record & Traceability Assistant' tab.")
    else:
        for idx, note in enumerate(all_notes):
            created_str = note.get('created_at', '')[:19].replace('T', ' ')
            words = len(note['transcript'].split())
            items = note.get("action_items", [])
            utterances = note.get("utterances", [])
            summary = note.get("summary", "")
            audio_path = note.get("audio_path", "")
            
            with st.expander(f"📝 {note['title']}  •  {created_str}  ({words} words • {len(items)} task(s))"):
                # Audio Playback from audio_path with interactive seek
                seek_key = f"saved_seek_{note['id']}"
                saved_seek = int(st.session_state.get(seek_key, 0))
                
                if audio_path and os.path.exists(audio_path):
                    st.markdown("#### 🎧 Verified Audio Recording")
                    if saved_seek > 0:
                        st.caption(f"Playing from: **{format_time(saved_seek)}**")
                    with open(audio_path, "rb") as af:
                        audio_data = af.read()
                    st.audio(audio_data, format="audio/wav", start_time=saved_seek)
                elif audio_path:
                    st.caption("ℹ️ Audio file was moved or deleted from disk.")
                
                if summary:
                    st.markdown("**📌 Executive Summary:**")
                    st.info(summary)
                    
                # Action Items with Clickable Timestamp Buttons
                if items:
                    st.markdown("**✅ Verified Action Items & Decisions:**")
                    st.caption("Click any timestamp button to verify against the original audio:")
                    for i_idx, itm in enumerate(items):
                        col_seek_btn, col_task = st.columns([1, 3])
                        
                        task_text = itm.get("task", itm.get("text", "")) if isinstance(itm, dict) else str(itm)
                        task_spk = itm.get("speaker", "Speaker 1") if isinstance(itm, dict) else "Speaker 1"
                        task_st = float(itm.get("start_time", 0.0)) if isinstance(itm, dict) else 0.0
                        
                        with col_seek_btn:
                            if audio_path and os.path.exists(audio_path):
                                if st.button(f"▶ {format_time(task_st)} • {task_spk}", key=f"saved_seek_btn_{note['id']}_{i_idx}"):
                                    st.session_state[seek_key] = int(task_st)
                                    st.rerun()
                            else:
                                st.markdown(f"`{format_time(task_st)} • {task_spk}`")
                        with col_task:
                            st.checkbox(task_text, value=False, key=f"saved_chk_{note['id']}_{i_idx}")

                # Diarized Utterances Timeline
                if utterances:
                    st.markdown("**🗣️ Diarized Speaker Segments:**")
                    with st.expander("Show Timestamped Transcript Timeline"):
                        for u_idx, u in enumerate(utterances):
                            col_u_btn, col_u_text = st.columns([1, 3])
                            u_st = float(u.get("start", 0.0))
                            u_spk = u.get("speaker", "Speaker 1")
                            with col_u_btn:
                                if audio_path and os.path.exists(audio_path):
                                    if st.button(f"▶ {format_time(u_st)} • {u_spk}", key=f"saved_u_btn_{note['id']}_{u_idx}"):
                                        st.session_state[seek_key] = int(u_st)
                                        st.rerun()
                                else:
                                    st.markdown(f"`{format_time(u_st)} • {u_spk}`")
                            with col_u_text:
                                st.write(f"**{u_spk}:** {u.get('text', '')}")
                else:
                    st.markdown("**Transcript:**")
                    st.write(note["transcript"])

                st.caption(f"Note ID: `{note['id']}` • Created: {created_str} UTC")
                
                # Download payload with tasks, timestamps, and speakers
                action_text_lines = []
                for x in items:
                    if isinstance(x, dict):
                        action_text_lines.append(f"- [ ] [{format_time(x.get('start_time', 0.0))}] ({x.get('speaker', 'Speaker 1')}): {x.get('task', '')}")
                    else:
                        action_text_lines.append(f"- [ ] {x}")
                
                download_text = f"Title: {note['title']}\nDate: {created_str}\n\nExecutive Summary:\n{summary}\n\nAction Items & Decisions:\n" + \
                                "\n".join(action_text_lines) + f"\n\nFull Transcript:\n{note['transcript']}"
                
                col_down, col_del = st.columns([1, 1])
                with col_down:
                    st.download_button(
                        "📥 Download .TXT (with Tasks & Timestamps)",
                        data=download_text,
                        file_name=f"{note['title'].replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"dl_note_{note['id']}",
                        use_container_width=True
                    )
                with col_del:
                    # Deletes SQLite row, ChromaDB vector, AND physical audio file from disk
                    if st.button("🗑️ Delete Note & Audio", key=f"del_note_{note['id']}", type="secondary", use_container_width=True):
                        try:
                            delete_note(note['id'])
                            delete_note_from_vector_db(note['id'])
                            st.toast(f"Deleted '{note['title']}' and removed audio from storage.", icon="🗑️")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete note: {e}")
