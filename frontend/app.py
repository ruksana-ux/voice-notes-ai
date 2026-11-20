# Users can record their voice, transcribe it, edit the transcript, save notes, search using AI and download notes.

import os
import socket
import threading
import time
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

st.set_page_config(
    page_title="AI Voice Notes",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ AI Voice Notes Application")

st.write(
    "Record your voice, convert it into text "
    "and search your notes using AI."
)

st.info(
    "Speech recognition can make mistakes. "
    "Please review your transcript before saving."
)


# Record audio
st.subheader("1. Record your voice")

if "recorder_key" not in st.session_state:
    st.session_state["recorder_key"] = 0

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key=f"voice_recorder_{st.session_state['recorder_key']}"
)


if audio and audio.get("bytes"):

    audio_bytes = audio["bytes"]

    st.audio(
        audio_bytes,
        format="audio/wav"
    )

    col_transcribe, col_discard = st.columns(2)

    with col_transcribe:
        # Transcribe audio
        if st.button("Transcribe with Deepgram", type="primary"):

            try:

                response = requests.post(
                    f"{BACKEND}/transcribe",
                    files={
                        "file": (
                            "recording.wav",
                            audio_bytes,
                            "audio/wav"
                        )
                    },
                    timeout=120
                )

                response.raise_for_status()

                transcript = response.json()["transcript"]

                st.session_state["transcript"] = transcript

                st.success("Transcription completed!")

            except requests.RequestException as e:

                st.error(f"Transcription error: {e}")

    with col_discard:
        # Delete / Discard this recording
        if st.button("🗑️ Discard Recording"):
            st.session_state["transcript"] = ""
            st.session_state["recorder_key"] += 1
            st.rerun()


# Edit transcript
st.subheader("2. Review and edit transcript")

if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""

transcript = st.text_area(
    "Your transcript",
    value=st.session_state["transcript"],
    height=180,
    key="edited_transcript"
)


# Save note
st.subheader("3. Save your voice note")

title = st.text_input(
    "Enter note title",
    placeholder="Example: Machine Learning Class"
)

if st.button("Save Note"):

    if not title.strip() or not transcript.strip():

        st.warning("Enter a title and transcript.")

    else:

        try:

            response = requests.post(
                f"{BACKEND}/notes",
                json={
                    "title": title,
                    "transcript": transcript
                },
                timeout=60
            )

            response.raise_for_status()

            st.success("Your note has been saved!")
            st.session_state["transcript"] = ""
            st.session_state["recorder_key"] += 1
            st.rerun()

        except requests.RequestException as e:

            st.error(f"Saving failed: {e}")


# Semantic search
st.divider()

st.subheader("4. AI Semantic Search")

query = st.text_input(
    "Search your notes by meaning",
    placeholder="Example: What did I discuss in my meeting?"
)

if st.button("Search with AI"):

    if query.strip():

        try:

            response = requests.get(
                f"{BACKEND}/search",
                params={"query": query},
                timeout=60
            )

            response.raise_for_status()

            results = response.json()

            if results:

                for result in results:

                    st.markdown(
                        f"### {result['metadata']['title']}"
                    )

                    st.write(result["transcript"])

                    st.caption(
                        f"Similarity distance: "
                        f"{result['distance']:.4f}"
                    )

            else:

                st.info("No matching notes found.")

        except requests.RequestException as e:

            st.error(f"Search failed: {e}")


# Display saved notes
st.divider()

col_header, col_ref = st.columns([4, 1])
with col_header:
    st.subheader("5. Your Saved Voice Notes")
with col_ref:
    if st.button("🔄 Refresh"):
        st.rerun()

try:

    response = requests.get(
        f"{BACKEND}/notes",
        timeout=30
    )

    response.raise_for_status()

    notes = response.json()

    if notes:

        for note in notes:

            with st.expander(f"📝 {note['title']}"):

                st.write(note["transcript"])

                st.caption(f"Created: {note['created_at']}")

                col_dl, col_del = st.columns([1, 1])

                with col_dl:
                    st.download_button(
                        "📥 Download TXT",
                        data=note["transcript"],
                        file_name=f"{note['title']}.txt",
                        mime="text/plain",
                        key=f"download_{note['id']}"
                    )

                with col_del:
                    if st.button("🗑️ Delete Note", key=f"delete_{note['id']}", type="secondary"):

                        try:

                            del_resp = requests.delete(
                                f"{BACKEND}/notes/{note['id']}",
                                timeout=30
                            )

                            del_resp.raise_for_status()

                            st.success(f"Deleted '{note['title']}'!")
                            st.rerun()

                        except requests.RequestException as e:

                            st.error(f"Could not delete note: {e}")

    else:

        st.info("No saved notes yet.")

except requests.RequestException as e:

    st.error(f"Could not load notes: {e}")
