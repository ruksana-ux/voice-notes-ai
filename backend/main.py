import base64
import os
from typing import List, Optional, Any
from uuid import uuid4
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from .database import init_db, save_note, get_all_notes, delete_note
from .deepgram_service import transcribe_audio
from .vector_store import add_note_to_vector_db, semantic_search, delete_note_from_vector_db
from .ai_service import analyze_transcript, ask_cross_meeting_memory
from .config import MAX_AUDIO_SIZE, AUDIO_DIR

app = FastAPI(title="Deepgram AI Voice Assistant", version="3.0")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def home():
    return {
        "name": "AI Voice Assistant API",
        "status": "healthy",
        "version": "3.0",
        "features": [
            "Deepgram Nova-3 Diarization & Utterances",
            "ChromaDB Semantic Vector Search",
            "Timestamped AI Action-Item Extractor",
            "Zero-Orphan Audio Storage"
        ]
    }


# 1. Speech-to-Text with Diarization, Utterances & AI Analysis (In-Memory Only)
@app.post("/transcribe")
@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    allowed_types = [
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/mpeg",
        "audio/mp4",
        "audio/ogg"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio format."
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty."
        )

    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file is too large. Maximum supported limit is {MAX_AUDIO_SIZE // (1024 * 1024)}MB."
        )

    try:
        # Transcribe with diarize=true and utterances=true (never saved to disk here)
        transcribe_res = transcribe_audio(
            audio_bytes,
            file.content_type
        )
        transcript = transcribe_res.get("transcript", "")
        dg_summary = transcribe_res.get("deepgram_summary", "")
        utterances = transcribe_res.get("utterances", [])

        if not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail="No speech detected. Please record again."
            )

        # AI Action Item & Summary Extraction with timestamp linking
        analysis = analyze_transcript(transcript, dg_summary, utterances)

        return {
            "transcript": transcript,
            "summary": analysis.get("summary", ""),
            "action_items": analysis.get("action_items", []),
            "utterances": utterances
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Transcription failed: {str(e)}"
        )


# 2. Extract action items directly from text
class AnalyzeRequest(BaseModel):
    transcript: str
    utterances: Optional[List[Any]] = []


@app.post("/notes/analyze")
@app.post("/api/notes/analyze")
def analyze_text(req: AnalyzeRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")
    return analyze_transcript(req.transcript.strip(), utterances=req.utterances)


# 3. Save note with action items, utterances and audio bytes
class NoteRequest(BaseModel):
    title: str
    transcript: str
    summary: Optional[str] = ""
    action_items: Optional[List[Any]] = []
    utterances: Optional[List[Any]] = []
    audio_base64: Optional[str] = None
    audio_ext: Optional[str] = "wav"


@app.post("/notes")
@app.post("/api/notes")
def create_note(note: NoteRequest):
    if not note.title.strip() or not note.transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="Title and transcript cannot be empty."
        )

    note_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    audio_path = ""

    # Persist audio to disk ONLY upon user confirmation (Save)
    if note.audio_base64:
        try:
            raw_bytes = base64.b64decode(note.audio_base64)
            ext = (note.audio_ext or "wav").lstrip(".")
            dest = AUDIO_DIR / f"{note_id}.{ext}"
            with open(dest, "wb") as f:
                f.write(raw_bytes)
            audio_path = str(dest)
        except Exception as e:
            # Continue even if audio save fails
            pass

    try:
        save_note(
            note_id,
            note.title.strip(),
            note.transcript.strip(),
            created_at,
            summary=note.summary.strip() if note.summary else "",
            action_items=note.action_items or [],
            audio_path=audio_path,
            utterances=note.utterances or []
        )

        add_note_to_vector_db(
            note_id,
            note.title.strip(),
            note.transcript.strip(),
            created_at,
            summary=note.summary.strip() if note.summary else "",
            utterances=note.utterances or []
        )

        return {
            "message": "Note saved successfully",
            "note_id": note_id,
            "audio_path": audio_path
        }

    except Exception as e:
        # Cleanup audio file if database save failed
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Could not save or index note: {str(e)}"
        )


# 4. Display saved notes
@app.get("/notes")
@app.get("/api/notes")
def list_notes():
    return get_all_notes()


# 5. AI semantic search
@app.get("/search")
@app.get("/api/search")
def search_voice_notes(query: str):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a search query."
        )

    return semantic_search(query)


# 6. Delete a note (Deletes SQLite row, ChromaDB vector, and physical audio file)
@app.delete("/notes/{note_id}")
@app.delete("/api/notes/{note_id}")
def delete_voice_note(note_id: str):
    deleted = delete_note(note_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Note not found."
        )

    delete_note_from_vector_db(note_id)

    return {
        "message": "Note and associated audio file deleted successfully",
        "note_id": note_id
    }


# 7. Cross-Meeting Conversational Memory RAG Q&A with Clickable Timestamp Citations
class MemoryAskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 6


@app.post("/memory/ask")
@app.post("/api/memory/ask")
def memory_ask(req: MemoryAskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        return ask_cross_meeting_memory(
            question=req.question.strip(),
            top_k=req.top_k or 6
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cross-meeting memory query failed: {str(e)}"
        )

