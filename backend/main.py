# importing libraries
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from uuid import uuid4
from typing import List, Optional

from .database import init_db, save_note, get_all_notes, delete_note
from .deepgram_service import transcribe_audio
from .vector_store import add_note_to_vector_db, semantic_search, delete_note_from_vector_db
from .ai_service import analyze_transcript
from .config import MAX_AUDIO_SIZE


app = FastAPI(title="Deepgram AI Voice Assistant", version="2.0")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def home():
    return {
        "name": "AI Voice Assistant API",
        "status": "healthy",
        "features": ["Deepgram Nova-3 Transcription", "ChromaDB Semantic Search", "AI Action Item Extractor"]
    }


# 1. Speech-to-Text with AI Analysis
@app.post("/transcribe")
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
            detail="Audio file is too large."
        )

    try:
        transcribe_res = transcribe_audio(
            audio_bytes,
            file.content_type
        )
        transcript = transcribe_res.get("transcript", "")
        dg_summary = transcribe_res.get("deepgram_summary", "")

        if not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail="No speech detected. Please record again."
            )

        # AI Action Item & Summary Extraction
        analysis = analyze_transcript(transcript, dg_summary)

        return {
            "transcript": transcript,
            "summary": analysis.get("summary", ""),
            "action_items": analysis.get("action_items", [])
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

@app.post("/notes/analyze")
def analyze_text(req: AnalyzeRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")
    return analyze_transcript(req.transcript.strip())


# 3. Save note with action items and summary
class NoteRequest(BaseModel):
    title: str
    transcript: str
    summary: Optional[str] = ""
    action_items: Optional[List[str]] = []


@app.post("/notes")
def create_note(note: NoteRequest):
    if not note.title.strip() or not note.transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="Title and transcript cannot be empty."
        )

    note_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        save_note(
            note_id,
            note.title.strip(),
            note.transcript.strip(),
            created_at,
            summary=note.summary.strip() if note.summary else "",
            action_items=note.action_items or []
        )

        add_note_to_vector_db(
            note_id,
            note.title.strip(),
            note.transcript.strip(),
            created_at,
            summary=note.summary.strip() if note.summary else ""
        )

        return {
            "message": "Note saved successfully",
            "note_id": note_id
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save or index note: {str(e)}"
        )


# 4. Display saved notes
@app.get("/notes")
def list_notes():
    return get_all_notes()


# 5. AI semantic search
@app.get("/search")
def search_voice_notes(query: str):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a search query."
        )

    return semantic_search(query)


# 6. Delete a note
@app.delete("/notes/{note_id}")
def delete_voice_note(note_id: str):
    deleted = delete_note(note_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Note not found."
        )

    delete_note_from_vector_db(note_id)

    return {
        "message": "Note deleted successfully",
        "note_id": note_id
    }
