# importing libraries
import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# Deepgram configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "multi")

DEEPGRAM_MIP_OPT_OUT = (
    os.getenv("DEEPGRAM_MIP_OPT_OUT", "true").lower() == "true"
)

# Determine writable data directory safely
# First try project root 'data' directory
candidate_data_dir = BASE_DIR / "data"
DATA_DIR = candidate_data_dir

try:
    candidate_data_dir.mkdir(parents=True, exist_ok=True)
    # Test writability by creating and removing a temporary canary file
    test_file = candidate_data_dir / ".write_canary"
    test_file.touch()
    test_file.unlink()
except Exception:
    # Fallback to system temp directory (essential for read-only mounts e.g. Streamlit Cloud)
    DATA_DIR = Path(tempfile.gettempdir()) / "voice_notes_ai_data"
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        DATA_DIR = Path(tempfile.gettempdir())

DATABASE_PATH = DATA_DIR / "voice_notes.db"
CHROMA_PATH = DATA_DIR / "chroma_db"
AUDIO_DIR = DATA_DIR / "audio"

try:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    AUDIO_DIR = Path(tempfile.gettempdir()) / "voice_notes_ai_audio"
    try:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        AUDIO_DIR = Path(tempfile.gettempdir())

MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100MB audio limit
MAX_FILE_SIZE = MAX_AUDIO_SIZE


