# importing libraries
import os
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

# Data directories
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "voice_notes.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)

MAX_AUDIO_SIZE = 20 * 1024 * 1024
