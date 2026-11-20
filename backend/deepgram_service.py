import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")


def transcribe_audio(audio_bytes, content_type="audio/wav"):
    url = "https://api.deepgram.com/v1/listen"

    key = os.getenv("DEEPGRAM_API_KEY") or API_KEY
    if not key:
        raise ValueError("DEEPGRAM_API_KEY is not configured. Please add it to your .env or Streamlit Secrets.")

    headers = {
        "Authorization": f"Token {key}",
        "Content-Type": content_type
    }

    params = {
        "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
        "smart_format": "true",
        "punctuate": "true",
    }

    response = requests.post(
        url,
        headers=headers,
        params=params,
        data=audio_bytes
    )

    response.raise_for_status()
    result = response.json()

    channels = result.get("results", {}).get("channels", [{}])
    alternatives = channels[0].get("alternatives", [{}]) if channels else [{}]
    transcript = alternatives[0].get("transcript", "").strip()

    summary = (
        result.get("results", {}).get("summary", {}).get("short", "") or
        alternatives[0].get("summary", {}).get("short", "")
    )

    return {
        "transcript": transcript,
        "deepgram_summary": summary
    }
