import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")


def transcribe_audio(audio_bytes, content_type="audio/wav"):
    url = "https://api.deepgram.com/v1/listen"

    key = os.getenv("DEEPGRAM_API_KEY") or API_KEY
    if not key:
        raise ValueError("DEEPGRAM_API_KEY is not configured. Please add your Deepgram API key in the sidebar, .env, or Streamlit Secrets.")

    # Clean key: remove whitespace, outer quotes, and accidental prefixes
    key = str(key).strip().strip("'\"")
    if key.lower().startswith("token "):
        key = key[6:].strip()
    elif key.lower().startswith("bearer "):
        key = key[7:].strip()

    if "your_" in key.lower() or "placeholder" in key.lower():
        raise ValueError(f"You are using a placeholder key ('{key}'). Please replace it with your real Deepgram API key from https://console.deepgram.com")

    headers = {
        "Authorization": f"Token {key}",
        "Content-Type": content_type
    }

    params = {
        "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
        "smart_format": "true",
        "punctuate": "true",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            data=audio_bytes
        )
        response.raise_for_status()
    except requests.HTTPError as err:
        try:
            err_json = response.json()
            err_msg = err_json.get("err_msg") or err_json.get("error") or err_json.get("message")
            if response.status_code == 401:
                raise ValueError(f"Deepgram 401 Unauthorized: {err_msg or 'Invalid API key or expired credits'}. Please verify your key at https://console.deepgram.com")
            elif err_msg:
                raise ValueError(f"Deepgram Error ({response.status_code}): {err_msg}")
        except ValueError:
            raise
        except Exception:
            pass
        raise err

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
