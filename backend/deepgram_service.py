
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")

def transcribe_audio(audio_bytes, content_type="audio/wav"):

    url = "https://api.deepgram.com/v1/listen"

    headers = {
        "Authorization": f"Token {API_KEY}",
        "Content-Type": content_type
    }

    params = {
        "model": "nova-3",
        "smart_format": "true",
        "language": "en"
    }

    response = requests.post(
        url,
        headers=headers,
        params=params,
        data=audio_bytes
    )

    response.raise_for_status()

    result = response.json()

    transcript = (
        result["results"]["channels"][0]
        ["alternatives"][0]["transcript"]
    )

    return transcript
