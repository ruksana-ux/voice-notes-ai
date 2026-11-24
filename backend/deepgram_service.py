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
        "diarize": "true",
        "utterances": "true",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            data=audio_bytes,
            timeout=180.0
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

    # Parse utterances with 1-indexed speaker labels and exact timestamps
    raw_utterances = result.get("results", {}).get("utterances", [])
    parsed_utterances = []

    if raw_utterances:
        for u in raw_utterances:
            spk_idx = u.get("speaker", 0)
            spk_label = f"Speaker {int(spk_idx) + 1}"
            start_t = round(float(u.get("start", 0.0)), 2)
            end_t = round(float(u.get("end", 0.0)), 2)
            text_val = u.get("transcript", "").strip()
            if text_val:
                parsed_utterances.append({
                    "speaker": spk_label,
                    "start": start_t,
                    "end": end_t,
                    "text": text_val
                })

    # Fallback to words grouping if raw_utterances is missing
    if not parsed_utterances and alternatives and alternatives[0].get("words"):
        words = alternatives[0].get("words", [])
        curr_speaker = None
        curr_start = 0.0
        curr_end = 0.0
        curr_words = []

        for w in words:
            w_spk = f"Speaker {int(w.get('speaker', 0)) + 1}"
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", 0.0))
            w_punct = w.get("punctuated_word") or w.get("word") or ""

            if curr_speaker is None or w_spk != curr_speaker:
                if curr_words and curr_speaker is not None:
                    parsed_utterances.append({
                        "speaker": curr_speaker,
                        "start": round(curr_start, 2),
                        "end": round(curr_end, 2),
                        "text": " ".join(curr_words).strip()
                    })
                curr_speaker = w_spk
                curr_start = w_start
                curr_end = w_end
                curr_words = [w_punct]
            else:
                curr_words.append(w_punct)
                curr_end = w_end

        if curr_words and curr_speaker is not None:
            parsed_utterances.append({
                "speaker": curr_speaker,
                "start": round(curr_start, 2),
                "end": round(curr_end, 2),
                "text": " ".join(curr_words).strip()
            })

    # Absolute fallback: single segment
    if not parsed_utterances and transcript:
        duration = float(result.get("metadata", {}).get("duration", 0.0))
        parsed_utterances = [{
            "speaker": "Speaker 1",
            "start": 0.0,
            "end": round(duration, 2),
            "text": transcript
        }]

    return {
        "transcript": transcript,
        "deepgram_summary": summary,
        "utterances": parsed_utterances
    }
