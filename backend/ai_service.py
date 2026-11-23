import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def format_timestamped_transcript(transcript: str, utterances: list = None) -> str:
    """Formats utterances into a readable transcript with timestamps and speaker tags."""
    if utterances:
        lines = []
        for u in utterances:
            start_val = u.get("start", 0.0)
            speaker = u.get("speaker", "Speaker 1")
            text = u.get("text", "")
            lines.append(f"[{start_val:.1f}s] {speaker}: {text}")
        return "\n".join(lines)
    return f"[0.0s] Speaker 1: {transcript}"


def extract_with_gemini(formatted_transcript: str, api_key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    prompt = f"""
You are an expert executive meeting assistant.
Analyze this timestamped voice transcript and extract:
1. "summary": A concise 2-3 sentence executive summary of key discussions and decisions.
2. "action_items": A list of specific actionable tasks, to-dos, or decisions mentioned.
For each action item, link it to the exact speaker who committed to it or was assigned, and its start and end timestamp in seconds.

Timestamped Transcript:
\"\"\"{formatted_transcript}\"\"\"

Format your output as strict JSON with keys:
- "summary": string
- "action_items": list of objects, each with:
    - "task": string (action item or decision description)
    - "speaker": string (e.g. "Speaker 1")
    - "start_time": float (start timestamp in seconds)
    - "end_time": float (end timestamp in seconds)

Example output:
{{
    "summary": "Speaker 1 discussed project deadlines and Speaker 2 agreed to review the budget.",
    "action_items": [
        {{
            "task": "Review and submit the budget report by Thursday",
            "speaker": "Speaker 2",
            "start_time": 12.4,
            "end_time": 18.2
        }}
    ]
}}
Do NOT wrap in markdown code blocks if possible, return raw valid JSON.
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
    }
    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return json.loads(text)


def extract_with_groq(formatted_transcript: str, api_key: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"""
Analyze this timestamped voice transcript and extract:
1. "summary": A concise 2-3 sentence executive summary.
2. "action_items": A list of specific actionable tasks, to-dos, or decisions with their speaker and timestamps.

Timestamped Transcript:
\"\"\"{formatted_transcript}\"\"\"

Format your output as strict JSON with keys "summary" (string) and "action_items" (list of objects with "task", "speaker", "start_time", "end_time").
"""
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def extract_rule_based(transcript: str, deepgram_summary: str = None, utterances: list = None):
    """
    Intelligent NLP fallback when no external LLM API key is configured.
    Identifies task-oriented sentences and links them to exact utterance timestamps and speakers.
    """
    action_items = []
    task_triggers = [
        r'\b(need to|have to|must|should|ought to)\b',
        r'\b(remember to|don\'t forget to|make sure to)\b',
        r'\b(will|going to|plan to|aim to)\b',
        r'\b(call|email|send|write|meet|submit|review|update|fix|check|buy|schedule)\b',
        r'\b(todo|to-do|action item|task)\b',
        r'\b(deadline|by tomorrow|by friday|by monday|by next week|at \d{1,2}(:\d{2})?\s*(am|pm)?)\b'
    ]
    combined_regex = re.compile('|'.join(task_triggers), re.IGNORECASE)

    if utterances:
        for u in utterances:
            speaker = u.get("speaker", "Speaker 1")
            start_t = round(float(u.get("start", 0.0)), 2)
            end_t = round(float(u.get("end", 0.0)), 2)
            text = u.get("text", "")
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())

            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 8 and combined_regex.search(s_clean):
                    cleaned = re.sub(r'^(and|so|also|then|well)\s+', '', s_clean, flags=re.IGNORECASE)
                    action_items.append({
                        "task": cleaned,
                        "speaker": speaker,
                        "start_time": start_t,
                        "end_time": end_t
                    })

        # Fallback if no specific trigger matched
        if not action_items and utterances:
            for u in utterances[:2]:
                action_items.append({
                    "task": u.get("text", ""),
                    "speaker": u.get("speaker", "Speaker 1"),
                    "start_time": round(float(u.get("start", 0.0)), 2),
                    "end_time": round(float(u.get("end", 0.0)), 2)
                })
    else:
        # Fallback without utterances
        sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 8 and combined_regex.search(s_clean):
                cleaned = re.sub(r'^(and|so|also|then|well)\s+', '', s_clean, flags=re.IGNORECASE)
                action_items.append({
                    "task": cleaned,
                    "speaker": "Speaker 1",
                    "start_time": 0.0,
                    "end_time": 0.0
                })

    summary = deepgram_summary if deepgram_summary else (
        transcript[:200] + ("..." if len(transcript) > 200 else "")
    )

    return {
        "summary": summary.strip(),
        "action_items": action_items[:6]
    }


def normalize_action_items(raw_items: list, utterances: list = None) -> list:
    """Ensures every action item follows the schema: {task, speaker, start_time, end_time}."""
    normalized = []
    for itm in raw_items:
        if isinstance(itm, str):
            task_text = itm.strip()
            spk = "Speaker 1"
            st_t = 0.0
            end_t = 0.0
            if utterances:
                for u in utterances:
                    if task_text.lower() in u.get("text", "").lower() or u.get("text", "").lower() in task_text.lower():
                        spk = u.get("speaker", "Speaker 1")
                        st_t = u.get("start", 0.0)
                        end_t = u.get("end", 0.0)
                        break
            normalized.append({
                "task": task_text,
                "speaker": spk,
                "start_time": round(float(st_t), 2),
                "end_time": round(float(end_t), 2)
            })
        elif isinstance(itm, dict):
            task_text = str(itm.get("task") or itm.get("text") or "").strip()
            spk = str(itm.get("speaker") or "Speaker 1")
            st_t = round(float(itm.get("start_time", itm.get("start", 0.0))), 2)
            end_t = round(float(itm.get("end_time", itm.get("end", 0.0))), 2)
            normalized.append({
                "task": task_text,
                "speaker": spk,
                "start_time": st_t,
                "end_time": end_t
            })
    return normalized


def analyze_transcript(transcript: str, deepgram_summary: str = None, utterances: list = None) -> dict:
    """Primary dispatcher: Passes timestamped transcript to Gemini -> Groq -> Rule-based NLP."""
    if not transcript or not transcript.strip():
        return {"summary": "", "action_items": []}

    formatted_transcript = format_timestamped_transcript(transcript, utterances)
    result = None

    # 1. Try Gemini
    gemini_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if gemini_key:
        try:
            result = extract_with_gemini(formatted_transcript, gemini_key)
        except Exception:
            pass

    # 2. Try Groq
    if not result:
        groq_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if groq_key:
            try:
                result = extract_with_groq(formatted_transcript, groq_key)
            except Exception:
                pass

    # 3. Built-in NLP fallback
    if not result:
        result = extract_rule_based(transcript, deepgram_summary, utterances)

    # Normalize action items schema
    summary_text = result.get("summary", "")
    action_items = normalize_action_items(result.get("action_items", []), utterances)

    return {
        "summary": summary_text,
        "action_items": action_items
    }
