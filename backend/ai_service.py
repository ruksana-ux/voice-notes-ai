import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def extract_with_gemini(transcript: str, api_key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    prompt = f"""
Analyze this voice note transcript and extract:
1. A concise 2-3 sentence executive summary.
2. A list of specific actionable to-do items / tasks mentioned (if any). If no specific tasks are mentioned, provide key takeaways.

Transcript:
\"\"\"{transcript}\"\"\"

Format your output as strict JSON with keys "summary" (string) and "action_items" (list of strings).
Example format:
{{
    "summary": "The speaker discussed preparing for the upcoming quarterly review and finalizing slides.",
    "action_items": [
        "Finalize slides by Thursday afternoon",
        "Send agenda to team members",
        "Review budget sheet"
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


def extract_with_groq(transcript: str, api_key: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"""
Analyze this voice note transcript and extract:
1. A concise 2-3 sentence executive summary.
2. A list of actionable to-dos / tasks.

Transcript:
\"\"\"{transcript}\"\"\"

Return strict JSON only with keys "summary" (string) and "action_items" (list of strings).
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


def extract_rule_based(transcript: str, deepgram_summary: str = None):
    """
    Intelligent NLP fallback when no external LLM API key is configured.
    Identifies task-oriented sentences and converts them into action items.
    """
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
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
    
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 8 and combined_regex.search(s_clean):
            # Clean up and normalize sentence as a bullet item
            cleaned = re.sub(r'^(and|so|also|then|well)\s+', '', s_clean, flags=re.IGNORECASE)
            action_items.append(cleaned)
            
    # If no specific action items were found, use key sentences
    if not action_items and sentences:
        action_items = [sentences[0]]
        if len(sentences) > 1:
            action_items.append(sentences[-1])
            
    summary = deepgram_summary if deepgram_summary else (
        sentences[0] if len(sentences) == 1 else f"{sentences[0]} {' '.join(sentences[1:3])}"
    )
    
    return {
        "summary": summary.strip(),
        "action_items": action_items[:5]  # Cap at top 5
    }


def analyze_transcript(transcript: str, deepgram_summary: str = None) -> dict:
    """
    Primary dispatcher: Uses Gemini -> Groq -> OpenAI -> Rule-based NLP fallback.
    """
    if not transcript or not transcript.strip():
        return {"summary": "", "action_items": []}
        
    # 1. Try Gemini
    if GEMINI_API_KEY:
        try:
            return extract_with_gemini(transcript, GEMINI_API_KEY)
        except Exception:
            pass
            
    # 2. Try Groq
    if GROQ_API_KEY:
        try:
            return extract_with_groq(transcript, GROQ_API_KEY)
        except Exception:
            pass
            
    # 3. Built-in pattern recognition fallback
    return extract_rule_based(transcript, deepgram_summary)
