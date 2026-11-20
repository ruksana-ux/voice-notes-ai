# 🎙️ Deepgram AI Voice Notes

An AI-powered voice note application that records your speech, transcribes it using **Deepgram Nova-3**, allows you to review and save notes in an SQLite database, and enables **semantic AI search** over your voice notes using **ChromaDB** and **Sentence-Transformers**.

---

## ✨ Features

- 🎙️ **In-browser Audio Recording**: Record voice notes directly from your microphone using `streamlit-mic-recorder`.
- ⚡ **Ultra-Fast Speech-to-Text**: Powered by the **Deepgram Nova-3** AI model.
- ✏️ **Review & Edit**: Edit transcripts before saving.
- 💾 **Local & Cloud Storage**: Notes are preserved with timestamps in SQLite.
- 🔍 **AI Semantic Search**: Search your voice notes by concept and meaning (not just keywords) using ChromaDB vector embeddings.
- 🗑️ **Delete & Discard**: Discard unwanted recordings before saving or delete notes from both SQLite and ChromaDB.
- 📥 **Export to Text**: Download transcripts as `.txt` files with a single click.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, SQLite
- **AI / Speech-to-Text**: Deepgram API (Nova-3)
- **Vector Database**: ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Frontend**: Streamlit, Streamlit Mic Recorder

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/deepgram-voice-notes-ai.git
cd deepgram-voice-notes-ai
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
DEEPGRAM_API_KEY=your_deepgram_api_key_here
DEEPGRAM_MODEL=nova-3
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
BACKEND_API_URL=http://127.0.0.1:8000
```

### 5. Run the Application
You can run the Streamlit app directly (it will automatically launch the backend server if not running):
```bash
python -m streamlit run frontend/app.py
```

Or run backend and frontend separately:
```bash
# Terminal 1: Backend
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
python -m streamlit run frontend/app.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **New app** and select:
   - **Repository**: `your-username/deepgram-voice-notes-ai`
   - **Branch**: `main`
   - **Main file path**: `frontend/app.py`
4. Click **Advanced settings** &rarr; **Secrets** and add:
   ```toml
   DEEPGRAM_API_KEY = "your_actual_deepgram_api_key"
   DEEPGRAM_MODEL = "nova-3"
   ```
5. Click **Deploy!** 🚀
