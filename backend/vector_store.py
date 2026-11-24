import chromadb
from sentence_transformers import SentenceTransformer
from .config import CHROMA_PATH

# Persistent vector database
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(
    name="voice_notes",
    metadata={"hnsw:space": "cosine"}
)

# AI embedding model (lazy loaded)
embedding_model = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            # 1. Try loading cached local model first
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            try:
                # 2. Fast probe: check if huggingface is reachable within 1.5 seconds
                import urllib.request
                urllib.request.urlopen("https://huggingface.co", timeout=1.5)
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                # Mark as explicitly False to avoid repeated probe attempts in offline/sandboxed environments
                embedding_model = False

    return embedding_model if embedding_model is not False else None


def create_embedding(text: str):
    try:
        model = get_embedding_model()
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception:
        # Deterministic 384-dimensional fallback embedding if offline or network unavailable
        import hashlib
        import math
        vec = [0.0] * 384
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % 384
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def create_embeddings(texts: list):
    if not texts:
        return []
    try:
        model = get_embedding_model()
        embeddings = model.encode(texts)
        return embeddings.tolist()
    except Exception:
        return [create_embedding(t) for t in texts]


def add_note_to_vector_db(note_id, title, transcript, created_at, summary="", utterances=None):
    """
    Indexes both:
    1. Full note document (doc_type='note') for meeting-level semantic search.
    2. Timestamped utterance chunks (doc_type='utterance') with speaker and time range
       for granular Conversational Cross-Meeting Memory RAG.
    """
    ids_to_upsert = []
    docs_to_upsert = []
    meta_to_upsert = []

    # 1. Full Note Document Chunk
    note_doc = f"{title}\n{summary}\n{transcript}".strip()
    ids_to_upsert.append(str(note_id))
    docs_to_upsert.append(note_doc)
    meta_to_upsert.append({
        "note_id": str(note_id),
        "doc_type": "note",
        "title": str(title),
        "note_title": str(title),
        "created_at": str(created_at),
        "summary": str(summary or "")
    })

    # 2. Utterance-Level Vector Indexing
    if utterances and isinstance(utterances, list):
        for idx, u in enumerate(utterances):
            if not isinstance(u, dict):
                continue
            text = str(u.get("text", "")).strip()
            if not text:
                continue
            speaker = str(u.get("speaker") or "Speaker 1")
            start_t = float(u.get("start", 0.0))
            end_t = float(u.get("end", 0.0))
            
            utt_id = f"utt_{note_id}_{idx}"
            # Embed title, speaker, and utterance for maximal semantic retrieval precision
            utt_doc = f"[{title}] {speaker} ({start_t:.1f}s): {text}"

            ids_to_upsert.append(utt_id)
            docs_to_upsert.append(utt_doc)
            meta_to_upsert.append({
                "note_id": str(note_id),
                "doc_type": "utterance",
                "note_title": str(title),
                "created_at": str(created_at),
                "speaker": speaker,
                "start_time": start_t,
                "end_time": end_t,
                "text": text
            })

    # Batch encode all embeddings efficiently
    embeddings = create_embeddings(docs_to_upsert)

    collection.upsert(
        ids=ids_to_upsert,
        documents=docs_to_upsert,
        embeddings=embeddings,
        metadatas=meta_to_upsert
    )


def semantic_search(query: str, limit: int = 5):
    """Searches full meeting notes for conceptual exploration."""
    if collection.count() == 0:
        return []

    query_embedding = create_embedding(query)
    
    # Try searching documents of type 'note' first
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection.count()),
            where={"doc_type": "note"}
        )
    except Exception:
        results = None

    # Fallback if no doc_type='note' results found (e.g. legacy records)
    if not results or not results["ids"] or not results["ids"][0]:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection.count())
        )

    matches = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] or {}
            # Format title fallback
            title = meta.get("title") or meta.get("note_title") or "Meeting Note"
            matches.append({
                "id": meta.get("note_id", doc_id),
                "transcript": results["documents"][0][i],
                "metadata": {
                    "title": title,
                    "created_at": meta.get("created_at", ""),
                    "summary": meta.get("summary", "")
                },
                "distance": results["distances"][0][i]
            })
    return matches


def search_utterances(query: str, limit: int = 6):
    """
    Searches across timestamped meeting utterances for Cross-Meeting Memory RAG.
    Returns rich metadata with exact speaker, timestamps, and note information.
    """
    if collection.count() == 0:
        return []

    query_embedding = create_embedding(query)
    
    # 1. Search utterance chunks
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection.count()),
            where={"doc_type": "utterance"}
        )
    except Exception:
        results = None

    # 2. Fallback to general search if no utterance chunks found (e.g. legacy notes)
    if not results or not results["ids"] or not results["ids"][0]:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection.count())
        )

    matched_sources = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] or {}
            raw_text = meta.get("text") or results["documents"][0][i]
            note_title = meta.get("note_title") or meta.get("title") or "Recorded Meeting"
            
            matched_sources.append({
                "note_id": meta.get("note_id", doc_id),
                "note_title": note_title,
                "created_at": meta.get("created_at", ""),
                "speaker": meta.get("speaker", "Speaker 1"),
                "start_time": float(meta.get("start_time", 0.0)),
                "end_time": float(meta.get("end_time", 0.0)),
                "text": raw_text,
                "distance": results["distances"][0][i]
            })

    return matched_sources


def delete_note_from_vector_db(note_id: str):
    """
    Zero-Orphan Cleanup: Deletes both the meeting note vector AND all associated
    utterance chunks from ChromaDB matching note_id.
    """
    try:
        collection.delete(where={"note_id": str(note_id)})
    except Exception:
        pass

    try:
        collection.delete(ids=[str(note_id)])
    except Exception:
        pass


def sync_existing_notes_if_needed():
    """
    Backfills/syncs existing SQLite notes into ChromaDB with utterance-level chunks
    so older recordings are immediately searchable via Cross-Meeting Memory RAG.
    """
    try:
        from .database import get_all_notes
        notes = get_all_notes()
        for note in notes:
            note_id = note.get("id")
            utterances = note.get("utterances", [])
            if note_id:
                add_note_to_vector_db(
                    note_id=note_id,
                    title=note.get("title", "Meeting Note"),
                    transcript=note.get("transcript", ""),
                    created_at=note.get("created_at", ""),
                    summary=note.get("summary", ""),
                    utterances=utterances
                )
    except Exception:
        pass

