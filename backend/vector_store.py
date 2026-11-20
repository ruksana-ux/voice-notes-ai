import chromadb
from sentence_transformers import SentenceTransformer
from .config import CHROMA_PATH

# Persistent vector database
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(
    name="voice_notes",
    metadata={"hnsw:space": "cosine"}
)

# AI embedding model
embedding_model = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


def create_embedding(text):
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()


def add_note_to_vector_db(note_id, title, transcript, created_at, summary=""):
    # Index both title, summary, and transcript for richer semantic context
    doc_to_embed = f"{title}\n{summary}\n{transcript}".strip()
    embedding = create_embedding(doc_to_embed)

    collection.upsert(
        ids=[note_id],
        documents=[transcript],
        embeddings=[embedding],
        metadatas=[{
            "title": title,
            "created_at": created_at,
            "summary": summary
        }]
    )


def semantic_search(query, limit=5):
    if collection.count() == 0:
        return []

    query_embedding = create_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, collection.count())
    )

    matches = []
    for i, note_id in enumerate(results["ids"][0]):
        matches.append({
            "id": note_id,
            "transcript": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    return matches


def delete_note_from_vector_db(note_id):
    try:
        collection.delete(ids=[note_id])
    except Exception:
        pass
