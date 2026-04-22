import chromadb
from functools import lru_cache

client = chromadb.PersistentClient(path="../outputs/chroma_db")

collection = client.get_or_create_collection(name="curriculum_intelligence")


@lru_cache(maxsize=512)
def _query_cached(query: str, top_k: int) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]

    if not docs:
        return "No relevant data found."

    return " ".join(docs)


def get_response(query, top_k=2):
    cleaned_query = str(query).strip()
    if not cleaned_query:
        return "No relevant data found."

    return _query_cached(cleaned_query, top_k)