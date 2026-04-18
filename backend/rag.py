import chromadb

client = chromadb.PersistentClient(path="../outputs/chroma_db")

collection = client.get_or_create_collection(name="curriculum_intelligence")


def get_response(query):
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    docs = results.get("documents", [[]])[0]

    if not docs:
        return "No relevant data found."

    return " ".join(docs)