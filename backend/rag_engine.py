from dotenv import load_dotenv
import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

INSTITUTE       = 'msrit'
CHROMA_PATH     = '../outputs/chroma_db'
COLLECTION_NAME = 'curriculum_intelligence'
EMBED_MODEL     = 'all-MiniLM-L6-v2'
GROQ_MODEL      = 'llama-3.1-8b-instant'
TOP_K           = 8
TOTAL_COMPANIES = 10

load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

groq_client = Groq(api_key=GROQ_API_KEY)

import spacy
nlp_qe = spacy.load("en_core_web_sm", disable=["ner", "parser"])

QUERY_SYNONYMS: dict[str, list[str]] = {
    "ai":       ["machine learning", "deep learning", "nlp", "llm", "generative ai"],
    "ml":       ["machine learning", "deep learning", "mlops"],
    "cloud":    ["aws", "azure", "gcp", "kubernetes", "serverless"],
    "security": ["vulnerability", "zero trust", "siem", "pentest"],
    "data":     ["sql", "spark", "analytics", "etl", "databricks"],
    "devops":   ["ci/cd", "jenkins", "helm", "ansible", "devsecops"],
    "web":      ["react", "angular", "node.js", "html", "typescript"],
    "mobile":   ["flutter", "swift", "kotlin", "react native"],
    "gap":      ["not taught", "missing", "curriculum gap", "priority"],
    "urgent":   ["highest priority", "debt score", "cli", "overdue"],
    "emerging": ["upcoming", "watch list", "early warning", "new skill"],
}

INTENT_FILTERS: dict[str, str] = {
    "gap":        "gap_summary",
    "missing":    "gap_summary",
    "curriculum": "syllabus_card",
    "syllabus":   "syllabus_card",
    "watch":      "watch_list",
    "emerging":   "watch_list",
    "upcoming":   "trajectory_summary",
    "trajectory": "trajectory_summary",
}

def expand_query(query: str) -> tuple[str, str | None]:
    """
    Returns:
        expanded_query : original query + synonym expansions joined
        inferred_filter: chroma filter_type inferred from intent, or None
    """
    doc = nlp_qe(query.lower())
    lemmas = [t.lemma_ for t in doc if not t.is_space]

    extra_terms = []
    for lemma in lemmas:
        if lemma in QUERY_SYNONYMS:
            extra_terms.extend(QUERY_SYNONYMS[lemma])
    expanded = query if not extra_terms else query + " " + " ".join(extra_terms)

    inferred_filter = None
    for kw, ftype in INTENT_FILTERS.items():
        if kw in query.lower():
            inferred_filter = ftype
            break

    return expanded, inferred_filter


client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


print('Loading embedding model (local, no API)...')
embedder = SentenceTransformer(EMBED_MODEL)
print('Model loaded.')


def retrieve_nlp(query: str, top_k: int = TOP_K, filter_type: str = None) -> list:
    """NLP-expanded retrieval with hybrid re-ranking."""
    expanded_q, inferred_filter = expand_query(query)
    active_filter = filter_type or inferred_filter

    query_vec = embedder.encode([expanded_q])[0].tolist()
    where = {"type": active_filter} if active_filter else None

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    chunks = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

    for chunk in chunks:
        sem_score        = 1 - chunk["distance"]
        nlp_prio         = float(chunk["metadata"].get("priority_score", 0))
        chunk["hybrid_score"] = round(0.7 * sem_score + 0.3 * nlp_prio, 4)

    chunks.sort(key=lambda c: c["hybrid_score"], reverse=True)
    return chunks


def build_context_nlp(query: str, user_type: str = "staff") -> str:
    """NLP-enhanced context builder using retrieve_nlp."""
    gap_chunks   = retrieve_nlp(query, top_k=1, filter_type="gap_summary")
    skill_chunks = retrieve_nlp(query, top_k=5, filter_type="skill_card")
    traj_chunks  = retrieve_nlp(query, top_k=1, filter_type="trajectory_summary")
    watch_chunks = retrieve_nlp(query, top_k=1, filter_type="watch_list")

    if user_type == "staff":
        syl_chunks = retrieve_nlp(query, top_k=3, filter_type="syllabus_card")
        all_chunks = gap_chunks + traj_chunks + skill_chunks + syl_chunks
    else:
        all_chunks = watch_chunks + traj_chunks + skill_chunks

    seen, unique = set(), []
    for c in all_chunks:
        key = c["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return "\n\n".join([
        f"[{i+1}] (hybrid_score={c['hybrid_score']:.3f}) {c['text']}"
        for i, c in enumerate(unique)
    ])


SYSTEM_PROMPTS = {

    'staff': (
        "You are a Curriculum Intelligence Assistant for engineering institutes in Bangalore.\n"
        "You help faculty make data-driven decisions about curriculum updates based on industry skill demand data.\n\n"
        f"You have access to skill demand data from {TOTAL_COMPANIES} companies "
        "(Microsoft, Oracle, SAP Labs, HPE, IBM, Morgan Stanley, NatWest, BT Group, Thales, Zebra Technologies) "
        "over 2021-2025.\n"
        "Key metrics in the data:\n"
        "- CLI (Curriculum Lag Index): years a skill has been demanded before curriculum responds.\n"
        "- Priority Score: urgency to add a skill (0-1, higher = more urgent).\n"
        "- Debt Score: accumulated curriculum lag (higher = more overdue).\n"
        "- Trajectory: Traditional (established), Transitional (shifting), Upcoming (emerging).\n"
        "- Propagated Gap Score: indirectly at-risk skills via co-occurrence graph.\n\n"
        "Rules:\n"
        """- Base your answer primarily on the retrieved context. If the query is general (e.g., greeting), respond naturally.If data is missing, say "Data not available". Never invent numbers.\n"""
        "- Always cite CLI, Priority Score, or Debt Score when recommending skills.\n"
        "- Mention co-occurring skills so faculty can plan clusters, not isolated additions.\n"
        "- Be specific and actionable. Faculty need to justify changes to committees.\n"
        "- If the context does not contain enough information, say: Data not available."
    ),

    'student': (
        "You are a Career Skills Advisor for engineering students in Bangalore.\n"
        "You help students understand what skills to learn based on real industry demand data.\n\n"
        f"You have access to skill demand data from {TOTAL_COMPANIES} major companies over 2021-2025.\n"
        "Key information available:\n"
        "- Which skills are Upcoming (learn now), Transitional (stable), Traditional (established).\n"
        "- A Watch List of skills that will become important soon.\n"
        "- Which skills are commonly demanded together (learn as a cluster).\n\n"
        "Rules:\n"
        "- Give practical, prioritised advice a student can act on today.\n"
        "- Recommend a clear learning order, not just a flat list.\n"
        "- Mention which companies demand each skill so students know the context.\n"
        "- Highlight Watch List skills as early opportunities to stand out.\n"
        "- Keep language simple and encouraging.\n"
        "- Answer ONLY from the retrieved context. If not in context, say: Data not available."
    )
}


def ask_nlp(query: str, user_type: str = "staff", verbose: bool = False) -> str:
    """
    Full NLP-augmented RAG pipeline — exact copy from notebook cell 8.
    Called by main.py for both /api/rag/query and /api/chat/query.
    """
 
    resolved_type = "staff" if user_type in ("staff", "admin") else "student"

    context = build_context_nlp(query, user_type=resolved_type)

    if verbose:
        print("=== NLP-EXPANDED CONTEXT ===")
        print(context)
        print("=== END CONTEXT ===\n")

    user_message = (
        f"Use the following retrieved curriculum intelligence data to answer the question.\n"
        f"Answer ONLY from this context. If something is not in the context, "
        f"say: Data not available.\n\n"
        f"--- RETRIEVED CONTEXT ---\n"
        f"{context}\n"
        f"--- END CONTEXT ---\n\n"
        f"Question: {query}"
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[resolved_type]},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    return response.choices[0].message.content
