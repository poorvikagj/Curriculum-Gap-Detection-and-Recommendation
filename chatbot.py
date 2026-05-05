"""
Curriculum Intelligence System — Terminal Chatbot
Reads REAL data from your output CSVs → ChromaDB → Groq LLM
Usage:  python chatbot.py --mode staff   (or student)
        python chatbot.py --build        (rebuild ChromaDB index first)
"""

import os, sys, argparse, textwrap, re
from pathlib import Path
from collections import defaultdict, deque

# ── Paths ────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent
OUTPUTS       = BASE / "outputs"
CHROMA_PATH   = str(OUTPUTS / "chroma_db")
DOTENV_PATH   = BASE / ".env"

CSV = {
    "analysis"    : OUTPUTS / "skill_analysis.csv",
    "industry"    : OUTPUTS / "cleaned_industry.csv",
    "syllabus"    : OUTPUTS / "cleaned_syllabus.csv",
    "cooccurrence": OUTPUTS / "skill_cooccurrence.csv",
}

COLLECTION_NAME = "curriculum_intelligence"
EMBED_MODEL     = "all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
INSTITUTE       = "msrit"
TOP_K           = 8
TOTAL_COMPANIES = 10
KNOWN_COMPANIES = [
    "microsoft","oracle","sap labs","hpe","ibm",
    "morgan stanley","natwest","bt group","thales","zebra"
]

# ── Colour helpers (terminal) ────────────────────────────────────────────────
class C:
    TEAL   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

def _c(color, text): return f"{color}{text}{C.RESET}"

# ── Environment & imports (lazy, with clear error messages) ──────────────────
def _load_env():
    if DOTENV_PATH.exists():
        from dotenv import load_dotenv
        load_dotenv(DOTENV_PATH)
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        print(_c(C.RED, "\n[ERROR] GROQ_API_KEY not found."))
        print("  Add it to outputs/.env  →  GROQ_API_KEY=gsk_...")
        print("  Or set it in your shell:   export GROQ_API_KEY=gsk_...\n")
        sys.exit(1)
    return key

def _check_csvs():
    missing = [name for name, p in CSV.items() if not p.exists()]
    if missing:
        print(_c(C.RED, f"\n[ERROR] Missing output CSVs: {missing}"))
        print("  Run notebooks 01–04 first to generate the outputs/ folder.\n")
        sys.exit(1)

# ── Domain / query config (mirrors rag_engine.py) ───────────────────────────
ROLE_TO_DOMAIN = {
    "ml engineer":"AI/ML","machine learning engineer":"AI/ML",
    "data engineer":"Data","data scientist":"Data","data analyst":"Data",
    "devops engineer":"DevOps","cloud engineer":"Cloud","cloud architect":"Cloud",
    "security engineer":"Security","cybersecurity":"Security",
    "frontend developer":"Web/Mobile","backend developer":"Systems/Infra",
    "full stack":"Web/Mobile","mobile developer":"Web/Mobile",
    "sre":"DevOps","site reliability":"DevOps","systems engineer":"Systems/Infra",
}
DOMAIN_KEYWORDS = {
    "ai":"AI/ML","ml":"AI/ML","machine learning":"AI/ML","deep learning":"AI/ML",
    "nlp":"AI/ML","llm":"AI/ML","generative":"AI/ML","mlops":"AI/ML",
    "cloud":"Cloud","aws":"Cloud","azure":"Cloud","gcp":"Cloud",
    "kubernetes":"Cloud","docker":"Cloud","serverless":"Cloud",
    "data":"Data","sql":"Data","spark":"Data","databricks":"Data","etl":"Data",
    "devops":"DevOps","ci/cd":"DevOps","jenkins":"DevOps","helm":"DevOps",
    "terraform":"DevOps","devsecops":"DevOps",
    "security":"Security","zero trust":"Security","siem":"Security",
    "web":"Web/Mobile","react":"Web/Mobile","angular":"Web/Mobile",
    "linux":"Systems/Infra","networking":"Systems/Infra",
}
DOMAIN_SKILL_HINTS = {
    "AI/ML":["tensorflow","pytorch","mlops","llm","generative ai","nlp",
              "deep learning","machine learning","hugging face","langchain",
              "vector database","rag","computer vision","xgboost","scikit"],
    "Cloud":["aws","azure","gcp","kubernetes","docker","serverless","cloud",
              "helm","terraform","cloud native","eks","aks"],
    "Data":["sql","spark","databricks","etl","data pipeline","kafka","airflow",
              "dbt","analytics","data warehouse","snowflake","pandas","redshift"],
    "DevOps":["ci/cd","jenkins","ansible","helm","devsecops","github actions",
               "argocd","prometheus","grafana","devops","pipeline"],
    "Security":["zero trust","siem","pentest","vulnerability","cybersecurity",
                 "oauth","iam","devsecops","api security","encryption"],
    "Web/Mobile":["react","angular","node.js","typescript","flutter","swift",
                   "kotlin","graphql","rest api","next.js","vue","tailwind"],
    "Systems/Infra":["linux","networking","embedded","grpc","microservices",
                      "service mesh","istio","posix","c++","rust"],
}
QUERY_SYNONYMS = {
    "ai":["machine learning","deep learning","nlp","llm","generative ai"],
    "ml":["machine learning","deep learning","mlops"],
    "cloud":["aws","azure","gcp","kubernetes","serverless"],
    "security":["vulnerability","zero trust","siem","pentest"],
    "data":["sql","spark","analytics","etl","databricks"],
    "devops":["ci/cd","jenkins","helm","ansible","devsecops"],
    "gap":["not taught","missing","curriculum gap","priority"],
    "urgent":["highest priority","debt score","cli","overdue"],
    "emerging":["upcoming","watch list","early warning","new skill"],
}
INTENT_FILTERS = {
    "gap":"gap_summary","missing":"gap_summary",
    "curriculum":"syllabus_card","syllabus":"syllabus_card",
    "watch":"watch_list","emerging":"watch_list",
    "upcoming":"trajectory_summary","trajectory":"trajectory_summary",
}

# ── System prompts (exact from NB05) ────────────────────────────────────────
SYSTEM_PROMPTS = {
    "staff": (
        "You are ARIA — a Curriculum Intelligence Assistant for engineering institutes in Bangalore.\n"
        "You help faculty make data-driven decisions about curriculum updates based on industry skill demand data.\n\n"
        f"You have access to skill demand data from {TOTAL_COMPANIES} companies "
        "(Microsoft, Oracle, SAP Labs, HPE, IBM, Morgan Stanley, NatWest, BT Group, Thales, Zebra Technologies) "
        "over 2021–2025.\n"
        "Key metrics in the data:\n"
        "- CLI (Curriculum Lag Index): years a skill has been demanded before curriculum responds.\n"
        "- Priority Score: urgency to add a skill (0–1, higher = more urgent).\n"
        "- Debt Score: accumulated curriculum lag (higher = more overdue).\n"
        "- Trajectory: Traditional (established), Transitional (shifting), Upcoming (emerging).\n"
        "- Propagated Gap Score: indirectly at-risk skills via co-occurrence graph.\n\n"
        "Rules:\n"
        "- Answer ONLY from the retrieved context provided. Never invent numbers.\n"
        "- Always cite CLI, Priority Score, or Debt Score when recommending skills.\n"
        "- Mention co-occurring skills so faculty can plan clusters, not isolated additions.\n"
        "- Be specific and actionable. Faculty need to justify changes to committees.\n"
        "- If context does not contain enough information, say: Data not available in current index."
    ),
    "student": (
        "You are SAGE — a Career Skills Advisor for engineering students in Bangalore.\n"
        "You help students understand what skills to learn based on real industry demand data.\n\n"
        f"You have access to skill demand data from {TOTAL_COMPANIES} major companies over 2021–2025.\n"
        "Key information available:\n"
        "- Which skills are Upcoming (learn now), Transitional (stable), Traditional (established).\n"
        "- A Watch List of skills that will become important soon.\n"
        "- Which skills are commonly demanded together (learn as a cluster).\n\n"
        "Rules:\n"
        "- Give practical, prioritised advice a student can act on today.\n"
        "- Recommend a clear learning order, not just a flat list.\n"
        "- Mention which companies demand each skill.\n"
        "- Highlight Watch List skills as early opportunities to stand out.\n"
        "- Keep language simple and encouraging.\n"
        "- Answer ONLY from the retrieved context. If not in context: Data not available."
    ),
}

# ── Query expansion (mirrors rag_engine.py expand_query) ────────────────────
def expand_query(query: str):
    q_low = query.lower()
    words = re.findall(r'\b\w+\b', q_low)
    extra = []
    for w in words:
        if w in QUERY_SYNONYMS:
            extra.extend(QUERY_SYNONYMS[w])
    expanded = query + (" " + " ".join(extra) if extra else "")

    intent_filter = None
    for kw, ftype in INTENT_FILTERS.items():
        if kw in q_low:
            intent_filter = ftype
            break

    detected_company = None
    for co in KNOWN_COMPANIES:
        if co in q_low:
            detected_company = co
            break

    detected_domain = None
    for phrase, dom in ROLE_TO_DOMAIN.items():
        if phrase in q_low:
            detected_domain = dom
            break
    if not detected_domain:
        for kw, dom in DOMAIN_KEYWORDS.items():
            if kw in q_low:
                detected_domain = dom
                break

    return expanded, intent_filter, detected_company, detected_domain

def _domain_exclusivity(text: str, target: str) -> float:
    t = text.lower()
    t_hits = sum(1 for h in DOMAIN_SKILL_HINTS.get(target,[]) if h in t)
    if t_hits == 0: return 0.0
    all_hits = sum(
        sum(1 for h in hints if h in t)
        for hints in DOMAIN_SKILL_HINTS.values()
    )
    return t_hits / max(all_hits, 1)

def _domain_relevance(text: str, domain: str) -> float:
    hints = DOMAIN_SKILL_HINTS.get(domain, [])
    if not hints: return 0.0
    t = text.lower()
    return min(sum(1 for h in hints if h in t) / max(len(hints)*0.25, 1), 1.0)

# ── Index builder ─────────────────────────────────────────────────────────────
def build_index():
    import pandas as pd
    import numpy as np
    import chromadb
    from sentence_transformers import SentenceTransformer

    _check_csvs()
    print(_c(C.TEAL, "\n[INDEX] Loading CSVs..."))

    analysis     = pd.read_csv(CSV["analysis"])
    syllabus     = pd.read_csv(CSV["syllabus"])
    cooccurrence = pd.read_csv(CSV["cooccurrence"])
    industry     = pd.read_csv(CSV["industry"])

    # company / role summary per skill
    company_summary = (
        industry.groupby("Skill")
        .agg(
            Top_Companies=("Company", lambda x: ", ".join(x.value_counts().head(3).index.tolist())),
            Top_Roles    =("Role",    lambda x: ", ".join(x.value_counts().head(3).index.tolist())),
            Year_Range   =("Year",    lambda x: f"{x.min()}–{x.max()}"),
        )
        .reset_index()
    )
    analysis = analysis.merge(company_summary, on="Skill", how="left")

    syllabus["Institute"] = syllabus["Institute"].str.lower().str.strip()

    documents = []

    # ── 1. Skill cards ───────────────────────────────────────────────────────
    for _, row in analysis.iterrows():
        skill   = row["Skill"]
        taught  = ("currently taught at MSRIT"
                   if row.get("Is_Taught_MSRIT", 0) == 1
                   else "NOT taught at MSRIT — curriculum gap")
        watch   = " On the Early Warning Watch List." if row.get("Watch_List", 0) == 1 else ""
        transit = (f" Transitioning: {row['Transition_Path']}."
                   if "Transition_Path" in row and pd.notna(row.get("Transition_Path")) else "")
        coocc   = (f" Co-occurs with: {row['Top_Cooccurring']}."
                   if "Top_Cooccurring" in row and pd.notna(row.get("Top_Cooccurring")) else "")
        companies = (f" Demanded by: {row['Top_Companies']}."
                     if "Top_Companies" in row and pd.notna(row.get("Top_Companies")) else "")
        roles     = (f" Common roles: {row['Top_Roles']}."
                     if "Top_Roles" in row and pd.notna(row.get("Top_Roles")) else "")
        yr_range  = (f" Active years: {row['Year_Range']}."
                     if "Year_Range" in row and pd.notna(row.get("Year_Range")) else "")

        text = (
            f"Skill: {skill}. "
            f"Domain: {row.get('Domain','Unknown')}. "
            f"Trajectory: {row.get('Trajectory','Unknown')}. "
            f"Status: {taught}.{watch}{transit} "
            f"Priority Score: {row.get('Priority_Score', row.get('NLP_Gap_Priority', 0)):.4f}. "
            f"Debt Score: {row.get('Debt_Score',0):.4f}. "
            f"CLI: {row.get('CLI',0)} years. "
            f"Velocity: {row.get('Velocity',0):.4f}. "
            f"Volatility: {row.get('Volatility',0):.4f}. "
            f"Company Spread: {row.get('Company_Spread',0)}/{TOTAL_COMPANIES} companies. "
            f"Frequency: {row.get('Freq',0)}. "
            f"Institutional Gap Score: {row.get('Institutional_Gap_Score',0):.4f}. "
            f"Propagated Gap Score: {row.get('Propagated_Gap_Score',0):.4f}. "
            f"Recommendation Prob: {row.get('Recommendation_Prob_MSRIT',0):.4f}. "
            f"Watch Urgency: {row.get('Watch_Urgency',0):.4f}."
            f"{coocc}{companies}{roles}{yr_range}"
        )
        p_score = float(row.get("Priority_Score", row.get("NLP_Gap_Priority", 0)) or 0)
        documents.append({
            "id"      : f"skill_{skill.lower().replace(' ','_').replace('/','_')}",
            "text"    : text,
            "metadata": {
                "type"          : "skill_card",
                "skill"         : skill,
                "domain"        : str(row.get("Domain","Unknown")),
                "trajectory"    : str(row.get("Trajectory","Unknown")),
                "is_gap"        : int(row.get("Is_Taught_MSRIT",0) == 0),
                "watch_list"    : int(row.get("Watch_List",0)),
                "cli"           : int(row.get("CLI",0)),
                "priority_score": round(p_score, 4),
                "debt_score"    : round(float(row.get("Debt_Score",0) or 0), 4),
            }
        })

    # ── 2. Gap summary ───────────────────────────────────────────────────────
    gaps = analysis[analysis.get("Is_Taught_MSRIT", analysis.get("Is_Taught_MSRIT")) == 0].copy()
    if "Priority_Score" not in gaps.columns and "NLP_Gap_Priority" in gaps.columns:
        gaps = gaps.rename(columns={"NLP_Gap_Priority":"Priority_Score"})
    top_gaps = gaps.nlargest(20, "Priority_Score") if "Priority_Score" in gaps.columns else gaps.head(20)

    for domain, grp in gaps.groupby("Domain"):
        top_d = grp.nlargest(10, "Priority_Score") if "Priority_Score" in grp.columns else grp.head(10)
        lines = []
        for _, r in top_d.iterrows():
            p = float(r.get("Priority_Score",0) or 0)
            d = float(r.get("Debt_Score",0) or 0)
            lines.append(f"  {r['Skill']} (Priority:{p:.3f}, Debt:{d:.2f}, CLI:{r.get('CLI',0)}yr, Traj:{r.get('Trajectory','?')})")
        text = (
            f"Domain gap summary — {domain}. "
            f"Gap skills in this domain ({len(grp)} total):\n" + "\n".join(lines)
        )
        documents.append({
            "id"      : f"gap_domain_{domain.lower().replace('/','_').replace(' ','_')}",
            "text"    : text,
            "metadata": {"type":"gap_summary","domain":domain,"skill":"ALL","priority_score":0.5}
        })

    # overall top-20 gap summary
    lines = []
    for _, r in top_gaps.iterrows():
        p = float(r.get("Priority_Score",0) or 0)
        d = float(r.get("Debt_Score",0) or 0)
        lines.append(f"  {r['Skill']} | domain:{r.get('Domain','?')} | priority:{p:.3f} | debt:{d:.2f} | cli:{r.get('CLI',0)}yr | traj:{r.get('Trajectory','?')}")
    documents.append({
        "id"      : "gap_overall_top20",
        "text"    : "Overall top-20 curriculum gap skills at MSRIT by priority:\n" + "\n".join(lines),
        "metadata": {"type":"gap_summary","domain":"ALL","skill":"ALL","priority_score":0.9}
    })

    # ── 3. Watch list ─────────────────────────────────────────────────────────
    watch = analysis[analysis.get("Watch_List", pd.Series(0, index=analysis.index)) == 1]
    if not watch.empty:
        lines = []
        for _, r in watch.iterrows():
            lines.append(
                f"  {r['Skill']} (Domain:{r.get('Domain','?')}, Urgency:{r.get('Watch_Urgency',0):.4f}, "
                f"Velocity:{r.get('Velocity',0):.4f}, Trajectory:{r.get('Trajectory','?')})"
            )
        documents.append({
            "id"      : "watch_list_all",
            "text"    : "Early Warning Watch List — upcoming skills with stable demand signal:\n" + "\n".join(lines),
            "metadata": {"type":"watch_list","domain":"ALL","skill":"ALL","priority_score":0.7}
        })

    # ── 4. Trajectory summaries ───────────────────────────────────────────────
    for traj, grp in analysis.groupby("Trajectory"):
        sample = grp.head(15)
        skills = ", ".join(sample["Skill"].tolist())
        text = (
            f"Trajectory summary — {traj} skills ({len(grp)} total). "
            f"These skills are classified as {traj} based on 2021–2025 demand pattern. "
            f"Sample: {skills}."
        )
        documents.append({
            "id"      : f"traj_{traj.lower()}",
            "text"    : text,
            "metadata": {"type":"trajectory_summary","domain":"ALL","skill":traj,"priority_score":0.4}
        })

    # ── 5. Syllabus cards (per institute) ────────────────────────────────────
    for inst, grp in syllabus.groupby("Institute"):
        for sem, sgrp in grp.groupby("Semester"):
            skills_in_sem = sgrp["Skill"].dropna().unique().tolist()
            courses       = sgrp["Course"].dropna().unique().tolist()
            text = (
                f"Syllabus — {inst.upper()}, Semester {sem}. "
                f"Courses: {', '.join(courses[:10])}. "
                f"Skills taught: {', '.join(skills_in_sem[:30])}."
            )
            documents.append({
                "id"      : f"syl_{inst}_sem{sem}",
                "text"    : text,
                "metadata": {
                    "type"    : "syllabus_card",
                    "institute": inst,
                    "semester" : str(sem),
                    "domain"  : "ALL",
                    "skill"   : "ALL",
                    "priority_score": 0.3,
                }
            })

    # ── 6. Co-occurrence summaries ────────────────────────────────────────────
    if not cooccurrence.empty:
        top_pairs = cooccurrence.nlargest(50, "Co_Count")
        for _, r in top_pairs.iterrows():
            text = (
                f"Skills '{r['Skill_A']}' and '{r['Skill_B']}' co-occur together "
                f"in job postings {r['Co_Count']} times. "
                f"Teaching them together is strongly recommended."
            )
            documents.append({
                "id"      : f"coocc_{r['Skill_A'].lower().replace(' ','_')}_{r['Skill_B'].lower().replace(' ','_')}",
                "text"    : text,
                "metadata": {
                    "type"    : "cooccurrence",
                    "skill"   : f"{r['Skill_A']},{r['Skill_B']}",
                    "domain"  : "ALL",
                    "priority_score": min(float(r["Co_Count"])/100, 1.0),
                }
            })

    # ── Embed + store ─────────────────────────────────────────────────────────
    print(_c(C.TEAL, f"[INDEX] Embedding {len(documents)} documents with {EMBED_MODEL}..."))
    embedder = SentenceTransformer(EMBED_MODEL)
    texts     = [d["text"] for d in documents]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)

    cli = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        cli.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = cli.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
    col.add(
        ids       =[d["id"] for d in documents],
        embeddings=embeddings.tolist(),
        documents =texts,
        metadatas =[d["metadata"] for d in documents],
    )
    print(_c(C.GREEN, f"[INDEX] Done — {col.count()} documents indexed at {CHROMA_PATH}"))
    return col, embedder

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(collection, embedder, query: str, top_k: int = TOP_K,
             filter_type=None, domain=None, company=None):
    expanded, intent_filt, _, _ = expand_query(query)
    active_filter = filter_type or intent_filt

    q_vec = embedder.encode([expanded])[0].tolist()
    where = {"type": active_filter} if active_filter else None

    fetch_n = min(top_k * 4, collection.count())
    results = collection.query(
        query_embeddings=[q_vec],
        n_results     =fetch_n,
        where         =where,
        include       =["documents","metadatas","distances"],
    )

    chunks = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    # hybrid scoring
    for c in chunks:
        sem  = 1 - c["distance"]
        prio = float(c["metadata"].get("priority_score", 0))
        dom_rel = _domain_relevance(c["text"], domain) if domain else 0
        dom_exc = _domain_exclusivity(c["text"], domain) if domain else 0
        co_rel  = 1.0 if company and company.lower() in c["text"].lower() else 0
        c["score"] = round(
            0.25*sem + 0.10*prio + 0.35*dom_rel + 0.30*dom_exc + 0.20*co_rel, 4
        )

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:top_k]

def build_context(collection, embedder, query: str, user_type: str = "staff") -> str:
    _, intent_f, company, domain = expand_query(query)

    if domain or company:
        n_gap, n_skill, n_traj, n_watch, n_syl = 2, 6, 2, 2, 4
    else:
        n_gap, n_skill, n_traj, n_watch, n_syl = 1, 5, 1, 1, 3

    def _get(n, ftype=None):
        return retrieve(collection, embedder, query, top_k=n,
                        filter_type=ftype, domain=domain, company=company)

    sections = []
    gap_chunks  = _get(n_gap,  "gap_summary")
    skill_chunks= _get(n_skill,"skill_card")
    traj_chunks = _get(n_traj, "trajectory_summary")
    watch_chunks= _get(n_watch,"watch_list")

    if gap_chunks:
        sections.append("=== GAP SUMMARIES ===\n" +
                        "\n---\n".join(c["text"] for c in gap_chunks))
    if skill_chunks:
        sections.append("=== INDIVIDUAL SKILL DATA ===\n" +
                        "\n---\n".join(c["text"] for c in skill_chunks))
    if traj_chunks:
        sections.append("=== TRAJECTORY DATA ===\n" +
                        "\n---\n".join(c["text"] for c in traj_chunks))
    if watch_chunks:
        sections.append("=== WATCH LIST ===\n" +
                        "\n---\n".join(c["text"] for c in watch_chunks))

    if user_type == "staff":
        syl_chunks = _get(n_syl, "syllabus_card")
        if syl_chunks:
            sections.append("=== SYLLABUS DATA ===\n" +
                            "\n---\n".join(c["text"] for c in syl_chunks))

    coocc_chunks = _get(2, "cooccurrence")
    if coocc_chunks:
        sections.append("=== CO-OCCURRENCE DATA ===\n" +
                        "\n---\n".join(c["text"] for c in coocc_chunks))

    return "\n\n".join(sections)

# ── Generation ────────────────────────────────────────────────────────────────
def ask(groq_client, collection, embedder,
        query: str, user_type: str = "staff",
        history: list = None, verbose: bool = False) -> str:

    context = build_context(collection, embedder, query, user_type)

    if verbose:
        print(_c(C.DIM, "\n--- RETRIEVED CONTEXT ---"))
        print(_c(C.DIM, context[:2000]))
        print(_c(C.DIM, "--- END CONTEXT ---\n"))

    user_msg = (
        "Use ONLY the following retrieved curriculum intelligence data to answer.\n"
        "If the answer is not in the context, say: 'Data not available in current index.'\n\n"
        f"--- RETRIEVED CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {query}"
    )

    messages = [{"role":"system","content":SYSTEM_PROMPTS[user_type]}]
    if history:
        messages.extend(history[-6:])      # last 3 turns for context
    messages.append({"role":"user","content":user_msg})

    resp = groq_client.chat.completions.create(
        model      =GROQ_MODEL,
        messages   =messages,
        temperature=0.2,
        max_tokens =1024,
    )
    return resp.choices[0].message.content

# ── Terminal UI ───────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║       CURRICULUM INTELLIGENCE SYSTEM — RAG CHATBOT              ║
║       Data: 10 companies · 5 institutes · 2021–2025             ║
╚══════════════════════════════════════════════════════════════════╝"""

STAFF_SUGGESTIONS = [
    "What are the top 5 gap skills by debt score?",
    "Which AI/ML skills are missing from MSRIT?",
    "What cloud skills are most urgent to add?",
    "Show the watch list skills",
    "Compare gap coverage across all institutes",
    "Which skills have the highest CLI?",
    "What should we add to the 6th semester cloud elective?",
    "Which security skills are missing?",
]
STUDENT_SUGGESTIONS = [
    "I want to become an ML engineer — what should I learn?",
    "What skills should I learn before placement?",
    "What does Microsoft look for in Bangalore hiring?",
    "What skills should I learn alongside Kubernetes?",
    "What are the upcoming skills I should learn now?",
    "I want to get into data engineering — roadmap?",
]

def print_banner(mode: str):
    print(_c(C.TEAL, BANNER))
    if mode == "staff":
        print(_c(C.BOLD, "\n  Mode: ARIA — Curriculum Intelligence Analyst (Faculty)\n"))
    else:
        print(_c(C.BOLD, "\n  Mode: SAGE — Skills & Career Advisor (Student)\n"))
    print(_c(C.DIM,  "  Commands:  /verbose  /mode  /suggestions  /quit\n"))

def print_suggestions(mode: str):
    s = STAFF_SUGGESTIONS if mode == "staff" else STUDENT_SUGGESTIONS
    print(_c(C.DIM, "\n  Suggested questions:"))
    for i, q in enumerate(s, 1):
        print(_c(C.DIM, f"    {i}. {q}"))
    print()

def wrap_response(text: str, width: int = 90) -> str:
    lines = text.split("\n")
    out   = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, width=width))
    return "\n".join(out)

def run_chat(mode: str, groq_key: str, collection, embedder, verbose: bool = False):
    from groq import Groq
    groq_client = Groq(api_key=groq_key)

    print_banner(mode)
    print_suggestions(mode)

    history = []
    persona = "ARIA" if mode == "staff" else "SAGE"

    while True:
        try:
            user_input = input(_c(C.YELLOW, "You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(_c(C.DIM, "\nBye!"))
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit","/exit","quit","exit"):
            print(_c(C.DIM, "Session ended.")); break
        if cmd == "/verbose":
            verbose = not verbose
            print(_c(C.DIM, f"Verbose: {verbose}")); continue
        if cmd == "/suggestions":
            print_suggestions(mode); continue
        if cmd == "/mode":
            mode = "student" if mode == "staff" else "staff"
            persona = "ARIA" if mode == "staff" else "SAGE"
            print_suggestions(mode)
            print(_c(C.TEAL, f"  Switched to {mode.upper()} mode ({persona})\n")); continue

    
        if user_input.isdigit():
            idx = int(user_input) - 1
            suggestions = STAFF_SUGGESTIONS if mode == "staff" else STUDENT_SUGGESTIONS
            if 0 <= idx < len(suggestions):
                user_input = suggestions[idx]
                print(_c(C.DIM, f"  → {user_input}"))

        print(_c(C.DIM, f"\n{persona}: thinking...\n"), end="", flush=True)

        try:
            answer = ask(groq_client, collection, embedder,
                         user_input, user_type=mode,
                         history=history, verbose=verbose)
        except Exception as e:
            print(_c(C.RED, f"\n[ERROR] {e}\n"))
            continue

        print(f"\r{_c(C.TEAL, persona+':')} {wrap_response(answer)}\n")
        print_suggestions(mode)

        history.append({"role":"user",    "content":user_input})
        history.append({"role":"assistant","content":answer})

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CIS Terminal Chatbot")
    parser.add_argument("--mode",    default="staff", choices=["staff","student"])
    parser.add_argument("--build",   action="store_true", help="(Re)build ChromaDB index from CSVs")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved context")
    args = parser.parse_args()

    groq_key = _load_env()

    if args.build:
        collection, embedder = build_index()
    else:
        # try loading existing index
        import chromadb
        from sentence_transformers import SentenceTransformer
        db_path = Path(CHROMA_PATH)
        if not db_path.exists() or not any(db_path.iterdir()):
            print(_c(C.YELLOW,
                "\n[INFO] No ChromaDB index found. Building from CSVs...\n"
                "  (Run with --build to rebuild any time)\n"))
            collection, embedder = build_index()
        else:
            print(_c(C.DIM, f"[INFO] Loading existing index from {CHROMA_PATH}..."))
            cli = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = cli.get_collection(COLLECTION_NAME)
            embedder   = SentenceTransformer(EMBED_MODEL)
            print(_c(C.DIM, f"[INFO] {collection.count()} documents loaded.\n"))

    run_chat(args.mode, groq_key, collection, embedder, verbose=args.verbose)

if __name__ == "__main__":
    main()
