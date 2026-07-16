# AI-Based Industry Skill Demand Modeling and Curriculum Recommendation System

An AI-powered platform that analyzes engineering curricula against evolving industry skill demands to identify curriculum gaps and generate intelligent recommendations for both faculty and students. The system combines Machine Learning, Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and semantic search to bridge the gap between academic education and real-world industry requirements. :contentReference[oaicite:0]{index=0}

---

## Features

### Staff Dashboard
- Detects missing skills in the curriculum.
- Identifies high-priority technologies based on industry trends.
- Calculates Curriculum Lag Index (CLI), Priority Score, and Trend Velocity.
- Provides explainable curriculum improvement recommendations.
- AI-powered chatbot for curriculum analysis using RAG.

### Student Dashboard
- Personalized skill recommendations.
- Learning roadmap with prerequisite skills.
- Placement-oriented preparation suggestions.
- AI chatbot for career guidance based on industry demand.

### AI & Analytics
- Automatic syllabus PDF parsing.
- NLP-based skill extraction and normalization.
- Semantic similarity search using embeddings.
- Machine Learning-based curriculum gap prediction.
- Context-aware responses using Retrieval-Augmented Generation (RAG).

---

## Tech Stack

### Backend
- Python
- FastAPI
- Pydantic

### Machine Learning & NLP
- Scikit-Learn
- XGBoost
- spaCy
- RapidFuzz
- Sentence Transformers
- Imbalanced-Learn (SMOTE)
- NetworkX
- Ruptures

### AI
- Groq API
- Llama 3.1 8B Instant
- ChromaDB
- RAG Pipeline

### Frontend
- HTML
- JavaScript
- Tailwind CSS

### Data Processing
- Pandas
- NumPy
- PyMuPDF

---

## System Architecture

```
                     Industry Job Data
                            │
                            ▼
                  Data Cleaning & NLP
                            │
                            ▼
                 Feature Engineering
                            │
                            ▼
             Machine Learning Prediction
                            │
                            ▼
                 ChromaDB Vector Store
                            │
                            ▼
                 Retrieval-Augmented AI
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
 Staff Dashboard                         Student Dashboard
```

---

## Project Workflow

1. Extract text from engineering syllabus PDFs.
2. Collect and preprocess industry job skill datasets.
3. Normalize and standardize skill names using NLP.
4. Generate analytical features such as:
   - Curriculum Lag Index (CLI)
   - Trend Velocity
   - Priority Score
   - Skill Transition
5. Train Machine Learning models to predict curriculum gaps.
6. Store processed knowledge inside ChromaDB.
7. Retrieve relevant context using semantic search.
8. Generate intelligent recommendations using Llama 3.1 through a RAG pipeline. :contentReference[oaicite:1]{index=1}

---

## Machine Learning

Models evaluated:

- Random Forest
- XGBoost

Feature Engineering includes:

- Skill frequency
- Trend slope
- Growth velocity
- Demand volatility
- Change-point detection
- Skill co-occurrence analysis

The final system uses a tuned Random Forest classifier for recommendation generation after comparing model performance. :contentReference[oaicite:2]{index=2}

---

## Retrieval-Augmented Generation (RAG)

The chatbot uses:

- Sentence Transformer embeddings
- ChromaDB vector database
- Semantic retrieval
- Hybrid document ranking
- Groq Llama 3.1 8B Instant

This enables explainable, context-aware recommendations instead of generic LLM responses. :contentReference[oaicite:3]{index=3}

---

## Project Structure

```
project/
│
├── backend/
│   ├── rag_engine.py
│   └── main.py
│
├── frontend/
│
├── data/
│
├── outputs/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/poorvikagj/Curriculum-Gap-Detection-and-Recommendation.git

cd Curriculum_Gap_Detection_And_Recommendation
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_api_key
```

---

## Running the Backend

```bash
cd backend

uvicorn app.main:app --reload
```

API Documentation

```
http://localhost:8000/docs
```

---

## Future Improvements

- Real-time LinkedIn and Naukri integration.
- Support for multiple universities.
- Time-series forecasting of emerging skills.
- Integration with online learning platforms (NPTEL, Coursera, edX).
- Personalized curriculum recommendations using student performance data. :contentReference[oaicite:4]{index=4}

---

## Contributors

- Poorvika G J
- M Abhinaya
- Manasvi K
- Parnika N

---

## Acknowledgements

This project was developed as part of the Information Science & Engineering Mini Project at **Ramaiah Institute of Technology (MSRIT)** under the guidance of **Dr. Savita K Shetty**. :contentReference[oaicite:5]{index=5}

---

