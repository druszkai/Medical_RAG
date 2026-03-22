# Medical RAG LLM System — Documentation

## 1. Project Overview

This system is a **Retrieval-Augmented Generation (RAG) application** for the medical domain, focused on cardiology and lipidology. It enables users to ask medical questions in Hungarian and receive answers grounded strictly in a curated corpus of medical articles scraped from PubMed and other web sources.

The system is designed as a **baseline RAG pipeline** with the following capabilities:
- Multi-source document ingestion (PubMed, web scraping via AgentQL)
- HTML cleaning and Hungarian-to-English translation of source documents
- Vector database indexing using ChromaDB with multilingual sentence embeddings
- Retrieval with keyword expansion and cross-encoder reranking
- LLM-based answer generation (Llama 3.3 70B via Groq) with a strict context-only prompt
- Automated evaluation using the RAGAS framework

**Tech Stack:**

| Component        | Technology                                      |
|------------------|-------------------------------------------------|
| Vector Database  | ChromaDB (persistent)                           |
| Embedding Model  | `paraphrase-multilingual-MiniLM-L12-v2`         |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-12-v2`         |
| LLM (main)       | `llama-3.3-70b-versatile` via Groq              |
| LLM (keywords)   | `openai/gpt-oss-20b` via Groq                   |
| Evaluation       | RAGAS                                           |
| Translation      | Google Translator (`deep-translator`)           |
| Language         | Python 3.x, LangChain                          |

---

## 2. Project Structure
```
Coding_new/
├── .venv/
└── src/
    ├── data/
    │   ├── database/
    │   │   └── chroma_db/          # ChromaDB vector store
    │   ├── eval/                   # RAGAS evaluation results
    │   ├── processed/              # Cleaned, chunked documents
    │   └── raw/                    # Scraped raw documents
    └── py/
        ├── agents/
        │   └── rag_agent.py        # Core RAG retrieval + generation
        ├── scrapers/
        │   ├── agentql_query.py    # AgentQL-based web scraper
        │   ├── pubmed_scraper.py   # PubMed-specific scraper
        │   ├── general_cardio_lipids_articles.csv
        │   └── urls.txt
        ├── tools/
        │   ├── evaluation/
        │   │   ├── ragas_eval.py   # Runs RAGAS metrics
        │   │   └── ragas_testgen.py# Generates eval test sets
        │   ├── database_builder.py # Embeds + indexes documents
        │   └── doc_cleaner.py      # Cleans raw scraped text
        ├── app.py                  # Entry point
        ├── config.py               # Paths, keys, model names
        └── merge.py                # Merges data sources
```

## 3. Architecture
```
╔══════════════════════════════════════════════════════════════════╗
║                  MEDICAL RAG LLM — ARCHITECTURE                  ║
╚══════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────┐
 │                     DATA INGESTION                          │
 │                                                             │
 │  ┌─────────────────┐   ┌─────────────┐                      │
 │  │ pubmed_scraper  │   │agentql_query│                      │
 │  └────────┬────────┘   └──────┬──────┘                      │
 │           └──────────────┬────┘                             │
 │                          ▼                                  │
 │                    data/raw/                                │
 └──────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                   PROCESSING PIPELINE                       │
 │                                                             │
 │        data/raw/  ──►  doc_cleaner.py  ──►  data/processed/ │
 │                        · strip HTML                         │
 │                        · translate HU→EN                    │
 └──────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                   VECTOR DATABASE                           │
 │                                                             │
 │   data/processed/ ──► database_builder.py                   │
 │                       · chunk (1500 chars / 200 overlap)    │
 │                       · embed (multilingual-MiniLM)         │
 │                       · index → chroma_db/                  │
 │                         collection: medical_collection      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                     RAG AGENT (query time)                  │
 │                                                             │
 │  User Query (HU) ──► keyword expansion (gpt-oss-20b)        │
 │                             │                               │
 │                             ▼                               │
 │                   similarity_search (k=20)                  │
 │                      chroma_db/                             │
 │                             │                               │
 │                             ▼                               │
 │               cross-encoder reranker → top 5 docs           │
 │                             │                               │
 │                             ▼                               │
 │            llama-3.3-70b (Groq) + strict HU prompt          │
 │                             │                               │
 │                             ▼                               │
 │                    Answer (Hungarian)                       │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                     EVALUATION (RAGAS)                      │ 
 │                              │                              │
 │                              ▼                              │
 │                       ragas_eval.py                         │
 │                       · English prompt swap                 │
 │                       · 20 samples evaluated                │
 │                              │                              │
 │                              ▼                              │
 │                    ragas_eval_results.csv                   │
 │                                                             │
 │    Metrics: Context Recall · Context Precision              │
 │             Faithfulness · Response Relevancy               │
 │             Factual Correctness                             │ 
 └─────────────────────────────────────────────────────────────┘
```
---
## 4. Data Pipeline

### 4.1 Scraping

Three scrapers populate `data/raw/` with JSON files, each document containing `document_id`, `source`, `language`, `title`, and `text` fields:

- **`pubmed_scraper.py`** — fetches articles from PubMed; produces `pubmed_1000_en.json`
- **`natural_scraper.py`** — scrapes general cardiology/lipids articles; produces `webmd_formatted_articles.json`
- **`agentql_query.py`** — uses the AgentQL API to scrape structured content from URLs listed in `urls.txt`; produces `cleaned_data_en_hu.json`

Scraped sources are merged into `all_merged_articles.json` by `merge.py`.

### 4.2 Cleaning (`doc_cleaner.py`)

Each raw document passes through two steps:

1. **HTML stripping** — `BeautifulSoup` removes all HTML tags, producing clean plain text.
2. **Hungarian → English translation** — documents where `language == "hu"` are translated using `GoogleTranslator` (`deep-translator`). Long texts are chunked into ≤4900-character segments before translation to stay within API limits.

Cleaned documents are written to `data/processed/`.

**Input files:** `pubmed_1000_en.json`, `webmd_formatted_articles.json`, `cleaned_data_en_hu.json`

---
## 5. Vector Database (`database_builder.py`)

### Chunking

Documents are split using `RecursiveCharacterTextSplitter` with:

| Parameter      | Value |
|----------------|-------|
| `chunk_size`   | 1500 characters |
| `chunk_overlap`| 200 characters  |

Each chunk is prepended with the document title: `"Title: {title}\n\n{chunk_text}"` to improve retrieval relevance.

### Embedding

| Parameter       | Value                                              |
|-----------------|----------------------------------------------------|
| Model           | `paraphrase-multilingual-MiniLM-L12-v2`            |
| Provider        | `SentenceTransformerEmbeddingFunction` (ChromaDB)  |

### ChromaDB Collection

| Parameter         | Value                  |
|-------------------|------------------------|
| Client type       | `PersistentClient`     |
| Path              | `data/database/chroma_db/` |
| Collection name   | `medical_collection`   |

Each chunk is stored with metadata: `document_id`, `source`, `language`, `title`, `chunk_index`.

### Running the Builder

```bash
python database_builder.py
```

Processes `all_merged_articles.json` from `data/processed/` by default.

---
## 6. RAG Agent (`rag_agent.py`)

The `MedicalRAG` class handles the full query pipeline.

### Query Pipeline

```
User query (Hungarian)
        │
        ▼
 Keyword Expansion
 · Model: gpt-oss-20b (Groq)
 · Generates 3–5 English medical keywords
        │
        ▼
 Similarity Search
 · Combined query: original question + keywords
 · ChromaDB similarity_search, k=20
        │
        ▼
 Cross-Encoder Reranking
 · Model: cross-encoder/ms-marco-MiniLM-L-12-v2
 · Scores all 20 candidate docs
 · Keeps top 5
        │
        ▼
 Answer Generation
 · Model: llama-3.3-70b-versatile (Groq), temperature=0.0
 · Strict context-only prompt in Hungarian
 · If answer not in context → fixed refusal message
        │
        ▼
 Response (Hungarian)
```


## 7. Evaluation (RAGAS)

### 7.1 Test Set Generation (`ragas_testgen.py`)

Generates a labeled test set from the processed corpus.

| Parameter     | Value                        |
|---------------|------------------------------|
| Input         | `all_merged_articles.json`   |
| Sample size   | 150 documents (random)       |
| Chunk size    | 2000 chars / 200 overlap     |
| Test set size | 50 question-answer pairs     |
| Output        | `data/processed/ragas_testset.csv` |
| LLM           | `gpt-oss-20b` (Groq)         |
| Embedding     | `paraphrase-multilingual-MiniLM-L12-v2` |

Two personas guide question generation: a **student** seeking conceptual understanding and a **medical professional** seeking detailed clinical information.

### 7.2 Evaluation (`ragas_eval.py`)

Runs the RAG pipeline on a random sample of the test set and scores it with RAGAS.

| Parameter     | Value                                   |
|---------------|-----------------------------------------|
| Sample size   | 20 questions (random, seed=42)          |
| Input         | `data/processed/ragas_testset.csv`      |
| Output        | `data/processed/ragas_eval_results.csv` |
| Eval LLM      | `gpt-oss-20b` (Groq), temp=0.0          |
| Eval Embedding| `paraphrase-multilingual-MiniLM-L12-v2` |

**Important:** The evaluation swaps the Hungarian system prompt for an English one (`patch_english`) so RAGAS metrics — which expect English — function correctly. This does not affect the production agent.

### Metrics

| Metric                          | What it measures                                          |
|---------------------------------|-----------------------------------------------------------|
| `LLMContextRecall`              | Are the reference answers covered by retrieved context?   |
| `LLMContextPrecisionWithReference` | Is the retrieved context free of irrelevant chunks?    |
| `Faithfulness`                  | Is the answer supported by the retrieved context?         |
| `ResponseRelevancy`             | Is the answer relevant to the question asked?             |
| `FactualCorrectness`            | Does the answer match the reference answer factually?     |

Results are saved as mean ± std per metric to `ragas_eval_results.csv`.

---

## 8. How to Run

### Generate evaluation test set (one-time)

```bash
python tools/eval/ragas_testgen.py
```

### Evaluate rag application

```bash
python tools/eval/ragas_eval.py
```

### Run main medical rag

```bash
python app.py
```

## 10. Known Limitations & Next Steps

**Current limitations:**

- Embedding model (`multilingual-MiniLM`) is a general-purpose model, not domain-specific for medical text.
- Chunking is purely character-based (`RecursiveCharacterTextSplitter`); semantic chunking is implemented but commented out.
- Evaluation is run on only 20 samples due to API rate limits.
- `app.py` entry point is incomplete — the RAG agent is not yet wired into the CLI loop.
- No reranking between different embedding model variants (only one is used at a time).

**Suggested next steps:**

- [ ] Switch embedding to `MedEmbed-small-v0.1` and compare RAGAS scores
- [ ] Enable semantic chunking and evaluate retrieval quality
- [ ] Wire `MedicalRAG` into `app.py` CLI loop
- [ ] Increase evaluation sample size (≥50) for more reliable metrics
- [ ] Move API keys to `.env` / secret manager
- [ ] Add a web UI (e.g., Streamlit or Gradio)
