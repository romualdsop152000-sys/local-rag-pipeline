# Local RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) pipeline built on open-source tools — no cloud API required. Combines **PostgreSQL + pgvector** for vector storage, **Ollama** for local LLM inference, and **LangChain** for orchestration. Includes ingestion, interactive querying, side-by-side RAG vs. no-RAG comparison, and a multi-metric evaluation suite with automatic translation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Ingest documents](#1-ingest-documents)
  - [2. Query the RAG pipeline](#2-query-the-rag-pipeline)
  - [3. Compare RAG vs. no-RAG](#3-compare-rag-vs-no-rag)
  - [4. Evaluate response quality](#4-evaluate-response-quality)
  - [5. Generate report schemas](#5-generate-report-schemas)
- [Evaluation Results](#evaluation-results)
- [Pipeline Details](#pipeline-details)
  - [Ingestion pipeline](#ingestion-pipeline)
  - [Query pipeline](#query-pipeline)
  - [Evaluation pipeline](#evaluation-pipeline)
- [Infrastructure](#infrastructure)
- [Limitations](#limitations)

---

## Overview

This project implements a complete RAG pipeline that:

1. **Ingests** documents from the HuggingFace dataset [`neural-bridge/rag-dataset-12000`](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000), splits them into chunks, generates 768-dimensional embeddings via `nomic-embed-text`, and stores everything in PostgreSQL with the pgvector extension.
2. **Queries** the knowledge base by embedding the user's question, retrieving the top-5 most semantically similar chunks via cosine similarity, injecting them into a prompt, and generating a contextualized answer with `llama3.1:8b`.
3. **Compares** RAG vs. no-RAG responses side by side in real time, demonstrating hallucination elimination and factual grounding.
4. **Evaluates** response quality with 7 complementary metrics — including ROUGE-1/2/L, BLEU, BERTScore F1, and cosine similarity — with automatic translation of the reference to match the response language.

Everything runs **entirely locally** inside a Linux VM (Ubuntu 22.04) without any external API calls.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host Machine                             │
│                   Windows 11 / RTX 4090                        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              VMware Workstation — Ubuntu 22.04            │  │
│  │                                                           │  │
│  │   ┌─────────────────┐      ┌──────────────────────────┐  │  │
│  │   │   Python Scripts │      │     Ollama (CPU mode)    │  │  │
│  │   │                 │─────▶│  llama3.1:8b  (gen.)     │  │  │
│  │   │  ingest.py      │      │  nomic-embed-text (emb.) │  │  │
│  │   │  query.py       │      │  port 11434              │  │  │
│  │   │  evaluate.py    │      └──────────────────────────┘  │  │
│  │   │  compare.py     │                                     │  │
│  │   │                 │      ┌──────────────────────────┐  │  │
│  │   │                 │─────▶│  Docker — PostgreSQL 16  │  │  │
│  │   └─────────────────┘      │  + pgvector 0.8.3        │  │  │
│  │                            │  port 5432               │  │  │
│  │   HuggingFace ────────────▶│  4 978 vectors (768D)    │  │  │
│  │   (dataset download)       └──────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:**
- `Python → Ollama (port 11434)` — embedding generation + LLM inference
- `Python → PostgreSQL (port 5432)` — vector storage and cosine similarity search
- `HuggingFace → Python` — automatic dataset download on first run (cached locally)

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| LLM (generation) | Ollama — llama3.1:8b | 0.30.10 |
| Embedding model | Ollama — nomic-embed-text | 768 dimensions |
| Vector database | PostgreSQL + pgvector | 16 / 0.8.3 |
| Orchestration | LangChain Python | 0.3.25 |
| Containerization | Docker Compose | 5.1.4 |
| Dataset | neural-bridge/rag-dataset-12000 | 9 600 entries |
| OS | Ubuntu 22.04 LTS (VMware) | — |
| Evaluation | ROUGE, BLEU, BERTScore, cosine sim. | — |
| Translation | deep-translator + langdetect | 1.11.4 / 1.0.9 |

---

## Project Structure

```
local-rag-pipeline/
│
├── ingest.py               # Ingestion pipeline: HuggingFace → chunks → embeddings → pgvector
├── query.py                # Interactive RAG query: question → top-5 chunks → LLM answer
├── compare.py              # Side-by-side RAG vs. no-RAG comparison in real time
├── evaluate.py             # Multi-metric evaluator with automatic reference translation
├── generate_schemas.py     # Generates workflow and architecture diagrams (matplotlib)
│
├── compose.yml             # Docker Compose: PostgreSQL 16 + pgvector
├── requirements.txt        # Python dependencies (pinned versions)
├── .env.example            # Environment variable template (copy to .env)
├── .gitignore

```

---

## Prerequisites

- **Docker** and **Docker Compose** installed
- **Ollama** installed and running (`ollama serve`)
- **Python 3.10+** with pip
- At least **8 GB RAM** and **10 GB free disk space**
- Internet access on first run (to download models and dataset)

> **GPU note:** This pipeline runs in CPU-only mode inside a VMware VM. Response times range from 40 to 190 seconds per query. In a native GPU environment (WSL2 + CUDA), this drops to 2–5 seconds.

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:romualdsop152000-sys/local-rag-pipeline.git
cd local-rag-pipeline
```

### 2. Create and activate a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Ollama models

```bash
# LLM for generation (4.9 GB — downloaded once)
ollama pull llama3.1:8b

# Embedding model (274 MB — 768-dimensional vectors)
ollama pull nomic-embed-text

# Verify
ollama list
```

### 5. Start PostgreSQL + pgvector

```bash
docker compose up -d

# Verify the container is healthy
docker compose ps

# Verify the pgvector extension is loaded
docker exec -it rag_postgres psql -U rag_user -d rag_db \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
```

Expected output:
```
 extname | extversion
---------+------------
 vector  | 0.8.3
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=rag_user
DB_PASSWORD=rag_password
OLLAMA_BASE_URL=http://localhost:11434
```

> The default values match the `compose.yml` configuration exactly. The `.env` file is optional — scripts fall back to these defaults automatically if the file is absent. **Never commit `.env` to version control.**

---

## Usage

### 1. Ingest documents

Runs the full ingestion pipeline: downloads the dataset, splits texts into chunks, generates embeddings, and stores vectors in pgvector.

```bash
python3 ingest.py
```

**What happens:**
- Downloads `neural-bridge/rag-dataset-12000` from HuggingFace (cached after first run)
- Selects the first 500 entries (configurable via `MAX_ENTRIES`)
- Splits each entry into chunks of 500 characters with 50-character overlap
- Generates a 768-dimensional vector for each chunk using `nomic-embed-text`
- Inserts chunks and vectors into PostgreSQL in batches of 20

**Result:** `4 978 chunks` stored, each with a 768D vector.

```bash
# Verify ingestion
docker exec -it rag_postgres psql -U rag_user -d rag_db \
  -c "SELECT COUNT(*) FROM documents;"
```

> Run `ingest.py` once only. The chunks persist in Docker volumes across restarts. Re-run only if you drop and recreate the database.

---

### 2. Query the RAG pipeline

Interactive session: type a question, get a contextualized RAG answer with similarity scores.

```bash
python3 query.py
```

**Example session:**

```
Question : What is the Berry Export Summary 2028?

[RAG] Top 5 chunks retrieved:
  Chunk 1 — similarity: 0.8251
  Chunk 2 — similarity: 0.7845
  ...

[LLM Response]
Le Berry Export Summary 2028 est un document qui cartographie la position actuelle
des secteurs de l'exportation de framboises, de fraises et de myrtilles...

Type 'exit' to quit.
```

---

### 3. Compare RAG vs. no-RAG

Generates two answers in parallel for each question: one from the raw LLM (no context) and one enriched by the pgvector knowledge base. Differences are highlighted.

```bash
python3 compare.py
```

**Example output:**

```
Question : Best short-term markets for blackberry/raspberry industry?

── Sans RAG (LLM seul) ─────────────────────────────
[LLM invente] Chine, Japon, États-Unis...    (80s)

── Avec RAG (pgvector + LLM) ───────────────────────
Hong Kong, Singapour, Émirats arabes unis, Canada  (42s)
Similarité max : 0.876
```

Key findings from the comparison:
- **Hallucination elimination:** the raw LLM invents markets (China, Japan) and fabricates statistics (FAO 3–4%); the RAG pipeline cites exact dataset values.
- **Factual grounding:** RAG answers include precise figures (100% growth, 4 978 tonnes) traceable to the ingested documents.
- **Honest out-of-scope handling:** when information is absent from the context, the RAG model explicitly says so instead of guessing.

---

### 4. Evaluate response quality

Interactive evaluator that scores a RAG response against a ground-truth reference using 7 metrics. The reference is automatically translated to match the response language before scoring.

```bash
python3 evaluate.py
```

**Metrics computed:**

| Metric | Measures | Good threshold |
|---|---|---|
| Similarity max | Relevance of the best pgvector chunk | ≥ 0.75 |
| Similarity avg | Average relevance of the top-5 chunks | ≥ 0.65 |
| ROUGE-1 F1 | Unigram overlap with reference | ≥ 0.35 |
| ROUGE-2 F1 | Bigram overlap with reference | ≥ 0.20 |
| ROUGE-L F1 | Longest common subsequence | ≥ 0.35 |
| BLEU | N-gram precision vs. reference | ≥ 0.20 |
| BERTScore F1 | Deep semantic similarity | ≥ 0.80 |

**Why automatic translation?** The RAG prompt forces `llama3.1:8b` to answer in French regardless of the question language. The dataset references are in English. Without translation, ROUGE-1 drops to 0.14; with automatic translation of the reference to French, ROUGE-1 reaches 0.60+ — a 4× improvement in measurement accuracy.

---

### 5. Generate report schemas

Generates the workflow and architecture diagrams as PNG files using matplotlib.

```bash
python3 generate_schemas.py
# → rapport/schemas/workflow_rag.png
# → rapport/schemas/architecture_env.png
```

---

## Evaluation Results

Results across 3 evaluated questions (averages):

| Metric | Q1 — Berry Export 2028 | Q2 — Short-term markets | Q3 — Export growth 2016–2017 | **Average** |
|---|---|---|---|---|
| Similarity max | 0.8251 | 0.8757 | 0.8773 | **0.8594** |
| Similarity avg | 0.6959 | 0.7603 | 0.7629 | **0.7397** |
| ROUGE-1 F1 | 0.5972 | 0.7857 | 0.8837 | **0.7555** |
| ROUGE-2 F1 | 0.3099 | 0.5926 | 0.7805 | **0.5610** |
| ROUGE-L F1 | 0.3750 | 0.7857 | 0.8837 | **0.6815** |
| BLEU | 0.2615 | 0.4029 | 0.7045 | **0.4563** |
| BERTScore F1 | 0.8978 | 0.9622 | 0.9792 | **0.9464** |
| Response time | 83.41s | 2 522.32s | 834.79s | 1 146.84s (CPU) |

**Key takeaways:**
- **BERTScore F1 avg = 0.9464** — the RAG responses are semantically near-identical to the ground truth references, confirming that the LLM faithfully extracts and reformulates the document content.
- **ROUGE-1 avg = 0.7555** — over 75% of reference words appear in the RAG answers on average.
- **BLEU avg = 0.4563** — well above the 0.20 threshold, indicating precise n-gram alignment.
- Q3 (a single precise figure: 100% growth) achieves near-perfect scores: BLEU 0.7045, BERTScore 0.9792 — the ideal use case for RAG on factual numeric questions.

---

## Pipeline Details

### Ingestion pipeline

```
HuggingFace dataset
      │
      ▼
  load_dataset('neural-bridge/rag-dataset-12000')
      │  500 entries selected (MAX_ENTRIES)
      ▼
  RecursiveCharacterTextSplitter
      │  chunk_size=500, chunk_overlap=50
      │  → 4 978 chunks
      ▼
  OllamaEmbeddings('nomic-embed-text')
      │  → 768-dimensional float vector per chunk
      ▼
  PostgreSQL / pgvector
      │  INSERT in batches of 20
      ▼
  documents table
  (id, content, source, embedding vector(768))
```

### Query pipeline

```
User question (text)
      │
      ▼
  OllamaEmbeddings.embed_query()
      │  → 768D vector
      ▼
  pgvector cosine similarity search
      │  SELECT ... ORDER BY embedding <=> query_vector LIMIT 5
      │  → top-5 chunks with similarity scores
      ▼
  RAG prompt construction
      │  [context: 5 chunks] + [question]
      ▼
  OllamaLLM('llama3.1:8b').invoke(prompt)
      │
      ▼
  Response (French) + similarity scores displayed
```

### Evaluation pipeline

```
User provides: question + reference answer
      │
      ▼
  RAG pipeline runs → response (French)
      │
      ▼
  langdetect: detect response language
      │
      ▼
  deep-translator: translate reference → response language
      │
      ▼
  Compute metrics:
    - cosine similarity (max + avg of top-5 chunks)
    - ROUGE-1 / ROUGE-2 / ROUGE-L  (rouge-score)
    - BLEU                          (nltk)
    - BERTScore F1                  (bert-score, multilingual)
      │
      ▼
  Display per-metric scores + session summary (avg/min/max)
```

---

## Infrastructure

### Docker Compose — PostgreSQL + pgvector

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: rag_postgres
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_db"]
      interval: 10s
      retries: 5
```

The `postgres_data` Docker volume persists all ingested vectors across container restarts — no need to re-run `ingest.py` after a `docker compose down` / `up`.

---

## Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| CPU-only inference (VMware GPU passthrough unavailable) | 40–2500s response time | Migrate to WSL2 + CUDA for 2–5s |
| 500/9600 entries ingested | Reduced knowledge coverage | Run `nohup python3 ingest.py &` overnight to ingest all 9600 |
| Automatic translation (deep-translator) | Residual bias in ROUGE/BLEU scores | Use multilingual BERTScore as primary metric |
| French prompt forces French output | Dataset is English; RAG works cross-lingually | Translation pipeline compensates during evaluation |
