"""
ingest.py
Pipeline d'ingestion RAG :
1. Chargement du dataset HuggingFace (neural-bridge/rag-dataset-12000)
2. Découpage en chunks (chunking)
3. Génération des embeddings via Ollama (nomic-embed-text)
4. Stockage des vecteurs dans PostgreSQL/pgvector
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     os.getenv("DB_PORT",     "5432"),
    "dbname":   os.getenv("DB_NAME",     "rag_db"),
    "user":     os.getenv("DB_USER",     "rag_user"),
    "password": os.getenv("DB_PASSWORD", "rag_password"),
}

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL    = "nomic-embed-text"
DATASET_NAME       = "neural-bridge/rag-dataset-12000"
DATASET_SPLIT      = "train"
MAX_ENTRIES        = 500          # Limite pour éviter un temps d'ingestion trop long
CHUNK_SIZE         = 500          # Taille d'un chunk en caractères
CHUNK_OVERLAP      = 50           # Chevauchement entre chunks

# ── Initialisation de la base de données ───────────────────────────────────────
def init_db(conn):
    """Crée l'extension pgvector et la table documents si elles n'existent pas."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id        SERIAL PRIMARY KEY,
                content   TEXT    NOT NULL,
                source    TEXT,
                embedding vector(768)
            );
        """)
        conn.commit()
    print("[OK] Base de données initialisée (extension vector + table documents)")

# ── Chargement du dataset ──────────────────────────────────────────────────────
def load_data():
    """Charge le dataset HuggingFace et retourne les contextes."""
    print(f"[...] Chargement du dataset '{DATASET_NAME}' (split={DATASET_SPLIT})...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    entries = dataset.select(range(min(MAX_ENTRIES, len(dataset))))
    print(f"[OK] {len(entries)} entrées chargées")
    return entries

# ── Découpage en chunks ────────────────────────────────────────────────────────
def chunk_documents(entries):
    """Découpe les contextes en chunks avec chevauchement."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for i, entry in enumerate(entries):
        context = entry["context"]
        splits  = splitter.split_text(context)
        for split in splits:
            chunks.append({
                "content": split,
                "source":  f"entry_{i}",
            })
    print(f"[OK] {len(chunks)} chunks générés depuis {len(entries)} entrées")
    return chunks

# ── Génération des embeddings et stockage ─────────────────────────────────────
def embed_and_store(conn, chunks):
    """Génère les embeddings via Ollama et les insère dans pgvector."""
    embeddings_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    print(f"[...] Génération des embeddings avec '{EMBEDDING_MODEL}'...")
    batch_size = 20
    total      = len(chunks)

    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch   = chunks[i: i + batch_size]
            texts   = [c["content"] for c in batch]
            sources = [c["source"]  for c in batch]

            vectors = embeddings_model.embed_documents(texts)

            rows = [
                (texts[j], sources[j], vectors[j])
                for j in range(len(batch))
            ]
            execute_values(
                cur,
                "INSERT INTO documents (content, source, embedding) VALUES %s",
                rows,
                template="(%s, %s, %s::vector)",
            )
            conn.commit()
            print(f"    [{i + len(batch)}/{total}] chunks insérés")

    print(f"[OK] Ingestion terminée — {total} chunks stockés dans pgvector")

# ── Point d'entrée ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PIPELINE D'INGESTION RAG")
    print("=" * 60)

    # Connexion PostgreSQL
    print(f"[...] Connexion à PostgreSQL ({DB_CONFIG['host']}:{DB_CONFIG['port']})...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("[OK] Connexion établie")

    # Étapes du pipeline
    init_db(conn)
    entries = load_data()
    chunks  = chunk_documents(entries)
    embed_and_store(conn, chunks)

    conn.close()
    print("=" * 60)
    print("  INGESTION TERMINÉE AVEC SUCCÈS")
    print("=" * 60)

if __name__ == "__main__":
    main()
