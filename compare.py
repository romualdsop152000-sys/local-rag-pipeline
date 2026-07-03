"""
compare.py
Comparaison interactive en temps réel — RAG vs Sans RAG.
L'utilisateur entre une question, la comparaison se fait immédiatement :
  1. Sans RAG : question envoyée directement au LLM
  2. Avec RAG  : question enrichie avec le contexte pgvector
Les résultats sont affichés côte à côte dans le terminal.
"""

import os
import time
import psycopg2
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     os.getenv("DB_PORT",     "5432"),
    "dbname":   os.getenv("DB_NAME",     "rag_db"),
    "user":     os.getenv("DB_USER",     "rag_user"),
    "password": os.getenv("DB_PASSWORD", "rag_password"),
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL       = "llama3.1:8b"
TOP_K           = 5

# ── Prompts ────────────────────────────────────────────────────────────────────
PROMPT_WITHOUT_RAG = PromptTemplate(
    input_variables=["question"],
    template="""Tu es un assistant intelligent. Réponds à la question suivante
du mieux que tu peux avec tes connaissances générales uniquement.
Ne mentionne pas que tu utilises des connaissances générales.

Question : {question}

Réponse :"""
)

PROMPT_WITH_RAG = PromptTemplate(
    input_variables=["context", "question"],
    template="""Tu es un assistant intelligent. Utilise uniquement le contexte
fourni ci-dessous pour répondre à la question. Si le contexte ne contient pas
l'information nécessaire, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :"""
)

# ── Recherche vectorielle ──────────────────────────────────────────────────────
def search_chunks(conn, embedding):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, source,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (embedding, embedding, TOP_K))
        return cur.fetchall()

# ── Comparaison en temps réel ──────────────────────────────────────────────────
def compare(question, llm, emb_model, conn):
    print("\n" + "=" * 70)
    print(f"  QUESTION : {question}")
    print("=" * 70)

    # ── SANS RAG ──
    print("\n[1/2] Génération SANS RAG...")
    prompt_no_rag = PROMPT_WITHOUT_RAG.format(question=question)
    t0            = time.time()
    resp_no_rag   = llm.invoke(prompt_no_rag).strip()
    time_no_rag   = round(time.time() - t0, 2)

    # ── AVEC RAG ──
    print("[2/2] Génération AVEC RAG...")
    query_emb   = emb_model.embed_query(question)
    chunks      = search_chunks(conn, query_emb)
    sim_max     = round(chunks[0][2], 4) if chunks else 0.0
    sim_avg     = round(sum(c[2] for c in chunks) / len(chunks), 4) if chunks else 0.0
    context     = "\n\n---\n\n".join([c[0] for c in chunks])
    prompt_rag  = PROMPT_WITH_RAG.format(context=context, question=question)
    t1          = time.time()
    resp_rag    = llm.invoke(prompt_rag).strip()
    time_rag    = round(time.time() - t1, 2)

    # ── AFFICHAGE CÔTE À CÔTE ──
    print("\n" + "─" * 70)
    print("  SANS RAG")
    print("─" * 70)
    print(resp_no_rag)
    print(f"\n  Temps : {time_no_rag}s")

    print("\n" + "─" * 70)
    print("  AVEC RAG")
    print("─" * 70)
    print(resp_rag)
    print(f"\n  Temps         : {time_rag}s")
    print(f"  Similarité max : {sim_max}")
    print(f"  Similarité moy : {sim_avg}")
    print(f"  Chunks utilisés: {len(chunks)}")

    print("\n" + "─" * 70)
    print("  DIFFÉRENCE CLÉE")
    print("─" * 70)
    print(f"  Sans RAG → réponse générique basée sur les données d entraînement")
    print(f"  Avec RAG → réponse précise basée sur la base de connaissance locale")
    print("=" * 70)

# ── Point d'entrée interactif ──────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  COMPARAISON INTERACTIVE — RAG vs SANS RAG")
    print(f"  Modèle LLM  : {LLM_MODEL}")
    print(f"  Embedding   : {EMBEDDING_MODEL}")
    print(f"  Base de données : PostgreSQL + pgvector")
    print("=" * 70)
    print("\n  Initialisation des modèles...")

    conn      = psycopg2.connect(**DB_CONFIG)
    llm       = OllamaLLM(model=LLM_MODEL,             base_url=OLLAMA_BASE_URL)
    emb_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    print("  [OK] Prêt ! Tape ta question et la comparaison se lance immédiatement.")
    print("  Tape 'exit' pour quitter.\n")

    while True:
        try:
            question = input("Ta question : ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("\nAu revoir !")
                break
            compare(question, llm, emb_model, conn)
        except KeyboardInterrupt:
            print("\n\nInterruption. Au revoir !")
            break

    conn.close()

if __name__ == "__main__":
    main()