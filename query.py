"""
query.py
Pipeline d'interrogation RAG :
1. Prise en charge de la question utilisateur
2. Génération de l'embedding de la question
3. Recherche des chunks les plus pertinents dans pgvector
4. Construction du prompt enrichi avec le contexte RAG
5. Envoi au LLM Ollama (llama3.1:8b) et retour de la réponse
"""

import os
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

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL  = "nomic-embed-text"
LLM_MODEL        = "llama3.1:8b"
TOP_K            = 5   # Nombre de chunks pertinents à récupérer

# ── Template du prompt RAG ─────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """
Tu es un assistant intelligent. Utilise uniquement le contexte fourni ci-dessous
pour répondre à la question de l'utilisateur. Si le contexte ne contient pas
l'information nécessaire, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :
"""

# ── Recherche vectorielle dans pgvector ────────────────────────────────────────
def search_similar_chunks(conn, query_embedding, top_k=TOP_K):
    """Recherche les chunks les plus proches sémantiquement dans pgvector."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, source,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))
        results = cur.fetchall()
    return results

# ── Pipeline RAG complet ───────────────────────────────────────────────────────
def rag_query(question: str, conn):
    """
    Pipeline complet :
    question → embedding → recherche pgvector → contexte → LLM → réponse
    """
    print("\n" + "=" * 60)
    print(f"  QUESTION : {question}")
    print("=" * 60)

    # 1. Génération de l'embedding de la question
    print("[...] Génération de l'embedding de la question...")
    embeddings_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    query_embedding = embeddings_model.embed_query(question)
    print("[OK] Embedding généré")

    # 2. Recherche des chunks pertinents dans pgvector
    print(f"[...] Recherche des {TOP_K} chunks les plus pertinents...")
    results = search_similar_chunks(conn, query_embedding)
    if not results:
        print("[!] Aucun chunk pertinent trouvé.")
        return None

    print(f"[OK] {len(results)} chunks récupérés :")
    for i, (content, source, similarity) in enumerate(results):
        print(f"    [{i+1}] source={source} | similarité={similarity:.4f}")

    # 3. Construction du contexte
    context = "\n\n---\n\n".join([row[0] for row in results])

    # 4. Construction du prompt enrichi
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )
    prompt = prompt_template.format(context=context, question=question)

    # 5. Envoi au LLM Ollama
    print(f"[...] Envoi au LLM '{LLM_MODEL}'...")
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    response = llm.invoke(prompt)
    print("[OK] Réponse reçue")

    # 6. Affichage de la réponse
    print("\n" + "-" * 60)
    print("  RÉPONSE DU LLM (avec RAG) :")
    print("-" * 60)
    print(response)
    print("=" * 60)

    return response

# ── Point d'entrée interactif ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PIPELINE D'INTERROGATION RAG")
    print("  Modèle LLM  : llama3.1:8b")
    print("  Embedding   : nomic-embed-text")
    print("  Base de données : PostgreSQL + pgvector")
    print("=" * 60)

    # Connexion PostgreSQL
    print(f"[...] Connexion à PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("[OK] Connexion établie")

    # Boucle interactive
    print("\nTape 'exit' pour quitter.\n")
    while True:
        question = input("Ta question : ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Au revoir !")
            break
        if not question:
            continue
        rag_query(question, conn)

    conn.close()

if __name__ == "__main__":
    main()
