"""
evaluate.py
Évaluateur interactif en temps réel du pipeline RAG.
L'utilisateur entre :
  1. Sa question (dans n'importe quelle langue)
  2. La réponse de référence attendue (dans n'importe quelle langue)
Le script génère la réponse RAG, détecte la langue de la réponse,
traduit automatiquement la référence dans la même langue,
puis calcule toutes les métriques :
  - Score de similarité vectorielle (pgvector)
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BLEU
  - BERTScore F1
  - Temps de réponse
"""

import os
import time
import psycopg2
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from deep_translator import GoogleTranslator
from langdetect import detect
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── Ressources NLTK ────────────────────────────────────────────────────────────
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

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

# ── Prompt RAG ─────────────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
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

# ── Détection et traduction ────────────────────────────────────────────────────
def detect_language(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

def translate_to(text, target_lang):
    try:
        if target_lang == "en":
            return text
        translated = GoogleTranslator(
            source="auto", target=target_lang
        ).translate(text)
        return translated
    except Exception as e:
        print(f"    [!] Traduction échouée : {e} — référence utilisée telle quelle")
        return text

# ── Métriques ──────────────────────────────────────────────────────────────────
def compute_rouge(hypothesis, reference):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }

def compute_bleu(hypothesis, reference):
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothie   = SmoothingFunction().method4
    return round(sentence_bleu([ref_tokens], hyp_tokens,
                               smoothing_function=smoothie), 4)

def compute_bertscore(hypothesis, reference):
    try:
        from bert_score import score as bert_score
        _, _, F1 = bert_score(
            [hypothesis], [reference],
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        return round(F1[0].item(), 4)
    except Exception as e:
        print(f"    [!] BERTScore non disponible : {e}")
        return None

# ── Interprétation des scores ──────────────────────────────────────────────────
def interpret(score, thresholds):
    if score is None:
        return "N/A"
    for val, label in thresholds:
        if score >= val:
            return label
    return "Faible"

ROUGE_LEVELS = [(0.5, "Excellent"), (0.35, "Bon"), (0.2, "Moyen")]
BLEU_LEVELS  = [(0.4, "Excellent"), (0.2, "Bon"),  (0.1, "Moyen")]
BERT_LEVELS  = [(0.9, "Excellent"), (0.8, "Bon"),  (0.7, "Moyen")]
SIM_LEVELS   = [(0.85, "Excellent"),(0.75, "Bon"),  (0.65,"Moyen")]

# ── Évaluation d'une question ──────────────────────────────────────────────────
def evaluate(question, reference, llm, emb_model, conn):
    print("\n" + "=" * 70)
    print(f"  QUESTION  : {question}")
    print(f"  RÉFÉRENCE : {reference[:80]}...")
    print("=" * 70)

    # 1. Embedding + recherche
    print("\n[1/5] Recherche vectorielle...")
    query_emb = emb_model.embed_query(question)
    chunks    = search_chunks(conn, query_emb)
    sim_max   = round(chunks[0][2], 4) if chunks else 0.0
    sim_avg   = round(sum(c[2] for c in chunks) / len(chunks), 4) if chunks else 0.0
    context   = "\n\n---\n\n".join([c[0] for c in chunks])
    print(f"    Similarité max={sim_max} | moy={sim_avg} | {len(chunks)} chunks")

    # 2. Génération RAG
    print("[2/5] Génération de la réponse RAG...")
    prompt   = RAG_PROMPT.format(context=context, question=question)
    t0       = time.time()
    response = llm.invoke(prompt).strip()
    elapsed  = round(time.time() - t0, 2)
    print(f"    Réponse reçue en {elapsed}s")

    # 3. Détection langue + traduction référence
    print("[3/5] Détection de la langue et alignement...")
    response_lang = detect_language(response)
    print(f"    Langue détectée dans la réponse : {response_lang}")

    if response_lang != "en":
        print(f"    Traduction de la référence vers '{response_lang}'...")
        reference_aligned = translate_to(reference, response_lang)
        print(f"    Référence traduite : {reference_aligned[:80]}...")
    else:
        reference_aligned = reference
        print(f"    Référence déjà en anglais — pas de traduction nécessaire")

    # 4. Calcul des métriques
    print("[4/5] Calcul ROUGE + BLEU...")
    rouge = compute_rouge(response, reference_aligned)
    bleu  = compute_bleu(response, reference_aligned)

    print("[5/5] Calcul BERTScore...")
    bert  = compute_bertscore(response, reference_aligned)

    # 5. Affichage des résultats
    print("\n" + "─" * 70)
    print("  RÉPONSE RAG")
    print("─" * 70)
    print(response)

    print("\n" + "─" * 70)
    print("  RÉFÉRENCE ALIGNÉE (même langue que la réponse)")
    print("─" * 70)
    print(reference_aligned)

    print("\n" + "─" * 70)
    print("  MÉTRIQUES D'ÉVALUATION")
    print("─" * 70)
    print(f"  {'Métrique':<22} {'Score':>8}   Interprétation")
    print("  " + "-" * 50)
    print(f"  {'Similarité max':<22} {sim_max:>8}   {interpret(sim_max, SIM_LEVELS)}")
    print(f"  {'Similarité moy':<22} {sim_avg:>8}   {interpret(sim_avg, SIM_LEVELS)}")
    print(f"  {'ROUGE-1 F1':<22} {rouge['rouge1']:>8}   {interpret(rouge['rouge1'], ROUGE_LEVELS)}")
    print(f"  {'ROUGE-2 F1':<22} {rouge['rouge2']:>8}   {interpret(rouge['rouge2'], ROUGE_LEVELS)}")
    print(f"  {'ROUGE-L F1':<22} {rouge['rougeL']:>8}   {interpret(rouge['rougeL'], ROUGE_LEVELS)}")
    print(f"  {'BLEU':<22} {bleu:>8}   {interpret(bleu, BLEU_LEVELS)}")
    print(f"  {'BERTScore F1':<22} {str(bert) if bert else 'N/A':>8}   {interpret(bert, BERT_LEVELS)}")
    print(f"  {'Temps de réponse':<22} {elapsed:>7}s")
    print("=" * 70)

    return {
        "question":         question,
        "reference":        reference,
        "reference_aligned":reference_aligned,
        "response":         response,
        "response_lang":    response_lang,
        "similarity_max":   sim_max,
        "similarity_avg":   sim_avg,
        "rouge1":           rouge["rouge1"],
        "rouge2":           rouge["rouge2"],
        "rougeL":           rouge["rougeL"],
        "bleu":             bleu,
        "bertscore":        bert,
        "time_s":           elapsed,
    }

# ── Résumé de session ──────────────────────────────────────────────────────────
def print_session_summary(session):
    if not session:
        return
    print("\n" + "=" * 70)
    print(f"  RÉSUMÉ DE SESSION — {len(session)} question(s) évaluée(s)")
    print("=" * 70)
    keys = ["similarity_max", "similarity_avg", "rouge1", "rouge2",
            "rougeL", "bleu", "bertscore", "time_s"]
    print(f"  {'Métrique':<22} {'Moyenne':>8}  {'Min':>8}  {'Max':>8}")
    print("  " + "-" * 55)
    for key in keys:
        vals = [r[key] for r in session if r.get(key) is not None]
        if vals:
            print(f"  {key:<22} {sum(vals)/len(vals):>8.4f}  "
                  f"{min(vals):>8.4f}  {max(vals):>8.4f}")
    print("=" * 70)

# ── Point d'entrée interactif ──────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  ÉVALUATEUR RAG INTERACTIF — MÉTRIQUES EN TEMPS RÉEL")
    print(f"  Modèle LLM  : {LLM_MODEL}")
    print(f"  Embedding   : {EMBEDDING_MODEL}")
    print(f"  Métriques   : Similarité · ROUGE · BLEU · BERTScore · Temps")
    print(f"  Traduction  : Automatique (référence alignée sur la langue réponse)")
    print("=" * 70)
    print("\n  Initialisation des modèles...")

    conn      = psycopg2.connect(**DB_CONFIG)
    llm       = OllamaLLM(model=LLM_MODEL,             base_url=OLLAMA_BASE_URL)
    emb_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    print("  [OK] Prêt !\n")
    print("  Instructions :")
    print("  1. Entre ta question (français ou anglais)")
    print("  2. Entre la réponse de référence (depuis le dataset)")
    print("  3. La référence est traduite automatiquement si nécessaire")
    print("  4. Les métriques s'affichent immédiatement")
    print("  Tape 'exit' pour quitter et voir le résumé de session.\n")

    session = []

    while True:
        try:
            question = input("Question : ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                break

            reference = input("Réponse de référence : ").strip()
            if not reference:
                print("  [!] Réponse de référence vide — ignorée.")
                continue

            result = evaluate(question, reference, llm, emb_model, conn)
            session.append(result)

        except KeyboardInterrupt:
            print("\n\nInterruption.")
            break

    print_session_summary(session)
    conn.close()
    print("\nAu revoir !")

if __name__ == "__main__":
    main()