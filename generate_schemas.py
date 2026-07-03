"""
generate_schemas.py
Génère les deux schémas obligatoires du rapport en PNG :
  1. workflow_rag.png   — Pipeline d'ingestion + interrogation RAG
  2. architecture_env.png — Architecture de l'environnement
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

os.makedirs("rapport/schemas", exist_ok=True)

# ── Couleurs ───────────────────────────────────────────────────────────────────
BLUE      = "#185FA5"
TEAL      = "#0F6E56"
PURPLE    = "#534AB7"
AMBER     = "#854F0B"
GREEN     = "#3B6D11"
GRAY      = "#5F5E5A"
WHITE     = "#FFFFFF"
LIGHTBLUE = "#E6F1FB"
LIGHTGRAY = "#F1EFE8"

# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMA 1 — WORKFLOW RAG
# ══════════════════════════════════════════════════════════════════════════════
def draw_box(ax, x, y, w, h, label, sublabel, color, text_color=WHITE):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=0.8,
        edgecolor=color,
        facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.03, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color=text_color)
    if sublabel:
        ax.text(x, y - 0.07, sublabel, ha='center', va='center',
                fontsize=7, color=text_color, alpha=0.9)

def draw_arrow(ax, x, y1, y2, color=GRAY):
    ax.annotate('', xy=(x, y2 + 0.01), xytext=(x, y1 - 0.01),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

def generate_workflow():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # ── Titre Pipeline 1 ──
    ax.text(5, 13.5, "Pipeline 1 — Ingestion des documents",
            ha='center', va='center', fontsize=11, fontweight='bold', color=BLUE)

    # Boîtes pipeline 1
    boxes_p1 = [
        (5, 12.7, "Dataset HuggingFace",      "neural-bridge/rag-dataset-12000", BLUE),
        (5, 11.8, "Chargement",                "load_dataset() — 9 600 entrées",  TEAL),
        (5, 10.9, "Découpage (chunking)",      "RecursiveCharacterTextSplitter",  TEAL),
        (5,  9.9, "Génération des embeddings", "Ollama — nomic-embed-text",       PURPLE),
        (5,  9.0, "Stockage vectoriel",        "PostgreSQL + pgvector — 4 978 chunks", GREEN),
    ]

    for x, y, label, sub, color in boxes_p1:
        draw_box(ax, x, y, 5.5, 0.55, label, sub, color)

    for i in range(len(boxes_p1) - 1):
        draw_arrow(ax, 5, boxes_p1[i][1] - 0.28, boxes_p1[i+1][1] + 0.28)

    # ── Séparateur ──
    ax.axhline(y=8.3, xmin=0.05, xmax=0.95,
               color=GRAY, linewidth=0.8, linestyle='--', alpha=0.5)

    # ── Titre Pipeline 2 ──
    ax.text(5, 8.0, "Pipeline 2 — Interrogation RAG",
            ha='center', va='center', fontsize=11, fontweight='bold', color=BLUE)

    # Boîtes pipeline 2
    boxes_p2 = [
        (5, 7.3, "Question utilisateur",       "",                                GRAY),
        (5, 6.4, "Embedding de la question",   "nomic-embed-text via Ollama",     PURPLE),
        (5, 5.5, "Recherche vectorielle",      "Top-5 chunks pertinents (pgvector)", GREEN),
        (5, 4.6, "Construction du prompt RAG", "Contexte + question enrichie",    TEAL),
        (5, 3.7, "Réponse enrichie LLM",       "Ollama — llama3.1:8b",            AMBER),
    ]

    for x, y, label, sub, color in boxes_p2:
        draw_box(ax, x, y, 5.5, 0.55, label, sub, color)

    for i in range(len(boxes_p2) - 1):
        draw_arrow(ax, 5, boxes_p2[i][1] - 0.28, boxes_p2[i+1][1] + 0.28)

    # ── Légende ──
    legend_items = [
        mpatches.Patch(color=BLUE,   label='Source de données'),
        mpatches.Patch(color=TEAL,   label='Traitement'),
        mpatches.Patch(color=PURPLE, label='Embedding'),
        mpatches.Patch(color=GREEN,  label='Base vectorielle'),
        mpatches.Patch(color=AMBER,  label='LLM'),
        mpatches.Patch(color=GRAY,   label='Entrée utilisateur'),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              ncol=3, fontsize=8, framealpha=0.8,
              bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout()
    path = "rapport/schemas/workflow_rag.png"
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Schéma workflow généré : {path}")

# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMA 2 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
def draw_container(ax, x, y, w, h, label, color, alpha=0.15, lw=1.5):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        linewidth=lw,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(rect)
    ax.text(x + 0.15, y + h - 0.18, label,
            ha='left', va='top', fontsize=9,
            fontweight='bold', color=color)

def draw_component(ax, x, y, w, h, label, sublabel, color):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        linewidth=0.8,
        edgecolor=color,
        facecolor=color,
        alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.04, label,
            ha='center', va='center', fontsize=8.5,
            fontweight='bold', color=WHITE)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.1, sublabel,
                ha='center', va='center', fontsize=7, color=WHITE, alpha=0.9)

def generate_architecture():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # ── Machine hôte Windows (conteneur extérieur) ──
    draw_container(ax, 0.3, 0.3, 11.4, 8.4,
                   "Machine hôte — Windows 11  |  AMD Ryzen 7  ·  RTX 4090  ·  32 Go RAM",
                   GRAY, alpha=0.08, lw=2)

    # ── VM Ubuntu (conteneur intérieur) ──
    draw_container(ax, 0.7, 0.6, 10.6, 7.8,
                   "VM Ubuntu 22.04 LTS — VMware Workstation  |  8 Go RAM  ·  100 Go",
                   BLUE, alpha=0.10, lw=1.5)

    # ── Ollama ──
    draw_component(ax, 1.0, 4.5, 4.0, 2.5,
                   "Ollama 0.30.10", "llama3.1:8b  ·  nomic-embed-text\nMode CPU  ·  Port 11434", PURPLE)

    # ── Docker / PostgreSQL ──
    draw_component(ax, 6.5, 4.5, 4.0, 2.5,
                   "Docker Compose", "PostgreSQL 16 + pgvector 0.8.3\n4 978 chunks  ·  Port 5432", TEAL)

    # ── Flèche Ollama <-> PostgreSQL ──
    ax.annotate('', xy=(6.5, 5.75), xytext=(5.0, 5.75),
                arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.2))
    ax.text(5.75, 5.95, "embeddings", ha='center', fontsize=7, color=GRAY)

    # ── Python venv ──
    draw_component(ax, 1.0, 1.2, 9.5, 2.8,
                   "Python 3.12.3 — venv",
                   "ingest.py  ·  query.py  ·  compare.py  ·  evaluate.py\n"
                   "LangChain 0.3.25  ·  psycopg2  ·  datasets  ·  rouge-score  ·  bert-score",
                   AMBER)

    # ── Flèches Python -> Ollama et Python -> Docker ──
    ax.annotate('', xy=(3.0, 4.5), xytext=(3.0, 4.0),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))
    ax.annotate('', xy=(8.5, 4.5), xytext=(8.5, 4.0),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))

    # ── HuggingFace (externe) ──
    draw_component(ax, 1.0, 7.5, 3.5, 0.7,
                   "HuggingFace", "rag-dataset-12000", "#993C1D")
    ax.annotate('', xy=(2.75, 7.5), xytext=(2.75, 7.0),
                arrowprops=dict(arrowstyle='->', color="#993C1D", lw=1.0))
    ax.text(3.5, 7.2, "téléchargement\nautomatique",
            ha='center', fontsize=6.5, color="#993C1D")

    # ── Utilisateur (externe) ──
    draw_component(ax, 7.0, 7.5, 3.5, 0.7,
                   "Utilisateur", "Questions / Réponses RAG", GREEN)
    ax.annotate('', xy=(8.75, 7.5), xytext=(8.75, 7.0),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.0))

    # ── Titre ──
    ax.text(6, 8.7, "Architecture de l'environnement RAG",
            ha='center', va='center', fontsize=12,
            fontweight='bold', color=BLUE)

    plt.tight_layout()
    path = "rapport/schemas/architecture_env.png"
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Schéma architecture généré : {path}")

# ── Point d'entrée ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_workflow()
    generate_architecture()
    print("\n[OK] Les deux schémas sont dans rapport/schemas/")
    print("     workflow_rag.png")
    print("     architecture_env.png")