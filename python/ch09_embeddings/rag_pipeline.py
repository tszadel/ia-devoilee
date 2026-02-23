"""
Chapitre 9 — Embeddings & RAG
==============================
1. Similarité cosinus entre embeddings
2. Pipeline RAG minimal : index → retrieval → génération

Dépendances : sentence-transformers, chromadb, openai
"""

from __future__ import annotations

import os
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

client_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ---------------------------------------------------------------------------
# 1. Similarité cosinus
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def demo_cosine():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    pairs = [
        ("chien", "chat"),
        ("chien", "automobile"),
        ("Paris est la capitale de la France", "La France a pour capitale Paris"),
        ("Paris est la capitale de la France", "Le soleil se lève à l'est"),
    ]

    print("=== Similarités cosinus ===\n")
    for a, b in pairs:
        va = model.encode(a)
        vb = model.encode(b)
        sim = cosine(va, vb)
        print(f"  {a!r}")
        print(f"  {b!r}")
        print(f"  → {sim:.3f}\n")


# ---------------------------------------------------------------------------
# 2. Pipeline RAG minimal
# ---------------------------------------------------------------------------

DOCUMENTS = [
    "La tokenisation BPE divise les mots rares en sous-unités fréquentes.",
    "L'attention multi-têtes permet au modèle de se concentrer sur plusieurs parties de la séquence simultanément.",
    "Le fine-tuning adapte un modèle pré-entraîné à une tâche spécifique avec peu de données.",
    "Les embeddings de phrases représentent le sens d'un texte dans un espace vectoriel dense.",
    "Le RAG (Retrieval-Augmented Generation) combine recherche documentaire et génération de texte.",
    "La fenêtre de contexte limite la quantité de texte qu'un LLM peut traiter en une seule fois.",
    "Le temperature contrôle le caractère aléatoire de la génération : 0 = déterministe, 2 = très créatif.",
    "LoRA réduit le nombre de paramètres à entraîner en factorisant les mises à jour de matrices.",
]


class RAGPipeline:
    """
    Pipeline RAG minimaliste :
    - Indexation dans ChromaDB (vecteurs en mémoire)
    - Retrieval par similarité cosinus
    - Génération avec GPT-4o-mini
    """

    def __init__(self, embed_model: str = "BAAI/bge-small-en-v1.5", top_k: int = 3):
        self.encoder = SentenceTransformer(embed_model)
        self.top_k   = top_k
        self.db      = chromadb.Client()
        self.col     = self.db.create_collection("rag_demo")

    def index(self, docs: list[str]) -> None:
        embeddings = self.encoder.encode(docs).tolist()
        self.col.add(
            documents=docs,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(docs))],
        )
        print(f"✅ {len(docs)} documents indexés.")

    def retrieve(self, query: str) -> list[str]:
        q_vec = self.encoder.encode(query).tolist()
        results = self.col.query(query_embeddings=[q_vec], n_results=self.top_k)
        return results["documents"][0]

    def generate(self, query: str, context_chunks: list[str]) -> str:
        context = "\n".join(f"- {c}" for c in context_chunks)
        messages = [
            {"role": "system", "content": (
                "Tu es un assistant pédagogique expert en LLM. "
                "Réponds uniquement en te basant sur le contexte fourni. "
                "Si la réponse n'y figure pas, dis-le explicitement."
            )},
            {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {query}"},
        ]
        resp = client_oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content

    def ask(self, query: str) -> str:
        chunks = self.retrieve(query)
        print(f"\n📎 Chunks récupérés :")
        for c in chunks:
            print(f"  • {textwrap.shorten(c, 80)}")
        return self.generate(query, chunks)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Similarités
    demo_cosine()

    # 2. RAG
    print("\n=== Pipeline RAG ===\n")
    rag = RAGPipeline(top_k=3)
    rag.index(DOCUMENTS)

    questions = [
        "Comment fonctionne le RAG ?",
        "Qu'est-ce que LoRA ?",
        "Quel est le rôle du temperature ?",
    ]
    for q in questions:
        print(f"\n❓ {q}")
        answer = rag.ask(q)
        print(f"💬 {answer}")
