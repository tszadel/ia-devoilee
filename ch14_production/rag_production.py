"""
Chapitre 14 — Production
========================
1. Pipeline RAG avec reranking cross-encodeur
2. Construction de contexte anti-lost-in-the-middle
3. Compression progressive de l'historique de conversation

Dépendances : sentence-transformers, openai, numpy
"""

from __future__ import annotations

import os
import asyncio
import numpy as np
from sentence_transformers import CrossEncoder
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ---------------------------------------------------------------------------
# 1. Reranking cross-encodeur
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: list[str], model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[str]:
    """
    Réordonne les chunks candidats par pertinence réelle.
    Le bi-encodeur (FAISS) fait le premier filtre rapide,
    le cross-encodeur fait le classement précis.
    """
    reranker = CrossEncoder(model)
    pairs    = [(query, c) for c in candidates]
    scores   = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked]


# ---------------------------------------------------------------------------
# 2. Construction de contexte optimisée (anti-lost-in-the-middle)
# ---------------------------------------------------------------------------

def build_optimized_context(
    system_prompt: str,
    top_chunks:    list[str],   # chunks rerangés, par ordre de pertinence décroissante
    max_tokens:    int = 3000,
    tok_per_char:  float = 0.25,  # heuristique : ~4 chars par token
) -> str:
    """
    Place les chunks les plus pertinents en DÉBUT et FIN du contexte,
    les moins pertinents au milieu — stratégie anti-lost-in-the-middle.

    Référence : Liu et al., "Lost in the Middle", TACL 2024.
    """
    budget   = max_tokens - int(len(system_prompt) * tok_per_char)
    selected = []
    used     = 0

    for chunk in top_chunks:
        cost = int(len(chunk) * tok_per_char)
        if used + cost > budget:
            break
        selected.append(chunk)
        used += cost

    if not selected:
        return ""

    # Interleave : plus pertinent → début/fin, moins pertinent → milieu
    ordered = []
    left, right = 0, len(selected) - 1
    toggle = True
    while left <= right:
        if toggle:
            ordered.append(selected[left]); left += 1
        else:
            ordered.append(selected[right]); right -= 1
        toggle = not toggle

    return "\n\n---\n\n".join(ordered)


# ---------------------------------------------------------------------------
# 3. Compression progressive de l'historique
# ---------------------------------------------------------------------------

async def compress_history(
    history:    list[dict],
    max_tokens: int = 2000,
    keep_last:  int = 4,        # tours récents à garder intacts
) -> list[dict]:
    """
    Compresse les anciens messages en un résumé structuré
    quand l'historique dépasse max_tokens.

    Stratégie :
    - Garde les `keep_last` derniers échanges intacts (contexte immédiat)
    - Résume le reste en un message system synthétique
    """
    if not history:
        return history

    # Estimation grossière du volume en tokens (4 chars ≈ 1 token)
    total_chars = sum(len(m.get("content", "")) for m in history)
    if total_chars * 0.25 <= max_tokens:
        return history  # pas besoin de compresser

    old    = history[:-keep_last] if len(history) > keep_last else []
    recent = history[-keep_last:]

    if not old:
        return recent

    # Résumé par LLM
    old_text = "\n".join(
        f"[{m['role'].upper()}] {m.get('content', '')}" for m in old
    )
    resp = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Résume l'historique de conversation suivant en 5 bullet points maximum. "
                "Conserve les faits clés, décisions, et préférences de l'utilisateur."
            )},
            {"role": "user", "content": old_text},
        ],
        temperature=0,
        max_tokens=300,
    )
    summary = resp.choices[0].message.content
    compressed = [{"role": "system", "content": f"[Résumé des échanges précédents]\n{summary}"}]
    return compressed + recent


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Reranking
    query = "Comment réduire les hallucinations d'un LLM en production ?"
    candidates = [
        "Le RAG ancre les réponses dans des documents vérifiés et réduit les inventions.",
        "Le temperature contrôle le caractère aléatoire de la génération.",
        "Les guardrails de sortie vérifient la cohérence factuelle des réponses.",
        "LoRA réduit le nombre de paramètres fine-tunables.",
        "La citation obligatoire force le modèle à référencer ses sources.",
    ]
    ranked = rerank(query, candidates)
    print("=== Reranking ===")
    for i, c in enumerate(ranked, 1):
        print(f"  {i}. {c[:70]}")

    # 2. Contexte optimisé
    context = build_optimized_context(
        system_prompt="Tu es un assistant RAG.",
        top_chunks=ranked,
        max_tokens=500,
    )
    print(f"\n=== Contexte optimisé ({len(context)} chars) ===\n{context[:300]}...")

    # 3. Compression d'historique
    history = [
        {"role": "user",      "content": "Bonjour, je développe un chatbot pour un cabinet juridique."},
        {"role": "assistant", "content": "Bonjour ! Quel est votre cas d'usage principal ?"},
        {"role": "user",      "content": "Répondre aux questions sur les contrats de travail."},
        {"role": "assistant", "content": "Avez-vous une base documentaire de référence ?"},
        {"role": "user",      "content": "Oui, 500 PDF de jurisprudence."},
        {"role": "assistant", "content": "RAG est la bonne approche. Quel LLM envisagez-vous ?"},
        {"role": "user",      "content": "Claude Sonnet ou GPT-4o, pas encore décidé."},
        {"role": "assistant", "content": "Les deux sont excellents. Quel est votre budget mensuel ?"},
    ]
    compressed = asyncio.run(compress_history(history, max_tokens=100, keep_last=2))
    print(f"\n=== Historique compressé ({len(compressed)} messages) ===")
    for m in compressed:
        print(f"  [{m['role']}] {m['content'][:80]}")
