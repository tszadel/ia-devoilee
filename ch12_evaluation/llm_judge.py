"""
Chapitre 12 — Évaluation : LLM-as-a-Judge
==========================================
Évaluateur structuré qui note une réponse LLM de 1 à 5
sur plusieurs critères, avec justification.

Dépendances : openai
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ---------------------------------------------------------------------------
# Prompt du juge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
Tu es un évaluateur expert et strict. On te fournit :
- une QUESTION posée à un assistant IA
- une RÉPONSE produite par cet assistant
- un CRITÈRE d'évaluation

Ta mission : noter la réponse de 1 (très mauvais) à 5 (excellent) sur ce critère,
et justifier en 1-2 phrases concises.

Réponds UNIQUEMENT en JSON avec le schéma suivant :
{
  "score": <entier 1-5>,
  "justification": "<string>"
}

---
QUESTION : {question}
RÉPONSE  : {response}
CRITÈRE  : {criterion}
"""

CRITERES = [
    "Exactitude factuelle : les informations sont-elles correctes et vérifiables ?",
    "Clarté : la réponse est-elle facile à comprendre pour un développeur senior ?",
    "Complétude : la réponse couvre-t-elle tous les aspects importants de la question ?",
    "Concision : la réponse évite-t-elle les répétitions et le remplissage inutile ?",
]


# ---------------------------------------------------------------------------
# Évaluateur
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    criterion: str
    score: int          # 1-5
    justification: str

    def __str__(self):
        stars = "★" * self.score + "☆" * (5 - self.score)
        return f"[{stars}] {self.criterion[:50]}\n     {self.justification}"


def judge(question: str, response: str, criterion: str, model: str = "gpt-4o-mini") -> EvalResult:
    """Évalue une réponse sur un critère unique."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        criterion=criterion,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return EvalResult(
        criterion=criterion,
        score=int(data["score"]),
        justification=data["justification"],
    )


def evaluate(question: str, response: str) -> list[EvalResult]:
    """Évalue une réponse sur tous les critères. Retourne la liste des résultats."""
    results = [judge(question, response, c) for c in CRITERES]
    avg = sum(r.score for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"Question : {question[:70]}")
    print(f"Réponse  : {response[:70]}...")
    print(f"{'─'*60}")
    for r in results:
        print(f"  {r}")
    print(f"{'─'*60}")
    print(f"  Score moyen : {avg:.1f}/5\n")
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY manquante — démo désactivée.")
    else:
        evaluate(
            question="Qu'est-ce que le RAG et pourquoi l'utiliser ?",
            response=(
                "RAG signifie Retrieval-Augmented Generation. "
                "C'est une architecture qui combine un système de recherche documentaire "
                "avec un LLM génératif. Plutôt que de répondre depuis ses paramètres, "
                "le modèle récupère d'abord des passages pertinents, puis les utilise "
                "comme contexte pour générer une réponse ancrée dans des faits vérifiables. "
                "Principal avantage : réduit les hallucinations sur des domaines spécialisés."
            ),
        )

        evaluate(
            question="Qu'est-ce que le RAG et pourquoi l'utiliser ?",
            response="RAG c'est bien, ça aide les LLM à mieux répondre.",
        )
