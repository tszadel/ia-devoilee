"""
Chapitre 17 — Optimisation des coûts
======================================
1. Calculateur de coût par requête
2. Prompt caching Anthropic
3. Cascade avec score de confiance
4. Batch processing Anthropic (−50%)
5. Router multi-provider avec circuit breaker

Dépendances : anthropic, openai
"""

from __future__ import annotations

import os
import time
import json
import asyncio
from collections import deque
from dataclasses import dataclass
from enum import Enum

import anthropic
from openai import AsyncOpenAI

aclient_oai  = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
client_anth  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


# ---------------------------------------------------------------------------
# 1. Calculateur de coût
# ---------------------------------------------------------------------------

@dataclass
class ModelPricing:
    name:                  str
    input_per_million:     float   # USD
    output_per_million:    float   # USD
    cached_per_million:    float = 0.0  # prompt cache hit


PRICING = {
    "claude-3-5-haiku":    ModelPricing("claude-3-5-haiku",    0.80,  4.00, 0.08),
    "claude-3-5-sonnet":   ModelPricing("claude-3-5-sonnet",   3.00, 15.00, 0.30),
    "claude-opus-4":       ModelPricing("claude-opus-4",       15.00, 75.00, 1.50),
    "gpt-4o-mini":         ModelPricing("gpt-4o-mini",         0.15,  0.60),
    "gpt-4o":              ModelPricing("gpt-4o",              2.50, 10.00),
}


def estimate_cost(
    model:          str,
    input_tokens:   int,
    output_tokens:  int,
    cached_tokens:  int = 0,
) -> float:
    """Retourne le coût estimé en USD."""
    p              = PRICING[model]
    fresh_tokens   = input_tokens - cached_tokens
    cost_input     = fresh_tokens  / 1_000_000 * p.input_per_million
    cost_cached    = cached_tokens / 1_000_000 * p.cached_per_million
    cost_output    = output_tokens / 1_000_000 * p.output_per_million
    return cost_input + cost_cached + cost_output


def compare_models(input_tokens: int, output_tokens: int):
    print(f"\n=== Coût pour {input_tokens:,} tokens in / {output_tokens:,} tokens out ===\n")
    print(f"  {'Modèle':<25} {'Coût USD':>12}   {'Coût 10k req':>14}")
    print("  " + "─" * 55)
    for model in PRICING:
        c    = estimate_cost(model, input_tokens, output_tokens)
        c10k = c * 10_000
        print(f"  {model:<25} ${c:>11.4f}   ${c10k:>13.2f}")


# ---------------------------------------------------------------------------
# 2. Prompt caching Anthropic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LONG = (
    "Tu es un assistant expert en droit du travail français. "
    "Voici la base documentaire de référence : " + "Lorem ipsum " * 500  # simuler un long system prompt
)


def anthropic_with_cache(user_question: str) -> anthropic.types.Message:
    """
    Utilise le prompt caching d'Anthropic pour le system prompt.
    Après le premier appel, le system prompt est mis en cache (économie ~90%).
    """
    return client_anth.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=[
            {
                "type":       "text",
                "text":       SYSTEM_PROMPT_LONG,
                "cache_control": {"type": "ephemeral"},   # ← cache
            }
        ],
        messages=[{"role": "user", "content": user_question}],
    )


# ---------------------------------------------------------------------------
# 3. Cascade avec score de confiance
# ---------------------------------------------------------------------------

async def cascading_pipeline(query: str) -> tuple[str, str]:
    """
    Essaie d'abord le modèle rapide/pas cher.
    Si le LLM-juge estime que la réponse est insuffisante, escalade vers le modèle puissant.
    Retourne (réponse, modèle_utilisé).
    """
    FAST_MODEL   = "gpt-4o-mini"
    STRONG_MODEL = "gpt-4o"
    CONFIDENCE_THRESHOLD = 0.75

    # Passe 1 : modèle rapide
    resp_fast = await aclient_oai.chat.completions.create(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=0.2,
        max_tokens=400,
    )
    answer_fast = resp_fast.choices[0].message.content

    # Juge : évalue la confiance
    judge_resp = await aclient_oai.chat.completions.create(
        model=FAST_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Question : {query}\nRéponse : {answer_fast}\n\n"
                "Note la confiance de cette réponse de 0.0 à 1.0. "
                "Réponds uniquement avec un float."
            ),
        }],
        temperature=0,
    )
    try:
        confidence = float(judge_resp.choices[0].message.content.strip())
    except ValueError:
        confidence = 0.5

    if confidence >= CONFIDENCE_THRESHOLD:
        return answer_fast, FAST_MODEL

    # Passe 2 : escalade vers le modèle fort
    resp_strong = await aclient_oai.chat.completions.create(
        model=STRONG_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=0.2,
        max_tokens=800,
    )
    return resp_strong.choices[0].message.content, STRONG_MODEL


# ---------------------------------------------------------------------------
# 4. Circuit breaker multi-provider
# ---------------------------------------------------------------------------

class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI    = "openai"


@dataclass
class CircuitBreaker:
    provider:        Provider
    failure_window:  int   = 60       # secondes
    failure_limit:   int   = 5        # erreurs avant ouverture
    cooldown:        int   = 30       # secondes avant retry

    _failures:       deque = None
    _open_since:     float = 0.0

    def __post_init__(self):
        self._failures = deque()

    @property
    def is_open(self) -> bool:
        now = time.time()
        # Nettoie les erreurs hors fenêtre
        while self._failures and now - self._failures[0] > self.failure_window:
            self._failures.popleft()
        # Vérifie si le cooldown est écoulé
        if self._open_since and now - self._open_since < self.cooldown:
            return True
        self._open_since = 0.0
        return len(self._failures) >= self.failure_limit

    def record_failure(self):
        self._failures.append(time.time())
        if len(self._failures) >= self.failure_limit:
            self._open_since = time.time()

    def record_success(self):
        self._failures.clear()
        self._open_since = 0.0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Comparaison de coûts
    compare_models(input_tokens=2_000, output_tokens=500)
    compare_models(input_tokens=10_000, output_tokens=1_000)

    # 2. Cascade (nécessite OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        queries = [
            "Quelle est la capitale de la France ?",     # simple → modèle rapide
            "Compare RLHF et DPO en détaillant les gradients et les implications pratiques.",  # complexe → escalade
        ]
        for q in queries:
            answer, model = asyncio.run(cascading_pipeline(q))
            print(f"\n❓ {q[:60]}")
            print(f"🤖 [{model}] {answer[:120]}...")

    # 3. Circuit breaker demo
    cb = CircuitBreaker(Provider.ANTHROPIC, failure_limit=3, cooldown=5)
    for i in range(4):
        cb.record_failure()
        print(f"  Après {i+1} erreur(s) — circuit ouvert : {cb.is_open}")
