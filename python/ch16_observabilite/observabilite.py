"""
Chapitre 16 — Observabilité & CI/CD
=====================================
1. Tracing LLM avec OpenTelemetry
2. Versioning de prompts (YAML)
3. Runner d'évaluation pour pipeline CI/CD
4. Shadow mode & feature flags

Dépendances : opentelemetry-api, opentelemetry-sdk, pyyaml, openai
"""

from __future__ import annotations

import os
import yaml
import json
import asyncio
import hashlib
import random
from dataclasses import dataclass
from typing import Optional
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# ---------------------------------------------------------------------------
# 1. Tracing OpenTelemetry
# ---------------------------------------------------------------------------

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Configuration minimale : export console (en prod → OTLP vers Jaeger/Tempo)
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm.tracer")


async def llm_with_tracing(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """Appel LLM instrumenté avec OpenTelemetry."""
    with tracer.start_as_current_span("llm.completion") as span:
        span.set_attribute("llm.model",       model)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.prompt_len",  len(prompt))

        resp = await aclient.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        content     = resp.choices[0].message.content
        usage       = resp.usage

        span.set_attribute("llm.input_tokens",  usage.prompt_tokens)
        span.set_attribute("llm.output_tokens", usage.completion_tokens)
        span.set_attribute("llm.total_tokens",  usage.total_tokens)

        return content


# ---------------------------------------------------------------------------
# 2. Versioning de prompts
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
# prompts/rag_assistant.yaml
version: "2.3.1"
created: "2026-01-15"
author: "thomas.szadel"
changelog:
  - "2.3.1 : ajout instruction anti-hallucination"
  - "2.3.0 : restructuration du contexte"
  - "2.2.0 : ajout citation obligatoire"

system: |
  Tu es un assistant RAG expert. Réponds UNIQUEMENT en te basant
  sur les passages fournis dans <context>. Si la réponse ne s'y
  trouve pas, dis-le explicitement. Cite tes sources entre [crochets].

user_template: |
  <context>
  {context}
  </context>

  Question : {question}
"""


@dataclass
class VersionedPrompt:
    version: str
    system: str
    user_template: str
    author: str
    changelog: list[str]

    @classmethod
    def from_yaml(cls, content: str) -> "VersionedPrompt":
        data = yaml.safe_load(content)
        return cls(
            version=data["version"],
            system=data["system"],
            user_template=data["user_template"],
            author=data["author"],
            changelog=data.get("changelog", []),
        )

    def render(self, **kwargs) -> tuple[str, str]:
        """Retourne (system_prompt, user_prompt) après interpolation."""
        return self.system, self.user_template.format(**kwargs)


# ---------------------------------------------------------------------------
# 3. Runner d'évaluation CI/CD
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    id:            str
    question:      str
    context:       str
    expected_keywords: list[str]   # présence obligatoire dans la réponse
    forbidden:         list[str]   # mots interdits (hallucinations connues)


@dataclass
class EvalReport:
    case_id:  str
    passed:   bool
    score:    float
    details:  str


async def run_eval(case: EvalCase, prompt: VersionedPrompt) -> EvalReport:
    """Exécute un cas d'évaluation et retourne le rapport."""
    system, user = prompt.render(context=case.context, question=case.question)
    resp = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content.lower()

    # Vérifications mécaniques
    found     = [kw for kw in case.expected_keywords if kw.lower() in answer]
    forbidden = [kw for kw in case.forbidden         if kw.lower() in answer]
    score     = len(found) / max(len(case.expected_keywords), 1)
    passed    = score >= 0.8 and len(forbidden) == 0

    detail = (
        f"Keywords trouvés : {found} / {case.expected_keywords}\n"
        f"Mots interdits   : {forbidden}"
    )
    return EvalReport(case_id=case.id, passed=passed, score=score, details=detail)


# ---------------------------------------------------------------------------
# 4. Shadow mode & feature flags
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model:       str
    temperature: float
    max_tokens:  int


def get_model_config(user_id: str, rollout_pct: float = 0.10) -> ModelConfig:
    """
    Feature flag LLM : route un pourcentage d'utilisateurs vers le nouveau modèle.
    Utilise un hash du user_id pour une attribution stable (pas aléatoire à chaque appel).
    """
    bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    if bucket < rollout_pct * 100:
        return ModelConfig(model="gpt-4o",     temperature=0.3, max_tokens=800)
    else:
        return ModelConfig(model="gpt-4o-mini", temperature=0.3, max_tokens=800)


async def shadow_router(query: str, shadow_rate: float = 0.10) -> str:
    """
    Shadow mode : exécute la requête sur le modèle de prod,
    et en parallèle sur le nouveau modèle (sans servir sa réponse).
    Permet de comparer latence et qualité sans impacter l'utilisateur.
    """
    prod_config   = ModelConfig("gpt-4o-mini", 0.2, 500)
    shadow_config = ModelConfig("gpt-4o",      0.2, 500)

    async def call(config: ModelConfig) -> tuple[str, str]:
        resp = await aclient.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": query}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return config.model, resp.choices[0].message.content

    if random.random() < shadow_rate:
        # Lancement en parallèle — on ne retourne que le résultat prod
        results = await asyncio.gather(call(prod_config), call(shadow_config))
        prod_model, prod_answer   = results[0]
        shadow_model, shadow_ans  = results[1]
        # En prod : log shadow_ans pour analyse, ne pas le retourner
        print(f"[Shadow] {shadow_model}: {shadow_ans[:60]}...")
        return prod_answer
    else:
        _, answer = await call(prod_config)
        return answer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Feature flags
    users = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"]
    print("=== Feature flags (10% rollout) ===")
    for u in users:
        cfg = get_model_config(u, rollout_pct=0.10)
        print(f"  {u:<10} → {cfg.model}")

    # Prompt versionné
    prompt = VersionedPrompt.from_yaml(PROMPT_TEMPLATE)
    print(f"\n=== Prompt v{prompt.version} par {prompt.author} ===")
    sys_p, user_p = prompt.render(
        context="Le RAG combine retrieval et génération.",
        question="Qu'est-ce que le RAG ?",
    )
    print(f"System : {sys_p[:80].strip()}...")
    print(f"User   : {user_p[:80].strip()}...")

    # Shadow mode + évaluation (nécessite OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        answer = asyncio.run(shadow_router("Explique le RAG en 2 phrases.", shadow_rate=1.0))
        print(f"\n=== Réponse prod ===\n{answer}")
