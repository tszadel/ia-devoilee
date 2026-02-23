"""
Chapitre 11 — Génération de texte
===================================
1. Inspection du forward pass (distribution de probabilité sur le vocabulaire)
2. Similarité cosinus sur embeddings OpenAI
3. Structured output (JSON Schema)
4. Comptage de tokens avec tiktoken

Dépendances : torch, transformers, tiktoken, openai, numpy
"""

from __future__ import annotations

import os
import json
import numpy as np
import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# ---------------------------------------------------------------------------
# 1. Forward pass : voir les 10 tokens les plus probables
# ---------------------------------------------------------------------------

def inspect_next_token(prompt: str, model_id: str = "gpt2", top_k: int = 10):
    """
    Charge GPT-2 localement et affiche les top-k tokens suivants.
    Illustre concrètement ce que produit un LLM : une distribution.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    inputs  = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits  = model(**inputs).logits[0, -1, :]   # (vocab_size,)
    probs   = torch.softmax(logits, dim=-1)

    top_idx  = probs.topk(top_k).indices
    top_prob = probs.topk(top_k).values

    print(f"\nPrompt : {prompt!r}")
    print(f"{'Token':<20} {'ID':>8}   {'Prob':>8}")
    print("-" * 40)
    for idx, prob in zip(top_idx.tolist(), top_prob.tolist()):
        tok = tokenizer.decode([idx])
        print(f"{tok!r:<20} {idx:>8}   {prob:>8.4%}")


# ---------------------------------------------------------------------------
# 2. Similarité cosinus sur embeddings OpenAI
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compare_embeddings(texts: list[str], model: str = "text-embedding-3-small"):
    """Encode une liste de textes et affiche la matrice de similarité."""
    resp = client.embeddings.create(input=texts, model=model)
    vecs = [np.array(e.embedding) for e in resp.data]

    print(f"\n=== Matrice de similarité ({model}) ===\n")
    header = "".join(f"{i:>8}" for i in range(len(texts)))
    print(f"{'':>4}{header}")
    for i, vi in enumerate(vecs):
        row = "".join(f"{cosine(vi, vj):>8.3f}" for vj in vecs)
        print(f"  {i}  {row}   {texts[i][:30]!r}")


# ---------------------------------------------------------------------------
# 3. Structured output (JSON Schema)
# ---------------------------------------------------------------------------

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment":   {"type": "string", "enum": ["positif", "négatif", "neutre"]},
        "score":       {"type": "number", "minimum": -1, "maximum": 1},
        "justification": {"type": "string", "maxLength": 200},
    },
    "required": ["sentiment", "score", "justification"],
    "additionalProperties": False,
}


def analyze_sentiment(text: str) -> dict:
    """Analyse de sentiment avec garantie de format JSON via JSON Schema."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu analyses le sentiment de textes en français."},
            {"role": "user",   "content": f"Analyse : {text}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name":   "sentiment_analysis",
                "strict": True,
                "schema": SENTIMENT_SCHEMA,
            },
        },
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# 4. Comptage de tokens
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Forward pass local (GPT-2, ~500Mo)
    inspect_next_token("La capitale de la France est")

    # 2. Embeddings (nécessite OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        compare_embeddings([
            "Le chat dort sur le canapé.",
            "Le félin sommeille sur le sofa.",
            "La voiture roule sur l'autoroute.",
        ])

        # 3. Structured output
        avis = "Ce restaurant était absolument fantastique, je recommande !"
        result = analyze_sentiment(avis)
        print(f"\nSentiment de {avis!r[:40]} :")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 4. Comptage
    prompt = "Compare GPT et BERT comme si j'avais 10 ans."
    print(f"\nTokens dans {prompt!r} : {count_tokens(prompt)}")
