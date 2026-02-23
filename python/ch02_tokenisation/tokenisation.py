"""
Chapitre 2 — Tokenisation
=========================
Illustration de la tokenisation naïve par espaces vs. BPE (tiktoken).
"""

import tiktoken


# ---------------------------------------------------------------------------
# 1. Tokenisation naïve : split sur les espaces
# ---------------------------------------------------------------------------

def tokenize_naive(text: str) -> list[str]:
    return text.split(" ")


# ---------------------------------------------------------------------------
# 2. Tokenisation BPE via tiktoken (GPT-4)
# ---------------------------------------------------------------------------

def tokenize_bpe(text: str, model: str = "gpt-4") -> list[int]:
    enc = tiktoken.encoding_for_model(model)
    return enc.encode(text)


def decode_bpe(tokens: list[int], model: str = "gpt-4") -> str:
    enc = tiktoken.encoding_for_model(model)
    return enc.decode(tokens)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exemples = [
        "Combien font 1 + 1 ?",
        "Le transformeur est une architecture révolutionnaire.",
        "tokenization != tokenisation",   # illustre les sous-mots
        "ChatGPT",
    ]

    enc = tiktoken.encoding_for_model("gpt-4")

    for texte in exemples:
        tokens = enc.encode(texte)
        mots   = tokenize_naive(texte)
        print(f"\nTexte    : {texte!r}")
        print(f"  Mots   : {len(mots):3d}  → {mots}")
        print(f"  Tokens : {len(tokens):3d}  → {tokens}")
        print(f"  Décodé : {[enc.decode([t]) for t in tokens]}")
