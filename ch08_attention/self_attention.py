"""
Chapitre 8 — Mécanisme d'attention
===================================
Self-attention causale minimale et implémentation MultiHeadAttention
en PyTorch — sans bibliothèque externe hormis torch.

Référence : Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. Masque causal (pour la génération auto-régressive)
# ---------------------------------------------------------------------------

def causal_mask(T: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Retourne un masque booléen (T, T) : True = position à masquer.
    Le token i ne peut voir que les tokens 0..i.
    """
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


def scaled_dot_product_attention(
    Q: torch.Tensor,   # (B, H, T, d_k)
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, H, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)   # (B, H, T, T)
    return weights @ V                    # (B, H, T, d_k)


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention avec masque causal optionnel.

    Paramètres
    ----------
    d_model : dimension du modèle (ex. 512)
    n_heads : nombre de têtes (ex. 8)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, n_heads, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, T, d_k) → (B, T, d_model)"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        x: torch.Tensor,              # (B, T, d_model)
        causal: bool = True,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self._split_heads(self.W_q(x))   # (B, H, T, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        mask = causal_mask(T, x.device) if causal else None
        out  = scaled_dot_product_attention(Q, K, V, mask)   # (B, H, T, d_k)
        out  = self._merge_heads(out)                         # (B, T, d_model)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, d_model, n_heads = 2, 16, 64, 4
    mha = MultiHeadAttention(d_model, n_heads)

    x   = torch.randn(B, T, d_model)
    out = mha(x, causal=True)

    print(f"Entrée  : {x.shape}")
    print(f"Sortie  : {out.shape}")
    print(f"Paramètres MHA : {sum(p.numel() for p in mha.parameters()):,}")

    # Vérification : le masque causal fait bien disparaître le futur
    mask = causal_mask(T)
    print(f"\nMasque causal (4×4) :\n{mask[:4, :4].int()}")
