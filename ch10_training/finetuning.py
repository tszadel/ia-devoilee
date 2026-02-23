"""
Chapitre 10 — Entraînement & Fine-tuning
==========================================
1. Gradient clipping (stabilité d'entraînement)
2. Gradient checkpointing (économie mémoire)
3. Fine-tuning DPO avec la librairie TRL
4. Collecte de signal implicite pour data flywheel

Dépendances : torch, transformers, trl, datasets
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Gradient clipping — snippet autonome
# ---------------------------------------------------------------------------

def train_step_with_clipping(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    max_norm: float = 1.0,
) -> float:
    """
    Effectue une étape d'optimisation avec gradient clipping.
    Retourne la norme des gradients AVANT clipping (utile pour le monitoring).
    """
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    return float(grad_norm)


# ---------------------------------------------------------------------------
# 2. Gradient checkpointing
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Active le gradient checkpointing sur un modèle HuggingFace.
    Réduit la mémoire GPU de ~40% au prix d'un recalcul partiel.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing activé.")
    else:
        print("⚠️  Ce modèle ne supporte pas gradient_checkpointing_enable().")


# ---------------------------------------------------------------------------
# 3. Fine-tuning DPO
# ---------------------------------------------------------------------------

def run_dpo_training(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    output_dir: str = "./dpo-output",
    num_train_epochs: int = 1,
):
    """
    Fine-tuning DPO minimal avec TRL.

    Le dataset attendu a trois colonnes :
      - prompt   : str
      - chosen   : str  (réponse préférée)
      - rejected : str  (réponse non préférée)
    """
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Dataset jouet
    data = [
        {
            "prompt": "Explique la tokenisation en une phrase.",
            "chosen": "La tokenisation découpe le texte en sous-unités (tokens) que le modèle peut traiter.",
            "rejected": "La tokenisation c'est quand on met des tokens dans le modèle.",
        },
        {
            "prompt": "Qu'est-ce qu'un embedding ?",
            "chosen": "Un embedding est une représentation vectorielle dense qui capture le sens sémantique d'un texte.",
            "rejected": "C'est un nombre.",
        },
    ]
    dataset = Dataset.from_list(data)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        beta=0.1,             # force du signal DPO
        logging_steps=1,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    print(f"✅ DPO terminé. Modèle sauvegardé dans {output_dir}")


# ---------------------------------------------------------------------------
# 4. Collecte de signal implicite (data flywheel)
# ---------------------------------------------------------------------------

@dataclass
class InteractionSignal:
    """Capture un signal implicite de qualité sur une réponse LLM."""

    session_id:       str
    prompt:           str
    response:         str
    timestamp:        datetime   = field(default_factory=datetime.utcnow)
    thumbs_up:        bool       = False
    thumbs_down:      bool       = False
    copy_paste:       bool       = False       # l'utilisateur a copié la réponse
    time_on_page_s:   float      = 0.0         # temps passé à lire
    follow_up_query:  Optional[str] = None     # question de relance ?

    @property
    def implicit_score(self) -> float:
        """
        Score heuristique 0..1 combinant les signaux implicites.
        À remplacer par un vrai modèle de préférence en production.
        """
        score = 0.5
        if self.thumbs_up:   score += 0.4
        if self.thumbs_down: score -= 0.4
        if self.copy_paste:  score += 0.15
        score += min(self.time_on_page_s / 120, 0.1)  # max +0.1 pour 2 min
        return max(0.0, min(1.0, score))

    def to_dpo_pair(self, negative: "InteractionSignal") -> dict:
        """Construit une paire (chosen, rejected) pour DPO."""
        assert self.prompt == negative.prompt
        return {
            "prompt":   self.prompt,
            "chosen":   self.response if self.implicit_score >= negative.implicit_score else negative.response,
            "rejected": negative.response if self.implicit_score >= negative.implicit_score else self.response,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Gradient clipping
    model = nn.Linear(10, 1)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x     = torch.randn(4, 10)
    loss  = ((model(x) - 1) ** 2).mean()
    norm  = train_step_with_clipping(model, opt, loss, max_norm=1.0)
    print(f"Gradient norm avant clipping : {norm:.4f}")

    # Signal implicite
    sig_a = InteractionSignal(
        session_id="s1", prompt="Qu'est-ce que RAG ?",
        response="RAG combine retrieval et génération.",
        thumbs_up=True, copy_paste=True, time_on_page_s=90,
    )
    sig_b = InteractionSignal(
        session_id="s2", prompt="Qu'est-ce que RAG ?",
        response="Je ne sais pas.",
        thumbs_down=True,
    )
    print(f"\nScore A : {sig_a.implicit_score:.2f}")
    print(f"Score B : {sig_b.implicit_score:.2f}")
    print(f"Paire DPO : {sig_a.to_dpo_pair(sig_b)}")

    # DPO (décommenter pour exécuter — nécessite ~1 Go RAM)
    # run_dpo_training()
