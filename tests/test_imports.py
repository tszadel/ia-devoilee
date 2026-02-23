"""Tests d'import et de fonctions sans appel API."""
import sys
sys.path.insert(0, ".")

def test_ch02_tokenise_naive():
    from ch02_tokenisation.tokenisation import tokenize_naive
    assert tokenize_naive("hello world") == ["hello", "world"]

def test_ch08_causal_mask():
    import torch
    from ch08_attention.self_attention import causal_mask
    mask = causal_mask(4)
    assert mask.shape == (4, 4)
    assert mask[0, 1] == True   # le token 0 ne voit pas le token 1
    assert mask[1, 0] == False  # le token 1 voit le token 0

def test_ch08_mha_shape():
    import torch
    from ch08_attention.self_attention import MultiHeadAttention
    mha = MultiHeadAttention(64, 4)
    x   = torch.randn(2, 8, 64)
    out = mha(x, causal=True)
    assert out.shape == (2, 8, 64)

def test_ch10_implicit_score():
    from ch10_training.finetuning import InteractionSignal
    s = InteractionSignal(session_id="x", prompt="?", response="!", thumbs_up=True)
    assert s.implicit_score > 0.5
    s2 = InteractionSignal(session_id="y", prompt="?", response="!", thumbs_down=True)
    assert s2.implicit_score < 0.5

def test_ch14_causal_mask_context():
    from ch14_production.rag_production import build_optimized_context
    chunks  = ["Chunk A " * 20, "Chunk B " * 20, "Chunk C " * 20]
    context = build_optimized_context("System.", chunks, max_tokens=200)
    assert "Chunk A" in context

def test_ch17_cost_estimate():
    from ch17_couts.optimisation_couts import estimate_cost
    cost = estimate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=200)
    assert cost > 0
    assert cost < 0.01  # moins d'1 centime pour 1200 tokens

def test_ch17_circuit_breaker():
    from ch17_couts.optimisation_couts import CircuitBreaker, Provider
    cb = CircuitBreaker(Provider.OPENAI, failure_limit=2, cooldown=1)
    assert not cb.is_open
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open
