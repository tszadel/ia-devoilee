# Python

Exemples Python du livre — chapitres 2, 8 à 12, 14 à 17.

## Prérequis

Python 3.11+

```bash
pip install -r requirements.txt
```

## Fichiers

| Fichier                                  | Chapitre | Concept clé                                      |
|------------------------------------------|----------|--------------------------------------------------|
| `ch02_tokenisation/tokenisation.py`      | 2        | BPE vs split naïf, tiktoken                      |
| `ch08_attention/self_attention.py`       | 8        | MultiHeadAttention + masque causal (PyTorch)     |
| `ch09_embeddings/rag_pipeline.py`        | 9        | Similarité cosinus, pipeline RAG ChromaDB        |
| `ch10_training/finetuning.py`            | 10       | DPO (TRL), gradient clipping, data flywheel      |
| `ch11_generation/generation.py`          | 11       | Forward pass GPT-2, structured output JSON       |
| `ch12_evaluation/llm_judge.py`           | 12       | LLM-as-a-judge multi-critères                    |
| `ch14_production/rag_production.py`      | 14       | Reranking cross-encodeur, anti-lost-in-middle    |
| `ch15_agents/agent_loop.py`              | 15       | ReAct, tools JSON Schema, mémoire épisodique     |
| `ch16_observabilite/observabilite.py`    | 16       | OpenTelemetry, versioning prompts, shadow mode   |
| `ch17_couts/optimisation_couts.py`       | 17       | Pricing, prompt cache, cascade, circuit breaker  |

## Tests

```bash
# Tests sans API key (CI)
pytest tests/test_imports.py -v

# Tests avec API key (local uniquement)
pytest tests/test_api.py -v -m api
```
