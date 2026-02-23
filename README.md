# L'IA dévoilée — Code de référence

> Code d'accompagnement du livre **"L'IA dévoilée : voyage au cœur de la boîte noire"**  
> Thomas SZADEL — 2026

Ce dépôt contient tous les exemples de code du livre, organisés par chapitre.  
Chaque dossier est autonome et exécutable indépendamment.

---

## Structure

```
.
├── ch02_tokenisation/       # Chapitre 2  — Tokenisation
├── ch08_attention/          # Chapitre 8  — Mécanisme d'attention
├── ch09_embeddings/         # Chapitre 9  — Embeddings & RAG
├── ch10_training/           # Chapitre 10 — Entraînement & fine-tuning
├── ch11_generation/         # Chapitre 11 — Génération de texte
├── ch12_evaluation/         # Chapitre 12 — Évaluation LLM-as-a-judge
├── ch14_production/         # Chapitre 14 — Mise en production
├── ch15_agents/             # Chapitre 15 — Agents & multi-agents
├── ch16_observabilite/      # Chapitre 16 — Observabilité & CI/CD
└── ch17_couts/              # Chapitre 17 — Optimisation des coûts
```

## Prérequis

Python 3.11+ recommandé.

```bash
pip install -r requirements.txt
```

Pour les exemples Java (chapitres 9, 14, 15), Maven 3.9+ et Java 21+.

## Lancer un exemple

```bash
# Exemple : attention minimale (chapitre 8)
python ch08_attention/self_attention.py

# Exemple : agent ReAct (chapitre 15)
python ch15_agents/agent_loop.py
```

## Variables d'environnement

Certains exemples appellent des API externes. Créez un fichier `.env` à la racine :

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Licence

MIT — libre d'utilisation pour l'apprentissage et les projets personnels.  
Merci de citer le livre si vous réutilisez ces exemples dans un contexte professionnel.
