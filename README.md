# L'IA dévoilée — Code de référence

> Code d'accompagnement du livre **"L'IA dévoilée : voyage au cœur de la boîte noire"**  
> Thomas SZADEL — 2026

---

## Organisation

Le code est organisé **par langage**, puis par chapitre.  
Chaque dossier langage est autonome : ses propres dépendances, ses propres tests, son propre CI.

```
.
├── python/          ← exemples Python (chapitres 2, 8–12, 14–17)
└── java/            ← exemples Java   (chapitres 9, 14–16)
```

Ajouter un nouveau langage ? Créer un dossier `typescript/`, `kotlin/`, `rust/`…  
et suivre la même convention `chXX_nom/`.

---

## Couverture par chapitre

| Chapitre | Sujet                        | Python | Java |
|----------|------------------------------|:------:|:----:|
| 2        | Tokenisation                 | ✅     |      |
| 8        | Mécanisme d'attention        | ✅     |      |
| 9        | Embeddings & RAG             | ✅     | ✅   |
| 10       | Entraînement & fine-tuning   | ✅     |      |
| 11       | Génération de texte          | ✅     |      |
| 12       | Évaluation LLM-as-a-judge    | ✅     |      |
| 14       | Mise en production           | ✅     | ✅   |
| 15       | Agents & multi-agents        | ✅     | ✅   |
| 16       | Observabilité & CI/CD        | ✅     | ✅   |
| 17       | Optimisation des coûts       | ✅     |      |

---

## Démarrage rapide

### Python

```bash
cd python
pip install -r requirements.txt

# Exemple : attention multi-têtes from scratch
python ch08_attention/self_attention.py

# Exemple : agent ReAct
python ch15_agents/agent_loop.py

# Tests (sans API key)
pytest tests/test_imports.py
```

### Java

```bash
cd java

# Démarrer l'infra locale (pgvector + Weaviate + Jaeger)
docker compose -f docker-compose.infra.yml up -d

# Compiler et tester
mvn clean verify

# Un seul module
mvn clean verify -pl ch09_embeddings
```

---

## Variables d'environnement

Les exemples qui appellent des APIs externes lisent ces variables.  
Créez un fichier `.env` à la racine (chargé automatiquement en Python via `python-dotenv`) :

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Aucun exemple ne fonctionne *uniquement* avec une API key — tous ont un mode dégradé ou local.

---

## Infrastructure locale (Java / ch14–16)

```bash
docker compose -f java/docker-compose.infra.yml up -d
```

| Service    | URL                       | Utilisé par                    |
|------------|---------------------------|--------------------------------|
| pgvector   | `localhost:5432`          | Spring AI, ch14                |
| Weaviate   | `http://localhost:8080`   | LangChain4j, ch09 + ch14       |
| Jaeger UI  | `http://localhost:16686`  | Traces OpenTelemetry, ch16     |

---

## CI/CD

| Workflow            | Déclenché sur              | Contenu                              |
|---------------------|----------------------------|--------------------------------------|
| `ci-python.yml`     | push `python/**`           | ruff + pytest (sans API)             |
| `ci-java.yml`       | push `java/**`             | `mvn verify` (sans API)              |
| job `api-tests`     | merge sur `main` seulement | tests d'intégration avec vraies APIs |

---

## Licence

MIT — libre d'utilisation pour l'apprentissage et les projets personnels.  
Merci de citer le livre si vous réutilisez ces exemples dans un contexte professionnel.
