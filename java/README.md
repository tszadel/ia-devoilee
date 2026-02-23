# Java

Exemples Java du livre — chapitres 9, 14, 15, 16.  
Projet Maven multi-modules, Java 21, LangChain4j + Spring AI.

## Prérequis

- Java 21+
- Maven 3.9+
- Docker (infra locale)

## Fichiers

| Module              | Fichier principal          | Chapitre | Concept clé                                   |
|---------------------|----------------------------|----------|-----------------------------------------------|
| `ch09_embeddings`   | `CosineSimilarity.java`    | 9.1      | Similarité cosinus, recherche sémantique      |
| `ch09_embeddings`   | `RagPipeline.java`         | 9.4      | RAG LangChain4j (InMemory + Weaviate)         |
| `ch14_production`   | `RagConfig.java`           | 14.2     | Spring AI + pgvector + QuestionAnswerAdvisor  |
| `ch14_production`   | `CascadeService.java`      | 14.3     | Cascade Ollama → Claude + classifieur         |
| `ch15_agents`       | `JiraAgent.java`           | 15.2     | Boucle ReAct via `@Tool` LangChain4j          |
| `ch16_observabilite`| `RagPipelineService.java`  | 16.1     | Traces OTel + métriques Micrometer            |
| `ch16_observabilite`| `OtelConfig.java`          | 16.1     | Profils dev (console) / prod (OTLP)           |

## Compilation & tests

```bash
# Tout compiler et tester
mvn clean verify

# Un seul module
mvn clean verify -pl ch09_embeddings

# Tests uniquement
mvn test -pl ch16_observabilite
```

## Infra locale

```bash
docker compose -f docker-compose.infra.yml up -d
```

| Service   | URL                     | Utilisé par        |
|-----------|-------------------------|--------------------|
| pgvector  | `localhost:5432`        | ch14 Spring AI     |
| Weaviate  | `http://localhost:8080` | ch09 LangChain4j   |
| Jaeger UI | `http://localhost:16686`| ch16 OpenTelemetry |
