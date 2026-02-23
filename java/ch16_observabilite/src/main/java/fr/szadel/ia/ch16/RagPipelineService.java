package fr.szadel.ia.ch16;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * Chapitre 16 — Pipeline RAG instrumenté avec OpenTelemetry + Micrometer.
 *
 * <p>Stratégie d'observabilité :
 * <ul>
 *   <li><b>Traces (OTel)</b> : timeline request → retrieval → generation, avec attributs
 *       métier (nb chunks, score top-1, tokens in/out). Exporté vers Jaeger ou Tempo.</li>
 *   <li><b>Métriques (Micrometer → Prometheus)</b> : latences p50/p95/p99,
 *       compteurs guardrails, tokens par requête.</li>
 * </ul>
 *
 * <p>Dashboard Grafana fourni dans {@code src/main/resources/grafana-dashboard.json}.
 *
 * <p>Référence livre : section 16.1 "Traces, métriques, logs"
 */
@Service
public class RagPipelineService {

    // -------------------------------------------------------------------------
    // Dépendances
    // -------------------------------------------------------------------------

    private final Tracer        tracer;
    private final MeterRegistry registry;

    // Métriques Micrometer — enregistrées une fois, réutilisées à chaque appel
    private final Timer   retrievalTimer;
    private final Timer   generationTimer;
    private final Counter guardrailCounter;
    private final Counter tokenCounter;

    public RagPipelineService(Tracer tracer, MeterRegistry registry) {
        this.tracer   = tracer;
        this.registry = registry;

        this.retrievalTimer = Timer.builder("llm.retrieval.duration")
            .description("Latence du retrieval vectoriel")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);

        this.generationTimer = Timer.builder("llm.generation.duration")
            .description("Latence de génération LLM")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);

        this.guardrailCounter = Counter.builder("llm.guardrail.triggered")
            .description("Nombre de guardrails déclenchés")
            .tag("type", "low_grounding")
            .register(registry);

        this.tokenCounter = Counter.builder("llm.tokens.total")
            .description("Total de tokens consommés")
            .register(registry);
    }

    // -------------------------------------------------------------------------
    // Pipeline instrumenté
    // -------------------------------------------------------------------------

    /**
     * Exécute la requête RAG complète avec tracing et métriques.
     *
     * @param userQuery  question de l'utilisateur
     * @param sessionId  identifiant de session (pour correler les traces)
     * @return           réponse générée
     */
    public String ragQuery(String userQuery, String sessionId) {

        // Span racine — visible dans Jaeger comme "rag-request"
        Span rootSpan = tracer.spanBuilder("rag-request")
            .setAttribute("user.query.length", userQuery.length())
            .setAttribute("session.id",        sessionId)
            .setAttribute("pipeline.version",  "2.1.4")
            .startSpan();

        try (var scope = rootSpan.makeCurrent()) {

            // --- Étape 1 : Retrieval ---
            List<Chunk> chunks = timedSpan(
                "retrieval",
                retrievalTimer,
                () -> {
                    var results = vectorStore().search(userQuery, 5);
                    Span.current().setAttribute("retrieval.chunks_count", results.size());
                    if (!results.isEmpty()) {
                        Span.current().setAttribute("retrieval.top_score", results.get(0).score());
                    }
                    return results;
                }
            );

            // Guardrail d'entrée : score de pertinence insuffisant → ne pas générer
            if (chunks.isEmpty() || chunks.get(0).score() < 0.6) {
                guardrailCounter.increment();
                rootSpan.setAttribute("guardrail.triggered", "low_retrieval_score");
                return "Je n'ai pas trouvé de document suffisamment pertinent pour répondre à cette question.";
            }

            // --- Étape 2 : Génération ---
            String prompt = buildPrompt(userQuery, chunks);

            String response = timedSpan(
                "generation",
                generationTimer,
                () -> {
                    int promptTokens = estimateTokens(prompt);
                    Span.current().setAttribute("generation.prompt_tokens",  promptTokens);
                    Span.current().setAttribute("generation.model",          "claude-sonnet-4");

                    String result = llmClient().generate(prompt);

                    int outputTokens = estimateTokens(result);
                    Span.current().setAttribute("generation.output_tokens", outputTokens);
                    tokenCounter.increment(promptTokens + outputTokens);

                    return result;
                }
            );

            // --- Guardrail de sortie : score d'ancrage ---
            double groundingScore = groundingScore(response, chunks);
            rootSpan.setAttribute("quality.grounding_score", groundingScore);

            if (groundingScore < 0.5) {
                guardrailCounter.increment();
                rootSpan.setAttribute("guardrail.triggered", "low_grounding");
            }

            return response;

        } catch (Exception e) {
            rootSpan.setStatus(StatusCode.ERROR, e.getMessage());
            throw e;
        } finally {
            rootSpan.end();
        }
    }

    // -------------------------------------------------------------------------
    // Utilitaires privés
    // -------------------------------------------------------------------------

    /** Lance une opération dans un sous-span OTel tout en mesurant via Micrometer. */
    private <T> T timedSpan(String spanName, Timer timer, Supplier<T> op) {
        long start = System.nanoTime();
        Span span  = tracer.spanBuilder(spanName).startSpan();
        try (var s = span.makeCurrent()) {
            return op.get();
        } catch (Exception e) {
            span.setStatus(StatusCode.ERROR, e.getMessage());
            throw e;
        } finally {
            timer.record(System.nanoTime() - start, TimeUnit.NANOSECONDS);
            span.end();
        }
    }

    // Stubs — à remplacer par vos vraies implémentations
    private VectorStore vectorStore()  { return new VectorStore.Stub(); }
    private LlmClient   llmClient()    { return new LlmClient.Stub(); }

    private String buildPrompt(String query, List<Chunk> chunks) {
        var sb = new StringBuilder("Contexte :\n");
        chunks.forEach(c -> sb.append("- ").append(c.text()).append("\n"));
        sb.append("\nQuestion : ").append(query);
        return sb.toString();
    }

    private int estimateTokens(String text) {
        return text.length() / 4;   // heuristique : ~4 chars par token
    }

    private double groundingScore(String response, List<Chunk> chunks) {
        // Simplifié : proportion de mots de la réponse présents dans les chunks
        String[] words    = response.toLowerCase().split("\\s+");
        String   context  = chunks.stream().map(Chunk::text).reduce("", String::concat).toLowerCase();
        long     found    = java.util.Arrays.stream(words).filter(context::contains).count();
        return words.length > 0 ? (double) found / words.length : 0.0;
    }

    // -------------------------------------------------------------------------
    // Types auxiliaires
    // -------------------------------------------------------------------------

    public record Chunk(String text, double score) {}

    interface VectorStore {
        List<Chunk> search(String query, int topK);

        class Stub implements VectorStore {
            @Override
            public List<Chunk> search(String query, int topK) {
                return List.of(
                    new Chunk("Le RAG combine retrieval et génération.", 0.92),
                    new Chunk("Les embeddings représentent le sens sémantique.", 0.78)
                );
            }
        }
    }

    interface LlmClient {
        String generate(String prompt);

        class Stub implements LlmClient {
            @Override
            public String generate(String prompt) {
                return "Le RAG est une architecture qui combine recherche vectorielle et génération LLM [1].";
            }
        }
    }
}
