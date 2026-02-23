package fr.szadel.ia.ch14;

import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;

import java.time.Duration;

/**
 * Chapitre 14 — Cascade de modèles avec LangChain4j.
 *
 * <p>Stratégie : tenter d'abord le modèle local/rapide (Ollama),
 * escalader vers Claude si le classifieur de confiance détecte
 * une réponse hésitante.
 *
 * <p>Avantage : ~80% des requêtes simples résolues localement,
 * zéro coût API. Les 20% complexes vont sur Claude Sonnet.
 *
 * <p>Référence livre : section 14.3 "Cascade et routage intelligent"
 */
public class CascadeService {

    // -------------------------------------------------------------------------
    // Interfaces LangChain4j — le proxy est généré à la compilation
    // -------------------------------------------------------------------------

    interface FastAssistant {
        @SystemMessage("Tu es un assistant concis. Réponds en 1-2 phrases maximum.")
        String answer(String question);
    }

    interface DeepAssistant {
        @SystemMessage("""
            Tu es un expert. Analyse la question en profondeur,
            structure ta réponse avec des sections claires,
            et justifie chaque affirmation.
            """)
        String answer(String question);
    }

    // -------------------------------------------------------------------------
    // Classifieur de confiance — heuristique basique
    // (en production : fine-tuner un petit classifieur ou utiliser un LLM-juge)
    // -------------------------------------------------------------------------

    static class ConfidenceClassifier {

        private static final String[] UNCERTAINTY_MARKERS = {
            "je ne suis pas sûr", "je ne sais pas", "peut-être",
            "il est possible", "je pense", "probablement",
            "i'm not sure", "i don't know", "it depends",
        };

        /**
         * Retourne true si la réponse semble confiante.
         * Retourne false → escalade vers le modèle puissant.
         */
        public boolean isConfident(String response) {
            String lower = response.toLowerCase();
            for (String marker : UNCERTAINTY_MARKERS) {
                if (lower.contains(marker)) return false;
            }
            // Réponse trop courte = probable aveu d'ignorance
            return response.trim().length() > 30;
        }
    }

    // -------------------------------------------------------------------------
    // Service de cascade
    // -------------------------------------------------------------------------

    private final FastAssistant fast;
    private final DeepAssistant deep;
    private final ConfidenceClassifier classifier = new ConfidenceClassifier();

    public CascadeService(String anthropicApiKey, String ollamaBaseUrl) {
        ChatLanguageModel ollamaModel = OllamaChatModel.builder()
            .baseUrl(ollamaBaseUrl)           // ex. "http://localhost:11434"
            .modelName("llama3.2:3b")
            .timeout(Duration.ofSeconds(30))
            .build();

        ChatLanguageModel claudeModel = AnthropicChatModel.builder()
            .apiKey(anthropicApiKey)
            .modelName("claude-haiku-4-5-20251001")   // haiku = rapide + pas cher
            .maxTokens(1000)
            .build();

        this.fast = AiServices.create(FastAssistant.class, ollamaModel);
        this.deep = AiServices.create(DeepAssistant.class, claudeModel);
    }

    /**
     * Tente Ollama en premier. Escalade vers Claude si confiance insuffisante.
     *
     * @return record contenant la réponse et le modèle utilisé
     */
    public record CascadeResult(String answer, String modelUsed, boolean escalated) {}

    public CascadeResult answer(String question) {
        String fastAnswer = fast.answer(question);

        if (classifier.isConfident(fastAnswer)) {
            return new CascadeResult(fastAnswer, "llama3.2:3b", false);
        }

        // Escalade : le contexte de la question est retransmis intégralement
        String deepAnswer = deep.answer(question);
        return new CascadeResult(deepAnswer, "claude-haiku-4-5", true);
    }

    // -------------------------------------------------------------------------
    // Demo
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        // Sans Ollama : montrer la logique du classifieur seul
        var classifier = new ConfidenceClassifier();

        var exemples = new String[][] {
            {"Je ne suis pas sûr, mais peut-être que le RAG utilise des embeddings.", "false"},
            {"Le RAG combine un vector store et un LLM pour ancrer les réponses.", "true"},
            {"Ça dépend du contexte.", "false"},
            {"LoRA réduit le nombre de paramètres entraînables en factorisant les deltas de poids.", "true"},
        };

        System.out.println("=== Classifieur de confiance ===\n");
        for (var ex : exemples) {
            boolean confident = classifier.isConfident(ex[0]);
            String expected   = ex[1];
            String status     = String.valueOf(confident).equals(expected) ? "✅" : "❌";
            System.out.printf("  %s [confident=%b] %s%n", status, confident,
                ex[0].substring(0, Math.min(60, ex[0].length())));
        }

        System.out.println("""
            
            Pour tester la cascade complète, lancez Ollama localement :
              docker run -d -p 11434:11434 ollama/ollama
              docker exec ollama ollama pull llama3.2:3b
            
            Puis définissez ANTHROPIC_API_KEY et décommentez main().
            """);
    }
}
