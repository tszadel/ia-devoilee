package fr.szadel.ia.ch09;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.weaviate.WeaviateEmbeddingStore;

import java.util.List;

/**
 * Chapitre 9 — Pipeline RAG complet avec LangChain4j.
 *
 * <p>Deux variantes proposées :
 * <ol>
 *   <li>{@link InMemoryRagPipeline} — zero infrastructure, idéal pour les tests</li>
 *   <li>{@link WeaviateRagPipeline} — production avec Weaviate (docker-compose inclus)</li>
 * </ol>
 *
 * <p>Référence livre : section 9.4 "RAG — la boîte à outils du développeur"
 */
public class RagPipeline {

    // -------------------------------------------------------------------------
    // Interface commune : le LLM ne voit pas le vector store utilisé
    // -------------------------------------------------------------------------

    interface Assistant {
        @SystemMessage("""
            Tu es un assistant pédagogique expert en LLM.
            Réponds UNIQUEMENT en te basant sur le contexte fourni.
            Si la réponse ne s'y trouve pas, dis-le explicitement.
            Cite tes sources entre [crochets].
            """)
        String answer(String question);
    }

    // -------------------------------------------------------------------------
    // Variante 1 : In-Memory — aucune infrastructure requise
    // -------------------------------------------------------------------------

    public static class InMemoryRagPipeline {

        private final Assistant assistant;

        public InMemoryRagPipeline(String openAiApiKey, List<String> documents) {
            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            // Store en mémoire : parfait pour les tests et les démos
            EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();

            // Indexation : découpage → embedding → stockage en une passe
            var ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(400, 50))
                .embeddingModel(embeddingModel)
                .embeddingStore(store)
                .build();

            documents.stream()
                .map(Document::from)
                .forEach(ingestor::ingest);

            // Retriever : top-5 par similarité cosinus
            var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.6)   // filtre les chunks peu pertinents
                .build();

            this.assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(
                    OpenAiChatModel.builder()
                        .apiKey(openAiApiKey)
                        .modelName("gpt-4o-mini")
                        .temperature(0.2)
                        .build()
                )
                .contentRetriever(retriever)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
        }

        public String ask(String question) {
            return assistant.answer(question);
        }
    }

    // -------------------------------------------------------------------------
    // Variante 2 : Weaviate — production
    // -------------------------------------------------------------------------

    public static class WeaviateRagPipeline {

        private final Assistant assistant;

        /**
         * @param anthropicApiKey  clé API Anthropic
         * @param weaviateHost     ex. "localhost:8080" ou "cluster.weaviate.io"
         * @param weaviateScheme   "http" (local) ou "https" (cloud)
         */
        public WeaviateRagPipeline(String anthropicApiKey,
                                   String weaviateHost,
                                   String weaviateScheme) {
            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            // Weaviate : self-hosted ou cloud
            EmbeddingStore<TextSegment> store = WeaviateEmbeddingStore.builder()
                .scheme(weaviateScheme)
                .host(weaviateHost)
                .className("DocChunk")
                .build();

            var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.65)
                .build();

            this.assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(
                    AnthropicChatModel.builder()
                        .apiKey(anthropicApiKey)
                        .modelName("claude-haiku-4-5-20251001")
                        .maxTokens(800)
                        .build()
                )
                .contentRetriever(retriever)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
        }

        /**
         * Indexe des documents dans Weaviate.
         * À appeler une fois lors de l'initialisation du pipeline.
         */
        public void ingest(List<String> documents) {
            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
            EmbeddingStore<TextSegment> store = WeaviateEmbeddingStore.builder()
                .scheme("http").host("localhost:8080")
                .className("DocChunk")
                .build();

            var ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(400, 50))
                .embeddingModel(embeddingModel)
                .embeddingStore(store)
                .build();

            documents.stream()
                .map(Document::from)
                .forEach(ingestor::ingest);
        }

        public String ask(String question) {
            return assistant.answer(question);
        }
    }

    // -------------------------------------------------------------------------
    // Demo autonome
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        var documents = List.of(
            "La tokenisation BPE divise les mots rares en sous-unités fréquentes.",
            "L'attention multi-têtes permet au modèle de se concentrer sur plusieurs parties simultanément.",
            "Le RAG combine recherche documentaire et génération de texte pour ancrer les réponses.",
            "LoRA réduit le nombre de paramètres entraînables en factorisant les mises à jour de matrices.",
            "La fenêtre de contexte limite la quantité de texte qu'un LLM traite en une seule fois.",
            "Le temperature contrôle l'aléatoire de la génération : 0 = déterministe, 2 = très créatif."
        );

        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null) {
            System.err.println("⚠️  OPENAI_API_KEY manquante — démo désactivée.");
            return;
        }

        var pipeline = new InMemoryRagPipeline(apiKey, documents);

        var questions = List.of(
            "Comment fonctionne le RAG ?",
            "Qu'est-ce que LoRA ?",
            "Quel est le rôle du temperature ?"
        );

        for (var q : questions) {
            System.out.printf("%n❓ %s%n💬 %s%n", q, pipeline.ask(q));
        }
    }
}
