package fr.szadel.ia.ch14;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.PgVectorStore;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.beans.factory.annotation.Autowired;

/**
 * Chapitre 14 — Pipeline RAG complet avec Spring AI.
 *
 * <p>Architecture :
 * <pre>
 *   HTTP POST /ask
 *       ↓
 *   ChatClient  (Spring AI)
 *       ↓ advisor
 *   QuestionAnswerAdvisor  ← injecte les chunks pertinents automatiquement
 *       ↓
 *   PgVectorStore  ← pgvector (PostgreSQL + extension vector)
 *       ↓
 *   LLM (OpenAI / Anthropic / Ollama — même interface)
 * </pre>
 *
 * <p>Prérequis : PostgreSQL avec l'extension pgvector installée.
 * Voir {@code src/main/resources/docker-compose.yml}.
 *
 * <p>Référence livre : section 14.2 "Architectures de production"
 */
@Configuration
public class RagConfig {

    /**
     * Le ChatClient orchestre le pipeline complet.
     * Les advisors s'enchaînent dans l'ordre de déclaration.
     */
    @Bean
    public ChatClient chatClient(ChatModel model, VectorStore vectorStore) {
        return ChatClient.builder(model)
            .defaultAdvisors(
                // RAG : injecte automatiquement les k chunks pertinents dans le prompt
                new QuestionAnswerAdvisor(
                    vectorStore,
                    SearchRequest.defaults().withTopK(5)
                )
                // Guardrail d'entrée : à ajouter ici (SafeGuardAdvisor, LoggingAdvisor…)
            )
            .defaultSystemText("""
                Tu es un assistant expert. Réponds UNIQUEMENT en te basant
                sur le contexte fourni. Si la réponse ne s'y trouve pas,
                dis-le explicitement. Cite tes sources.
                """)
            .build();
    }

    /**
     * Vector store : pgvector via Spring Data JDBC.
     * Configuration dans application.yml : spring.datasource.*
     */
    @Bean
    public VectorStore vectorStore(EmbeddingModel embeddingModel, JdbcTemplate jdbcTemplate) {
        return new PgVectorStore(jdbcTemplate, embeddingModel);
    }
}


// =============================================================================
// Endpoint REST : le pipeline RAG en 3 lignes métier
// =============================================================================

@RestController
class AssistantController {

    @Autowired
    private ChatClient chatClient;

    /**
     * Répond à une question en exploitant le RAG.
     * Le retrieval, l'injection du contexte et la génération sont transparents.
     *
     * <pre>
     * curl -X POST http://localhost:8080/ask \
     *   -H "Content-Type: text/plain" \
     *   -d "Quelle est la procédure de remboursement ?"
     * </pre>
     */
    @PostMapping("/ask")
    public String ask(@RequestBody String question) {
        return chatClient.prompt()
            .user(question)
            .call()
            .content();
    }
}
