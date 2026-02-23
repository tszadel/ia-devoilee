package fr.szadel.ia.ch09;

import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Chapitre 9 — Similarité cosinus entre embeddings.
 *
 * <p>Illustre concrètement l'espace vectoriel sémantique :
 * deux phrases synonymes sont proches, deux phrases sans rapport sont loin.
 *
 * <p>Référence livre : section 9.1 "L'intuition géométrique"
 */
public class CosineSimilarity {

    private final AllMiniLmL6V2EmbeddingModel model = new AllMiniLmL6V2EmbeddingModel();

    // -------------------------------------------------------------------------
    // Calcul de similarité
    // -------------------------------------------------------------------------

    /**
     * Similarité cosinus entre deux vecteurs.
     * Retourne 1.0 si identiques, 0.0 si orthogonaux, -1.0 si opposés.
     */
    public static double cosine(float[] a, float[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-9);
    }

    /**
     * Encode un texte et retourne son vecteur brut.
     */
    public float[] encode(String text) {
        Embedding embedding = model.embed(TextSegment.from(text)).content();
        return embedding.vector();
    }

    /**
     * Calcule et affiche la matrice de similarité pour une liste de textes.
     */
    public void printSimilarityMatrix(List<String> texts) {
        float[][] vectors = texts.stream()
            .map(this::encode)
            .toArray(float[][]::new);

        System.out.printf("%n=== Matrice de similarité (AllMiniLM-L6-v2) ===%n%n");
        System.out.printf("%-4s", "");
        for (int i = 0; i < texts.size(); i++) System.out.printf("%8d", i);
        System.out.println();

        for (int i = 0; i < vectors.length; i++) {
            System.out.printf("%-4d", i);
            for (int j = 0; j < vectors.length; j++) {
                System.out.printf("%8.3f", cosine(vectors[i], vectors[j]));
            }
            System.out.printf("   %s%n", texts.get(i).substring(0, Math.min(40, texts.get(i).length())));
        }
    }

    /**
     * Recherche sémantique naïve : retourne les k textes les plus proches de la requête.
     */
    public List<String> search(String query, List<String> corpus, int topK) {
        float[] queryVec = encode(query);

        record Scored(String text, double score) {}

        return corpus.stream()
            .map(text -> new Scored(text, cosine(queryVec, encode(text))))
            .sorted(Comparator.comparingDouble(Scored::score).reversed())
            .limit(topK)
            .map(Scored::text)
            .toList();
    }

    // -------------------------------------------------------------------------
    // Demo
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        var sim = new CosineSimilarity();

        // Matrice de similarité
        sim.printSimilarityMatrix(List.of(
            "Le chat dort sur le canapé.",
            "Le félin sommeille sur le sofa.",
            "La voiture roule sur l'autoroute.",
            "Un modèle de langage prédit le prochain token."
        ));

        // Recherche sémantique
        var corpus = List.of(
            "La tokenisation BPE découpe les mots rares en sous-mots.",
            "L'attention permet de pondérer l'importance de chaque token.",
            "Le RAG ancre les réponses dans des documents vérifiés.",
            "LoRA réduit les paramètres entraînables par factorisation matricielle.",
            "La température contrôle l'aléatoire de la génération."
        );

        String query = "Comment réduire le coût du fine-tuning ?";
        System.out.printf("%n=== Recherche : %s ===%n", query);
        sim.search(query, corpus, 2).forEach(r -> System.out.println("  → " + r));
    }
}
