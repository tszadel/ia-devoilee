package fr.szadel.ia.ch09;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

class CosineSimilarityTest {

    private final CosineSimilarity sim = new CosineSimilarity();

    @Test
    void identicalVectorsHaveSimilarityOne() {
        float[] v = {1f, 2f, 3f};
        assertThat(CosineSimilarity.cosine(v, v)).isCloseTo(1.0, within(1e-6));
    }

    @Test
    void orthogonalVectorsHaveSimilarityZero() {
        float[] a = {1f, 0f, 0f};
        float[] b = {0f, 1f, 0f};
        assertThat(CosineSimilarity.cosine(a, b)).isCloseTo(0.0, within(1e-6));
    }

    @Test
    void synonymsAreCloserThanUnrelatedSentences() {
        float[] cat1  = sim.encode("Le chat dort sur le canapé.");
        float[] cat2  = sim.encode("Le félin sommeille sur le sofa.");
        float[] car   = sim.encode("La voiture roule sur l'autoroute.");

        double simSynonyms = CosineSimilarity.cosine(cat1, cat2);
        double simUnrelated = CosineSimilarity.cosine(cat1, car);

        assertThat(simSynonyms).isGreaterThan(simUnrelated);
    }

    @Test
    void searchReturnsTopKResults() {
        var corpus = List.of(
            "La tokenisation BPE découpe les mots rares.",
            "Le RAG ancre les réponses dans des documents.",
            "LoRA réduit les paramètres entraînables.",
            "La température contrôle l'aléatoire."
        );
        List<String> results = sim.search("fine-tuning paramètres", corpus, 2);
        assertThat(results).hasSize(2);
    }
}
