package fr.szadel.ia.ch14;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.assertj.core.api.Assertions.*;

class CascadeServiceTest {

    private final CascadeService.ConfidenceClassifier classifier =
        new CascadeService.ConfidenceClassifier();

    @ParameterizedTest(name = "{0} → confident={1}")
    @CsvSource({
        "'Je ne suis pas sûr de la réponse.',                                false",
        "'Le RAG combine vector store et LLM pour ancrer les réponses.',     true",
        "'Peut-être, ça dépend.',                                            false",
        "'LoRA réduit les paramètres entraînables via factorisation delta.', true",
        "'I dont know.',                                                      false",
    })
    void classifierDetectsUncertainty(String response, boolean expectedConfident) {
        assertThat(classifier.isConfident(response)).isEqualTo(expectedConfident);
    }

    @Test
    void shortResponseIsNotConfident() {
        assertThat(classifier.isConfident("Ok.")).isFalse();
    }

    @Test
    void longConfidentResponsePasses() {
        String response = "Le mécanisme d'attention calcule des scores de similarité "
            + "entre chaque paire de tokens, permettant au modèle de pondérer "
            + "dynamiquement l'importance de chaque position dans la séquence.";
        assertThat(classifier.isConfident(response)).isTrue();
    }
}
