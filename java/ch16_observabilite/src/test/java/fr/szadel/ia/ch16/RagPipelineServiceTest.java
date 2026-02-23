package fr.szadel.ia.ch16;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class RagPipelineServiceTest {

    private InMemorySpanExporter spanExporter;
    private RagPipelineService   service;
    private SimpleMeterRegistry  meterRegistry;

    @BeforeEach
    void setUp() {
        spanExporter = InMemorySpanExporter.create();

        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
            .addSpanProcessor(SimpleSpanProcessor.create(spanExporter))
            .build();

        OpenTelemetry otel = OpenTelemetrySdk.builder()
            .setTracerProvider(tracerProvider)
            .build();

        Tracer tracer = otel.getTracer("test");
        meterRegistry = new SimpleMeterRegistry();
        service       = new RagPipelineService(tracer, meterRegistry);
    }

    @Test
    void ragQueryProducesRootSpanAndChildSpans() {
        service.ragQuery("Qu'est-ce que le RAG ?", "session-test-1");

        var spans     = spanExporter.getFinishedSpanItems();
        var spanNames = spans.stream().map(s -> s.getName()).toList();

        assertThat(spanNames).contains("rag-request", "retrieval", "generation");
    }

    @Test
    void ragQueryRecordsTimerMetrics() {
        service.ragQuery("Qu'est-ce que le RAG ?", "session-test-2");

        assertThat(meterRegistry.find("llm.retrieval.duration").timer()).isNotNull();
        assertThat(meterRegistry.find("llm.generation.duration").timer()).isNotNull();
    }

    @Test
    void ragQueryReturnsNonEmptyResponse() {
        String response = service.ragQuery("Qu'est-ce que le RAG ?", "session-test-3");
        assertThat(response).isNotBlank();
    }

    @Test
    void rootSpanContainsSessionAttribute() {
        service.ragQuery("Test ?", "my-session-id");

        var rootSpan = spanExporter.getFinishedSpanItems().stream()
            .filter(s -> "rag-request".equals(s.getName()))
            .findFirst();

        assertThat(rootSpan).isPresent();
        assertThat(rootSpan.get().getAttributes().get(
            io.opentelemetry.api.common.AttributeKey.stringKey("session.id")
        )).isEqualTo("my-session-id");
    }
}
