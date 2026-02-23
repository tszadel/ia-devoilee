package fr.szadel.ia.ch16;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import io.opentelemetry.exporter.logging.LoggingSpanExporter;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.semconv.resource.attributes.ResourceAttributes;
import io.opentelemetry.api.common.Attributes;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

/**
 * Chapitre 16 — Configuration OpenTelemetry.
 *
 * <p>Deux profils :
 * <ul>
 *   <li>{@code dev}  : export console (LoggingSpanExporter) — zéro infrastructure</li>
 *   <li>{@code prod} : export OTLP vers Jaeger/Tempo via BatchSpanProcessor</li>
 * </ul>
 *
 * <p>Activation : {@code spring.profiles.active=dev} dans application.yml.
 */
@Configuration
public class OtelConfig {

    private static final String SERVICE_NAME = "ia-devoilee-rag";

    private Resource serviceResource() {
        return Resource.getDefault().merge(
            Resource.create(Attributes.of(
                ResourceAttributes.SERVICE_NAME,    SERVICE_NAME,
                ResourceAttributes.SERVICE_VERSION, "2.1.4"
            ))
        );
    }

    // -------------------------------------------------------------------------
    // Profil DEV : export console — idéal pour le développement local
    // -------------------------------------------------------------------------

    @Bean
    @Profile("dev")
    public OpenTelemetry openTelemetryDev() {
        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
            .setResource(serviceResource())
            .addSpanProcessor(SimpleSpanProcessor.create(LoggingSpanExporter.create()))
            .build();

        return OpenTelemetrySdk.builder()
            .setTracerProvider(tracerProvider)
            .buildAndRegisterGlobal();
    }

    // -------------------------------------------------------------------------
    // Profil PROD : export OTLP → Jaeger, Grafana Tempo, Honeycomb…
    // -------------------------------------------------------------------------

    @Bean
    @Profile("prod")
    public OpenTelemetry openTelemetryProd() {
        String otlpEndpoint = System.getenv().getOrDefault(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"
        );

        SpanExporter exporter = OtlpGrpcSpanExporter.builder()
            .setEndpoint(otlpEndpoint)
            .build();

        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
            .setResource(serviceResource())
            // BatchSpanProcessor : bufferize et envoie par lots — moins de latence qu'un SimpleProcessor
            .addSpanProcessor(BatchSpanProcessor.builder(exporter)
                .setMaxExportBatchSize(512)
                .build())
            .build();

        return OpenTelemetrySdk.builder()
            .setTracerProvider(tracerProvider)
            .buildAndRegisterGlobal();
    }

    @Bean
    public Tracer tracer(OpenTelemetry openTelemetry) {
        return openTelemetry.getTracer(SERVICE_NAME);
    }
}
