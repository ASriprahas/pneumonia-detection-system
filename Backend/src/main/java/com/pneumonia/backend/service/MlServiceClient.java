package com.pneumonia.backend.service;

import com.pneumonia.backend.dto.prediction.FastApiEnsembleResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;

import java.io.File;

@Service
public class MlServiceClient {

    private static final Logger logger = LoggerFactory.getLogger(MlServiceClient.class);

    private final RestClient restClient;
    private final String internalApiKey;

    public MlServiceClient(
            @Value("${app.ml-service.url:http://localhost:8000}") String mlServiceUrl,
            @Value("${app.ml-service.internal-key:pneumonia-internal-secret-key-2026}") String internalApiKey
    ) {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(10000);
        requestFactory.setReadTimeout(60000);

        this.restClient = RestClient.builder()
                .baseUrl(mlServiceUrl)
                .requestFactory(requestFactory)
                .build();
        this.internalApiKey = internalApiKey;
        logger.info("Initialized MlServiceClient (HTTP/1.1) pointing to {}", mlServiceUrl);
    }

    public FastApiEnsembleResponse callEnsemblePrediction(File file) {
        logger.info("Sending image {} ({} bytes) to FastAPI ensemble endpoint", file.getName(), file.length());

        HttpHeaders fileHeaders = new HttpHeaders();
        fileHeaders.setContentType(MediaType.IMAGE_JPEG);
        HttpEntity<FileSystemResource> filePart = new HttpEntity<>(new FileSystemResource(file), fileHeaders);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", filePart);
        body.add("ensemble_strategy", "weighted_average");
        body.add("return_gradcam", "true");

        try {
            FastApiEnsembleResponse response = restClient.post()
                    .uri("/api/v1/predict/ensemble")
                    .header("X-Internal-API-Key", internalApiKey)
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(body)
                    .retrieve()
                    .body(FastApiEnsembleResponse.class);

            if (response == null || !response.isSuccess()) {
                throw new RuntimeException("FastAPI ML service returned unsuccessful response.");
            }

            logger.info("FastAPI ensemble inference succeeded. Consensus: {} (Confidence: {})",
                    response.getConsensusPrediction(), response.getConsensusConfidence());
            return response;
        } catch (Exception ex) {
            logger.error("Failed to communicate with FastAPI ML service: {}", ex.getMessage(), ex);
            throw new RuntimeException("ML inference service communication error: " + ex.getMessage(), ex);
        }
    }
}

