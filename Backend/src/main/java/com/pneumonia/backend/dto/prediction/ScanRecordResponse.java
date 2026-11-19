package com.pneumonia.backend.dto.prediction;

import com.pneumonia.backend.entity.ScanRecord;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class ScanRecordResponse {

    private Long id;
    private String originalFilename;
    private String storedImagePath;
    private String consensusPrediction;
    private Double consensusConfidence;
    private String ensembleStrategy;
    private Double totalInferenceTimeMs;
    private String gradcamHeatmap;
    private LocalDateTime createdAt;
    private List<ModelPredictionDto> predictions = new ArrayList<>();

    public ScanRecordResponse() {
    }

    public static ScanRecordResponse fromEntity(ScanRecord entity) {
        ScanRecordResponse dto = new ScanRecordResponse();
        dto.setId(entity.getId());
        dto.setOriginalFilename(entity.getOriginalFilename());
        dto.setStoredImagePath(entity.getStoredImagePath());
        dto.setConsensusPrediction(entity.getConsensusPrediction());
        dto.setConsensusConfidence(entity.getConsensusConfidence());
        dto.setEnsembleStrategy(entity.getEnsembleStrategy());
        dto.setTotalInferenceTimeMs(entity.getTotalInferenceTimeMs());
        dto.setGradcamHeatmap(entity.getGradcamHeatmap());
        dto.setCreatedAt(entity.getCreatedAt());

        if (entity.getPredictions() != null) {
            dto.setPredictions(
                entity.getPredictions().stream()
                    .map(ModelPredictionDto::fromEntity)
                    .collect(Collectors.toList())
            );
        }
        return dto;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getOriginalFilename() {
        return originalFilename;
    }

    public void setOriginalFilename(String originalFilename) {
        this.originalFilename = originalFilename;
    }

    public String getStoredImagePath() {
        return storedImagePath;
    }

    public void setStoredImagePath(String storedImagePath) {
        this.storedImagePath = storedImagePath;
    }

    public String getConsensusPrediction() {
        return consensusPrediction;
    }

    public void setConsensusPrediction(String consensusPrediction) {
        this.consensusPrediction = consensusPrediction;
    }

    public Double getConsensusConfidence() {
        return consensusConfidence;
    }

    public void setConsensusConfidence(Double consensusConfidence) {
        this.consensusConfidence = consensusConfidence;
    }

    public String getEnsembleStrategy() {
        return ensembleStrategy;
    }

    public void setEnsembleStrategy(String ensembleStrategy) {
        this.ensembleStrategy = ensembleStrategy;
    }

    public Double getTotalInferenceTimeMs() {
        return totalInferenceTimeMs;
    }

    public void setTotalInferenceTimeMs(Double totalInferenceTimeMs) {
        this.totalInferenceTimeMs = totalInferenceTimeMs;
    }

    public String getGradcamHeatmap() {
        return gradcamHeatmap;
    }

    public void setGradcamHeatmap(String gradcamHeatmap) {
        this.gradcamHeatmap = gradcamHeatmap;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public List<ModelPredictionDto> getPredictions() {
        return predictions;
    }

    public void setPredictions(List<ModelPredictionDto> predictions) {
        this.predictions = predictions;
    }
}
