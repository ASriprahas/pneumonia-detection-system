package com.pneumonia.backend.dto.prediction;

import com.pneumonia.backend.entity.ModelPrediction;

public class ModelPredictionDto {

    private Long id;
    private String modelName;
    private String prediction;
    private Double confidence;
    private Double normalProbability;
    private Double pneumoniaProbability;
    private Double inferenceTimeMs;

    public ModelPredictionDto() {
    }

    public static ModelPredictionDto fromEntity(ModelPrediction entity) {
        ModelPredictionDto dto = new ModelPredictionDto();
        dto.setId(entity.getId());
        dto.setModelName(entity.getModelName());
        dto.setPrediction(entity.getPrediction());
        dto.setConfidence(entity.getConfidence());
        dto.setNormalProbability(entity.getNormalProbability());
        dto.setPneumoniaProbability(entity.getPneumoniaProbability());
        dto.setInferenceTimeMs(entity.getInferenceTimeMs());
        return dto;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public Double getNormalProbability() {
        return normalProbability;
    }

    public void setNormalProbability(Double normalProbability) {
        this.normalProbability = normalProbability;
    }

    public Double getPneumoniaProbability() {
        return pneumoniaProbability;
    }

    public void setPneumoniaProbability(Double pneumoniaProbability) {
        this.pneumoniaProbability = pneumoniaProbability;
    }

    public Double getInferenceTimeMs() {
        return inferenceTimeMs;
    }

    public void setInferenceTimeMs(Double inferenceTimeMs) {
        this.inferenceTimeMs = inferenceTimeMs;
    }
}
