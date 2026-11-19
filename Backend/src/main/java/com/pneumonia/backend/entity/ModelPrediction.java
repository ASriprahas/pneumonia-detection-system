package com.pneumonia.backend.entity;

import jakarta.persistence.*;

@Entity
@Table(name = "model_predictions")
public class ModelPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "scan_record_id", nullable = false)
    private ScanRecord scanRecord;

    @Column(name = "model_name", nullable = false, length = 50)
    private String modelName;

    @Column(nullable = false, length = 50)
    private String prediction;

    @Column(nullable = false)
    private Double confidence;

    @Column(name = "normal_probability")
    private Double normalProbability;

    @Column(name = "pneumonia_probability")
    private Double pneumoniaProbability;

    @Column(name = "inference_time_ms")
    private Double inferenceTimeMs;

    public ModelPrediction() {
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public ScanRecord getScanRecord() {
        return scanRecord;
    }

    public void setScanRecord(ScanRecord scanRecord) {
        this.scanRecord = scanRecord;
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
