package com.pneumonia.backend.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "scan_records")
public class ScanRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "original_filename", nullable = false)
    private String originalFilename;

    @Column(name = "stored_image_path", nullable = false, length = 500)
    private String storedImagePath;

    @Column(name = "consensus_prediction", nullable = false, length = 50)
    private String consensusPrediction;

    @Column(name = "consensus_confidence", nullable = false)
    private Double consensusConfidence;

    @Column(name = "ensemble_strategy", length = 50)
    private String ensembleStrategy;

    @Column(name = "total_inference_time_ms")
    private Double totalInferenceTimeMs;

    @Lob
    @Column(name = "gradcam_heatmap", columnDefinition = "LONGTEXT")
    private String gradcamHeatmap;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @OneToMany(mappedBy = "scanRecord", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<ModelPrediction> predictions = new ArrayList<>();

    public ScanRecord() {
    }

    @PrePersist
    public void prePersist() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
        }
    }

    public void addPrediction(ModelPrediction prediction) {
        predictions.add(prediction);
        prediction.setScanRecord(this);
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
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

    public List<ModelPrediction> getPredictions() {
        return predictions;
    }

    public void setPredictions(List<ModelPrediction> predictions) {
        this.predictions = predictions;
    }
}
