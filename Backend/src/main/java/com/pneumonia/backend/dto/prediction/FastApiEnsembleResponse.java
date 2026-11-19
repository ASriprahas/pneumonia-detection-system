package com.pneumonia.backend.dto.prediction;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

public class FastApiEnsembleResponse {

    private boolean success;

    @JsonProperty("consensus_prediction")
    private String consensusPrediction;

    @JsonProperty("consensus_confidence")
    private Double consensusConfidence;

    @JsonProperty("ensemble_strategy")
    private String ensembleStrategy;

    @JsonProperty("total_inference_time_ms")
    private Double totalInferenceTimeMs;

    @JsonProperty("individual_results")
    private List<IndividualResult> individualResults;

    private GradCamInfo gradcam;

    public FastApiEnsembleResponse() {
    }

    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
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

    public List<IndividualResult> getIndividualResults() {
        return individualResults;
    }

    public void setIndividualResults(List<IndividualResult> individualResults) {
        this.individualResults = individualResults;
    }

    public GradCamInfo getGradcam() {
        return gradcam;
    }

    public void setGradcam(GradCamInfo gradcam) {
        this.gradcam = gradcam;
    }

    public static class IndividualResult {
        @JsonProperty("model_name")
        private String modelName;

        private String prediction;
        private Double confidence;
        private Map<String, Double> probabilities;

        @JsonProperty("inference_time_ms")
        private Double inferenceTimeMs;

        public IndividualResult() {
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

        public Map<String, Double> getProbabilities() {
            return probabilities;
        }

        public void setProbabilities(Map<String, Double> probabilities) {
            this.probabilities = probabilities;
        }

        public Double getInferenceTimeMs() {
            return inferenceTimeMs;
        }

        public void setInferenceTimeMs(Double inferenceTimeMs) {
            this.inferenceTimeMs = inferenceTimeMs;
        }
    }

    public static class GradCamInfo {
        private boolean generated;

        @JsonProperty("heatmap_base64")
        private String heatmapBase64;

        @JsonProperty("activation_intensity_score")
        private Double activationIntensityScore;

        public GradCamInfo() {
        }

        public boolean isGenerated() {
            return generated;
        }

        public void setGenerated(boolean generated) {
            this.generated = generated;
        }

        public String getHeatmapBase64() {
            return heatmapBase64;
        }

        public void setHeatmapBase64(String heatmapBase64) {
            this.heatmapBase64 = heatmapBase64;
        }

        public Double getActivationIntensityScore() {
            return activationIntensityScore;
        }

        public void setActivationIntensityScore(Double activationIntensityScore) {
            this.activationIntensityScore = activationIntensityScore;
        }
    }
}
