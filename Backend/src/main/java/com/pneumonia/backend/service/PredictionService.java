package com.pneumonia.backend.service;

import com.pneumonia.backend.dto.prediction.FastApiEnsembleResponse;
import com.pneumonia.backend.dto.prediction.ScanRecordResponse;
import com.pneumonia.backend.entity.ModelPrediction;
import com.pneumonia.backend.entity.ScanRecord;
import com.pneumonia.backend.entity.User;
import com.pneumonia.backend.repository.ScanRecordRepository;
import com.pneumonia.backend.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class PredictionService {

    private static final Logger logger = LoggerFactory.getLogger(PredictionService.class);

    private final StorageService storageService;
    private final MlServiceClient mlServiceClient;
    private final ScanRecordRepository scanRecordRepository;
    private final UserRepository userRepository;

    public PredictionService(StorageService storageService,
                             MlServiceClient mlServiceClient,
                             ScanRecordRepository scanRecordRepository,
                             UserRepository userRepository) {
        this.storageService = storageService;
        this.mlServiceClient = mlServiceClient;
        this.scanRecordRepository = scanRecordRepository;
        this.userRepository = userRepository;
    }

    @Transactional
    public ScanRecordResponse analyzeScan(MultipartFile file, String username) {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded file cannot be empty.");
        }

        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new IllegalArgumentException("User not found: " + username));

        // 1. Save uploaded X-ray image locally
        String storedFilePath = storageService.storeFile(file);
        java.io.File savedImageFile = new java.io.File(storedFilePath);

        // 2. Forward to FastAPI ensemble endpoint
        FastApiEnsembleResponse mlResponse = mlServiceClient.callEnsemblePrediction(savedImageFile);

        // 3. Construct ScanRecord entity
        ScanRecord scanRecord = new ScanRecord();
        scanRecord.setUser(user);
        scanRecord.setOriginalFilename(file.getOriginalFilename() != null ? file.getOriginalFilename() : "xray.jpg");
        scanRecord.setStoredImagePath(storedFilePath);
        scanRecord.setConsensusPrediction(mlResponse.getConsensusPrediction());
        scanRecord.setConsensusConfidence(mlResponse.getConsensusConfidence());
        scanRecord.setEnsembleStrategy(mlResponse.getEnsembleStrategy());
        scanRecord.setTotalInferenceTimeMs(mlResponse.getTotalInferenceTimeMs());

        if (mlResponse.getGradcam() != null && mlResponse.getGradcam().isGenerated()) {
            scanRecord.setGradcamHeatmap(mlResponse.getGradcam().getHeatmapBase64());
        }

        // 4. Construct individual model prediction records (VGG16, ResNet50, PneumoniaNet)
        if (mlResponse.getIndividualResults() != null) {
            for (FastApiEnsembleResponse.IndividualResult ind : mlResponse.getIndividualResults()) {
                ModelPrediction pred = new ModelPrediction();
                pred.setModelName(ind.getModelName());
                pred.setPrediction(ind.getPrediction());
                pred.setConfidence(ind.getConfidence());
                pred.setInferenceTimeMs(ind.getInferenceTimeMs());

                if (ind.getProbabilities() != null) {
                    pred.setNormalProbability(ind.getProbabilities().get("NORMAL"));
                    pred.setPneumoniaProbability(ind.getProbabilities().get("PNEUMONIA"));
                }
                scanRecord.addPrediction(pred);
            }
        }

        // 5. Save to MySQL
        ScanRecord saved = scanRecordRepository.save(scanRecord);
        logger.info("Successfully persisted ScanRecord #{} with {} model predictions for user '{}'",
                saved.getId(), saved.getPredictions().size(), username);

        return ScanRecordResponse.fromEntity(saved);
    }

    @Transactional(readOnly = true)
    public List<ScanRecordResponse> getUserHistory(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new IllegalArgumentException("User not found: " + username));

        List<ScanRecord> records = scanRecordRepository.findByUserWithPredictionsOrderByCreatedAtDesc(user);
        return records.stream()
                .map(ScanRecordResponse::fromEntity)
                .collect(Collectors.toList());
    }
}
