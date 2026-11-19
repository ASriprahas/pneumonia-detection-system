package com.pneumonia.backend.repository;

import com.pneumonia.backend.entity.ModelPrediction;
import com.pneumonia.backend.entity.ScanRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ModelPredictionRepository extends JpaRepository<ModelPrediction, Long> {
    List<ModelPrediction> findByScanRecord(ScanRecord scanRecord);
}
