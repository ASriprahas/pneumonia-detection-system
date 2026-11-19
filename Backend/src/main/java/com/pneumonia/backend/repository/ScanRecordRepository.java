package com.pneumonia.backend.repository;

import com.pneumonia.backend.entity.ScanRecord;
import com.pneumonia.backend.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ScanRecordRepository extends JpaRepository<ScanRecord, Long> {

    @Query("SELECT s FROM ScanRecord s LEFT JOIN FETCH s.predictions WHERE s.user = :user ORDER BY s.createdAt DESC")
    List<ScanRecord> findByUserWithPredictionsOrderByCreatedAtDesc(@Param("user") User user);
}
