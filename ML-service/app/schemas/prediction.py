from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class GradCamInfo(BaseModel):
    generated: bool
    heatmap_base64: Optional[str] = None
    target_layer: Optional[str] = None
    activation_intensity_score: Optional[float] = None

class PredictionResponse(BaseModel):
    success: bool
    model_name: str
    model_version: str = "1.0.0"
    prediction: str = Field(description="'NORMAL' or 'PNEUMONIA'")
    confidence: float = Field(ge=0.0, le=1.0)
    class_probabilities: Dict[str, float]
    is_high_confidence: bool
    inference_time_ms: float
    gradcam: Optional[GradCamInfo] = None

class IndividualModelResult(BaseModel):
    model_name: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float

class EnsembleResponse(BaseModel):
    success: bool
    consensus_prediction: str
    consensus_confidence: float
    ensemble_strategy: str
    individual_results: List[IndividualModelResult]
    total_inference_time_ms: float
    gradcam: Optional[GradCamInfo] = None
