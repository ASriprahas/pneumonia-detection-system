import time
from typing import Dict, Any, Optional
import torch
from app.core.config import settings
from app.core.logging import get_logger
from app.models.base import BasePneumoniaModel
from app.models.vgg16 import VGG16PneumoniaModel
from app.models.resnet50 import ResNet50PneumoniaModel
from app.models.pneumonianet import PneumoniaNetModel
from app.utils.image_processing import preprocess_image_bytes
from app.services.gradcam_service import generate_gradcam_overlay
from app.schemas.prediction import (
    PredictionResponse,
    GradCamInfo,
    EnsembleResponse,
    IndividualModelResult,
)

logger = get_logger(__name__)

class InferenceService:
    def __init__(self):
        # Hardware selection
        if settings.DEVICE_SETTING.lower() == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif settings.DEVICE_SETTING.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Initializing InferenceService on device: {self.device}")
        self.models: Dict[str, BasePneumoniaModel] = {}

    def load_models(self) -> None:
        """Load all 3 models into memory."""
        logger.info("Loading VGG16 model...")
        self.models["vgg16"] = VGG16PneumoniaModel(self.device)

        logger.info("Loading ResNet50 model...")
        self.models["resnet50"] = ResNet50PneumoniaModel(self.device)

        logger.info("Loading Custom PneumoniaNet model...")
        self.models["pneumonianet"] = PneumoniaNetModel(self.device)

        logger.info("All 3 models initialized and ready for inference.")

    def get_models_status(self) -> Dict[str, bool]:
        if not self.models:
            self.load_models()
        return {name: model.is_loaded for name, model in self.models.items()}

    def get_models_metadata(self) -> Dict[str, Any]:
        if not self.models:
            self.load_models()
        meta = {}
        for name, m in self.models.items():
            meta[name] = {
                "architecture": m.__class__.__name__,
                "weights_path": str(settings.WEIGHTS_DIR / f"{name}_chest_xray.pth"),
                "weights_source": m.weights_source,
                "target_layer_for_gradcam": str(m.get_gradcam_target_layer().__class__.__name__),
                "input_size": "224x224x3",
            }
        return meta

    def predict(
        self,
        image_bytes: bytes,
        model_name: str,
        return_gradcam: bool = True
    ) -> PredictionResponse:
        if not self.models:
            self.load_models()
        model_key = model_name.lower().strip()
        if model_key not in self.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(self.models.keys())}")

        selected_model = self.models[model_key]
        
        start_time = time.perf_counter()
        tensor, rgb_numpy = preprocess_image_bytes(image_bytes)
        
        # Inference
        pred_idx, confidence, probabilities = selected_model.predict_tensor(tensor)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        prediction_label = settings.CLASS_NAMES[pred_idx]
        is_high_conf = confidence >= 0.85

        # Grad-CAM generation
        gradcam_info = None
        if return_gradcam:
            try:
                target_layer = selected_model.get_gradcam_target_layer()
                gradcam_tensor = tensor.clone().to(self.device)
                heatmap_base64, intensity = generate_gradcam_overlay(
                    model=selected_model.model,
                    target_layer=target_layer,
                    input_tensor=gradcam_tensor,
                    class_idx=pred_idx,
                    original_rgb=rgb_numpy
                )
                gradcam_info = GradCamInfo(
                    generated=True,
                    heatmap_base64=heatmap_base64,
                    target_layer=target_layer.__class__.__name__,
                    activation_intensity_score=intensity
                )
            except Exception as e:
                logger.error(f"Grad-CAM generation failed for {model_key}: {e}")
                gradcam_info = GradCamInfo(generated=False)

        return PredictionResponse(
            success=True,
            model_name=model_key,
            model_version="1.0.0",
            prediction=prediction_label,
            confidence=round(confidence, 4),
            class_probabilities={k: round(v, 4) for k, v in probabilities.items()},
            is_high_confidence=is_high_conf,
            inference_time_ms=inference_time_ms,
            gradcam=gradcam_info
        )

    def predict_ensemble(
        self,
        image_bytes: bytes,
        strategy: str = "weighted_average",
        return_gradcam: bool = True
    ) -> EnsembleResponse:
        if not self.models:
            self.load_models()
        start_time = time.perf_counter()
        tensor, rgb_numpy = preprocess_image_bytes(image_bytes)

        # Model weights in ensemble (ResNet50: 0.40, VGG16: 0.35, PneumoniaNet: 0.25)
        weights = {"resnet50": 0.40, "vgg16": 0.35, "pneumonianet": 0.25}
        
        individual_results = []
        weighted_probs = {c: 0.0 for c in settings.CLASS_NAMES}

        for model_key in ["resnet50", "vgg16", "pneumonianet"]:
            m = self.models[model_key]
            m_start = time.perf_counter()
            pred_idx, conf, probs = m.predict_tensor(tensor)
            m_time = round((time.perf_counter() - m_start) * 1000, 2)
            
            individual_results.append(
                IndividualModelResult(
                    model_name=model_key,
                    prediction=settings.CLASS_NAMES[pred_idx],
                    confidence=round(conf, 4),
                    probabilities={k: round(v, 4) for k, v in probs.items()},
                    inference_time_ms=m_time
                )
            )
            
            w = weights.get(model_key, 0.33)
            for c in settings.CLASS_NAMES:
                weighted_probs[c] += probs[c] * w

        # Consensus determination
        consensus_pred = max(weighted_probs, key=weighted_probs.get)
        consensus_conf = weighted_probs[consensus_pred]

        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Grad-CAM using primary benchmark model (ResNet50)
        gradcam_info = None
        if return_gradcam:
            try:
                ref_model = self.models["resnet50"]
                pred_idx = settings.CLASS_NAMES.index(consensus_pred)
                gradcam_tensor = tensor.clone().to(self.device)
                heatmap_base64, intensity = generate_gradcam_overlay(
                    model=ref_model.model,
                    target_layer=ref_model.get_gradcam_target_layer(),
                    input_tensor=gradcam_tensor,
                    class_idx=pred_idx,
                    original_rgb=rgb_numpy
                )
                gradcam_info = GradCamInfo(
                    generated=True,
                    heatmap_base64=heatmap_base64,
                    target_layer="ResNet50.layer4.conv3",
                    activation_intensity_score=intensity
                )
            except Exception as e:
                logger.error(f"Ensemble Grad-CAM generation failed: {e}")
                gradcam_info = GradCamInfo(generated=False)

        return EnsembleResponse(
            success=True,
            consensus_prediction=consensus_pred,
            consensus_confidence=round(consensus_conf, 4),
            ensemble_strategy=strategy,
            individual_results=individual_results,
            total_inference_time_ms=total_time_ms,
            gradcam=gradcam_info
        )

# Global singleton instance
inference_service = InferenceService()
