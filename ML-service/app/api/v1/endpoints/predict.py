from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from app.api.deps import verify_internal_api_key
from app.services.inference_service import inference_service
from app.schemas.prediction import PredictionResponse, EnsembleResponse
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "application/octet-stream"}

def validate_image_file(file: UploadFile):
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image type '{file.content_type}'. Supported: JPEG, PNG."
        )

@router.post("/predict", response_model=PredictionResponse, summary="Predict Chest X-Ray with Selected Model")
async def predict_single(
    file: UploadFile = File(..., description="Chest X-Ray image file (JPEG/PNG)"),
    model_name: str = Form("resnet50", description="Model architecture: 'vgg16', 'resnet50', or 'pneumonianet'"),
    return_gradcam: bool = Form(True, description="Whether to compute and return the Grad-CAM visualization overlay"),
    _=Depends(verify_internal_api_key)
):
    validate_image_file(file)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        response = inference_service.predict(
            image_bytes=contents,
            model_name=model_name,
            return_gradcam=return_gradcam
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(e)}")

@router.post("/predict/ensemble", response_model=EnsembleResponse, summary="Predict with Multi-Model Ensemble")
async def predict_ensemble(
    file: UploadFile = File(..., description="Chest X-Ray image file (JPEG/PNG)"),
    ensemble_strategy: str = Form("weighted_average", description="Aggregation strategy: 'weighted_average' or 'majority_vote'"),
    return_gradcam: bool = Form(True, description="Whether to compute Grad-CAM for consensus"),
    _=Depends(verify_internal_api_key)
):
    validate_image_file(file)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        response = inference_service.predict_ensemble(
            image_bytes=contents,
            strategy=ensemble_strategy,
            return_gradcam=return_gradcam
        )
        return response
    except Exception as e:
        logger.error(f"Ensemble inference error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ensemble inference failed: {str(e)}")
