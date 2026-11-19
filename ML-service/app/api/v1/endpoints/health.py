from fastapi import APIRouter, Depends
from app.api.deps import verify_internal_api_key
from app.core.config import settings
from app.services.inference_service import inference_service
from app.schemas.health import HealthResponse, ModelsInfoResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="Service & Models Health Check")
def get_health(_=Depends(verify_internal_api_key)):
    models_status = inference_service.get_models_status()
    all_ready = all(models_status.values()) if models_status else False
    
    return HealthResponse(
        status="HEALTHY" if all_ready else "INITIALIZING",
        device=str(inference_service.device),
        models_loaded=models_status,
        version="1.0.0"
    )

@router.get("/models", response_model=ModelsInfoResponse, summary="Inspect Loaded Model Architectures")
def get_models_info(_=Depends(verify_internal_api_key)):
    return ModelsInfoResponse(models=inference_service.get_models_metadata())
