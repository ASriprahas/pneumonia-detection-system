from pydantic import BaseModel
from typing import Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: Dict[str, bool]
    version: str

class ModelMetadata(BaseModel):
    architecture: str
    weights_path: Optional[str] = None
    weights_source: str
    target_layer_for_gradcam: str
    input_size: str

class ModelsInfoResponse(BaseModel):
    models: Dict[str, ModelMetadata]
