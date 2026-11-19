"""
Integration and Sanity Test for ML Service
Verifies:
1. /api/v1/health returns HEALTHY and reports models loaded
2. /api/v1/models lists vgg16, resnet50, pneumonianet
3. /api/v1/predict returns prediction + valid Grad-CAM data URI
4. /api/v1/predict/ensemble returns consensus prediction
"""

import io
from PIL import Image
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from app.core.config import settings

client = TestClient(app)

def create_synthetic_chest_xray() -> bytes:
    """Generate a synthetic 224x224 grayscale image simulating a chest X-ray."""
    arr = np.random.randint(50, 200, size=(224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_endpoint():
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["HEALTHY", "INITIALIZING"]
    assert "vgg16" in data["models_loaded"]
    assert "resnet50" in data["models_loaded"]
    assert "pneumonianet" in data["models_loaded"]


def test_models_endpoint():
    response = client.get(f"{settings.API_V1_STR}/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "vgg16" in data["models"]
    assert "resnet50" in data["models"]
    assert "pneumonianet" in data["models"]


def test_predict_single_resnet50():
    img_bytes = create_synthetic_chest_xray()
    files = {"file": ("test_xray.jpg", img_bytes, "image/jpeg")}
    data = {"model_name": "resnet50", "return_gradcam": "true"}
    
    response = client.post(f"{settings.API_V1_STR}/predict", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["model_name"] == "resnet50"
    assert res["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "NORMAL" in res["class_probabilities"]
    assert "PNEUMONIA" in res["class_probabilities"]
    assert res["gradcam"]["generated"] is True
    assert res["gradcam"]["heatmap_base64"].startswith("data:image/jpeg;base64,")


def test_predict_single_pneumonianet():
    img_bytes = create_synthetic_chest_xray()
    files = {"file": ("test_xray.jpg", img_bytes, "image/jpeg")}
    data = {"model_name": "pneumonianet", "return_gradcam": "true"}
    
    response = client.post(f"{settings.API_V1_STR}/predict", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["model_name"] == "pneumonianet"
    assert res["prediction"] in ["NORMAL", "PNEUMONIA"]


def test_predict_ensemble():
    img_bytes = create_synthetic_chest_xray()
    files = {"file": ("test_xray.jpg", img_bytes, "image/jpeg")}
    data = {"ensemble_strategy": "weighted_average", "return_gradcam": "true"}
    
    response = client.post(f"{settings.API_V1_STR}/predict/ensemble", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["consensus_prediction"] in ["NORMAL", "PNEUMONIA"]
    assert len(res["individual_results"]) == 3
