import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

class Settings(BaseModel):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Pneumonia Detection ML Engine")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "pneumonia-internal-secret-key-2026")
    
    # Dataset path pointing directly to local Kaggle dataset
    DATASET_PATH: str = os.getenv("DATASET_PATH", r"C:\Kaggle\Chest_Xray\chest_xray")
    
    # Model Weights Directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    WEIGHTS_DIR: Path = Path(__file__).resolve().parent.parent / "weights"
    DOWNLOAD_IMAGENET_WEIGHTS: bool = os.getenv("DOWNLOAD_IMAGENET_WEIGHTS", "false").lower() == "true"
    
    # Hardware device
    DEVICE_SETTING: str = os.getenv("DEVICE", "auto")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Class mapping
    CLASS_NAMES: list[str] = ["NORMAL", "PNEUMONIA"]

settings = Settings()

# Ensure weights directory exists
settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
