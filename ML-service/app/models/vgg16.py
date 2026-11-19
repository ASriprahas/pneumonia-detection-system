import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights
from app.models.base import BasePneumoniaModel
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class VGG16PneumoniaModel(BasePneumoniaModel):
    def __init__(self, device: torch.device):
        super().__init__("vgg16", device)
        self.load_weights()

    def load_weights(self) -> None:
        checkpoint_path = settings.WEIGHTS_DIR / "vgg16_chest_xray.pth"
        
        # Load pre-trained weights if requested and available, else initialize cleanly
        if settings.DOWNLOAD_IMAGENET_WEIGHTS:
            try:
                self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
                self.weights_source = "ImageNet (pre-trained baseline)"
            except Exception as e:
                logger.warning(f"Could not load ImageNet weights for VGG16: {e}. Using unweighted architecture.")
                self.model = models.vgg16(weights=None)
                self.weights_source = "Initialized Architecture"
        else:
            self.model = models.vgg16(weights=None)
            self.weights_source = "Initialized Architecture (Fast Boot)"

        # Adapt classifier head for 2 classes: NORMAL (0) and PNEUMONIA (1)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )

        # Check if fine-tuned checkpoint exists
        if checkpoint_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.weights_source = f"Fine-tuned checkpoint ({checkpoint_path.name})"
                logger.info(f"Loaded fine-tuned VGG16 weights from {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error loading VGG16 checkpoint: {e}. Using base weights.")
        else:
            logger.info(f"No checkpoint found at {checkpoint_path}. Running with {self.weights_source}.")

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def get_gradcam_target_layer(self) -> nn.Module:
        # Find the last Conv2d layer in features
        for layer in reversed(self.model.features):
            if isinstance(layer, nn.Conv2d):
                return layer
        return self.model.features[-1]
