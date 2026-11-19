import os
from pathlib import Path
import torch
import torch.nn as nn
from app.models.base import BasePneumoniaModel
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class PneumoniaNetArchitecture(nn.Module):
    """
    Custom Lightweight CNN tailored for Chest X-Ray classification.
    Designed for high sensitivity, fast CPU/GPU inference, and clear Grad-CAM feature maps.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Block 1: 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Block 4 (Target for Grad-CAM): 28x28 -> 14x14
        self.target_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block4 = nn.Sequential(
            self.target_conv,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # Global Pooling and Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class PneumoniaNetModel(BasePneumoniaModel):
    def __init__(self, device: torch.device):
        super().__init__("pneumonianet", device)
        self.load_weights()

    def load_weights(self) -> None:
        checkpoint_path = settings.WEIGHTS_DIR / "pneumonianet_v1.pth"
        self.model = PneumoniaNetArchitecture(num_classes=2)
        self.weights_source = "Kaiming Initialized (Custom Architecture)"

        if checkpoint_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.weights_source = f"Fine-tuned checkpoint ({checkpoint_path.name})"
                logger.info(f"Loaded fine-tuned PneumoniaNet weights from {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error loading PneumoniaNet checkpoint: {e}. Using initialized architecture.")
        else:
            logger.info(f"No checkpoint found at {checkpoint_path}. Running with {self.weights_source}.")

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.model.target_conv
