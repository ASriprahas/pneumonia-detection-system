from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from app.core.config import settings

class BasePneumoniaModel(ABC):
    """Abstract base class for all pneumonia detection models."""

    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model: nn.Module = None
        self.weights_source: str = "uninitialized"
        self.is_loaded: bool = False

    @abstractmethod
    def load_weights(self) -> None:
        """Load pre-trained weights or custom checkpoint."""
        pass

    @abstractmethod
    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the target convolutional layer for Grad-CAM."""
        pass

    def predict_tensor(self, x: torch.Tensor) -> Tuple[int, float, Dict[str, float]]:
        """Run forward inference and return predicted class, confidence, and class probabilities."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            
            prob_dict = {
                settings.CLASS_NAMES[i]: float(probabilities[i].item())
                for i in range(len(settings.CLASS_NAMES))
            }
            
            predicted_idx = int(torch.argmax(probabilities).item())
            confidence = prob_dict[settings.CLASS_NAMES[predicted_idx]]
            
            return predicted_idx, confidence, prob_dict
