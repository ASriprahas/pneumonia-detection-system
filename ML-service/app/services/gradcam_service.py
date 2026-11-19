import io
import base64
from typing import Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from app.core.logging import get_logger

logger = get_logger(__name__)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Generates explainability heatmaps showing the exact regions of interest (e.g. infiltrates, opacities)
    driving the model's pneumonia vs normal prediction.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Computes the Grad-CAM activation map for the given class index.
        Returns a normalized 2D numpy array [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass with gradient tracking enabled
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        
        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations for Grad-CAM.")

        # Global average pooling of gradients: shape (C,)
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations: shape (1, 1, H, W)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU to retain only positive influences
        cam = F.relu(cam)
        
        # Interpolate to input image dimensions (224, 224)
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        return heatmap


def generate_gradcam_overlay(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    original_rgb: np.ndarray,
    alpha: float = 0.4
) -> Tuple[str, float]:
    """
    Generates Grad-CAM heatmap, overlays it onto original image,
    and returns a base64 encoded data URI alongside an activation intensity score.
    """
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam.generate_heatmap(input_tensor, class_idx)
    finally:
        gradcam.remove_hooks()

    # Compute intensity score (mean activation in upper 20th percentile)
    threshold = np.percentile(heatmap, 80)
    top_activations = heatmap[heatmap >= threshold]
    intensity_score = float(np.mean(top_activations)) if len(top_activations) > 0 else float(np.mean(heatmap))

    # Convert heatmap to colored heatmap using OpenCV JET colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    overlay = np.uint8(alpha * colored_heatmap + (1.0 - alpha) * original_rgb)

    # Encode to Base64 JPEG data URL
    pil_overlay = Image.fromarray(overlay)
    buffer = io.BytesIO()
    pil_overlay.save(buffer, format="JPEG", quality=90)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_str}"

    return data_uri, round(intensity_score, 4)
