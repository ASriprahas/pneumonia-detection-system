import io
from typing import Tuple
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Standard ImageNet normalization for chest X-rays
IMAGE_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def preprocess_image_bytes(image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Parses raw image bytes, converts to RGB, resizes to 224x224,
    and returns both the normalized PyTorch tensor (1, 3, 224, 224)
    and the raw RGB numpy array (224, 224, 3) for visualization.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image bytes: {str(e)}")

    # Resize for numpy reference
    pil_resized = pil_img.resize(IMAGE_SIZE)
    rgb_numpy = np.array(pil_resized, dtype=np.uint8)

    # Tensor representation
    tensor = inference_transforms(pil_img).unsqueeze(0)
    return tensor, rgb_numpy
