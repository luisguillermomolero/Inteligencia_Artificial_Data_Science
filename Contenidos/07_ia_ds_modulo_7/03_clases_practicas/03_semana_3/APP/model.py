from __future__ import annotations
import io
import logging
from typing import List, Tuple
import json

import numpy as np
import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger("cv_app.model")

# Clase principal: Clasificador de imagenes
class ImageClassifier:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self.labels = self._load_imagenet_labels()
        self.transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
    def _load_model(self) -> torch.nn.Module:
        logger.info("Descargando modelo MobileNet...")
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        logger.info("Modelo MovilNet fue descargado y cargado correctamente")
        return model
    
    def _load_imagenet_labels(self):
        try:
            from torchvision.models import get_model_weights
            weights= get_model_weights('mobilenet_v3_small')
            labels = weights.IMAGENET1K_V1.meta['categories']
            logger.info(f"Cargadas {len(labels)} etiquetas desde weights preentrenados")
            return labels
        except Exception as e:
            logger.warning(f"No se pudieron cargar las etiquetas desde weights: {e}")
            return [f"objeto_{i}" for i in range(1000)]
    
    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(image).unsqueeze(0)
        return x.to(self.device)
    
    @torch.inference_mode()
    def predict(self, image_bytes: bytes, topk: int = 5) -> List[Tuple[str, float]]:
        x = self.preprocess(image_bytes)
        logits = self.model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)
        top_probs = top_probs.cpu().numpy().flatten().tolist()
        top_idxs = top_idxs.cpu().numpy().flatten().tolist()
        
        labels = [self.labels[idx] for idx in top_idxs]
        return list(zip(labels, top_probs))
    
    