import io
from typing import Dict, Any
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models

class VisionModel:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        # Try to load TorchScript if available
        try:
            import pathlib
            ts_path = pathlib.Path('models/resnet18_traced.pt')
            if ts_path.exists():
                self.cls_model = torch.jit.load(str(ts_path), map_location=self.device)
                self.using_torchscript = True
            else:
                raise FileNotFoundError
        except Exception:
            # Load torchvision model as fallback
            weights = models.ResNet18_Weights.DEFAULT
            self.cls_model = models.resnet18(weights=weights)
            # Save categories for human-readable names
            self.imagenet_categories = weights.meta.get('categories', [])
            self.using_torchscript = False

        self.cls_model.eval().to(self.device)
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def _image_from_bytes(self, data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert('RGB')

    def predict_bytes(self, data: bytes) -> Dict[str, Any]:
        image = self._image_from_bytes(data)
        return self.predict_image(image)

    def predict_image(self, image: Image.Image) -> Dict[str, Any]:
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cls_model(x)
            # handle TorchScript output shapes
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.dim() == 2:
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
            else:
                probs = torch.nn.functional.softmax(logits, dim=0)
            top_prob, top_idx = torch.max(probs, 0)
        class_id = int(top_idx.item())
        class_name = None
        # Provide human-readable class name if available
        if hasattr(self, 'imagenet_categories') and self.imagenet_categories:
            if 0 <= class_id < len(self.imagenet_categories):
                class_name = self.imagenet_categories[class_id]
        return {'class_id': class_id, 'class_name': class_name, 'score': float(top_prob.item())}
