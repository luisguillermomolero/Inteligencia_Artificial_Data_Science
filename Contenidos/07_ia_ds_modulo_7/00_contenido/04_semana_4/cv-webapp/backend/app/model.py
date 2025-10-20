import io
import base64
from typing import List, Dict, Any, Optional
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
            self.cls_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.using_torchscript = False

        self.cls_model.eval().to(self.device)
        # Detection model (Faster R-CNN)
        try:
            self.det_model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        except Exception:
            # fallback if weights API differs
            self.det_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.det_model.eval().to(self.device)

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

    def predict_base64(self, b64: str) -> Dict[str, Any]:
        # allow data URI prefix
        if b64.startswith('data:'):
            b64 = b64.split(',', 1)[1]
        data = base64.b64decode(b64)
        return self.predict_bytes(data)

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
        return {'class_id': int(top_idx.item()), 'score': float(top_prob.item())}

    def detect_bytes(self, data: bytes, threshold: float = 0.5) -> List[Dict[str, Any]]:
        image = self._image_from_bytes(data)
        tensor = T.ToTensor()(image).to(self.device)
        with torch.no_grad():
            preds = self.det_model([tensor])[0]
        results = []
        for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
            if float(score) >= threshold:
                results.append({'box': [float(x) for x in box.tolist()], 'label': int(label.item()), 'score': float(score.item())})
        return results
