import torch
from torchvision import models
from pathlib import Path


def export_resnet18(output_path='models/resnet18_traced.pt'):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)
    model.eval()
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)
    traced.save(output_path)
    print('Saved TorchScript model to', output_path)

if __name__ == '__main__':
    export_resnet18()
