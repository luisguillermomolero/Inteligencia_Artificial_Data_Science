# Librerias de uso general del script
import gc
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Librerias de uso específico del script
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision 
from torchvision import transforms, datasets, models

# Configurar el logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('train_cnn')

@dataclass
class Config:
    # Creación de los parametros globales para el entrenamiento
    data_dir: str = 'data'
    model_name: str = 'simple_cnn' # ó resnet18
    pretrained: bool = False
    num_classes: int = 10
    epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir: str = 'checkpoint'

# Fijar semillas

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Función para liberar memoria
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Función para cargar los datos
# Descargar CIFAR-10 y preparar los DataLoaders de entrenamiento y validación
def get_dataloader(cfg: Config):
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_set = datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_set = datasets.CIFAR10(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader

# Construcción del Modelo (CNN)
def build_model(cfg: Config):
    
    if cfg.model_name == 'resnet18':
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if cfg.pretrained else None
        )
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cfg.num_classes)
        
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    
    elif cfg.model_name == 'simple_cnn':
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, cfg.num_classes),
        )
    
    else:
        raise ValueError(f'Modelo no soportado: {cfg.model_name}')
    
    return model

# Entrenamiento de una época
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        try:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning('Error de memoria, limpiando el caché...')
                clear_memory()
                continue
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 100 == 0:
            acc = correct / total if total > 0 else 0
            logger.info(f'Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f} - Acc: {acc:.4f}')
    
    return running_loss / total, correct / total

# Validar el modelo
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss =  criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total

# Guardado de checkpoint
def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info(f'Checkpoint guardado: {path}')

def main():
    start_time = time.time()
    cfg = Config()
    
    logger.info(f'Dispositivo {cfg.device}')
    logger.info(f'Configuración: batch_size={cfg.batch_size}, epochs={cfg.epochs}, model={cfg.model_name}')
    
    if cfg.model_name == 'simple_cnn':
        logger.info('CNN Simple - Configuración optimizada para velocidad')
        logger.info('Entrenamiento rápido para demostración')
        
    set_seed()
    
    try:
        train_loader, val_loader = get_dataloader(cfg)
        model = build_model(cfg).to(cfg.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.time()
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
            val_loss, val_acc = validate(model, val_loader, criterion, cfg.device)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f'Época {epoch}/{cfg.epochs} ({epoch_time:.1f}s): '
                        f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
                        f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

            scheduler.step(val_loss)
            
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, f'{cfg.output_dir}/model_epoch{epoch}.pt')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 3:
                    logger.info(f'Early stopping in epoch {epoch}')
                    break
            
            clear_memory()
    
    except Exception as e:
        logger.error(f'Error durante el entrenamiento: {e}')
        raise
    
    finally:
        clear_memory()
        total_time = time.time() -start_time
        logger.info(f'Entrenamiento completado en {total_time:.1f} segundos')
    
if __name__ == '__main__':
    main()
    

            