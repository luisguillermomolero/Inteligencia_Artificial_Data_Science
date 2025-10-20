from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger("main")

@dataclass
class ImageProcessingConfig:
    interpolation_resize: int = cv2.INTER_AREA

def load_image(path: str, as_gray: bool = False) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        logger.error("Archivo no encontrado: %s", path)
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    
    img = cv2.imread(str(p), flags=flag)
    if img is None:
        logger.error("cv2.imread devolvió None para %s", path)
        raise IOError(f"No se pudo leer la imagen: {path}")
    
    logger.info("Imagen cargada: %s -- shape=%s dtype=%s", path, img.shape, img.dtype)
    
    return img

def save_image(path: str, image: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    ok = cv2.imwrite(str(p), image)
    if not ok:
        logger.error("Error guardando la imagen en %s", path)
        raise IOError(f"Error guardando la imagen en {path}")
    
    logger.info("Imagen guardada en %s", path)

def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect: bool = True,
    config: ImageProcessingConfig = ImageProcessingConfig()
) -> np.ndarray:
    
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if keep_aspect:
        if width is None:
            ratio = height / float(h)
            dim = (int(w * ratio), height)
        elif height is None:
            ratio = width / float(w)
            dim = (width, int(h * ratio))
        else:
            dim = (width, height)
    else:
        dim = (width if width else w, height if height else h)

    resized = cv2.resize(image, dim, interpolation=config.interpolation_resize)
    
    logger.info("resize_image: %s -> %s", (w, h), resized.shape[:2])
    return resized

def rotate_image(
    image: np.ndarray, 
    angle: float, 
    center: Optional[Tuple[int, int]] = None, 
    scale: float = 1.0
) -> np.ndarray:
    
    (h, w) = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    logger.info("rotate_image: angle=%s center=%s scale=%s", angle, center, scale)
    return rotated

def crop_image(
    image: np.ndarray, 
    x: int, 
    y: int, 
    w: int, 
    h: int
) -> np.ndarray:
    return image[y:y + h, x:x + w].copy()

def histogram_equialization_gray(image_gray: np.ndarray) -> np.ndarray:
    if image_gray.ndim != 2:
        raise ValueError("La imagen debe ser en escala de grises(2D)...")
    
    eq = cv2.equalizeHist(image_gray)
    
    logger.info("histogram_equialization_gray: completado")
    return eq

def apply_clahe_gray(
    image_gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    res = clahe.apply(image_gray)
    
    logger.info("apply_clahe_gray: clip_limit=%s tile_grid_size=%s", clip_limit, tile_grid_size)
    return res

def gaussian_blur(
    image: np.ndarray, 
    ksize: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    return cv2.GaussianBlur(image, ksize, 0)

def canny_edge(
    image_gray: np.ndarray,
    threshold1: float = 100.0,
    threshold2: float = 200.0
) -> np.ndarray:
    edges = cv2.Canny(image_gray, threshold1, threshold2)
    logger.info("canny_edge: thresholds=(%s,%s)", threshold1, threshold2)
    return edges


def pipeline_demo(
    input_path: str,
    output_path: str
) -> None:
    
    img = load_image(input_path, as_gray=False)
    gray = to_gray(img)
    eq = apply_clahe_gray(gray)
    edges = canny_edge(eq, 50, 150)
    
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img, 0.6, edges_bgr, 0.4, 0)
    
    base_name = Path(output_path).stem
    save_image(f"{base_name}_original.jpg", img)
    save_image(f"{base_name}_bordes.jpg", edges)
    save_image(f"{base_name}_superpuesto.jpg", overlay)
    save_image(output_path, overlay)
    
    logger.info("pipeline_demo completado: %s", output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo de operaciones básicas con OpenCV")
    parser.add_argument("--input", required=True, help="Ruta de entrada de la imagen")
    parser.add_argument("--output", required=True, help="Ruta de salida de la imagen procesada")
    args = parser.parse_args()
    
    pipeline_demo(args.input, args.output)
    