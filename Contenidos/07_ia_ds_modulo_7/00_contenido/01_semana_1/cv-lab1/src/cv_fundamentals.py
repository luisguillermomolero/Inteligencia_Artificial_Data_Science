"""cv_fundamentals.py

Utilities for basic image operations using OpenCV and numpy.
The functions include: load_image, save_image, resize_image, rotate_image, crop_image and a helper to show images in notebooks.
The code follows PEP 8, uses typing and basic logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configure a module-level logger.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("cv_fundamentals")

@dataclass
class ImageProcessingConfig:
    """Configuration container for image processing parameters."""
    interpolation_resize: int = cv2.INTER_AREA

def load_image(path: str, as_gray: bool = False) -> np.ndarray:
    """Load an image from disk.

    Args:
        path: Path to the image file.
        as_gray: If True, load as grayscale.

    Returns:
        Image as a numpy ndarray.
    """
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(path)

    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(p), flags=flag)
    if img is None:
        logger.error("cv2.imread returned None for %s", path)
        raise IOError(f"Could not read image: {path}")
    logger.info("Loaded image %s shape=%s dtype=%s", path, img.shape, img.dtype)
    return img

def save_image(path: str, image: np.ndarray) -> None:
    """Save an image to disk, creating directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), image)
    if not ok:
        logger.error("Error saving image to %s", path)
        raise IOError(f"Error saving image to {path}")
    logger.info("Saved image to %s", path)

def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image (OpenCV) to RGB for plotting with matplotlib."""
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale."""
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None,
                 keep_aspect: bool = True, config: ImageProcessingConfig = ImageProcessingConfig()) -> np.ndarray:
    """Resize an image preserving aspect ratio by default.

    Args:
        image: Input image.
        width: Desired width in pixels.
        height: Desired height in pixels.
        keep_aspect: If True maintain aspect ratio.
        config: Interpolation configuration.
    """
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
    logger.info("Resized image %s -> %s", (w, h), resized.shape[:2])
    return resized

def rotate_image(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None,
                 scale: float = 1.0) -> np.ndarray:
    """Rotate image by angle (degrees, positive = counter-clockwise)."""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    logger.info("Rotated image angle=%s center=%s", angle, center)
    return rotated

def crop_image(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop region (x, y, w, h) and return a copy."""
    return image[y:y + h, x:x + w].copy()

def show_image_matplotlib(image: np.ndarray, title: str = 'Image') -> None:
    """Display image in notebooks converting BGR->RGB when needed."""
    img = to_rgb(image)
    plt.figure(figsize=(8, 6))
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
