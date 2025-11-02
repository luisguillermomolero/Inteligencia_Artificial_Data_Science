"""Lightweight configuration helpers for the API."""

from typing import List

# Frontend origins allowed by CORS
ALLOWED_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Max upload size in bytes (8MB)
MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024

# Inference device
DEVICE: str = "cpu"


