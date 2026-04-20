"""
preprocessing.py — Image loading, normalization, and contrast enhancement.

Pipeline position: Step 1 (entry point, before morphology).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def normalize_image_stem(image_arg: Optional[str]) -> str:
    """Normalize user-supplied image identifier to canonical ``PhieuQG.XXXX`` form.

    Accepts flexible inputs: ``0030``, ``PhieuQG.0030``,
    ``PhieuQG/PhieuQG.0030.jpg``, etc.

    Args:
        image_arg: Raw identifier string from CLI or UI.

    Returns:
        Canonical stem, e.g. ``"PhieuQG.0015"``.
    """
    if not image_arg:
        return "PhieuQG.0015"

    raw = image_arg.strip()
    if not raw:
        return "PhieuQG.0015"

    token = Path(raw).name
    token_lower = token.lower()
    if token_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        token = Path(token).stem

    if token.startswith("PhieuQG."):
        suffix = token.split(".", 1)[1]
    elif token.lower().startswith("phieuqg."):
        suffix = token.split(".", 1)[1]
    else:
        suffix = token

    if suffix.isdigit():
        suffix = suffix.zfill(4)

    return f"PhieuQG.{suffix}"


def load_image(stem: str, search_dir: Path) -> np.ndarray:
    """Try ``.jpg``, ``.jpeg``, ``.png`` in order; raise on failure.

    Args:
        stem: Canonical image stem (e.g. ``"PhieuQG.0015"``).
        search_dir: Directory containing the image files.

    Returns:
        Loaded BGR image as ``np.ndarray``.

    Raises:
        FileNotFoundError: If no matching image file is found.
    """
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".BMP"):
        candidate = search_dir / f"{stem}{ext}"
        if candidate.exists():
            img = cv2.imread(str(candidate))
            if img is not None:
                return img
    raise FileNotFoundError(
        f"No image found for stem '{stem}' in {search_dir}. "
        f"Tried: .jpg, .jpeg, .png, .bmp, .BMP"
    )


def preprocess_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to rescue faint / low-contrast scans.

    Args:
        image: Input BGR image.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
