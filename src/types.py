"""
types.py — Shared type definitions for the XLA_DeThiTHPT pipeline.

Provides type aliases and TypedDict definitions used across all processing
modules to ensure consistent data contracts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Contour = np.ndarray
"""A single OpenCV contour, shape (N, 1, 2), dtype int32."""

BoundingRect = Tuple[int, int, int, int]
"""OpenCV bounding rectangle: ``(x, y, w, h)``."""

Point2D = Tuple[float, float]
"""A 2-D point as ``(x, y)``."""

# ---------------------------------------------------------------------------
# NOTE: We use plain dictionaries (as the original code does) rather than
# TypedDict for now, so that all existing dict-returning functions remain
# compatible without any change.  The type aliases below document the
# *expected* keys for each dictionary kind and serve as reference contracts.
# A future PR can migrate to strict TypedDict enforcement.
# ---------------------------------------------------------------------------

# BoxInfo keys: "box", "x", "y", "w", "h", "center_y", "area"
BoxInfoKeys = Dict[str, object]

# GridInfo keys: "box_idx", "box_bounds", "region_quad", "region_size",
#   "grid_rows", "grid_cols", "cell_size", and optionally extra keys
GridInfoKeys = Dict[str, object]

# EvalResult keys: "box_idx", "row", "col", "fill_ratio", "filled",
#   "center", "radius"
EvalResultKeys = Dict[str, object]

# DigitResult keys: "decoded", "column_decisions"
DigitResultKeys = Dict[str, object]

# GridResult keys: "image_with_grid", "grid_info"
GridResultKeys = Dict[str, object]
