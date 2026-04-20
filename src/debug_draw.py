"""
debug_draw.py — Visualization and debug overlay functions.

All drawing, overlay, and print-summary helpers live here. These functions
never participate in the actual grading logic — they only visualize results.

Pipeline position: Final step (after evaluation / decoding).
"""

from __future__ import annotations

from src.log_config import logger

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================================================================
#  Overlay drawing
# =========================================================================


def draw_filled_cells_overlay(
    image: np.ndarray,
    evaluations: List[Dict[str, object]],
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """Draw a transparent overlay on cells classified as filled.

    Args:
        image: Original BGR image.
        evaluations: List of per-cell eval dicts.
        color: Overlay color.
        alpha: Overlay alpha.

    Returns:
        Image with overlay applied.
    """
    if image is None or image.size == 0 or not evaluations:
        return image

    overlay = image.copy()
    for item in evaluations:
        if not bool(item.get("filled", False)):
            continue
        quad = np.round(np.array(item["cell_quad"], dtype=np.float32)).astype(np.int32)
        cv2.fillConvexPoly(overlay, quad, color)

    out = image.copy()
    cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0, out)
    return out


def draw_binary_fillratio_debug(
    binary_image: np.ndarray,
    evaluations: List[Dict[str, object]],
    out_path: str,
) -> None:
    """Save a debug image showing fill-ratio labels on the binary image.

    Args:
        binary_image: Binary image used for grading.
        evaluations: Per-cell eval results.
        out_path: Output file path.
    """
    if binary_image is None or binary_image.size == 0:
        return

    if binary_image.ndim == 2:
        canvas = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = binary_image.copy()

    for item in evaluations:
        quad = np.round(np.array(item["cell_quad"], dtype=np.float32)).astype(np.int32)
        ratio = float(item.get("fill_ratio", 0.0))
        filled = bool(item.get("filled", False))
        color = (0, 255, 0) if filled else (0, 165, 255)

        cv2.polylines(canvas, [quad], True, color, 1)

        cxy = np.mean(quad, axis=0)
        tx = int(cxy[0]) - 10
        ty = int(cxy[1]) + 4
        cv2.putText(
            canvas, f"{ratio:.2f}", (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA,
        )

    cv2.imwrite(out_path, canvas)


def draw_digit_darkness_overlay(
    image: np.ndarray,
    result: Dict[str, object],
    color: Tuple[int, int, int],
    alpha: float = 0.40,
) -> np.ndarray:
    """Draw filled-circle overlay for bubbles selected by mean-darkness decoder.

    Args:
        image: Original BGR image.
        result: Mean-darkness decode result dict.
        color: Overlay circle color.
        alpha: Overlay alpha.

    Returns:
        Image with overlay applied.
    """
    if image is None or image.size == 0:
        return image

    evaluations = result.get("evaluations", [])
    if not isinstance(evaluations, list) or not evaluations:
        return image

    overlay = image.copy()
    for item in evaluations:
        if not bool(item.get("filled", False)):
            continue
        box = item.get("box")
        if box is None:
            continue

        x, y, w, h = cv2.boundingRect(np.array(box))
        cx = x + (w // 2)
        cy = y + (h // 2)
        rr = max(2, int(round(min(w, h) * 0.36)))
        cv2.circle(overlay, (cx, cy), rr, color, -1)
        cv2.circle(overlay, (cx, cy), rr, tuple(max(0, c - 80) for c in color), 1)

    out = image.copy()
    cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0, out)
    return out


def draw_rows_contours(
    image: np.ndarray,
    rows: List[List[np.ndarray]],
    color: Tuple[int, int, int],
    thickness: int,
) -> int:
    """Draw contours for a list of box rows. Returns total boxes drawn.

    Args:
        image: Target image (drawn in-place).
        rows: List of row lists of contour polygons.
        color: BGR color.
        thickness: Line thickness.

    Returns:
        Number of boxes drawn.
    """
    drawn = 0
    for row in rows:
        for poly in row:
            cv2.polylines(image, [poly], True, color, thickness)
            drawn += 1
    return drawn


# =========================================================================
#  Print summaries
# =========================================================================


def print_fill_summary(
    title: str, evaluations: List[Dict[str, object]], limit: int = 40,
) -> None:
    """Print a short summary of filled cells for a grid section.

    Args:
        title: Section name.
        evaluations: Per-cell eval results.
        limit: Max detail lines.
    """
    total = len(evaluations)
    filled_items = [e for e in evaluations if bool(e.get("filled", False))]
    hough_items = [e for e in evaluations if str(e.get("mask_mode", "")) == "hough-circle"]
    if hough_items:
        detected = sum(1 for e in hough_items if bool(e.get("circle_detected", False)))
        logger.info(f"{title}: filled {len(filled_items)}/{total}, hough circles {detected}/{len(hough_items)}")
    else:
        logger.info(f"{title}: filled {len(filled_items)}/{total}")
    for idx, item in enumerate(filled_items):
        if idx >= limit:
            logger.info(f"  ... and {len(filled_items) - limit} more")
            break
        logger.info(
            f"  box={item['box_idx']} r={item['row']} c={item['col']} "
            f"ratio={item['fill_ratio']:.3f}"
        )


def print_digit_darkness_summary(
    title: str, result: Dict[str, object], limit: int = 20,
) -> None:
    """Print a short summary of mean-darkness digit decode results.

    Args:
        title: Section name (SoBaoDanh / MaDe).
        result: Decode result dict.
        limit: Max detail lines.
    """
    decoded = str(result.get("decoded", ""))
    evals = [e for e in result.get("evaluations", []) if bool(e.get("filled", False))]
    decisions = result.get("column_decisions", [])
    if isinstance(decisions, list):
        filled_cols = sum(1 for d in decisions if bool(d.get("filled", False)))
        logger.info(f"{title} filled columns: {filled_cols}/{len(decisions)}")
    logger.info(f"{title} decoded: {decoded}")
    for idx, item in enumerate(evals):
        if idx >= limit:
            logger.info(f"  ... and {len(evals) - limit} more")
            break
        logger.info(
            f"  col={item['col']} digit={item['row']} darkness={item['mean_darkness']:.1f}"
        )


def print_grid_info(
    grid_info: List[Dict[str, object]],
    detail_formatter: Optional[Callable[[Dict[str, object]], str]] = None,
) -> None:
    """Print grid metadata in a uniform log format.

    Args:
        grid_info: List of grid metadata dicts.
        detail_formatter: Optional extra-info formatter per box.
    """
    logger.info(f"Grid drawn on {len(grid_info)} boxes")
    for info in grid_info:
        detail = detail_formatter(info) if detail_formatter is not None else ""
        suffix = f", {detail}" if detail else ""
        logger.info(
            f"  Box {info['box_idx']}: region {info['region_size']}, "
            f"cell_size ~{info['cell_size'][0]:.1f}x{info['cell_size'][1]:.1f}{suffix}"
        )
