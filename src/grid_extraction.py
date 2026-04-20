"""
grid_extraction.py — Quad geometry, grid cell extraction, and grid drawing.

Provides functions for:
- Converting contours to ordered quadrilaterals.
- Bilinear interpolation on perspective quads.
- Extracting grid cells with configurable offsets and patterns.
- Drawing grid lines and cells on images.

Pipeline position: Step 5 (after box grouping, before fill evaluation).
"""

from __future__ import annotations

from src.log_config import logger

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================================================================
#  Quad geometry helpers
# =========================================================================


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Args:
        points: Array with 4 points.

    Returns:
        ``(4, 2)`` float32 array in canonical order.
    """
    pts = points.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] != 4:
        raise ValueError("Expected 4 points to order a quadrilateral")

    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    return ordered


def _box_to_quad(box: np.ndarray) -> np.ndarray:
    """Normalize any contour to a stable 4-point quadrilateral.

    Falls back through ``approxPolyDP`` → ``minAreaRect`` if needed.

    Args:
        box: Input contour (polygon of any vertex count).

    Returns:
        Canonical 4-point quad.
    """
    contour = box.reshape(-1, 2).astype(np.float32)
    if contour.shape[0] == 4:
        return _order_quad_points(contour)

    peri = cv2.arcLength(contour.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(contour.reshape(-1, 1, 2), 0.02 * peri, True)
    if approx.shape[0] == 4:
        return _order_quad_points(approx.reshape(-1, 2))

    rect = cv2.minAreaRect(contour.reshape(-1, 1, 2))
    quad = cv2.boxPoints(rect)
    return _order_quad_points(quad)


def _lerp_point(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two points.

    Args:
        p0: Start point.
        p1: End point.
        t: Interpolation factor in [0, 1].

    Returns:
        Interpolated point on segment p0→p1.
    """
    return p0 + (p1 - p0) * float(t)


def _point_on_quad(quad: np.ndarray, u: float, v: float) -> np.ndarray:
    """Bilinear interpolation on an ordered quad with u, v in [0, 1].

    Args:
        quad: 4-point quad in canonical order.
        u: Normalized horizontal coordinate.
        v: Normalized vertical coordinate.

    Returns:
        Pixel coordinate after mapping from normalized domain.
    """
    top = _lerp_point(quad[0], quad[1], u)
    bottom = _lerp_point(quad[3], quad[2], u)
    return _lerp_point(top, bottom, v)


def _inner_quad(
    quad: np.ndarray,
    start_offset_x: float,
    start_offset_y: float,
    end_offset_x: float,
    end_offset_y: float,
) -> np.ndarray:
    """Create an inner working-area quad from offsets applied to the outer quad.

    Args:
        quad: Outer quad.
        start_offset_x: Left inset ratio.
        start_offset_y: Top inset ratio.
        end_offset_x: Right inset ratio.
        end_offset_y: Bottom inset ratio.

    Returns:
        Inner quad after safe clamped offsets.
    """
    u0 = float(np.clip(start_offset_x, 0.0, 0.95))
    v0 = float(np.clip(start_offset_y, 0.0, 0.95))
    u1 = float(np.clip(1.0 - end_offset_x, u0 + 1e-4, 1.0))
    v1 = float(np.clip(1.0 - end_offset_y, v0 + 1e-4, 1.0))

    return np.array(
        [
            _point_on_quad(quad, u0, v0),
            _point_on_quad(quad, u1, v0),
            _point_on_quad(quad, u1, v1),
            _point_on_quad(quad, u0, v1),
        ],
        dtype=np.float32,
    )


def _quad_cell_at(
    region_quad: np.ndarray, row: int, col: int, rows: int, cols: int,
) -> np.ndarray:
    """Get the quad for a single grid cell by row/col index.

    Args:
        region_quad: Quad of the grid region.
        row: Row index.
        col: Column index.
        rows: Total rows.
        cols: Total columns.

    Returns:
        4-point quad for the requested cell.
    """
    u0 = col / float(max(1, cols))
    u1 = (col + 1) / float(max(1, cols))
    v0 = row / float(max(1, rows))
    v1 = (row + 1) / float(max(1, rows))

    return np.array(
        [
            _point_on_quad(region_quad, u0, v0),
            _point_on_quad(region_quad, u1, v0),
            _point_on_quad(region_quad, u1, v1),
            _point_on_quad(region_quad, u0, v1),
        ],
        dtype=np.float32,
    )


def _shrink_quad_towards_center(quad: np.ndarray, margin_ratio: float) -> np.ndarray:
    """Shrink a quad towards its center to reduce border noise during scoring.

    Args:
        quad: Cell quad.
        margin_ratio: Shrink ratio (0 = no shrink, 0.45 = max).

    Returns:
        Shrunken quad.
    """
    ratio = float(np.clip(margin_ratio, 0.0, 0.45))
    if ratio <= 0:
        return quad.astype(np.float32)

    center = np.mean(quad.astype(np.float32), axis=0)
    return center + (quad.astype(np.float32) - center) * (1.0 - ratio)


# =========================================================================
#  Grid drawing helpers
# =========================================================================


def _draw_grid_lines_on_quad(
    image: np.ndarray,
    quad: np.ndarray,
    grid_cols: int,
    grid_rows: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw an evenly spaced grid on a perspective quad.

    Args:
        image: Target image (drawn in-place).
        quad: Quad defining the grid region.
        grid_cols: Number of columns.
        grid_rows: Number of rows.
        color: BGR line color.
        thickness: Line thickness.
    """
    cv2.polylines(image, [quad.astype(np.int32)], True, color, thickness)

    for col in range(1, grid_cols):
        t = col / float(grid_cols)
        p_top = _lerp_point(quad[0], quad[1], t)
        p_bottom = _lerp_point(quad[3], quad[2], t)
        cv2.line(
            image,
            tuple(np.round(p_top).astype(int)),
            tuple(np.round(p_bottom).astype(int)),
            color,
            thickness,
        )

    for row in range(1, grid_rows):
        t = row / float(grid_rows)
        p_left = _lerp_point(quad[0], quad[3], t)
        p_right = _lerp_point(quad[1], quad[2], t)
        cv2.line(
            image,
            tuple(np.round(p_left).astype(int)),
            tuple(np.round(p_right).astype(int)),
            color,
            thickness,
        )


def _draw_grid_cells_with_pattern(
    image: np.ndarray,
    quad: np.ndarray,
    grid_cols: int,
    grid_rows: int,
    row_col_patterns: Optional[List[List[int]]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw grid cells following a per-row column pattern.

    Args:
        image: Target image (drawn in-place).
        quad: Grid region quad.
        grid_cols: Total columns.
        grid_rows: Total rows.
        row_col_patterns: Which columns to draw per row.
        color: BGR cell outline color.
        thickness: Line thickness.
    """
    cv2.polylines(image, [quad.astype(np.int32)], True, color, thickness)

    if grid_cols <= 0 or grid_rows <= 0:
        return

    for row in range(grid_rows):
        if row_col_patterns and row < len(row_col_patterns):
            cols_to_draw = row_col_patterns[row]
        else:
            cols_to_draw = list(range(grid_cols))

        v0 = row / float(grid_rows)
        v1 = (row + 1) / float(grid_rows)
        for col in cols_to_draw:
            if col < 0 or col >= grid_cols:
                continue
            u0 = col / float(grid_cols)
            u1 = (col + 1) / float(grid_cols)

            cell = np.array(
                [
                    _point_on_quad(quad, u0, v0),
                    _point_on_quad(quad, u1, v0),
                    _point_on_quad(quad, u1, v1),
                    _point_on_quad(quad, u0, v1),
                ],
                dtype=np.float32,
            )
            cv2.polylines(image, [cell.astype(np.int32)], True, color, thickness)


# =========================================================================
#  Grid metadata builders
# =========================================================================


def _validate_box_dims(
    box: np.ndarray, box_idx: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Validate box dimensions before grid extraction.

    Args:
        box: Contour box.
        box_idx: Box index for warning messages.

    Returns:
        ``(x, y, w, h)`` if valid, ``None`` otherwise.
    """
    x, y, w, h = cv2.boundingRect(box)
    if w <= 0 or h <= 0:
        logger.info(f"Warning: Box {box_idx} has invalid dimensions: {w}x{h}")
        return None
    return x, y, w, h


def _build_grid_info(
    box_idx: int,
    box_bounds: Tuple[int, int, int, int],
    region_quad: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    extra: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    """Build grid metadata dict for a single box.

    Args:
        box_idx: Box index.
        box_bounds: ``(x, y, w, h)``.
        region_quad: Inner working-area quad.
        grid_rows: Number of rows.
        grid_cols: Number of columns.
        extra: Additional metadata to merge.

    Returns:
        Metadata dict if region is valid, ``None`` otherwise.
    """
    _, _, region_width, region_height = cv2.boundingRect(
        region_quad.astype(np.int32)
    )
    if region_width <= 0 or region_height <= 0:
        logger.info(
            f"Warning: Box {box_idx} has invalid region dimensions: "
            f"{region_width}x{region_height}"
        )
        return None

    info: Dict[str, object] = {
        "box_idx": box_idx,
        "box_bounds": box_bounds,
        "region_quad": region_quad.tolist(),
        "region_size": (region_width, region_height),
        "cell_size": (
            region_width / max(1, grid_cols),
            region_height / max(1, grid_rows),
        ),
        "grid_shape": (grid_rows, grid_cols),
    }
    if extra:
        info.update(extra)
    return info


# =========================================================================
#  Public grid extraction functions
# =========================================================================


def extract_grid_from_boxes(
    image: np.ndarray,
    boxes: List[np.ndarray],
    grid_cols: int = 4,
    grid_rows: int = 10,
    start_offset_ratio_x: float = 0.2,
    start_offset_ratio_y: float = 0.1,
    end_offset_ratio_x: float = 0.0,
    end_offset_ratio_y: float = 0.0,
    grid_color: Tuple[int, int, int] = (0, 255, 0),
    grid_thickness: int = 1,
) -> Dict[str, object]:
    """Draw a uniform grid on each box and collect grid metadata.

    Args:
        image: Input image (a copy is drawn on).
        boxes: List of contour polygons.
        grid_cols: Columns per box.
        grid_rows: Rows per box.
        start_offset_ratio_x: Left inset ratio.
        start_offset_ratio_y: Top inset ratio.
        end_offset_ratio_x: Right inset ratio.
        end_offset_ratio_y: Bottom inset ratio.
        grid_color: BGR color.
        grid_thickness: Line thickness.

    Returns:
        ``{"image_with_grid": ..., "grid_info": [...]}``.
    """
    output_image = image.copy()
    grid_info: List[Dict[str, object]] = []

    for box_idx, box in enumerate(boxes):
        box_bounds = _validate_box_dims(box, box_idx)
        if box_bounds is None:
            continue
        x, y, w, h = box_bounds

        quad = _box_to_quad(box)
        region_quad = _inner_quad(
            quad,
            start_offset_ratio_x,
            start_offset_ratio_y,
            end_offset_ratio_x,
            end_offset_ratio_y,
        )
        _draw_grid_lines_on_quad(
            output_image, region_quad, grid_cols, grid_rows,
            grid_color, grid_thickness,
        )

        info = _build_grid_info(
            box_idx=box_idx,
            box_bounds=(x, y, w, h),
            region_quad=region_quad,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            extra={
                "start_offset": (start_offset_ratio_x, start_offset_ratio_y),
                "end_offset": (end_offset_ratio_x, end_offset_ratio_y),
            },
        )
        if info is not None:
            grid_info.append(info)

    return {"image_with_grid": output_image, "grid_info": grid_info}


def extract_grid_from_boxes_variable_offsets(
    image: np.ndarray,
    boxes: List[np.ndarray],
    grid_cols: int = 4,
    grid_rows: int = 10,
    start_offset_ratios: Optional[List[Tuple[float, float]]] = None,
    end_offset_ratios_x: Optional[List[float]] = None,
    end_offset_ratios_y: Optional[List[float]] = None,
    grid_color: Tuple[int, int, int] = (0, 255, 0),
    grid_thickness: int = 1,
) -> Dict[str, object]:
    """Draw grid with per-box offsets and collect grid metadata.

    Args:
        image: Input image (a copy is drawn on).
        boxes: List of contour polygons.
        grid_cols: Columns per box.
        grid_rows: Rows per box.
        start_offset_ratios: Per-box ``(offset_x, offset_y)`` list.
        end_offset_ratios_x: Per-box right inset ratio list.
        end_offset_ratios_y: Per-box bottom inset ratio list.
        grid_color: BGR color.
        grid_thickness: Line thickness.

    Returns:
        ``{"image_with_grid": ..., "grid_info": [...]}``.
    """
    output_image = image.copy()
    grid_info: List[Dict[str, object]] = []
    default_offset = (0.2, 0.1)

    for box_idx, box in enumerate(boxes):
        if start_offset_ratios and box_idx < len(start_offset_ratios):
            offset_x, offset_y = start_offset_ratios[box_idx]
        else:
            offset_x, offset_y = default_offset

        end_offset_x = 0.0
        if end_offset_ratios_x and box_idx < len(end_offset_ratios_x):
            end_offset_x = end_offset_ratios_x[box_idx]

        end_offset_y = 0.0
        if end_offset_ratios_y and box_idx < len(end_offset_ratios_y):
            end_offset_y = end_offset_ratios_y[box_idx]

        box_bounds = _validate_box_dims(box, box_idx)
        if box_bounds is None:
            continue
        x, y, w, h = box_bounds

        quad = _box_to_quad(box)
        region_quad = _inner_quad(quad, offset_x, offset_y, end_offset_x, end_offset_y)
        _draw_grid_lines_on_quad(
            output_image, region_quad, grid_cols, grid_rows,
            grid_color, grid_thickness,
        )

        info = _build_grid_info(
            box_idx=box_idx,
            box_bounds=(x, y, w, h),
            region_quad=region_quad,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            extra={
                "offset_ratios": (offset_x, offset_y),
                "end_offset_x": end_offset_x,
                "end_offset_y": end_offset_y,
            },
        )
        if info is not None:
            grid_info.append(info)

    return {"image_with_grid": output_image, "grid_info": grid_info}


def extract_grid_from_boxes_custom_pattern(
    image: np.ndarray,
    boxes: List[np.ndarray],
    grid_cols: int = 4,
    grid_rows: int = 12,
    start_offset_ratio_x: float = 0.2,
    start_offset_ratio_y: float = 0.1,
    end_offset_ratio_x: float = 0.0,
    end_offset_ratio_y: float = 0.0,
    grid_color: Tuple[int, int, int] = (0, 255, 0),
    grid_thickness: int = 1,
    row_col_patterns: Optional[List[List[int]]] = None,
) -> Dict[str, object]:
    """Draw grid with per-row column patterns and collect grid metadata.

    Args:
        image: Input image (a copy is drawn on).
        boxes: List of contour polygons.
        grid_cols: Total columns.
        grid_rows: Total rows.
        start_offset_ratio_x: Left inset ratio.
        start_offset_ratio_y: Top inset ratio.
        end_offset_ratio_x: Right inset ratio.
        end_offset_ratio_y: Bottom inset ratio.
        grid_color: BGR color.
        grid_thickness: Line thickness.
        row_col_patterns: Which columns to draw per row, e.g.
            ``[[0], [1, 2], [0, 1, 2, 3], ...]``.

    Returns:
        ``{"image_with_grid": ..., "grid_info": [...]}``.
    """
    output_image = image.copy()
    grid_info: List[Dict[str, object]] = []

    for box_idx, box in enumerate(boxes):
        box_bounds = _validate_box_dims(box, box_idx)
        if box_bounds is None:
            continue
        x, y, w, h = box_bounds

        quad = _box_to_quad(box)
        region_quad = _inner_quad(
            quad,
            start_offset_ratio_x,
            start_offset_ratio_y,
            end_offset_ratio_x,
            end_offset_ratio_y,
        )

        if row_col_patterns:
            _draw_grid_cells_with_pattern(
                output_image, region_quad, grid_cols, grid_rows,
                row_col_patterns, grid_color, grid_thickness,
            )
        else:
            _draw_grid_lines_on_quad(
                output_image, region_quad, grid_cols, grid_rows,
                grid_color, grid_thickness,
            )

        info = _build_grid_info(
            box_idx=box_idx,
            box_bounds=(x, y, w, h),
            region_quad=region_quad,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            extra={
                "pattern": row_col_patterns,
                "start_offset": (start_offset_ratio_x, start_offset_ratio_y),
                "end_offset": (end_offset_ratio_x, end_offset_ratio_y),
            },
        )
        if info is not None:
            grid_info.append(info)

    return {"image_with_grid": output_image, "grid_info": grid_info}
