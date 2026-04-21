"""
fill_evaluation.py — Bubble fill-ratio measurement and grading.

Provides functions for computing fill-ratios within quad or circle masks,
Hough-based circle detection for robust bubble localization, and the
main ``evaluate_grid_fill_from_binary`` evaluator.

Pipeline position: Step 6 (after grid extraction, produces EvalResult dicts).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.grid_extraction import (
    _point_on_quad,
    _quad_cell_at,
    _shrink_quad_towards_center,
)


# =========================================================================
#  Low-level fill-ratio helpers
# =========================================================================


def _fill_ratio_in_quad(binary_image: np.ndarray, quad: np.ndarray) -> float:
    """White-pixel ratio inside a quadrilateral mask.

    Args:
        binary_image: Inverted binary image.
        quad: Quad region to measure.

    Returns:
        Fill ratio in [0, 1].
    """
    if binary_image is None or binary_image.size == 0:
        return 0.0

    h, w = binary_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(quad).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillConvexPoly(mask, pts, 255)

    pixels = binary_image[mask > 0]
    if pixels.size == 0:
        return 0.0
    return float(np.count_nonzero(pixels)) / float(pixels.size)


def _estimate_circle_from_quad(
    quad: np.ndarray, radius_scale: float = 0.46,
) -> Tuple[np.ndarray, float]:
    """Estimate bubble center and radius from a cell quad.

    Args:
        quad: Cell quad.
        radius_scale: Scale factor for the radius.

    Returns:
        ``(center, radius)`` tuple.
    """
    q = quad.astype(np.float32)
    center = np.mean(q, axis=0)

    top_w = float(np.linalg.norm(q[1] - q[0]))
    bottom_w = float(np.linalg.norm(q[2] - q[3]))
    left_h = float(np.linalg.norm(q[3] - q[0]))
    right_h = float(np.linalg.norm(q[2] - q[1]))

    avg_w = 0.5 * (top_w + bottom_w)
    avg_h = 0.5 * (left_h + right_h)
    base_radius = 0.5 * min(avg_w, avg_h)
    radius = max(1.0, base_radius * float(np.clip(radius_scale, 0.20, 0.60)))
    return center, radius


def _circle_polygon(
    center: np.ndarray, radius: float, n_pts: int = 28,
) -> np.ndarray:
    """Approximate a circle as a polygon for masking / overlay.

    Args:
        center: Circle center.
        radius: Circle radius.
        n_pts: Number of polygon vertices.

    Returns:
        Float32 polygon array.
    """
    cx, cy = float(center[0]), float(center[1])
    rr = float(max(1.0, radius))
    angles = np.linspace(0.0, 2.0 * np.pi, num=max(8, int(n_pts)), endpoint=False)
    pts = np.stack([cx + rr * np.cos(angles), cy + rr * np.sin(angles)], axis=1)
    return pts.astype(np.float32)


def _detect_single_circle_hough_in_quad(
    binary_image: np.ndarray,
    quad: np.ndarray,
    radius_scale: float = 0.46,
) -> Tuple[np.ndarray, float, bool]:
    """Detect a single bubble circle in a cell using Hough + NMS + weighted merging.

    Args:
        binary_image: Inverted binary image.
        quad: Cell quad.
        radius_scale: Initial radius estimate scale.

    Returns:
        ``(center, radius, found)`` — ``found`` is True when Hough succeeds.
    """
    est_center, est_radius = _estimate_circle_from_quad(quad, radius_scale=radius_scale)
    if binary_image is None or binary_image.size == 0:
        return est_center, est_radius, False

    h, w = binary_image.shape[:2]
    pts = np.round(quad).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    x, y, bw, bh = cv2.boundingRect(pts)
    if bw < 8 or bh < 8:
        return est_center, est_radius, False

    roi = binary_image[y : y + bh, x : x + bw]
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    local_quad = pts.copy()
    local_quad[:, 0] -= x
    local_quad[:, 1] -= y

    roi_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, local_quad, 255)

    masked = cv2.bitwise_and(roi, roi, mask=roi_mask)
    prep = cv2.GaussianBlur(masked, (5, 5), 0)
    edges = cv2.Canny(prep, 40, 120)

    min_r = max(2, int(round(est_radius * 0.80)))
    max_r = max(min_r + 1, int(round(est_radius * 1.10)))
    min_dist = max(4, int(round(est_radius * 2.2)))

    def _ring_edge_support(cx_local: float, cy_local: float, r_local: float) -> float:
        ring = np.zeros((bh, bw), dtype=np.uint8)
        cxy = (int(round(cx_local)), int(round(cy_local)))
        rr = max(1, int(round(r_local)))
        cv2.circle(ring, cxy, rr, 255, 2)
        ring = cv2.bitwise_and(ring, roi_mask)
        denom = int(np.count_nonzero(ring))
        if denom <= 0:
            return 0.0
        num = int(np.count_nonzero(cv2.bitwise_and(edges, ring)))
        return float(num) / float(denom)

    def _circle_iou(ca: Dict[str, float], cb: Dict[str, float]) -> float:
        r1 = float(max(1.0, ca["r"]))
        r2 = float(max(1.0, cb["r"]))
        dx = float(ca["cx"] - cb["cx"])
        dy = float(ca["cy"] - cb["cy"])
        d = float(np.hypot(dx, dy))

        if d >= (r1 + r2):
            inter = 0.0
        elif d <= abs(r1 - r2):
            inter = float(np.pi * min(r1, r2) ** 2)
        else:
            a1 = float(np.arccos(np.clip((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0)))
            a2 = float(np.arccos(np.clip((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0)))
            term = float((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
            inter = r1 * r1 * a1 + r2 * r2 * a2 - 0.5 * np.sqrt(max(0.0, term))

        area1 = float(np.pi * r1 * r1)
        area2 = float(np.pi * r2 * r2)
        union = area1 + area2 - inter
        if union <= 1e-6:
            return 0.0
        return float(inter / union)

    def _is_same_circle(ca: Dict[str, float], cb: Dict[str, float]) -> bool:
        center_dist = float(np.hypot(ca["cx"] - cb["cx"], ca["cy"] - cb["cy"]))
        radius_ref = float(max(1.0, max(ca["r"], cb["r"], est_radius)))
        radius_diff = abs(float(ca["r"] - cb["r"])) / radius_ref
        if center_dist <= 0.48 * radius_ref and radius_diff <= 0.34:
            return True
        return _circle_iou(ca, cb) >= 0.35

    candidates: List[Dict[str, float]] = []
    # Restore original combinations for maximum robustness
    for param2 in (14, 11, 8):
        circles = cv2.HoughCircles(
            prep, cv2.HOUGH_GRADIENT, dp=1.1, minDist=float(min_dist),
            param1=80, param2=param2, minRadius=min_r, maxRadius=max_r,
        )
        if circles is None or circles.size == 0:
            continue

        for c in circles[0][:8]:
            cx_local = float(c[0])
            cy_local = float(c[1])
            rr = float(max(1.0, c[2]))

            cx = cx_local + float(x)
            cy = cy_local + float(y)
            center_norm = float(np.hypot(cx - est_center[0], cy - est_center[1])) / float(max(1.0, est_radius))
            radius_norm = abs(rr - float(est_radius)) / float(max(1.0, est_radius))
            prior_score = max(0.0, 1.0 - (0.70 * center_norm) - (0.60 * radius_norm))
            edge_score = _ring_edge_support(cx_local, cy_local, rr)
            score = (0.58 * edge_score) + (0.42 * prior_score)

            candidates.append({"cx": cx, "cy": cy, "r": rr, "score": float(score)})

    if not candidates:
        return est_center, est_radius, False

    candidates = sorted(candidates, key=lambda it: float(it["score"]), reverse=True)

    kept: List[Dict[str, float]] = []
    for cand in candidates:
        if any(_is_same_circle(cand, k) for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= 8:
            break

    if not kept:
        return est_center, est_radius, False

    clusters: List[List[Dict[str, float]]] = [[] for _ in kept]
    for cand in candidates:
        best_idx = -1
        best_dist = float("inf")
        for idx_keep, keep in enumerate(kept):
            if not _is_same_circle(cand, keep):
                continue
            dist = float(np.hypot(cand["cx"] - keep["cx"], cand["cy"] - keep["cy"]))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx_keep
        if best_idx >= 0:
            clusters[best_idx].append(cand)

    merged: List[Dict[str, float]] = []
    for idx_keep, keep in enumerate(kept):
        cluster = clusters[idx_keep] if clusters[idx_keep] else [keep]
        ws = np.array([max(1e-3, float(c["score"])) for c in cluster], dtype=np.float32)
        cx_vals = np.array([float(c["cx"]) for c in cluster], dtype=np.float32)
        cy_vals = np.array([float(c["cy"]) for c in cluster], dtype=np.float32)
        r_vals = np.array([float(c["r"]) for c in cluster], dtype=np.float32)

        cxm = float(np.sum(ws * cx_vals) / np.sum(ws))
        cym = float(np.sum(ws * cy_vals) / np.sum(ws))
        rm = float(np.sum(ws * r_vals) / np.sum(ws))
        support_bonus = 0.08 * float(len(cluster) - 1)
        base_score = float(np.max([float(c["score"]) for c in cluster])) + support_bonus

        center_penalty = float(np.hypot(cxm - est_center[0], cym - est_center[1])) / float(max(1.0, est_radius))
        radius_penalty = abs(rm - float(est_radius)) / float(max(1.0, est_radius))
        final_score = base_score - (0.25 * center_penalty) - (0.20 * radius_penalty)

        merged.append({"cx": cxm, "cy": cym, "r": max(1.0, rm), "score": final_score})

    if not merged:
        return est_center, est_radius, False

    best = max(merged, key=lambda it: float(it["score"]))
    return np.array([best["cx"], best["cy"]], dtype=np.float32), float(best["r"]), True


def _fill_ratio_in_circle(
    binary_image: np.ndarray,
    quad: np.ndarray,
    radius_scale: float = 0.46,
    border_exclude_ratio: float = 0.10,
    use_hough_detection: bool = False,
) -> Tuple[float, np.ndarray, bool]:
    """White-pixel ratio inside a circular bubble mask.

    Args:
        binary_image: Inverted binary image.
        quad: Cell quad.
        radius_scale: Bubble radius scale factor.
        border_exclude_ratio: Border exclusion ratio.
        use_hough_detection: Enable Hough circle refinement.

    Returns:
        ``(fill_ratio, score_polygon, circle_found)``.
    """
    if binary_image is None or binary_image.size == 0:
        return 0.0, quad.astype(np.float32), False

    h, w = binary_image.shape[:2]
    qmask = np.zeros((h, w), dtype=np.uint8)
    qpts = np.round(quad).astype(np.int32)
    qpts[:, 0] = np.clip(qpts[:, 0], 0, w - 1)
    qpts[:, 1] = np.clip(qpts[:, 1], 0, h - 1)
    cv2.fillConvexPoly(qmask, qpts, 255)

    def _score_for_circle(center_in: np.ndarray, outer_r_in: float) -> Tuple[float, float]:
        inner_r_local = outer_r_in * (1.0 - float(np.clip(border_exclude_ratio, 0.0, 0.4)))
        inner_r_local = max(1.0, inner_r_local)

        cmask = np.zeros((h, w), dtype=np.uint8)
        cxy = tuple(np.round(center_in).astype(np.int32))
        cv2.circle(cmask, cxy, int(round(inner_r_local)), 255, -1)

        mask = cv2.bitwise_and(qmask, cmask)
        pixels = binary_image[mask > 0]
        if pixels.size == 0:
            return 0.0, inner_r_local
        return float(np.count_nonzero(pixels)) / float(pixels.size), inner_r_local

    if use_hough_detection:
        base_scale = float(np.clip(radius_scale, 0.20, 0.60))
        
        # Perform single scale hough sweep (Reduced from 3 to 1 for 3x speedup)
        center_s, outer_r_s, found_s = _detect_single_circle_hough_in_quad(
            binary_image, quad, radius_scale=base_scale,
        )
        
        if found_s:
            ratio_s, inner_r_s = _score_for_circle(center_s, outer_r_s)
            return ratio_s, _circle_polygon(center_s, inner_r_s), True

        # Fallback to estimate if Hough failed completely
        center, outer_r = _estimate_circle_from_quad(quad, radius_scale=base_scale)
        ratio, inner_r = _score_for_circle(center, outer_r)
        return ratio, _circle_polygon(center, inner_r), False

    center, outer_r = _estimate_circle_from_quad(quad, radius_scale=radius_scale)
    ratio, inner_r = _score_for_circle(center, outer_r)
    return ratio, _circle_polygon(center, inner_r), False


# =========================================================================
#  Main evaluator
# =========================================================================


def evaluate_grid_fill_from_binary(
    binary_image: np.ndarray,
    grid_info: List[Dict[str, object]],
    fill_ratio_thresh: float,
    inner_margin_ratio: float = 0.18,
    mask_mode: str = "quad",
    circle_radius_scale: float = 0.46,
    circle_border_exclude_ratio: float = 0.10,
) -> List[Dict[str, object]]:
    """Evaluate filled/unfilled status for each grid cell.

    Supports three mask modes:
    - ``"quad"``:         Raw quadrilateral region.
    - ``"circle"``:       Geometric circle estimate.
    - ``"hough-circle"``: Hough-detected circle (most robust).

    Args:
        binary_image: Inverted binary image.
        grid_info: Cell metadata from grid extraction.
        fill_ratio_thresh: Minimum ratio to classify as filled.
        inner_margin_ratio: Cell shrink ratio before measurement.
        mask_mode: ``"quad"``, ``"circle"``, or ``"hough-circle"``.
        circle_radius_scale: Bubble radius scale factor.
        circle_border_exclude_ratio: Border exclusion ratio.

    Returns:
        List of result dicts per cell (box_idx, row, col, fill_ratio, filled, …).
    """
    evaluations: List[Dict[str, object]] = []
    mode = str(mask_mode).lower()
    use_circle = mode in ("circle", "hough-circle")
    use_hough_circle = mode == "hough-circle"

    for info in grid_info:
        if "region_quad" not in info or "grid_shape" not in info:
            continue

        region_quad = np.array(info["region_quad"], dtype=np.float32)
        rows, cols = info["grid_shape"]
        rows = int(rows)
        cols = int(cols)
        pattern = info.get("pattern")

        for row in range(rows):
            if isinstance(pattern, list) and row < len(pattern):
                cols_to_check = [int(c) for c in pattern[row] if 0 <= int(c) < cols]
            else:
                cols_to_check = list(range(cols))

            for col in cols_to_check:
                cell_quad = _quad_cell_at(region_quad, row=row, col=col, rows=rows, cols=cols)
                inner_quad = _shrink_quad_towards_center(cell_quad, inner_margin_ratio)
                if use_circle:
                    fill_ratio, score_poly, circle_found = _fill_ratio_in_circle(
                        binary_image, inner_quad,
                        radius_scale=circle_radius_scale,
                        border_exclude_ratio=circle_border_exclude_ratio,
                        use_hough_detection=use_hough_circle,
                    )
                else:
                    fill_ratio = _fill_ratio_in_quad(binary_image, inner_quad)
                    score_poly = inner_quad
                    circle_found = False
                evaluations.append(
                    {
                        "box_idx": int(info.get("box_idx", -1)),
                        "row": row,
                        "col": col,
                        "fill_ratio": fill_ratio,
                        "filled": fill_ratio >= float(fill_ratio_thresh),
                        "cell_quad": score_poly,
                        "mask_mode": "hough-circle" if use_hough_circle else ("circle" if use_circle else "quad"),
                        "circle_detected": bool(circle_found),
                    }
                )

    return evaluations
