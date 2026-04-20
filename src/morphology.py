"""
morphology.py — Line detection, grid-point finding, and box extraction.

Provides the low-level morphological operations that turn a raw scan into
structure: detecting horizontal/vertical lines, intersecting them to find
grid points, and extracting enclosed rectangular contours.

Also includes corner-marker detection for affine correction.

Pipeline position: Step 2 (after preprocessing, before box grouping).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================================================================
#  Line filtering and alignment
# =========================================================================


def _filter_line_components_by_length(
    line_mask: np.ndarray,
    min_length: int,
    orientation: str,
) -> np.ndarray:
    """Keep only connected line components whose major axis ≥ *min_length*.

    Args:
        line_mask: Binary mask of detected lines.
        min_length: Minimum component length to keep.
        orientation: ``"vertical"`` or ``"horizontal"``.

    Returns:
        Filtered binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_mask, connectivity=8,
    )
    filtered = np.zeros_like(line_mask)
    axis = cv2.CC_STAT_HEIGHT if orientation == "vertical" else cv2.CC_STAT_WIDTH

    for i in range(1, num_labels):
        if int(stats[i, axis]) >= min_length:
            filtered[labels == i] = 255

    return filtered


def _align_vertical_lengths_by_row(
    vertical_mask: np.ndarray,
    row_tolerance: int = 25,
    min_group_size: int = 2,
) -> np.ndarray:
    """Elongate vertical line components in the same row to a uniform length.

    Args:
        vertical_mask: Binary mask of vertical lines.
        row_tolerance: Y-tolerance to group lines into the same row.
        min_group_size: Minimum group size for alignment.

    Returns:
        Aligned vertical line mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        vertical_mask, connectivity=8,
    )

    comps: List[Dict[str, int]] = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if h <= 0 or w <= 0:
            continue
        comps.append({
            "idx": i, "x": x, "y": y, "w": w, "h": h,
            "cy": y + h // 2, "y_end": y + h,
        })

    if not comps:
        return vertical_mask

    comps.sort(key=lambda c: c["cy"])
    groups: List[List[Dict[str, int]]] = []
    for comp in comps:
        placed = False
        for group in groups:
            mean_cy = int(round(sum(g["cy"] for g in group) / len(group)))
            if abs(comp["cy"] - mean_cy) <= row_tolerance:
                group.append(comp)
                placed = True
                break
        if not placed:
            groups.append([comp])

    aligned = np.zeros_like(vertical_mask)

    for group in groups:
        if len(group) < min_group_size:
            for g in group:
                aligned[labels == g["idx"]] = vertical_mask[labels == g["idx"]]
            continue

        target_h = int(round(float(np.median([g["h"] for g in group]))))
        target_h = max(1, target_h)

        for g in group:
            line_component = (labels == g["idx"]).astype(np.uint8) * 255
            extend_px = max(0, target_h - g["h"])
            if extend_px > 0:
                k = 2 * extend_px + 1
                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
                extended = cv2.dilate(line_component, v_kernel, iterations=1)
                aligned = cv2.bitwise_or(aligned, extended)
            else:
                aligned = cv2.bitwise_or(aligned, line_component)

    for i in range(1, num_labels):
        if not any(comp["idx"] == i for group in groups for comp in group):
            aligned = cv2.bitwise_or(aligned, (labels == i).astype(np.uint8) * 255)

    return aligned


# =========================================================================
#  Grid-point detection
# =========================================================================


def detect_grid_points(
    image: np.ndarray,
    vertical_scale: float = 0.015,
    horizontal_scale: float = 0.015,
    min_point_area: int = 8,
    block_size: int = 35,
    block_offset: int = 7,
    debug_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """Detect grid intersection points from vertical/horizontal morphology lines.

    Args:
        image: Input BGR or grayscale image.
        vertical_scale: Vertical kernel length as fraction of image height.
        horizontal_scale: Horizontal kernel length as fraction of image width.
        min_point_area: Minimum CC area for an intersection point.
        block_size: Adaptive threshold block size.
        block_offset: Adaptive threshold C constant.
        debug_prefix: Prefix for saving debug images (None = skip).

    Returns:
        Dict with intermediate masks, overlay, and list of point coords.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)

    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block_size, block_offset,
    )

    h, w = gray.shape
    v_len = max(3, int(h * vertical_scale))
    h_len = max(3, int(w * horizontal_scale))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))

    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    intersections = cv2.bitwise_and(vertical_lines, horizontal_lines)
    intersections = cv2.dilate(intersections, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(intersections)
    points: List[Tuple[int, int]] = []
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] < min_point_area:
            continue
        cx, cy = centroids[idx]
        points.append((int(round(cx)), int(round(cy))))

    points.sort(key=lambda p: (p[1], p[0]))

    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)

    debug_data = {
        "binary": binary,
        "vertical": vertical_lines,
        "horizontal": horizontal_lines,
        "intersections": intersections,
        "points_overlay": overlay,
        "points": points,
    }

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_binary.jpg", binary)
        cv2.imwrite(f"{debug_prefix}_vertical.jpg", vertical_lines)
        cv2.imwrite(f"{debug_prefix}_horizontal.jpg", horizontal_lines)
        cv2.imwrite(f"{debug_prefix}_intersections.jpg", intersections)
        cv2.imwrite(f"{debug_prefix}_points.jpg", overlay)

    return debug_data


# =========================================================================
#  Box detection from closed line regions
# =========================================================================


def detect_boxes_from_morph_lines(
    image: np.ndarray,
    vertical_scale: float = 0.015,
    horizontal_scale: float = 0.015,
    min_line_length: int = 30,
    align_vertical_rows: bool = True,
    vertical_row_tolerance: int = 25,
    block_size: int = 35,
    block_offset: int = 7,
    min_box_area: int = 100,
    min_box_width: int = 8,
    min_box_height: int = 8,
    close_kernel_size: int = 6,
    debug_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """Detect closed rectangular contours from morphological line grid.

    Pipeline:
    1. Extract vertical/horizontal line masks.
    2. Merge and close small gaps.
    3. Flood-fill background to isolate enclosed regions.
    4. Extract contours as box polygons.

    Args:
        image: Input BGR image.
        vertical_scale: Vertical kernel scale.
        horizontal_scale: Horizontal kernel scale.
        min_line_length: Min component length.
        align_vertical_rows: Align vertical line lengths by row.
        vertical_row_tolerance: Row grouping tolerance.
        block_size: Adaptive threshold block size.
        block_offset: Adaptive threshold C constant.
        min_box_area: Minimum contour area.
        min_box_width: Minimum contour width.
        min_box_height: Minimum contour height.
        close_kernel_size: Morphological closing kernel size.
        debug_prefix: Debug image prefix (None = skip).

    Returns:
        Dict with intermediate masks, overlay, and list of box contours.
    """
    grid = detect_grid_points(
        image=image,
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale,
        min_point_area=8,
        block_size=block_size,
        block_offset=block_offset,
        debug_prefix=None,
    )

    vertical = grid["vertical"]
    horizontal = grid["horizontal"]

    vertical = _filter_line_components_by_length(vertical, min_line_length, "vertical")
    horizontal = _filter_line_components_by_length(horizontal, min_line_length, "horizontal")

    if align_vertical_rows:
        vertical = _align_vertical_lengths_by_row(
            vertical, row_tolerance=vertical_row_tolerance, min_group_size=2,
        )

    lines = cv2.bitwise_or(vertical, horizontal)
    k = max(1, close_kernel_size)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    lines_closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    inv = cv2.bitwise_not(lines_closed)

    h, w = inv.shape
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=128)

    enclosed = np.where(flood == 255, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(enclosed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_box_area:
            continue

        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, bw, bh = cv2.boundingRect(approx)
        if bw < min_box_width or bh < min_box_height:
            continue

        boxes.append(approx)

    boxes.sort(key=lambda b: (int(b[0, 0, 1]), int(b[0, 0, 0])))

    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for poly in boxes:
        cv2.polylines(overlay, [poly], True, (0, 255, 0), 2)

    result = {
        "binary": grid["binary"],
        "vertical": vertical,
        "horizontal": horizontal,
        "lines": lines,
        "lines_closed": lines_closed,
        "enclosed": enclosed,
        "boxes_overlay": overlay,
        "boxes": boxes,
    }

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_vertical.jpg", vertical)
        cv2.imwrite(f"{debug_prefix}_horizontal.jpg", horizontal)
        cv2.imwrite(f"{debug_prefix}_lines.jpg", lines)
        cv2.imwrite(f"{debug_prefix}_boxes.jpg", overlay)

    return result


# =========================================================================
#  Corner marker detection for affine correction
# =========================================================================


def detect_black_corner_markers(
    image: np.ndarray,
    debug_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """Detect 4 black corner markers for perspective correction.

    Args:
        image: Input BGR image.
        debug_prefix: Debug image prefix (None = skip).

    Returns:
        Dict with ``corners``, ``ordered_corners``, ``found_count``,
        ``all_found``, ``candidate_count``, ``debug_image_path``.
    """
    h_img, w_img = image.shape[:2]
    if h_img <= 0 or w_img <= 0:
        return {
            "corners": {"top_left": None, "top_right": None, "bottom_right": None, "bottom_left": None},
            "ordered_corners": [], "found_count": 0, "all_found": False,
            "candidate_count": 0, "debug_image_path": None,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_thresh = int(np.clip(0.62 * float(otsu_val), 60.0, 130.0))
    _, binary_inv = cv2.threshold(blur, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    img_area = float(h_img * w_img)
    diag = float(np.hypot(w_img, h_img))

    selected: Dict[str, Optional[Tuple[int, int]]] = {
        "top_left": None, "top_right": None, "bottom_right": None, "bottom_left": None,
    }
    selected_bboxes: Dict[str, Optional[Tuple[int, int, int, int]]] = {
        "top_left": None, "top_right": None, "bottom_right": None, "bottom_left": None,
    }
    candidates: List[Dict[str, object]] = []

    def _pick_corner_from_roi(
        x0: int, y0: int, x1: int, y1: int,
        target: Tuple[float, float],
    ) -> Optional[Dict[str, object]]:
        x0 = int(np.clip(x0, 0, w_img - 1))
        y0 = int(np.clip(y0, 0, h_img - 1))
        x1 = int(np.clip(x1, x0 + 1, w_img))
        y1 = int(np.clip(y1, y0 + 1, h_img))

        roi = binary_inv[y0:y1, x0:x1]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        local_min_area = max(8.0, 0.000002 * img_area)
        local_max_area = max(local_min_area + 1.0, 0.030 * img_area)

        best = None
        best_score = -1e9
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < local_min_area or area > local_max_area:
                continue
            rx, ry, rw, rh = cv2.boundingRect(contour)
            if rw < 3 or rh < 3:
                continue
            gx, gy = x0 + rx, y0 + ry
            if gx <= 2 or gy <= 2 or (gx + rw) >= (w_img - 2) or (gy + rh) >= (h_img - 2):
                continue
            extent = area / float(max(1, rw * rh))
            aspect = float(rw) / float(rh)
            if extent < 0.10 or not (0.03 <= aspect <= 35.0):
                continue
            roi_gray = gray[gy:gy + rh, gx:gx + rw]
            if roi_gray.size == 0:
                continue
            mean_intensity = float(np.mean(roi_gray))
            if mean_intensity > 235.0:
                continue

            cx = float(gx + 0.5 * rw)
            cy = float(gy + 0.5 * rh)
            distance_norm = float(np.hypot(cx - target[0], cy - target[1])) / max(diag, 1.0)
            area_norm = area / max(img_area, 1.0)
            area_boost = min(1.0, area_norm / 0.0015)
            score = (0.35 * extent) + (0.10 * area_boost) - (4.0 * distance_norm)
            if score > best_score:
                best_score = score
                best = {
                    "bbox": (int(gx), int(gy), int(rw), int(rh)),
                    "center": (int(round(cx)), int(round(cy))),
                    "area": area, "extent": extent, "score": score,
                }
        return best

    roi_x = max(40, int(round(0.24 * w_img)))
    roi_y = max(40, int(round(0.24 * h_img)))
    corner_targets = {
        "top_left": (0.0, 0.0), "top_right": (float(w_img - 1), 0.0),
        "bottom_right": (float(w_img - 1), float(h_img - 1)),
        "bottom_left": (0.0, float(h_img - 1)),
    }
    corner_rois = {
        "top_left": (0, 0, roi_x, roi_y),
        "top_right": (w_img - roi_x, 0, w_img, roi_y),
        "bottom_right": (w_img - roi_x, h_img - roi_y, w_img, h_img),
        "bottom_left": (0, h_img - roi_y, roi_x, h_img),
    }

    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        target = corner_targets[name]
        x0, y0, x1, y1 = corner_rois[name]
        picked = _pick_corner_from_roi(x0, y0, x1, y1, target)
        if picked is None:
            roi_x2 = max(roi_x, int(round(0.34 * w_img)))
            roi_y2 = max(roi_y, int(round(0.34 * h_img)))
            if name == "top_left":
                picked = _pick_corner_from_roi(0, 0, roi_x2, roi_y2, target)
            elif name == "top_right":
                picked = _pick_corner_from_roi(w_img - roi_x2, 0, w_img, roi_y2, target)
            elif name == "bottom_right":
                picked = _pick_corner_from_roi(w_img - roi_x2, h_img - roi_y2, w_img, h_img, target)
            else:
                picked = _pick_corner_from_roi(0, h_img - roi_y2, roi_x2, h_img, target)

        if picked is not None:
            selected[name] = picked["center"]
            selected_bboxes[name] = picked["bbox"]
            candidates.append(picked)

    # Fallback: global contour search for missing corners.
    missing_names = [name for name, pt in selected.items() if pt is None]
    if missing_names:
        global_contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        global_candidates: List[Dict[str, object]] = []
        min_a = max(8.0, 0.000002 * img_area)
        max_a = max(min_a + 1.0, 0.035 * img_area)

        for contour in global_contours:
            area = float(cv2.contourArea(contour))
            if area < min_a or area > max_a:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3:
                continue
            if x <= 2 or y <= 2 or (x + w) >= (w_img - 2) or (y + h) >= (h_img - 2):
                continue
            extent = area / float(max(1, w * h))
            aspect = float(w) / float(h)
            if extent < 0.10 or not (0.03 <= aspect <= 35.0):
                continue
            roi_gray = gray[y:y + h, x:x + w]
            if roi_gray.size == 0:
                continue
            if float(np.mean(roi_gray)) > 235.0:
                continue
            cx = float(x + 0.5 * w)
            cy = float(y + 0.5 * h)
            global_candidates.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "center": (int(round(cx)), int(round(cy))),
                "extent": extent,
            })

        used_centers = {pt for pt in selected.values() if pt is not None}
        for name in missing_names:
            target = corner_targets[name]
            best = None
            best_score = float("inf")
            for cand in global_candidates:
                center = cand["center"]
                if center in used_centers:
                    continue
                cx, cy = float(center[0]), float(center[1])
                dist_norm = float(np.hypot(cx - target[0], cy - target[1])) / max(diag, 1.0)
                if name == "top_left":
                    quadrant_penalty = 0.0 if (cx <= 0.5 * w_img and cy <= 0.5 * h_img) else 0.7
                elif name == "top_right":
                    quadrant_penalty = 0.0 if (cx >= 0.5 * w_img and cy <= 0.5 * h_img) else 0.7
                elif name == "bottom_right":
                    quadrant_penalty = 0.0 if (cx >= 0.5 * w_img and cy >= 0.5 * h_img) else 0.7
                else:
                    quadrant_penalty = 0.0 if (cx <= 0.5 * w_img and cy >= 0.5 * h_img) else 0.7
                score = dist_norm + quadrant_penalty
                if score < best_score:
                    best_score = score
                    best = cand
            if best is not None:
                selected[name] = best["center"]
                selected_bboxes[name] = best["bbox"]
                candidates.append(best)
                used_centers.add(best["center"])

    # Distance-based pruning.
    max_corner_distance_ratio = 0.32
    for name, pt in list(selected.items()):
        if pt is None:
            continue
        target = corner_targets[name]
        dist_norm = float(np.hypot(float(pt[0]) - target[0], float(pt[1]) - target[1])) / max(diag, 1.0)
        if dist_norm > max_corner_distance_ratio:
            selected[name] = None
            selected_bboxes[name] = None

    # Quadrant sanity check.
    for name, pt in list(selected.items()):
        if pt is None:
            continue
        px, py = float(pt[0]), float(pt[1])
        invalid = False
        if name in ("top_left", "top_right") and py > 0.40 * h_img:
            invalid = True
        if name in ("bottom_left", "bottom_right") and py < 0.88 * h_img:
            invalid = True
        if name in ("top_left", "bottom_left") and px > 0.25 * w_img:
            invalid = True
        if name in ("top_right", "bottom_right") and px < 0.75 * w_img:
            invalid = True
        if invalid:
            selected[name] = None
            selected_bboxes[name] = None

    # Interpolate missing corners from the remaining three.
    tl, tr, br, bl = selected["top_left"], selected["top_right"], selected["bottom_right"], selected["bottom_left"]
    if br is None and tr is not None and bl is not None:
        selected["bottom_right"] = (int(tr[0]), int(bl[1]))
    if bl is None and tl is not None and br is not None:
        selected["bottom_left"] = (int(tl[0]), int(br[1]))
    if tr is None and tl is not None and br is not None:
        selected["top_right"] = (int(br[0]), int(tl[1]))
    if tl is None and tr is not None and bl is not None:
        selected["top_left"] = (int(bl[0]), int(tr[1]))

    ordered_corners: List[Tuple[int, int]] = []
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        pt = selected[name]
        if pt is not None:
            ordered_corners.append(pt)

    debug_image_path = None
    if debug_prefix:
        debug_img = image.copy() if image.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cand in candidates:
            x, y, w, h = cand["bbox"]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 180, 0), 2)
        draw_colors = {
            "top_left": (0, 255, 0), "top_right": (0, 255, 255),
            "bottom_right": (0, 128, 255), "bottom_left": (255, 0, 0),
        }
        for name, pt in selected.items():
            if pt is None:
                continue
            color = draw_colors.get(name, (0, 255, 0))
            cv2.circle(debug_img, pt, 8, color, -1)
            cv2.putText(debug_img, name, (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        debug_image_path = f"{debug_prefix}_corner_markers.jpg"
        cv2.imwrite(debug_image_path, debug_img)

    found_count = sum(1 for pt in selected.values() if pt is not None)
    return {
        "corners": selected,
        "ordered_corners": ordered_corners,
        "found_count": found_count,
        "all_found": found_count == 4,
        "candidate_count": len(candidates),
        "debug_image_path": debug_image_path,
    }
