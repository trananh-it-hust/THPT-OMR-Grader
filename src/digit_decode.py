"""
digit_decode.py — SoBaoDanh / MaDe digit decoding via mean-darkness.

Selects the darkest bubble per column to determine which digit (row index)
was marked by the student.

Pipeline position: Step 7 (after extrapolation, produces decoded strings).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


def _mean_darkness_in_box_circle(
    gray_image: np.ndarray,
    box: np.ndarray,
    radius_scale: float = 0.36,
) -> float:
    """Mean gray value inside a central circle within a bubble box.

    Lower values = darker = more likely filled.

    Args:
        gray_image: Grayscale input image.
        box: Contour box of the bubble.
        radius_scale: Scale factor for measurement circle.

    Returns:
        Mean darkness (0 = darkest, 255 = brightest).
    """
    if gray_image is None or gray_image.size == 0:
        return 255.0

    x, y, w, h = cv2.boundingRect(box)
    if w <= 0 or h <= 0:
        return 255.0

    cx = x + (w // 2)
    cy = y + (h // 2)
    rr = max(2, int(round(min(w, h) * float(np.clip(radius_scale, 0.15, 0.48)))))

    x1 = max(0, cx - rr)
    y1 = max(0, cy - rr)
    x2 = min(gray_image.shape[1], cx + rr + 1)
    y2 = min(gray_image.shape[0], cy + rr + 1)

    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return 255.0

    mask = np.zeros(roi.shape, dtype=np.uint8)
    local_center = (cx - x1, cy - y1)
    local_radius = max(1, min(rr, min(roi.shape[0], roi.shape[1]) // 2))
    cv2.circle(mask, local_center, local_radius, 255, -1)

    pixels = roi[mask > 0]
    if pixels.size == 0:
        return 255.0
    return float(np.mean(pixels))


def evaluate_digit_rows_mean_darkness(
    gray_image: np.ndarray,
    aligned_rows: List[Optional[List[np.ndarray]]],
    expected_cols: int,
    radius_scale: float = 0.36,
    abs_darkness_threshold: float = 195.0,
    min_second_gap: float = 8.0,
    min_median_gap: float = 12.0,
    min_abs_median_gap: float = 5.0,
    min_abs_second_gap: float = 5.0,
) -> Dict[str, object]:
    """Decode one digit per column via mean-darkness comparison.

    For each column, the row with the darkest bubble is selected as the
    filled digit.  Confidence checks (gap vs second-darkest, gap vs median)
    prevent false positives on noisy columns.

    Args:
        gray_image: Grayscale input image.
        aligned_rows: Rows of boxes (may contain ``None`` for missing rows).
        expected_cols: Number of digit columns to decode.
        radius_scale: Measurement circle radius scale.
        abs_darkness_threshold: Max gray value for a valid filled bubble.
        min_second_gap: Required gap from second-darkest.
        min_median_gap: Required gap from column median.
        min_abs_median_gap: Absolute minimum median gap.
        min_abs_second_gap: Absolute minimum second gap.

    Returns:
        Dict with ``"decoded"`` string, ``"evaluations"`` list,
        and ``"column_decisions"`` list.
    """
    evaluations: List[Dict[str, object]] = []
    decoded_chars: List[str] = []
    column_decisions: List[Dict[str, object]] = []

    if expected_cols <= 0:
        return {"decoded": "", "evaluations": evaluations}

    for col in range(expected_cols):
        col_items: List[Dict[str, object]] = []
        for row_idx, row in enumerate(aligned_rows):
            if row is None or col >= len(row):
                col_items.append(
                    {
                        "row": row_idx, "col": col,
                        "mean_darkness": 255.0, "filled": False,
                        "box": None, "valid": False,
                    }
                )
                continue

            box = row[col]
            darkness = _mean_darkness_in_box_circle(gray_image, box, radius_scale=radius_scale)
            col_items.append(
                {
                    "row": row_idx, "col": col,
                    "mean_darkness": darkness, "filled": False,
                    "box": box, "valid": True,
                }
            )

        valid_items = [it for it in col_items if it["valid"]]
        if valid_items:
            sorted_items = sorted(valid_items, key=lambda it: float(it["mean_darkness"]))
            best_item = sorted_items[0]
            best_dark = float(best_item["mean_darkness"])
            second_dark = float(sorted_items[1]["mean_darkness"]) if len(sorted_items) > 1 else 255.0
            median_dark = float(np.median([float(it["mean_darkness"]) for it in sorted_items]))

            has_abs_dark = best_dark <= float(abs_darkness_threshold)
            second_gap = second_dark - best_dark
            median_gap = median_dark - best_dark
            has_gap = second_gap >= float(min_second_gap) and median_gap >= float(min_median_gap)
            has_context_for_abs = (
                median_gap >= float(min_abs_median_gap)
                and second_gap >= float(min_abs_second_gap)
            )
            col_filled = bool(has_gap or (has_abs_dark and has_context_for_abs))

            if col_filled:
                best_item["filled"] = True
                decoded_chars.append(str(int(best_item["row"])))
            else:
                decoded_chars.append("?")

            column_decisions.append(
                {
                    "col": col,
                    "filled": col_filled,
                    "best_row": int(best_item["row"]),
                    "best_darkness": best_dark,
                    "second_darkness": second_dark,
                    "median_darkness": median_dark,
                    "second_gap": second_gap,
                    "median_gap": median_gap,
                }
            )
        else:
            decoded_chars.append("?")
            column_decisions.append(
                {
                    "col": col, "filled": False,
                    "best_row": None, "best_darkness": 255.0,
                    "second_darkness": 255.0, "median_darkness": 255.0,
                    "second_gap": 0.0, "median_gap": 0.0,
                }
            )

        evaluations.extend(col_items)

    return {
        "decoded": "".join(decoded_chars),
        "evaluations": evaluations,
        "column_decisions": column_decisions,
    }
