"""
pipeline.py — End-to-end pipeline orchestrator.

Replaces the original ``_demo()`` function in ``detect.py``.  Calls into
all other ``src.*`` modules in sequence, handles fallback logic (CLAHE,
affine retry, synthetic ID grids), and writes debug images to disk.

Pipeline position: Top-level entry point.
"""

from __future__ import annotations

from src.log_config import logger

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.preprocessing import normalize_image_stem, load_image, preprocess_clahe
from src.morphology import detect_boxes_from_morph_lines, detect_black_corner_markers
from src.box_grouping import (
    group_boxes_into_parts,
    _split_merged_boxes_for_grouping,
    _separate_upper_id_boxes,
    detect_sobao_danh_boxes,
    detect_ma_de_boxes,
    extrapolate_missing_rows,
    _rect_to_poly,
    _build_synthetic_id_rows_from_part_i,
    _build_synthetic_id_rows_fixed_image_position,
    _apply_affine_from_corner_markers,
)
from src.grid_extraction import (
    extract_grid_from_boxes,
    extract_grid_from_boxes_variable_offsets,
    extract_grid_from_boxes_custom_pattern,
)
from src.fill_evaluation import evaluate_grid_fill_from_binary
from src.digit_decode import evaluate_digit_rows_mean_darkness
from src.debug_draw import (
    draw_filled_cells_overlay,
    draw_binary_fillratio_debug,
    draw_digit_darkness_overlay,
    draw_rows_contours,
    print_fill_summary,
    print_digit_darkness_summary,
    print_grid_info,
)


# =========================================================================
#  Pipeline helpers
# =========================================================================


def _evaluate_section_fill(
    section_name: str,
    binary_threshold: Optional[np.ndarray],
    grid_info: List[Dict[str, object]],
    fill_ratio_thresh: float,
    inner_margin_ratio: float,
    circle_radius_scale: float,
    circle_border_exclude_ratio: float,
) -> List[Dict[str, object]]:
    """Evaluate filled cells for one grid section."""
    if binary_threshold is None or not grid_info:
        return []

    evals = evaluate_grid_fill_from_binary(
        binary_image=binary_threshold,
        grid_info=grid_info,
        fill_ratio_thresh=fill_ratio_thresh,
        inner_margin_ratio=inner_margin_ratio,
        mask_mode="hough-circle",
        circle_radius_scale=circle_radius_scale,
        circle_border_exclude_ratio=circle_border_exclude_ratio,
    )
    print_fill_summary(section_name, evals)
    return evals


def _parts_score(parts_local: Dict[str, object], page_h: int) -> float:
    """Heuristic score to pick the best detection result (base vs CLAHE)."""
    p1 = len(parts_local["part_i"])
    p2 = len(parts_local["part_ii"])
    p3 = len(parts_local["part_iii"])
    score = 0.0
    score += 3.0 if p1 == 4 else (p1 / 4.0)
    score += 4.0 * min(1.0, p2 / 8.0)
    score += 3.0 * min(1.0, p3 / 6.0)

    if p3 and page_h > 0:
        y_vals = [cv2.boundingRect(b)[1] for b in parts_local["part_iii"]]
        p3_ratio = float(np.mean(y_vals)) / float(page_h)
        score += max(0.0, min(1.0, (p3_ratio - 0.55) / 0.20))
    return score


def _run_detection_pipeline(
    src_img: np.ndarray, prefix: Optional[str],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run morphology + part grouping on a single image."""
    data_local = detect_boxes_from_morph_lines(
        src_img,
        vertical_scale=0.015,
        horizontal_scale=0.015,
        min_line_length=50,
        align_vertical_rows=True,
        vertical_row_tolerance=10,
        block_size=35,
        block_offset=7,
        min_box_area=200,
        min_box_width=15,
        min_box_height=15,
        close_kernel_size=3,
        debug_prefix=prefix,
    )
    parts_local = group_boxes_into_parts(data_local["boxes"], row_tolerance=30)
    return data_local, parts_local


# =========================================================================
#  Main pipeline
# =========================================================================


def run_pipeline(image_arg: Optional[str] = None) -> None:
    """Full detection → grading → debug-output pipeline.

    This is the refactored version of the original ``_demo()`` in
    ``detect.py``.  All logic is preserved; the only change is that
    function calls now go through the ``src.*`` module API.

    Args:
        image_arg: Image identifier (e.g. ``"0015"``).
    """
    base_image_name = normalize_image_stem(image_arg)
    candidate_paths = [
        Path("PhieuQG") / f"{base_image_name}.jpg",
        Path("PhieuQG") / f"{base_image_name}.jpeg",
        Path("PhieuQG") / f"{base_image_name}.png",
        Path("PhieuQG") / f"{base_image_name}.bmp",
        Path("PhieuQG") / f"{base_image_name}.BMP",
    ]
    image_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    out_dir = Path("output/detection")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        tried = ", ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(f"Cannot read image. Tried: {tried}")
    img_original = img.copy()

    debug_prefix = str(out_dir / image_path.stem)
    corner_markers = detect_black_corner_markers(img, debug_prefix=debug_prefix)
    logger.info(
        f"Corner markers: {corner_markers['found_count']}/4 "
        f"(candidates={corner_markers['candidate_count']})"
    )
    if corner_markers.get("debug_image_path"):
        logger.info(f"Corner debug image: {corner_markers['debug_image_path']}")
    for key in ("top_left", "top_right", "bottom_right", "bottom_left"):
        logger.info(f"  {key}: {corner_markers['corners'][key]}")

    data, parts = _run_detection_pipeline(img, debug_prefix)

    preprocess_mode = "base"
    page_h = img.shape[0] if img is not None else 0

    if len(parts["part_ii"]) < 8 or len(parts["part_iii"]) < 6:
        img_clahe = preprocess_clahe(img)
        data_clahe, parts_clahe = _run_detection_pipeline(img_clahe, None)

        base_score = _parts_score(parts, page_h)
        clahe_score = _parts_score(parts_clahe, page_h)
        if clahe_score > base_score + 0.05:
            data = data_clahe
            parts = parts_clahe
            preprocess_mode = "clahe"

    logger.info(f"Detected boxes: {len(data['boxes'])}")
    logger.info(f"Preprocess mode: {preprocess_mode}")
    logger.info(f"Part I boxes: {len(parts['part_i'])}")
    logger.info(f"Part II boxes: {len(parts['part_ii'])}")
    logger.info(f"Part III boxes: {len(parts['part_iii'])}")

    part_box_set = set(id(box) for box in parts["all_parts"])
    remaining_boxes = [box for box in data["boxes"] if id(box) not in part_box_set]

    remaining_for_upper = _split_merged_boxes_for_grouping(
        remaining_boxes, split_wide=True, split_tall=False,
    )

    sbd_candidates, ma_de_candidates, split_x = _separate_upper_id_boxes(
        remaining_for_upper, parts["part_i"],
    )

    sobao_danh = detect_sobao_danh_boxes(
        sbd_candidates, boxes_per_row=6, max_rows=10,
        row_tolerance=45, size_tolerance_ratio=0.45, debug=False,
    )
    logger.info(f"SoBaoDanh rows: {sobao_danh['row_count']}")
    logger.info(f"SoBaoDanh boxes: {len(sobao_danh['sobao_danh'])}")

    remaining_for_ma_de = _split_merged_boxes_for_grouping(
        ma_de_candidates, split_wide=False, split_tall=True,
    )
    ma_de = detect_ma_de_boxes(
        remaining_for_ma_de, boxes_per_row=3, max_rows=10,
        row_tolerance=35, size_tolerance_ratio=0.40, debug=False,
    )

    # --- Affine retry ---
    affine_retry_threshold = 4
    topview_allowed_by_rows = (
        sobao_danh["row_count"] <= affine_retry_threshold
        and ma_de["row_count"] <= affine_retry_threshold
    )
    if topview_allowed_by_rows:
        affine_img, affine_matrix = _apply_affine_from_corner_markers(
            img, corner_markers.get("corners", {}),
        )
        if affine_img is not None and affine_matrix is not None:
            data_affine, parts_affine = _run_detection_pipeline(affine_img, None)

            part_box_set_affine = set(id(box) for box in parts_affine["all_parts"])
            remaining_boxes_affine = [
                box for box in data_affine["boxes"]
                if id(box) not in part_box_set_affine
            ]
            remaining_for_upper_affine = _split_merged_boxes_for_grouping(
                remaining_boxes_affine, split_wide=True, split_tall=False,
            )
            sbd_candidates_affine, ma_de_candidates_affine, split_x_affine = (
                _separate_upper_id_boxes(remaining_for_upper_affine, parts_affine["part_i"])
            )
            sobao_danh_affine = detect_sobao_danh_boxes(
                sbd_candidates_affine, boxes_per_row=6, max_rows=10,
                row_tolerance=45, size_tolerance_ratio=0.45, debug=False,
            )
            remaining_for_ma_de_affine = _split_merged_boxes_for_grouping(
                ma_de_candidates_affine, split_wide=False, split_tall=True,
            )
            ma_de_affine = detect_ma_de_boxes(
                remaining_for_ma_de_affine, boxes_per_row=3, max_rows=10,
                row_tolerance=35, size_tolerance_ratio=0.40, debug=False,
            )

            old_score = int(sobao_danh["row_count"]) + int(ma_de["row_count"])
            new_score = int(sobao_danh_affine["row_count"]) + int(ma_de_affine["row_count"])
            if new_score > old_score:
                img = affine_img
                data = data_affine
                parts = parts_affine
                split_x = split_x_affine
                sobao_danh = sobao_danh_affine
                ma_de = ma_de_affine
                preprocess_mode = f"{preprocess_mode}+topview-id"
                logger.info(
                    f"[Topview] Retry improved ID detection: score {old_score} -> {new_score} "
                    f"(SBD={sobao_danh['row_count']}, MaDe={ma_de['row_count']})"
                )
            else:
                logger.info(
                    f"[Topview] Retry no improvement: score {old_score} -> {new_score} "
                    f"(SBD={sobao_danh_affine['row_count']}, MaDe={ma_de_affine['row_count']})"
                )
        else:
            logger.info("[Topview] Retry skipped: not enough reliable corner markers")
    else:
        logger.info(
            "[Topview] Retry blocked by row gate: "
            f"SBD={sobao_danh['row_count']}, MaDe={ma_de['row_count']} (>4 on at least one side)"
        )

    # --- ID fallback ---
    id_fallback_row_threshold = 4
    id_grid_top_ratio = 0.072
    id_grid_row_step_ratio = 0.0215
    id_sbd_x_range_ratio_fixed = (0.745, 0.865)
    id_made_x_range_ratio_fixed = (0.90, 0.96)
    sbd_x_range_ratio = id_sbd_x_range_ratio_fixed
    made_x_range_ratio = id_made_x_range_ratio_fixed

    if topview_allowed_by_rows and (
        sobao_danh["row_count"] <= id_fallback_row_threshold
        or ma_de["row_count"] <= id_fallback_row_threshold
    ):
        topview_img, _ = _apply_affine_from_corner_markers(
            img_original, corner_markers.get("corners", {}),
        )
        if topview_img is not None:
            topview_sbd_rows = _build_synthetic_id_rows_fixed_image_position(
                image_shape=topview_img.shape[:2], cols=6, rows=10,
                x_range_ratio=sbd_x_range_ratio,
                top_y_ratio=id_grid_top_ratio,
                row_step_ratio=id_grid_row_step_ratio,
            )
            topview_made_rows = _build_synthetic_id_rows_fixed_image_position(
                image_shape=topview_img.shape[:2], cols=3, rows=10,
                x_range_ratio=made_x_range_ratio,
                top_y_ratio=id_grid_top_ratio,
                row_step_ratio=id_grid_row_step_ratio,
            )

            gray_topview = cv2.cvtColor(topview_img, cv2.COLOR_BGR2GRAY)
            topview_sbd_digits = evaluate_digit_rows_mean_darkness(
                gray_topview, topview_sbd_rows, expected_cols=6,
            )
            topview_made_digits = evaluate_digit_rows_mean_darkness(
                gray_topview, topview_made_rows, expected_cols=3,
            )

            logger.info("\n=== Topview Mean-Darkness Decode ===")
            print_digit_darkness_summary("Topview SoBaoDanh", topview_sbd_digits)
            print_digit_darkness_summary("Topview MaDe", topview_made_digits)

            topview_debug = topview_img.copy()
            draw_rows_contours(topview_debug, topview_sbd_rows, (255, 128, 0), thickness=1)
            draw_rows_contours(topview_debug, topview_made_rows, (255, 255, 0), thickness=1)

            topview_debug = draw_digit_darkness_overlay(
                topview_debug, topview_sbd_digits, color=(0, 220, 255), alpha=0.40,
            )
            topview_debug = draw_digit_darkness_overlay(
                topview_debug, topview_made_digits, color=(0, 255, 255), alpha=0.40,
            )

            cv2.putText(topview_debug, f"Topview SBD: {topview_sbd_digits.get('decoded', '')}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(topview_debug, f"Topview MaDe: {topview_made_digits.get('decoded', '')}",
                        (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            topview_raw_path = f"{debug_prefix}_topview.jpg"
            topview_id_path = f"{debug_prefix}_topview_id_mean_darkness.jpg"
            cv2.imwrite(topview_raw_path, topview_img)
            cv2.imwrite(topview_id_path, topview_debug)
            logger.info(f"Topview image saved to {topview_raw_path}")
            logger.info(f"Topview ID debug saved to {topview_id_path}")
        else:
            logger.info("Topview debug skipped: affine transform unavailable (missing reliable corners)")

    fallback_threshold = max(0, int(id_fallback_row_threshold))
    if sobao_danh["row_count"] <= fallback_threshold:
        sobao_rows = _build_synthetic_id_rows_fixed_image_position(
            image_shape=img.shape[:2], cols=6, rows=10,
            x_range_ratio=sbd_x_range_ratio,
            top_y_ratio=id_grid_top_ratio,
            row_step_ratio=id_grid_row_step_ratio,
        )
        if sobao_rows:
            sobao_danh["sobao_danh_rows"] = sobao_rows
            sobao_danh["sobao_danh"] = [b for row in sobao_rows for b in row]
            sobao_danh["row_count"] = len(sobao_rows)
            logger.info(f"[Fallback] SoBaoDanh rows <= {fallback_threshold}, drew synthetic 6x10 grid (mode=fixed)")

    if ma_de["row_count"] <= fallback_threshold:
        ma_de_rows_synth = _build_synthetic_id_rows_fixed_image_position(
            image_shape=img.shape[:2], cols=3, rows=10,
            x_range_ratio=made_x_range_ratio,
            top_y_ratio=id_grid_top_ratio,
            row_step_ratio=id_grid_row_step_ratio,
        )
        if ma_de_rows_synth:
            ma_de["ma_de_rows"] = ma_de_rows_synth
            ma_de["ma_de"] = [b for row in ma_de_rows_synth for b in row]
            ma_de["row_count"] = len(ma_de_rows_synth)
            logger.info(f"[Fallback] MaDe rows <= {fallback_threshold}, drew synthetic 3x10 grid (mode=fixed)")

    # --- MaDe completion ---
    if ma_de["row_count"] < 10 and ma_de["ma_de_rows"] and sobao_danh["sobao_danh_rows"]:
        ref_positions = [
            int(round(float(np.mean([cv2.boundingRect(box)[1] for box in row]))))
            for row in sobao_danh["sobao_danh_rows"][:10] if row
        ]
        ma_de_rect_rows = []
        for row in ma_de["ma_de_rows"]:
            rects = sorted([cv2.boundingRect(box) for box in row], key=lambda r: r[0])
            if len(rects) == 3:
                ma_de_rect_rows.append(rects)

        if len(ref_positions) == 10 and ma_de_rect_rows:
            col_x = [int(round(float(np.median([rects[c][0] for rects in ma_de_rect_rows])))) for c in range(3)]
            col_w = [int(round(float(np.median([rects[c][2] for rects in ma_de_rect_rows])))) for c in range(3)]
            row_h = max(1, int(round(float(np.median([rects[0][3] for rects in ma_de_rect_rows])))))

            detected_rows_y = [int(round(float(np.mean([r[1] for r in rects])))) for rects in ma_de_rect_rows]
            align_tolerance = max(35, int(round(row_h * 1.2)))
            aligned_rows: List[Optional[List[np.ndarray]]] = [None] * 10

            used_ref_indices = set()
            for rects, row_y in sorted(zip(ma_de_rect_rows, detected_rows_y), key=lambda t: t[1]):
                candidates_idx = sorted(range(10), key=lambda idx: abs(ref_positions[idx] - row_y))
                chosen_idx = None
                for idx in candidates_idx:
                    if idx in used_ref_indices:
                        continue
                    if abs(ref_positions[idx] - row_y) <= align_tolerance:
                        chosen_idx = idx
                        break
                if chosen_idx is None:
                    continue
                row_polys = [_rect_to_poly(rx, ry, rw, rh) for rx, ry, rw, rh in rects]
                aligned_rows[chosen_idx] = row_polys
                used_ref_indices.add(chosen_idx)

            for idx in range(10):
                if aligned_rows[idx] is not None:
                    continue
                y_ref = ref_positions[idx]
                synthetic_row: List[np.ndarray] = []
                for c in range(3):
                    x = col_x[c]
                    w = max(1, col_w[c])
                    synthetic_row.append(_rect_to_poly(x, y_ref, w, row_h))
                aligned_rows[idx] = synthetic_row

            ma_de_completed_rows = [row for row in aligned_rows if row is not None]
            if len(ma_de_completed_rows) == 10:
                ma_de["ma_de_rows"] = ma_de_completed_rows
                ma_de["ma_de"] = [box for row in ma_de_completed_rows for box in row]
                ma_de["row_count"] = 10

    logger.info(f"MaDe rows: {ma_de['row_count']}")
    logger.info(f"MaDe boxes: {len(ma_de['ma_de'])}")
    logger.info(f"Upper split X: {split_x:.1f}")
    logger.info(f"SoBaoDanh rows (final): {sobao_danh['row_count']}")
    logger.info(f"MaDe rows (final): {ma_de['row_count']}")

    # --- Extrapolation ---
    combined_results = {
        "sobao_danh_rows": sobao_danh["sobao_danh_rows"],
        "ma_de_rows": ma_de["ma_de_rows"],
    }
    extrapolated = extrapolate_missing_rows(combined_results, target_rows=10, debug=False)

    logger.info(f"\nExtrapolation Summary:")
    logger.info(f"  SoBaoDanh: {extrapolated['sobao_detected_count']}/10 detected, {extrapolated['sobao_missing_count']} missing")
    logger.info(f"  MaDe: {extrapolated['ma_de_detected_count']}/10 detected, {extrapolated['ma_de_missing_count']} missing")

    aligned_ma_de = extrapolated.get("ma_de_rows_aligned", [])
    reference_positions = extrapolated.get("reference_positions", [])
    if aligned_ma_de and None in aligned_ma_de:
        logger.info(f"\n  Missing MaDe rows at positions:")
        for idx, row in enumerate(aligned_ma_de):
            if row is None:
                y_pos = reference_positions[idx] if idx < len(reference_positions) else "?"
                logger.info(f"    Row {idx + 1}: Y ≈ {y_pos}")

    # --- Digit decode ---
    gray_for_digits = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobao_rows_aligned = extrapolated.get("sobao_danh_rows_aligned", sobao_danh["sobao_danh_rows"])
    ma_de_rows_aligned = extrapolated.get("ma_de_rows_aligned", ma_de["ma_de_rows"])

    sbd_digits = evaluate_digit_rows_mean_darkness(gray_for_digits, sobao_rows_aligned, expected_cols=6)
    made_digits = evaluate_digit_rows_mean_darkness(gray_for_digits, ma_de_rows_aligned, expected_cols=3)

    logger.info("\n=== Mean-Darkness Digit Decode ===")
    print_digit_darkness_summary("SoBaoDanh", sbd_digits)
    print_digit_darkness_summary("MaDe", made_digits)

    # --- Visualization ---
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 1.5, 2

    part_configs = [
        ("part_i", "Part I (4)", (0, 255, 0)),
        ("part_ii", "Part II (8)", (0, 165, 255)),
        ("part_iii", "Part III (6)", (255, 0, 0)),
    ]
    for part_key, label, color in part_configs:
        for poly in parts[part_key]:
            cv2.polylines(overlay, [poly], True, color, 3)
        if parts[part_key]:
            min_y = min(cv2.boundingRect(p)[1] for p in parts[part_key])
            cv2.putText(overlay, label, (50, min_y - 20), font, font_scale, color, font_thickness)

    draw_rows_contours(overlay, sobao_danh["sobao_danh_rows"], (255, 128, 0), thickness=2)
    draw_rows_contours(overlay, ma_de["ma_de_rows"], (255, 255, 0), thickness=2)

    cv2.imwrite(f"{debug_prefix}_parts.jpg", overlay)
    logger.info(f"Parts visualization saved to {debug_prefix}_parts.jpg")

    # --- Grid drawing + fill evaluation ---
    logger.info(f"\n=== Drawing grids on all parts ===")
    combined_grid_image = img.copy()
    binary_threshold = data.get("binary")
    if binary_threshold is not None:
        logger.info("\nUsing binary threshold image for fill-ratio classification")

    part_i_evals: List[Dict[str, object]] = []
    if parts["part_i"]:
        logger.info(f"\n=== Part I: 4x10 grid (20% x, 10% y) ===")
        grid_result = extract_grid_from_boxes(
            combined_grid_image, boxes=parts["part_i"],
            grid_cols=4, grid_rows=10,
            start_offset_ratio_x=0.2, start_offset_ratio_y=0.1,
            end_offset_ratio_x=0.015, end_offset_ratio_y=0.015,
            grid_color=(0, 255, 0), grid_thickness=1,
        )
        combined_grid_image = grid_result["image_with_grid"]
        print_grid_info(grid_result["grid_info"])
        part_i_evals = _evaluate_section_fill(
            "Part I", binary_threshold, grid_result["grid_info"],
            fill_ratio_thresh=0.54, inner_margin_ratio=0.05,
            circle_radius_scale=0.6, circle_border_exclude_ratio=0.1,
        )

    part_ii_evals: List[Dict[str, object]] = []
    if parts["part_ii"]:
        logger.info(f"\n=== Part II: 2x4 grid (alternating offsets, 30% y, -5% bottom) ===")
        part_ii_count = len(parts["part_ii"])
        offset_ratios = [
            (0.3, 0.33) if (box_idx % 2 == 0) else (0.0, 0.33)
            for box_idx in range(part_ii_count)
        ]
        end_offset_x = [0.0] * part_ii_count
        end_offset_y = [0.03] * part_ii_count

        grid_result_ii = extract_grid_from_boxes_variable_offsets(
            combined_grid_image, boxes=parts["part_ii"],
            grid_cols=2, grid_rows=4,
            start_offset_ratios=offset_ratios,
            end_offset_ratios_x=end_offset_x,
            end_offset_ratios_y=end_offset_y,
            grid_color=(0, 165, 255), grid_thickness=1,
        )
        combined_grid_image = grid_result_ii["image_with_grid"]
        print_grid_info(
            grid_result_ii["grid_info"],
            detail_formatter=lambda info: (
                f"offset {info['offset_ratios']}, end_y -{info['end_offset_y'] * 100:.0f}%"
            ),
        )
        part_ii_evals = _evaluate_section_fill(
            "Part II", binary_threshold, grid_result_ii["grid_info"],
            fill_ratio_thresh=0.54, inner_margin_ratio=0.01,
            circle_radius_scale=0.6, circle_border_exclude_ratio=0.1,
        )

    part_iii_evals: List[Dict[str, object]] = []
    if parts["part_iii"]:
        logger.info(f"\n=== Part III: 4x12 grid with custom pattern (20% x, 10% y) ===")
        custom_pattern = [[0], [1, 2]] + [[0, 1, 2, 3] for _ in range(10)]
        grid_result_iii = extract_grid_from_boxes_custom_pattern(
            combined_grid_image, boxes=parts["part_iii"],
            grid_cols=4, grid_rows=12,
            start_offset_ratio_x=0.22, start_offset_ratio_y=0.16,
            end_offset_ratio_x=0.1, end_offset_ratio_y=0.015,
            grid_color=(255, 0, 0), grid_thickness=1,
            row_col_patterns=custom_pattern,
        )
        combined_grid_image = grid_result_iii["image_with_grid"]
        print_grid_info(
            grid_result_iii["grid_info"],
            detail_formatter=lambda info: f"pattern={info['pattern'][:2]}... (12 rows total)",
        )
        part_iii_evals = _evaluate_section_fill(
            "Part III", binary_threshold, grid_result_iii["grid_info"],
            fill_ratio_thresh=0.54, inner_margin_ratio=0.05,
            circle_radius_scale=0.6, circle_border_exclude_ratio=0.1,
        )

    # SBD / MaDe grid overlays
    if sobao_danh["sobao_danh_rows"]:
        logger.info(f"\n=== SoBaoDanh: drawing detected box grid ===")
        sobao_count = draw_rows_contours(
            combined_grid_image, sobao_danh["sobao_danh_rows"], (255, 128, 0), thickness=1,
        )
        logger.info(f"Grid drawn on {sobao_count} SoBaoDanh boxes")

    if ma_de["ma_de_rows"]:
        logger.info(f"\n=== MaDe: drawing detected box grid ===")
        ma_de_count = draw_rows_contours(
            combined_grid_image, ma_de["ma_de_rows"], (255, 255, 0), thickness=1,
        )
        logger.info(f"Grid drawn on {ma_de_count} MaDe boxes")

    combined_grid_image = draw_digit_darkness_overlay(
        combined_grid_image, sbd_digits, color=(0, 220, 255), alpha=0.40,
    )
    combined_grid_image = draw_digit_darkness_overlay(
        combined_grid_image, made_digits, color=(0, 255, 255), alpha=0.40,
    )

    all_evals = part_i_evals + part_ii_evals + part_iii_evals
    if all_evals:
        combined_grid_image = draw_filled_cells_overlay(
            combined_grid_image, all_evals, color=(0, 255, 0), alpha=0.35,
        )
        if binary_threshold is not None:
            binary_fillratio_path = f"{debug_prefix}_binary_fillratio_grid.jpg"
            draw_binary_fillratio_debug(binary_threshold, all_evals, binary_fillratio_path)
            logger.info(f"✓ Binary fill-ratio debug image saved to: {binary_fillratio_path}")

    combined_grid_path = f"{debug_prefix}_all_parts_with_grid.jpg"
    cv2.imwrite(combined_grid_path, combined_grid_image)
    logger.info(f"\n✓ Combined grid image saved to: {combined_grid_path}")


# =========================================================================
#  Public API — for Streamlit / web consumers
# =========================================================================


def process_image(
    image: np.ndarray,
    fill_ratio_part1: float = 0.55,
    fill_ratio_part2: float = 0.55,
    fill_ratio_part3: float = 0.55,
    debug_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """Run the full OMR pipeline on a pre-loaded image and return all results.

    This is the **primary entry point for the web UI** (``app.py``).
    Unlike :func:`run_pipeline`, it accepts an already-decoded
    ``np.ndarray`` instead of a file path, and returns a structured
    result dict rather than writing everything to disk.

    Args:
        image: Input BGR image as ``np.ndarray``.
        fill_ratio_part1: Bubble fill threshold for Part I (multiple-choice).
        fill_ratio_part2: Bubble fill threshold for Part II (true/false).
        fill_ratio_part3: Bubble fill threshold for Part III (numeric).
        debug_prefix: If set, intermediate debug images are written with this
            path prefix.  ``None`` skips all disk I/O.

    Returns:
        Dict with keys:

        - ``"preprocess_mode"`` — ``"base"`` or ``"clahe"`` (or ``"+topview-id"``)
        - ``"data"`` — raw morphology output (``boxes``, ``binary``, …)
        - ``"parts"`` — ``{"part_i": …, "part_ii": …, "part_iii": …, "all_parts": …}``
        - ``"sobao_danh"`` — SoBaoDanh detection result
        - ``"ma_de"`` — MaDe detection result
        - ``"split_x"`` — horizontal split coordinate
        - ``"extrapolated"`` — row extrapolation result
        - ``"sbd_digits"`` — digit-decode result for SBD
        - ``"made_digits"`` — digit-decode result for MaDe
        - ``"part_i_evals"`` — cell evaluation list for Part I
        - ``"part_ii_evals"`` — cell evaluation list for Part II
        - ``"part_iii_evals"`` — cell evaluation list for Part III
        - ``"parts_overlay"`` — BGR image with Part I/II/III outlines
        - ``"result_image"`` — BGR image with grids + filled-cell overlay
        - ``"binary_threshold"`` — binary mask used for fill scoring
    """
    if image is None or image.size == 0:
        raise ValueError("process_image() received an empty or None image array.")

    img = image.copy()
    page_h = img.shape[0]

    # ------------------------------------------------------------------
    # 1. Detection + CLAHE fallback
    # ------------------------------------------------------------------
    data, parts = _run_detection_pipeline(img, debug_prefix)
    preprocess_mode = "base"

    if len(parts["part_ii"]) < 8 or len(parts["part_iii"]) < 6:
        img_clahe = preprocess_clahe(img)
        data_clahe, parts_clahe = _run_detection_pipeline(img_clahe, None)

        base_score = _parts_score(parts, page_h)
        clahe_score = _parts_score(parts_clahe, page_h)
        if clahe_score > base_score + 0.05:
            data = data_clahe
            parts = parts_clahe
            preprocess_mode = "clahe"

    # ------------------------------------------------------------------
    # 2. Separate remaining boxes → SBD / MaDe candidates
    # ------------------------------------------------------------------
    part_box_set = set(id(box) for box in parts["all_parts"])
    remaining_boxes = [box for box in data["boxes"] if id(box) not in part_box_set]

    remaining_for_upper = _split_merged_boxes_for_grouping(
        remaining_boxes, split_wide=True, split_tall=False,
    )
    sbd_candidates, ma_de_candidates, split_x = _separate_upper_id_boxes(
        remaining_for_upper, parts["part_i"],
    )

    sobao_danh = detect_sobao_danh_boxes(
        sbd_candidates, boxes_per_row=6, max_rows=10,
        row_tolerance=45, size_tolerance_ratio=0.45, debug=False,
    )

    remaining_for_ma_de = _split_merged_boxes_for_grouping(
        ma_de_candidates, split_wide=False, split_tall=True,
    )
    ma_de = detect_ma_de_boxes(
        remaining_for_ma_de, boxes_per_row=3, max_rows=10,
        row_tolerance=35, size_tolerance_ratio=0.40, debug=False,
    )

    # ------------------------------------------------------------------
    # 2.0 Affine retry (if ID regions failed due to rotation/perspective)
    # ------------------------------------------------------------------
    affine_retry_threshold = 4
    topview_allowed_by_rows = (
        sobao_danh["row_count"] <= affine_retry_threshold
        and ma_de["row_count"] <= affine_retry_threshold
    )
    if topview_allowed_by_rows:
        corner_markers = detect_black_corner_markers(img, debug_prefix=None)
        affine_img, affine_matrix = _apply_affine_from_corner_markers(
            img, corner_markers.get("corners", {}),
        )
        if affine_img is not None and affine_matrix is not None:
            data_affine, parts_affine = _run_detection_pipeline(affine_img, None)

            part_box_set_affine = set(id(box) for box in parts_affine["all_parts"])
            remaining_boxes_affine = [
                box for box in data_affine["boxes"]
                if id(box) not in part_box_set_affine
            ]
            remaining_for_upper_affine = _split_merged_boxes_for_grouping(
                remaining_boxes_affine, split_wide=True, split_tall=False,
            )
            sbd_candidates_affine, ma_de_candidates_affine, split_x_affine = (
                _separate_upper_id_boxes(remaining_for_upper_affine, parts_affine["part_i"])
            )
            sobao_danh_affine = detect_sobao_danh_boxes(
                sbd_candidates_affine, boxes_per_row=6, max_rows=10,
                row_tolerance=45, size_tolerance_ratio=0.45, debug=False,
            )
            remaining_for_ma_de_affine = _split_merged_boxes_for_grouping(
                ma_de_candidates_affine, split_wide=False, split_tall=True,
            )
            ma_de_affine = detect_ma_de_boxes(
                remaining_for_ma_de_affine, boxes_per_row=3, max_rows=10,
                row_tolerance=35, size_tolerance_ratio=0.40, debug=False,
            )

            old_score = int(sobao_danh["row_count"]) + int(ma_de["row_count"])
            new_score = int(sobao_danh_affine["row_count"]) + int(ma_de_affine["row_count"])
            if new_score > old_score:
                img = affine_img
                data = data_affine
                parts = parts_affine
                split_x = split_x_affine
                sobao_danh = sobao_danh_affine
                ma_de = ma_de_affine
                preprocess_mode = f"{preprocess_mode}+topview-id"
                logger.info(
                    f"[Topview] Retry improved ID detection: score {old_score} -> {new_score} "
                    f"(SBD={sobao_danh['row_count']}, MaDe={ma_de['row_count']})"
                )
            else:
                logger.info(
                    f"[Topview] Retry no improvement: score {old_score} -> {new_score} "
                    f"(SBD={sobao_danh_affine['row_count']}, MaDe={ma_de_affine['row_count']})"
                )
        else:
            logger.info("[Topview] Retry skipped: not enough reliable corner markers")

    # ------------------------------------------------------------------
    # 2.1 Synthetic fallback for severely misaligned/faint SBD/MDT
    # ------------------------------------------------------------------
    id_fallback_row_threshold = 4
    if sobao_danh["row_count"] <= id_fallback_row_threshold:
        sobao_rows_synth = _build_synthetic_id_rows_fixed_image_position(
            image_shape=img.shape[:2], cols=6, rows=10,
            x_range_ratio=(0.745, 0.865),
            top_y_ratio=0.072,
            row_step_ratio=0.021,
        )
        if sobao_rows_synth:
            sobao_danh["sobao_danh_rows"] = sobao_rows_synth
            sobao_danh["sobao_danh"] = [b for row in sobao_rows_synth for b in row]
            sobao_danh["row_count"] = len(sobao_rows_synth)
            logger.info(f"[Fallback] SoBaoDanh rows <= {id_fallback_row_threshold}, applied synthetic grid.")

    if ma_de["row_count"] <= id_fallback_row_threshold:
        ma_de_rows_synth = _build_synthetic_id_rows_fixed_image_position(
            image_shape=img.shape[:2], cols=3, rows=10,
            x_range_ratio=(0.90, 0.96),
            top_y_ratio=0.072,
            row_step_ratio=0.021,
        )
        if ma_de_rows_synth:
            ma_de["ma_de_rows"] = ma_de_rows_synth
            ma_de["ma_de"] = [b for row in ma_de_rows_synth for b in row]
            ma_de["row_count"] = len(ma_de_rows_synth)
            logger.info(f"[Fallback] MaDe rows <= {id_fallback_row_threshold}, applied synthetic grid.")

    # ------------------------------------------------------------------
    # 3. MaDe completion (fill missing rows using SBD reference grid)
    # ------------------------------------------------------------------
    if ma_de["row_count"] < 10 and ma_de["ma_de_rows"] and sobao_danh["sobao_danh_rows"]:
        ref_positions = [
            int(round(float(np.mean([cv2.boundingRect(box)[1] for box in row]))))
            for row in sobao_danh["sobao_danh_rows"][:10] if row
        ]
        ma_de_rect_rows = []
        for row in ma_de["ma_de_rows"]:
            rects = sorted([cv2.boundingRect(box) for box in row], key=lambda r: r[0])
            if len(rects) == 3:
                ma_de_rect_rows.append(rects)

        if len(ref_positions) == 10 and ma_de_rect_rows:
            col_x = [int(round(float(np.median([rects[c][0] for rects in ma_de_rect_rows])))) for c in range(3)]
            col_w = [int(round(float(np.median([rects[c][2] for rects in ma_de_rect_rows])))) for c in range(3)]
            row_h = max(1, int(round(float(np.median([rects[0][3] for rects in ma_de_rect_rows])))))

            detected_rows_y = [int(round(float(np.mean([r[1] for r in rects])))) for rects in ma_de_rect_rows]
            align_tolerance = max(35, int(round(row_h * 1.2)))
            aligned_rows: List[Optional[List[np.ndarray]]] = [None] * 10

            used_ref = set()
            for rects, row_y in sorted(zip(ma_de_rect_rows, detected_rows_y), key=lambda t: t[1]):
                candidates_idx = sorted(range(10), key=lambda i: abs(ref_positions[i] - row_y))
                chosen = next(
                    (i for i in candidates_idx
                     if i not in used_ref and abs(ref_positions[i] - row_y) <= align_tolerance),
                    None,
                )
                if chosen is None:
                    continue
                aligned_rows[chosen] = [_rect_to_poly(rx, ry, rw, rh) for rx, ry, rw, rh in rects]
                used_ref.add(chosen)

            for idx in range(10):
                if aligned_rows[idx] is not None:
                    continue
                y_ref = ref_positions[idx]
                aligned_rows[idx] = [_rect_to_poly(col_x[c], y_ref, max(1, col_w[c]), row_h) for c in range(3)]

            ma_de_completed = [row for row in aligned_rows if row is not None]
            if len(ma_de_completed) == 10:
                ma_de["ma_de_rows"] = ma_de_completed
                ma_de["ma_de"] = [box for row in ma_de_completed for box in row]
                ma_de["row_count"] = 10

    # ------------------------------------------------------------------
    # 4. Extrapolation + digit decode
    # ------------------------------------------------------------------
    extrapolated = extrapolate_missing_rows(
        {"sobao_danh_rows": sobao_danh["sobao_danh_rows"], "ma_de_rows": ma_de["ma_de_rows"]},
        target_rows=10,
        debug=False,
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobao_rows_aligned = extrapolated.get("sobao_danh_rows_aligned", sobao_danh["sobao_danh_rows"])
    ma_de_rows_aligned = extrapolated.get("ma_de_rows_aligned", ma_de["ma_de_rows"])

    sbd_digits = evaluate_digit_rows_mean_darkness(gray, sobao_rows_aligned, expected_cols=6)
    made_digits = evaluate_digit_rows_mean_darkness(gray, ma_de_rows_aligned, expected_cols=3)

    # ------------------------------------------------------------------
    # 5. Parts overlay image
    # ------------------------------------------------------------------
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    part_configs = [
        ("part_i",   "Part I (4)",   (0, 255, 0)),
        ("part_ii",  "Part II (8)",  (0, 165, 255)),
        ("part_iii", "Part III (6)", (255, 0, 0)),
    ]
    for part_key, label, color in part_configs:
        for poly in parts[part_key]:
            cv2.polylines(overlay, [poly], True, color, 3)
        if parts[part_key]:
            min_y = min(cv2.boundingRect(p)[1] for p in parts[part_key])
            cv2.putText(overlay, label, (50, min_y - 20), font, 1.5, color, 2)

    draw_rows_contours(overlay, sobao_danh["sobao_danh_rows"], (255, 128, 0), thickness=2)
    draw_rows_contours(overlay, ma_de["ma_de_rows"],           (255, 255, 0), thickness=2)

    # ------------------------------------------------------------------
    # 6. Grid drawing + fill evaluation (per-part thresholds)
    # ------------------------------------------------------------------
    combined_grid_image = img.copy()
    binary_threshold = data.get("binary")

    part_i_evals: List[Dict[str, object]] = []
    if parts["part_i"]:
        grid_result = extract_grid_from_boxes(
            combined_grid_image, boxes=parts["part_i"],
            grid_cols=4, grid_rows=10,
            start_offset_ratio_x=0.2,   start_offset_ratio_y=0.1,
            end_offset_ratio_x=0.015,   end_offset_ratio_y=0.015,
            grid_color=(0, 255, 0),      grid_thickness=1,
        )
        combined_grid_image = grid_result["image_with_grid"]
        if binary_threshold is not None:
            part_i_evals = evaluate_grid_fill_from_binary(
                binary_image=binary_threshold,
                grid_info=grid_result["grid_info"],
                fill_ratio_thresh=float(fill_ratio_part1),
                inner_margin_ratio=0.05,
                mask_mode="hough-circle",
                circle_radius_scale=0.5,
                circle_border_exclude_ratio=0.0,
            )

    part_ii_evals: List[Dict[str, object]] = []
    if parts["part_ii"]:
        part_ii_count = len(parts["part_ii"])
        offset_ratios = [(0.3, 0.33) if i % 2 == 0 else (0.0, 0.33) for i in range(part_ii_count)]
        grid_result_ii = extract_grid_from_boxes_variable_offsets(
            combined_grid_image, boxes=parts["part_ii"],
            grid_cols=2, grid_rows=4,
            start_offset_ratios=offset_ratios,
            end_offset_ratios_x=[0.0] * part_ii_count,
            end_offset_ratios_y=[0.03] * part_ii_count,
            grid_color=(0, 165, 255), grid_thickness=1,
        )
        combined_grid_image = grid_result_ii["image_with_grid"]
        if binary_threshold is not None:
            part_ii_evals = evaluate_grid_fill_from_binary(
                binary_image=binary_threshold,
                grid_info=grid_result_ii["grid_info"],
                fill_ratio_thresh=float(fill_ratio_part2),
                inner_margin_ratio=0.05,
                mask_mode="hough-circle",
                circle_radius_scale=0.5,
                circle_border_exclude_ratio=0.0,
            )

    part_iii_evals: List[Dict[str, object]] = []
    if parts["part_iii"]:
        custom_pattern = [[0], [1, 2]] + [[0, 1, 2, 3] for _ in range(10)]
        grid_result_iii = extract_grid_from_boxes_custom_pattern(
            combined_grid_image, boxes=parts["part_iii"],
            grid_cols=4, grid_rows=12,
            start_offset_ratio_x=0.22, start_offset_ratio_y=0.16,
            end_offset_ratio_x=0.1,    end_offset_ratio_y=0.015,
            grid_color=(255, 0, 0),     grid_thickness=1,
            row_col_patterns=custom_pattern,
        )
        combined_grid_image = grid_result_iii["image_with_grid"]
        if binary_threshold is not None:
            part_iii_evals = evaluate_grid_fill_from_binary(
                binary_image=binary_threshold,
                grid_info=grid_result_iii["grid_info"],
                fill_ratio_thresh=float(fill_ratio_part3),
                inner_margin_ratio=0.05,
                mask_mode="hough-circle",
                circle_radius_scale=0.5,
                circle_border_exclude_ratio=0.0,
            )

    # SBD / MaDe grid overlays on result image
    draw_rows_contours(combined_grid_image, sobao_danh["sobao_danh_rows"], (255, 128, 0), thickness=1)
    draw_rows_contours(combined_grid_image, ma_de["ma_de_rows"],           (255, 255, 0), thickness=1)

    combined_grid_image = draw_digit_darkness_overlay(combined_grid_image, sbd_digits,  color=(0, 220, 255), alpha=0.40)
    combined_grid_image = draw_digit_darkness_overlay(combined_grid_image, made_digits, color=(0, 255, 255), alpha=0.40)

    all_evals = part_i_evals + part_ii_evals + part_iii_evals
    if all_evals:
        combined_grid_image = draw_filled_cells_overlay(combined_grid_image, all_evals, color=(0, 255, 0), alpha=0.35)

    # ------------------------------------------------------------------
    # 7. Return structured result dict
    # ------------------------------------------------------------------
    return {
        "preprocess_mode":  preprocess_mode,
        "data":             data,
        "parts":            parts,
        "sobao_danh":       sobao_danh,
        "ma_de":            ma_de,
        "split_x":          split_x,
        "extrapolated":     extrapolated,
        "sbd_digits":       sbd_digits,
        "made_digits":      made_digits,
        "part_i_evals":     part_i_evals,
        "part_ii_evals":    part_ii_evals,
        "part_iii_evals":   part_iii_evals,
        "parts_overlay":    overlay,
        "result_image":     combined_grid_image,
        "binary_threshold": binary_threshold,
    }
