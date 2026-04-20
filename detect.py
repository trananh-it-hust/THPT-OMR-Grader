"""
detect.py — Thin backward-compatibility bridge.

This file re-exports every public symbol from the ``src/`` package so that
existing code (e.g. ``import detect as core``) continues to work unchanged.

All actual logic now lives in ``src/``.  Run ``python detect.py 0015`` to
execute the full pipeline via ``src.pipeline.run_pipeline()``.
"""

from __future__ import annotations

import argparse

# Re-export all public symbols from src modules ---------------------------

# Preprocessing
from src.preprocessing import (
    normalize_image_stem as _normalize_image_stem,
    load_image,
    preprocess_clahe as _preprocess_clahe,
)

# Morphology
from src.morphology import (
    detect_grid_points,
    detect_boxes_from_morph_lines,
    detect_black_corner_markers,
)

# Box grouping
from src.box_grouping import (
    group_boxes_into_parts,
    detect_sobao_danh_boxes,
    detect_ma_de_boxes,
    extrapolate_missing_rows,
    _rect_to_poly,
    _build_box_info,
    _group_box_info_by_row,
    _is_uniform_size_group,
    _filter_rows_by_global_size_consistency,
    _trim_rows_to_consistent_window,
    _split_merged_boxes_for_grouping,
    _separate_upper_id_boxes,
    _build_synthetic_id_rows_from_part_i,
    _build_synthetic_id_rows_fixed_image_position,
    _apply_affine_from_corner_markers,
)

# Grid extraction
from src.grid_extraction import (
    _order_quad_points,
    _box_to_quad,
    _lerp_point,
    _point_on_quad,
    _inner_quad,
    _draw_grid_lines_on_quad,
    _draw_grid_cells_with_pattern,
    _validate_box_dims,
    _build_grid_info,
    _quad_cell_at,
    _shrink_quad_towards_center,
    extract_grid_from_boxes,
    extract_grid_from_boxes_variable_offsets,
    extract_grid_from_boxes_custom_pattern,
)

# Fill evaluation
from src.fill_evaluation import (
    _fill_ratio_in_quad,
    _estimate_circle_from_quad,
    _circle_polygon,
    _detect_single_circle_hough_in_quad,
    _fill_ratio_in_circle,
    evaluate_grid_fill_from_binary,
)

# Digit decode
from src.digit_decode import (
    _mean_darkness_in_box_circle,
    evaluate_digit_rows_mean_darkness,
)

# Debug draw
from src.debug_draw import (
    draw_filled_cells_overlay,
    draw_binary_fillratio_debug,
    draw_digit_darkness_overlay,
    draw_rows_contours as _draw_rows_contours,
    print_fill_summary as _print_fill_summary,
    print_digit_darkness_summary as _print_digit_darkness_summary,
    print_grid_info as _print_grid_info,
)

# Pipeline
from src.pipeline import run_pipeline as _demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phát hiện và vẽ lưới đáp án từ ảnh phiếu.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Định danh ảnh, ví dụ: 0015, 31, PhieuQG.0031, hoặc PhieuQG/PhieuQG.0031.jpg",
    )
    parser.add_argument(
        "--image",
        dest="image_opt",
        default=None,
        help="Định danh ảnh (chấp nhận cùng định dạng như tham số vị trí image).",
    )
    args = parser.parse_args()
    _demo(image_arg=args.image_opt or args.image)
