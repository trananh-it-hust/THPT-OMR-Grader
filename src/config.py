"""
config.py — Centralized constants and default parameters.

Every numeric threshold, ratio, and default value used across the pipeline
is defined here.  Processing functions receive these values as arguments
(with defaults matching these constants) so they remain testable in isolation.

See also: CoreRules/ARCH_RULES.md §5 (Configuration & Parameters).
"""

from __future__ import annotations

# =========================================================================
#  Morphology — line / box detection
# =========================================================================

VERTICAL_SCALE: float = 0.015
"""Fraction of image height used as vertical morphology kernel length."""

HORIZONTAL_SCALE: float = 0.015
"""Fraction of image width used as horizontal morphology kernel length."""

MIN_LINE_LENGTH: int = 50
"""Minimum pixel length to keep a detected line component."""

VERTICAL_ROW_TOLERANCE: int = 10
"""Y-tolerance (px) when aligning vertical line lengths by row."""

BLOCK_SIZE: int = 35
"""Adaptive threshold block size."""

BLOCK_OFFSET: int = 7
"""Adaptive threshold constant (C) subtracted from mean."""

MIN_BOX_AREA: int = 200
"""Minimum contour area (px²) to qualify as a valid closed box."""

MIN_CONTAINER_AREA: int = 5000
"""Minimum area for a large container block (Part I, II, III)."""

MIN_BOX_WIDTH: int = 15
"""Minimum contour width (px) to qualify as a valid closed box."""

MIN_BOX_HEIGHT: int = 15
"""Minimum contour height (px) to qualify as a valid closed box."""

CLOSE_KERNEL_SIZE: int = 3
"""Morphological closing kernel size to bridge small line gaps."""

# =========================================================================
#  Box grouping — Part I / II / III
# =========================================================================

GROUP_ROW_TOLERANCE: int = 30
"""Y-tolerance (px) when grouping container boxes into rows."""

GROUP_SIZE_TOLERANCE_RATIO: float = 0.15
"""Max relative area deviation for container box uniformity check."""

GROUP_MIN_BOXES_PER_GROUP: int = 3
"""Minimum boxes in a row group to consider it a valid part row."""

# =========================================================================
#  SoBaoDanh (Student ID)
# =========================================================================

SBD_BOXES_PER_ROW: int = 6
"""Expected bubble columns per SoBaoDanh row."""

SBD_MAX_ROWS: int = 10
"""Maximum SoBaoDanh digit rows."""

SBD_ROW_TOLERANCE: int = 25
"""Y-tolerance (px) for grouping SoBaoDanh bubbles into rows."""

SBD_SIZE_TOLERANCE_RATIO: float = 0.35
"""Max relative area deviation within a SoBaoDanh row."""

# =========================================================================
#  MaDe (Exam Code)
# =========================================================================

MADE_BOXES_PER_ROW: int = 3
"""Expected bubble columns per MaDe row."""

MADE_MAX_ROWS: int = 10
"""Maximum MaDe digit rows."""

MADE_ROW_TOLERANCE: int = 20
"""Y-tolerance (px) for grouping MaDe bubbles into rows."""

MADE_SIZE_TOLERANCE_RATIO: float = 0.3
"""Max relative area deviation within a MaDe row."""

# =========================================================================
#  Extrapolation
# =========================================================================

TARGET_ROWS: int = 10
"""Target number of ID rows after extrapolation for both SBD and MaDe."""

# =========================================================================
#  Fill evaluation — bubble grading
# =========================================================================

FILL_RATIO_THRESH: float = 0.54
"""Bubble classified as filled when white-pixel ratio ≥ this value."""

CIRCLE_RADIUS_SCALE: float = 0.60
"""Scale factor applied to estimated cell half-size to derive bubble radius."""

CIRCLE_BORDER_EXCLUDE: float = 0.10
"""Fraction of radius excluded at the edge to reduce border-noise impact."""

MASK_MODE: str = "hough-circle"
"""Default mask strategy for fill evaluation: 'quad', 'circle', or 'hough-circle'."""

# =========================================================================
#  Grid offsets — Part I
# =========================================================================

PART_I_START_OFFSET_X: float = 0.20
PART_I_START_OFFSET_Y: float = 0.10
PART_I_END_OFFSET_X: float = 0.015
PART_I_END_OFFSET_Y: float = 0.015
PART_I_GRID_COLS: int = 4
PART_I_GRID_ROWS: int = 10
PART_I_INNER_MARGIN: float = 0.05

# =========================================================================
#  Grid offsets — Part II
# =========================================================================

PART_II_GRID_COLS: int = 2
PART_II_GRID_ROWS: int = 4
PART_II_START_EVEN: tuple = (0.3, 0.33)
"""Start offset (x, y) for even-indexed Part II boxes."""
PART_II_START_ODD: tuple = (0.0, 0.33)
"""Start offset (x, y) for odd-indexed Part II boxes."""
PART_II_END_Y: float = 0.03
PART_II_INNER_MARGIN: float = 0.01

# =========================================================================
#  Grid offsets — Part III
# =========================================================================

PART_III_START_OFFSET_X: float = 0.22
PART_III_START_OFFSET_Y: float = 0.16
PART_III_END_OFFSET_X: float = 0.10
PART_III_END_OFFSET_Y: float = 0.015
PART_III_GRID_COLS: int = 4
PART_III_GRID_ROWS: int = 12
PART_III_INNER_MARGIN: float = 0.05

# =========================================================================
#  Digit decode — mean darkness
# =========================================================================

DARKNESS_RADIUS_SCALE: float = 0.36
"""Scale factor for the measurement circle inside a digit bubble."""

DARKNESS_ABS_THRESHOLD: float = 205.0
"""A column's darkest bubble must be below this gray value to count."""

DARKNESS_MIN_SECOND_GAP: float = 4.0
"""Required gap between darkest and second-darkest bubble in a column."""

DARKNESS_MIN_MEDIAN_GAP: float = 12.0
"""Required gap between darkest bubble and column median."""

DARKNESS_MIN_ABS_MEDIAN_GAP: float = 5.0
"""Required absolute gap between darkest bubble and column median."""

DARKNESS_MIN_ABS_SECOND_GAP: float = 4.0
"""Required absolute gap between darkest and second-darkest bubble."""

# =========================================================================
#  ID fallback — synthetic grid
# =========================================================================

ID_FALLBACK_ROW_THRESHOLD: int = 4
"""If detected rows ≤ this, build a synthetic 10-row grid via fixed position."""

ID_GRID_TOP_RATIO: float = 0.072
"""Top-Y ratio for the fixed-position synthetic ID grid."""

ID_GRID_ROW_STEP_RATIO: float = 0.0215
"""Row step ratio for the fixed-position synthetic ID grid."""

ID_SBD_X_RANGE_RATIO: tuple = (0.71, 0.83)
"""X-range ratio (left, right) for synthetic SoBaoDanh grid."""

ID_MADE_X_RANGE_RATIO: tuple = (0.86, 0.94)
"""X-range ratio (left, right) for synthetic MaDe grid."""

# =========================================================================
#  Affine retry
# =========================================================================

AFFINE_RETRY_THRESHOLD: int = 4
"""Row threshold: only attempt topview affine when both SBD and MaDe rows ≤ this."""
