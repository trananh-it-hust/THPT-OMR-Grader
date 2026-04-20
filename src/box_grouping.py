"""
box_grouping.py — Part I/II/III grouping, SBD/MaDe detection, and row extrapolation.

Contains the complex heuristic logic for:
- Grouping detected boxes into Part I (4 boxes), Part II (8 boxes), Part III (6 boxes).
- Detecting SoBaoDanh (6 cols × 10 rows) and MaDe (3 cols × 10 rows) regions.
- Extrapolating missing ID rows based on reference Y positions.
- Helper functions for box info building, row grouping, size consistency, etc.

Pipeline position: Step 3-4 (after morphology / before grid extraction).
"""

from __future__ import annotations

from src.log_config import logger

import itertools
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src import config


def _rect_to_poly(x: int, y: int, w: int, h: int) -> np.ndarray:
    # Hàm phụ chuyển đổi biểu diễn hình học giữa các dạng.
    """
    Chuyển bounding rectangle sang contour polygon 4 đỉnh.

    Args:
        x: Tọa độ trái.
        y: Tọa độ trên.
        w: Chiều rộng.
        h: Chiều cao.

    Returns:
        Contour đa giác hình chữ nhật dạng OpenCV.
    """
    return np.array(
        [
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]],
        ],
        dtype=np.int32,
    )


def _build_box_info(boxes: List[np.ndarray]) -> List[Dict[str, object]]:
    # Hàm phụ dựng cấu trúc dữ liệu trung gian dùng lại nhiều nơi.
    """
    Dựng metadata cơ bản cho từng box để thuận tiện cho các bước gom nhóm.

    Args:
        boxes: Danh sách contour box.

    Returns:
        Danh sách dictionary chứa vị trí, kích thước, tâm Y và diện tích mỗi box.
    """
    box_info: List[Dict[str, object]] = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        box_info.append(
            {
                "box": box,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "center_y": y + h // 2,
                "area": w * h,
            }
        )
    return box_info


def _group_box_info_by_row(
    box_info: List[Dict[str, object]],
    row_tolerance: int,
) -> List[List[Dict[str, object]]]:
    # Hàm phụ gom nhóm dữ liệu theo quy tắc hình học.
    """
    Gom danh sách box-info thành các hàng dựa trên độ gần nhau theo trục Y.

    Args:
        box_info: Danh sách metadata box.
        row_tolerance: Ngưỡng lệch theo Y để coi cùng một hàng.

    Returns:
        Danh sách nhóm hàng, mỗi phần tử là một list box-info.
    """
    sorted_info = sorted(box_info, key=lambda b: b["center_y"])
    groups: List[List[Dict[str, object]]] = []
    for box in sorted_info:
        placed = False
        for group in groups:
            mean_y = np.mean([b["center_y"] for b in group])
            if abs(box["center_y"] - mean_y) <= row_tolerance:
                group.append(box)
                placed = True
                break
        if not placed:
            groups.append([box])

    groups.sort(key=lambda g: np.mean([b["center_y"] for b in g]))
    return groups


def _is_uniform_size_group(group: List[Dict[str, object]], size_tolerance_ratio: float) -> bool:
    # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
    """
    Kiểm tra một nhóm box có đồng đều kích thước theo diện tích hay không.

    Args:
        group: Nhóm box-info cần kiểm tra.
        size_tolerance_ratio: Ngưỡng lệch tương đối tối đa cho phép.

    Returns:
        `True` nếu nhóm có độ đồng đều đạt yêu cầu.
    """
    areas = [b["area"] for b in group]
    mean_area = np.mean(areas)
    return max([abs(a - mean_area) / mean_area for a in areas]) <= size_tolerance_ratio if mean_area > 0 else True


def _filter_rows_by_global_size_consistency(
    rows: List[List[np.ndarray]],
    size_tolerance_ratio: float,
    debug: bool = False,
) -> List[List[np.ndarray]]:
    # Hàm phụ lọc nhiễu hoặc loại bỏ phần tử không đạt điều kiện.
    """
    Lọc các hàng có kích thước trung bình lệch quá xa so với mặt bằng chung.

    Args:
        rows: Danh sách hàng box.
        size_tolerance_ratio: Ngưỡng lệch tương đối cho phép.
        debug: Bật log chi tiết khi lọc.

    Returns:
        Danh sách hàng sau lọc nhiễu kích thước.
    """
    if not rows:
        return rows

    all_areas: List[int] = []
    for row in rows:
        for box in row:
            _, _, w, h = cv2.boundingRect(box)
            all_areas.append(w * h)

    if not all_areas:
        return rows

    global_mean_area = np.mean(all_areas)
    if global_mean_area <= 0:
        return rows

    filtered_rows: List[List[np.ndarray]] = []
    for row_idx, row in enumerate(rows):
        row_areas: List[int] = []
        for box in row:
            _, _, w, h = cv2.boundingRect(box)
            row_areas.append(w * h)

        row_mean_area = np.mean(row_areas) if row_areas else 0
        row_variance = abs(row_mean_area - global_mean_area) / global_mean_area if global_mean_area > 0 else 0
        if row_variance <= size_tolerance_ratio:
            filtered_rows.append(row)
        elif debug:
            logger.info(
                f"  ✗ Row {row_idx + 1} filtered out: row_avg={int(row_mean_area)}, "
                f"global_avg={int(global_mean_area)}, variance={row_variance:.2f}"
            )

    return filtered_rows


def _trim_rows_to_consistent_window(
    rows: List[List[np.ndarray]],
    max_rows: int,
) -> List[List[np.ndarray]]:
    # Hàm phụ cắt/chọn cửa sổ dữ liệu ổn định nhất.
    """
    Chọn cửa sổ liên tiếp có hình học ổn định nhất khi số hàng vượt giới hạn.

    Args:
        rows: Danh sách hàng box đầu vào.
        max_rows: Số hàng tối đa cần giữ.

    Returns:
        Danh sách hàng đã được cắt về cửa sổ ổn định nhất.
    """
    if len(rows) <= max_rows:
        return rows

    row_infos = []
    for row in rows:
        y_vals = [cv2.boundingRect(box)[1] for box in row]
        a_vals = [cv2.boundingRect(box)[2] * cv2.boundingRect(box)[3] for box in row]
        row_infos.append(
            {
                "row": row,
                "mean_y": float(np.mean(y_vals)) if y_vals else 0.0,
                "mean_area": float(np.mean(a_vals)) if a_vals else 0.0,
            }
        )

    row_infos.sort(key=lambda r: r["mean_y"])
    best_start = 0
    best_score = float("inf")

    for start in range(0, len(row_infos) - max_rows + 1):
        window = row_infos[start:start + max_rows]
        ys = [r["mean_y"] for r in window]
        areas = [r["mean_area"] for r in window]

        if len(ys) >= 2:
            spacings = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            mean_spacing = float(np.mean(spacings)) if spacings else 0.0
            spacing_cv = float(np.std(spacings) / mean_spacing) if mean_spacing > 0 else 1.0
        else:
            spacing_cv = 1.0

        mean_area = float(np.mean(areas)) if areas else 0.0
        area_cv = float(np.std(areas) / mean_area) if mean_area > 0 else 1.0
        score = spacing_cv + 0.2 * area_cv - 0.0005 * float(np.mean(ys))
        if score < best_score:
            best_score = score
            best_start = start

    return [r["row"] for r in row_infos[best_start:best_start + max_rows]]


def _split_merged_boxes_for_grouping(
    boxes: List[np.ndarray],
    split_wide: bool = False,
    split_tall: bool = False,
    min_area: int = 400,
    max_area: int = 10000,
) -> List[np.ndarray]:
    # Hàm phụ tách phần tử gộp để phục hồi cấu trúc mong muốn.
    """
    Tách các box bubble có khả năng bị dính để nhóm hàng/cột chính xác hơn.

    Args:
        boxes: Danh sách contour box đầu vào.
        split_wide: Bật tách box dính theo chiều ngang.
        split_tall: Bật tách box dính theo chiều dọc.
        min_area: Diện tích nhỏ nhất của box xét tách.
        max_area: Diện tích lớn nhất của box xét tách.

    Returns:
        Danh sách box sau khi tách các contour dính.
    """
    if not boxes:
        return boxes

    rects = [cv2.boundingRect(b) for b in boxes]
    sample_ws = [w for x, y, w, h in rects if min_area <= (w * h) <= max_area]
    sample_hs = [h for x, y, w, h in rects if min_area <= (w * h) <= max_area]

    if not sample_ws or not sample_hs:
        return boxes

    median_w = float(np.median(sample_ws))
    median_h = float(np.median(sample_hs))

    out: List[np.ndarray] = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        area = w * h

        if split_wide and min_area <= area <= max_area and w >= 1.75 * median_w and h <= 1.5 * median_h:
            w_left = w // 2
            w_right = w - w_left
            left_poly = np.array([[[x, y]], [[x + w_left, y]], [[x + w_left, y + h]], [[x, y + h]]], dtype=np.int32)
            right_poly = np.array([[[x + w_left, y]], [[x + w, y]], [[x + w, y + h]], [[x + w_left, y + h]]], dtype=np.int32)
            out.extend([left_poly, right_poly])
            continue

        if split_tall and min_area <= area <= max_area and h >= 1.75 * median_h and w <= 1.5 * median_w:
            h_top = h // 2
            h_bottom = h - h_top
            top_poly = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h_top]], [[x, y + h_top]]], dtype=np.int32)
            bottom_poly = np.array([[[x, y + h_top]], [[x + w, y + h_top]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
            out.extend([top_poly, bottom_poly])
            continue

        out.append(box)

    return out


def _separate_upper_id_boxes(
    boxes: List[np.ndarray],
    part_i_boxes: List[np.ndarray],
    top_margin: int = 10,
    min_area: int = 350,
    max_area: int = 6000,
    row_tolerance: int = 20,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    # Hàm phụ chia dữ liệu thành các nhánh xử lý độc lập.
    """
    Tách box vùng ID phía trên thành SoBaoDanh (trái) và Mã đề (phải).

    Args:
        boxes: Danh sách box ứng viên vùng phía trên.
        part_i_boxes: Danh sách box phần Part I để suy ra ngưỡng Y phía trên.
        top_margin: Biên an toàn tính từ đỉnh Part I.
        min_area: Diện tích tối thiểu cho box ID.
        max_area: Diện tích tối đa cho box ID.
        row_tolerance: Ngưỡng gom các box cùng hàng theo Y.

    Returns:
        Tuple `(sbd_boxes, ma_de_boxes, split_x)`.
    """
    if not boxes:
        return [], [], 0.0
    if not part_i_boxes:
        return boxes, [], 0.0

    part_i_top = min(cv2.boundingRect(b)[1] for b in part_i_boxes)
    y_limit = part_i_top - top_margin

    upper_items: List[Tuple[np.ndarray, float, float]] = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        area = w * h
        if y < y_limit and min_area <= area <= max_area:
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            upper_items.append((box, cx, cy))

    if not upper_items:
        return [], [], 0.0

    rows: List[List[Tuple[np.ndarray, float, float]]] = []
    for item in sorted(upper_items, key=lambda t: t[2]):
        placed = False
        for row in rows:
            mean_cy = float(np.mean([z[2] for z in row]))
            if abs(item[2] - mean_cy) <= row_tolerance:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    split_candidates: List[float] = []
    for row in rows:
        if len(row) < 8:
            continue
        xs = sorted([z[1] for z in row])
        gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        if not gaps:
            continue
        max_gap_idx = int(np.argmax(gaps))
        if gaps[max_gap_idx] > 20:
            split_candidates.append((xs[max_gap_idx] + xs[max_gap_idx + 1]) / 2.0)

    if split_candidates:
        split_x = float(np.median(split_candidates))
    else:
        split_x = float(np.percentile([z[1] for z in upper_items], 70))

    sbd_boxes = [z[0] for z in upper_items if z[1] < split_x]
    ma_de_boxes = [z[0] for z in upper_items if z[1] >= split_x]
    return sbd_boxes, ma_de_boxes, split_x


def group_boxes_into_parts(
    boxes: List[np.ndarray],
    row_tolerance: int = 30,
    size_tolerance_ratio: float = 0.15,
    min_boxes_per_group: int = 3,
) -> Dict[str, object]:
    # Gom nhóm dữ liệu để tạo cấu trúc phục vụ xử lý phía sau.
    """
    Gom nhóm box phát hiện được thành ba phần chính của phiếu: Part I, Part II, Part III.

    Chiến lược:
    1. Lọc các box ứng viên theo diện tích và loại trùng bằng IoU.
    2. Gom box theo hàng theo trục Y.
    3. Nhận diện lần lượt Part I (4), Part II (8), Part III (6) theo đặc trưng hình học.
    4. Áp dụng nhiều nhánh fallback để phục hồi khi scan thiếu/méo contour.

    Args:
        boxes: Danh sách contour box phát hiện từ bước morphology.
        row_tolerance: Dung sai gom nhóm theo hàng.
        size_tolerance_ratio: Dung sai đồng đều kích thước trong một nhóm.
        min_boxes_per_group: Số box tối thiểu để coi là một nhóm hợp lệ.

    Returns:
        Dictionary gồm `part_i`, `part_ii`, `part_iii`, và `all_parts`.
    """
    if not boxes:
        return {"part_i": [], "part_ii": [], "part_iii": [], "all_parts": []}

    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        union = (aw * ah) + (bw * bh) - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def _rect_to_poly(x: int, y: int, w: int, h: int) -> np.ndarray:
        # Hàm phụ chuyển đổi biểu diễn hình học giữa các dạng.
        return np.array(
            [
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]],
            ],
            dtype=np.int32,
        )

    def _is_uniform_size(group: List[Dict[str, object]], tol: float) -> bool:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        areas = [int(b["area"]) for b in group]
        if not areas:
            return False
        mean_area = float(np.mean(areas))
        if mean_area <= 0:
            return True
        max_rel = max(abs(a - mean_area) / mean_area for a in areas)
        return max_rel <= tol

    def _select_best_subset(group: List[Dict[str, object]], expected_count: int) -> List[Dict[str, object]]:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        if len(group) < expected_count:
            return []
        sorted_group = sorted(group, key=lambda b: int(b["x"]))
        if len(sorted_group) == expected_count:
            return sorted_group

        best_subset: List[Dict[str, object]] = []
        best_score = float("inf")

        for idx_tuple in itertools.combinations(range(len(sorted_group)), expected_count):
            subset = [sorted_group[idx] for idx in idx_tuple]
            areas = [float(b["area"]) for b in subset]
            mean_area = float(np.mean(areas))
            if mean_area <= 0:
                continue
            size_var = max(abs(a - mean_area) / mean_area for a in areas)

            xs = [int(b["x"]) for b in subset]
            widths = [int(b["w"]) for b in subset]
            centers = [x + w // 2 for x, w in zip(xs, widths)]
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            if gaps:
                gap_mean = float(np.mean(gaps))
                gap_var = float(np.std(gaps) / gap_mean) if gap_mean > 0 else 1.0
            else:
                gap_var = 0.0

            score = size_var + 0.25 * gap_var
            if score < best_score:
                best_score = score
                best_subset = subset

        return best_subset

    # Extract bounding box info
    box_info: List[Dict[str, object]] = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        area = w * h
        box_info.append({
            "box": box,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_y": y + h // 2,
            "area": area,
        })

    # Keep only large containers to avoid mixing with tiny answer bubbles.
    all_areas = np.array([int(b["area"]) for b in box_info], dtype=np.float64)
    # The 80th percentile fluctuates massively between 12085.80 and 35880.40 depending on scan.
    area_threshold = float(config.MIN_CONTAINER_AREA)
    candidates = [b for b in box_info if float(b["area"]) >= area_threshold]
    if not candidates:
        return {"part_i": [], "part_ii": [], "part_iii": [], "all_parts": []}

    # De-duplicate heavily overlapping boxes (common when both inner/outer borders are detected).
    candidates.sort(key=lambda b: int(b["area"]), reverse=True)
    deduped: List[Dict[str, object]] = []
    for cand in candidates:
        r1 = (int(cand["x"]), int(cand["y"]), int(cand["w"]), int(cand["h"]))
        replaced = False
        for idx, keep in enumerate(deduped):
            r2 = (int(keep["x"]), int(keep["y"]), int(keep["w"]), int(keep["h"]))
            if _bbox_iou(r1, r2) >= 0.85:
                if int(cand["area"]) > int(keep["area"]):
                    deduped[idx] = cand
                replaced = True
                break
        if not replaced:
            deduped.append(cand)

    # Group boxes by row (Y proximity). Use adaptive tolerance for perspective skew.
    median_h = float(np.median([int(b["h"]) for b in deduped])) if deduped else 0.0
    row_tol = max(row_tolerance, 45, int(median_h * 0.25))
    deduped.sort(key=lambda b: int(b["center_y"]))
    groups: List[List[Dict[str, object]]] = []
    for box in deduped:
        placed = False
        for group in groups:
            mean_y = float(np.mean([int(b["center_y"]) for b in group]))
            if abs(int(box["center_y"]) - mean_y) <= row_tol:
                group.append(box)
                placed = True
                break
        if not placed:
            groups.append([box])

    groups.sort(key=lambda g: float(np.mean([int(b["center_y"]) for b in g])))

    estimated_page_height = max(int(b["y"]) + int(b["h"]) for b in deduped)

    def _group_center_ratio(group: List[Dict[str, object]]) -> float:
        # Hàm phụ gom nhóm dữ liệu theo quy tắc hình học.
        cy = float(np.mean([int(b["center_y"]) for b in group]))
        return (cy / float(estimated_page_height)) if estimated_page_height > 0 else 0.0

    parts = {"part_i": [], "part_ii": [], "part_iii": []}
    group_idx = 0

    # Part I: 4 large boxes in one row.
    for i in range(group_idx, len(groups)):
        group = groups[i]
        center_ratio = _group_center_ratio(group)
        if center_ratio < 0.25 or center_ratio > 0.65:
            continue
        subset = _select_best_subset(group, expected_count=4)
        if subset and _is_uniform_size(subset, size_tolerance_ratio * 2.0):
            parts["part_i"] = [b["box"] for b in subset]
            group_idx = i + 1
            break

    # Fallback: some scans miss one Part I contour; recover from 3 evenly spaced boxes.
    if not parts["part_i"]:
        estimated_page_width = max(int(b["x"]) + int(b["w"]) for b in box_info) if box_info else 0
        for i in range(0, len(groups)):
            group = groups[i]
            center_ratio = _group_center_ratio(group)
            if center_ratio < 0.25 or center_ratio > 0.65:
                continue


            subset3 = _select_best_subset(group, expected_count=3)
            if not subset3:
                continue
            if not _is_uniform_size(subset3, size_tolerance_ratio * 2.5):
                continue

            subset3 = sorted(subset3, key=lambda b: int(b["x"]))
            xs = [int(b["x"]) for b in subset3]
            ws = [int(b["w"]) for b in subset3]
            ys = [int(b["y"]) for b in subset3]
            hs = [int(b["h"]) for b in subset3]
            centers = [x + w // 2 for x, w in zip(xs, ws)]
            gaps = [centers[j + 1] - centers[j] for j in range(len(centers) - 1)]
            if not gaps:
                continue

            mean_gap = float(np.mean(gaps))
            if mean_gap <= 0:
                continue
            gap_cv = float(np.std(gaps) / mean_gap) if mean_gap > 0 else 1.0
            if gap_cv > 0.2:
                continue

            mean_w = int(round(float(np.mean(ws))))
            mean_h = int(round(float(np.mean(hs))))
            mean_y = int(round(float(np.mean(ys))))

            left_margin = xs[0]
            right_edge = xs[-1] + ws[-1]
            right_margin = max(0, estimated_page_width - right_edge)

            # Choose missing side based on available margin.
            if left_margin > right_margin:
                missing_center = centers[0] - int(round(mean_gap))
            else:
                missing_center = centers[-1] + int(round(mean_gap))

            missing_x = int(round(missing_center - mean_w / 2.0))
            missing_x = max(0, missing_x)
            missing_box = _rect_to_poly(missing_x, mean_y, max(1, mean_w), max(1, mean_h))

            # Prefer a real detected contour close to the inferred missing slot.
            inferred_rect = cv2.boundingRect(missing_box)
            used_ids = set(id(item["box"]) for item in subset3)
            best_detected_box: Optional[np.ndarray] = None
            best_detected_score = -1.0
            for cand in box_info:
                if id(cand["box"]) in used_ids:
                    continue

                cw = int(cand["w"])
                ch = int(cand["h"])
                if mean_w > 0 and not (0.55 * mean_w <= cw <= 1.6 * mean_w):
                    continue
                if mean_h > 0 and not (0.55 * mean_h <= ch <= 1.6 * mean_h):
                    continue

                cand_rect = (int(cand["x"]), int(cand["y"]), cw, ch)
                overlap = _bbox_iou(inferred_rect, cand_rect)

                # If IoU is weak, still allow by center proximity.
                ix, iy, iw, ih = inferred_rect
                icx = ix + (iw / 2.0)
                icy = iy + (ih / 2.0)
                ccx = int(cand["x"]) + (cw / 2.0)
                ccy = int(cand["y"]) + (ch / 2.0)
                center_dist = ((icx - ccx) ** 2 + (icy - ccy) ** 2) ** 0.5
                dist_score = 1.0 / (1.0 + center_dist)

                score = overlap + 0.15 * dist_score
                if score > best_detected_score:
                    best_detected_score = score
                    best_detected_box = cand["box"]

            if best_detected_box is not None and best_detected_score >= 0.25:
                missing_box = best_detected_box

            recovered = [b["box"] for b in subset3]
            recovered.append(missing_box)
            recovered.sort(key=lambda poly: cv2.boundingRect(poly)[0])

            parts["part_i"] = recovered
            group_idx = i + 1
            break

    # Part II: ideally 8 boxes; some scans merge each column pair into 4 tall boxes.
    for i in range(group_idx, len(groups)):
        group = groups[i]
        center_ratio = _group_center_ratio(group)
        if center_ratio < 0.45 or center_ratio > 0.75:
            continue
        subset8 = _select_best_subset(group, expected_count=8)
        if subset8 and _is_uniform_size(subset8, size_tolerance_ratio * 2.2):
            parts["part_ii"] = [b["box"] for b in subset8]
            group_idx = i + 1
            break

        subset4 = _select_best_subset(group, expected_count=4)

        # Build parent candidates from largest boxes first (usually true outer containers).
        parent_sets: List[List[Dict[str, object]]] = []
        largest4 = sorted(group, key=lambda b: int(b["area"]), reverse=True)[:4]
        largest4 = sorted(largest4, key=lambda b: int(b["x"]))
        if len(largest4) == 4:
            parent_sets.append(largest4)
        if subset4 and _is_uniform_size(subset4, size_tolerance_ratio * 2.2):
            parent_sets.append(subset4)

        detected_part_ii: List[np.ndarray] = []
        selected_parent_set: List[Dict[str, object]] = []

        for parent_set in parent_sets:
            # Prefer 8 boxes that were truly detected inside these 4 merged containers.
            local_detected: List[np.ndarray] = []
            for parent in parent_set:
                px = int(parent["x"])
                py = int(parent["y"])
                pw = int(parent["w"])
                ph = int(parent["h"])
                p_area = float(parent["area"])

                child_candidates: List[Dict[str, object]] = []
                for cand in box_info:
                    cx = int(cand["x"])
                    cy = int(cand["y"])
                    cw = int(cand["w"])
                    ch = int(cand["h"])
                    ca = float(cand["area"])

                    # Candidate must be fully inside parent (with small margin) and smaller than parent.
                    if cx < px + 2 or cy < py + 2:
                        continue
                    if cx + cw > px + pw - 2 or cy + ch > py + ph - 2:
                        continue
                    if ca >= p_area * 0.9:
                        continue

                    # Part II inner boxes are typically around half parent width and near full height.
                    width_ratio = float(cw) / float(pw) if pw > 0 else 0.0
                    height_ratio = float(ch) / float(ph) if ph > 0 else 0.0
                    if not (0.30 <= width_ratio <= 0.70):
                        continue
                    if not (0.70 <= height_ratio <= 1.05):
                        continue

                    child_candidates.append(cand)

                child_candidates.sort(key=lambda b: (int(b["x"]), -int(b["area"])))

                # Keep best non-overlapping children by X; expect 2 (left/right).
                selected_children: List[Dict[str, object]] = []
                for cand in child_candidates:
                    overlap_x = False
                    for chosen in selected_children:
                        c1x, c1w = int(cand["x"]), int(cand["w"])
                        c2x, c2w = int(chosen["x"]), int(chosen["w"])
                        left = max(c1x, c2x)
                        right = min(c1x + c1w, c2x + c2w)
                        if right > left:
                            inter_w = right - left
                            min_w = max(1, min(c1w, c2w))
                            if (inter_w / float(min_w)) > 0.5:
                                overlap_x = True
                                break
                    if not overlap_x:
                        selected_children.append(cand)
                    if len(selected_children) == 2:
                        break

                if len(selected_children) == 2:
                    local_detected.extend([selected_children[0]["box"], selected_children[1]["box"]])

            if len(local_detected) == 8:
                detected_part_ii = local_detected
                selected_parent_set = parent_set
                break

        if len(detected_part_ii) == 8:
            parts["part_ii"] = detected_part_ii
            group_idx = i + 1
            break

        if selected_parent_set:
            subset4 = selected_parent_set

        if subset4 and _is_uniform_size(subset4, size_tolerance_ratio * 2.2):

            # Split each merged detected box into left/right halves (Cau 1-2, 3-4, ...).
            split_boxes: List[np.ndarray] = []
            for item in subset4:
                x = int(item["x"])
                y = int(item["y"])
                w = int(item["w"])
                h = int(item["h"])
                w_left = w // 2
                w_right = w - w_left
                split_boxes.append(_rect_to_poly(x, y, w_left, h))
                split_boxes.append(_rect_to_poly(x + w_left, y, w_right, h))
            parts["part_ii"] = split_boxes
            group_idx = i + 1
            break

    # Part III: 6 boxes in one row (lower section).
    for i in range(group_idx, len(groups)):
        group = groups[i]
        center_ratio = _group_center_ratio(group)
        if center_ratio < 0.60:
            continue
        subset = _select_best_subset(group, expected_count=6)
        if subset and _is_uniform_size(subset, size_tolerance_ratio * 2.5):
            parts["part_iii"] = [b["box"] for b in subset]
            group_idx = i + 1
            break

    # Fallback: recover Part III from all detected boxes when large-box filtering
    # drops valid columns (common in noisy scans with mixed contour sizes).
    if not parts["part_iii"]:
        all_row_tol = max(row_tolerance, 25)
        all_groups: List[List[Dict[str, object]]] = []
        for item in sorted(box_info, key=lambda b: int(b["center_y"])):
            placed = False
            for group in all_groups:
                mean_y = float(np.mean([int(b["center_y"]) for b in group]))
                if abs(int(item["center_y"]) - mean_y) <= all_row_tol:
                    group.append(item)
                    placed = True
                    break
            if not placed:
                all_groups.append([item])

        full_page_height = max(int(b["y"]) + int(b["h"]) for b in box_info) if box_info else 0
        part_i_ref_area = None
        if parts["part_i"]:
            part_i_ref_area = float(
                np.mean([
                    cv2.boundingRect(b)[2] * cv2.boundingRect(b)[3]
                    for b in parts["part_i"]
                ])
            )

        best_subset: List[Dict[str, object]] = []
        best_score = float("inf")

        for group in all_groups:
            subset6 = _select_best_subset(group, expected_count=6)
            if not subset6:
                continue
            if not _is_uniform_size(subset6, size_tolerance_ratio * 3.0):
                continue

            subset_center = float(np.mean([int(b["center_y"]) for b in subset6]))
            center_ratio = (subset_center / float(full_page_height)) if full_page_height > 0 else 0.0
            if center_ratio < 0.58:
                continue

            areas = [float(b["area"]) for b in subset6]
            area_mean = float(np.mean(areas)) if areas else 0.0
            if area_mean <= 0:
                continue
            if area_mean < 2500:
                continue
            if part_i_ref_area is not None and area_mean < part_i_ref_area * 0.10:
                continue

            widths = [int(b["w"]) for b in subset6]
            median_w = float(np.median(widths)) if widths else 0.0
            if median_w <= 0:
                continue
            if max(widths) > median_w * 2.2:
                continue

            subset_sorted = sorted(subset6, key=lambda b: int(b["x"]))
            centers = [int(b["x"]) + int(b["w"]) // 2 for b in subset_sorted]
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            gap_mean = float(np.mean(gaps)) if gaps else 0.0
            gap_cv = float(np.std(gaps) / gap_mean) if gap_mean > 0 else 1.0
            area_cv = float(np.std(areas) / area_mean) if area_mean > 0 else 1.0

            # Prefer stable geometry and rows closer to the lower section.
            score = area_cv + 0.2 * gap_cv - 0.0008 * subset_center - 0.00001 * area_mean
            if score < best_score:
                best_score = score
                best_subset = subset_sorted

        if best_subset:
            parts["part_iii"] = [b["box"] for b in best_subset]

    # Fallback: some scans merge the whole Part III row into one large container.
    if not parts["part_iii"]:
        best_container: Optional[Dict[str, object]] = None
        best_score = -1.0
        for group in groups:
            if not group:
                continue
            center_ratio = _group_center_ratio(group)
            if center_ratio < 0.65:
                continue

            # Prefer large lower single-box groups as Part III container candidates.
            if len(group) == 1:
                item = group[0]
                w = float(item["w"])
                h = float(item["h"])
                if h <= 0:
                    continue
                aspect = w / h
                if aspect < 1.8:
                    continue

                score = float(item["area"]) * (1.0 + 0.1 * aspect)
                if score > best_score:
                    best_score = score
                    best_container = item

        if best_container is not None:
            x = int(best_container["x"])
            y = int(best_container["y"])
            w = int(best_container["w"])
            h = int(best_container["h"])
            if w >= 6 and h >= 20:
                base_w = w // 6
                rem_w = w - (base_w * 6)
                split_boxes: List[np.ndarray] = []
                cur_x = x
                for idx in range(6):
                    col_w = base_w + (1 if idx < rem_w else 0)
                    col_w = max(1, col_w)
                    split_boxes.append(_rect_to_poly(cur_x, y, col_w, h))
                    cur_x += col_w
                parts["part_iii"] = split_boxes

    # Fallback for scans where Part II boxes are much smaller than Part I/III and
    # were excluded by the large-container area threshold.
    if not parts["part_ii"]:
        all_row_tol = max(row_tolerance, 25)
        all_groups: List[List[Dict[str, object]]] = []
        for item in sorted(box_info, key=lambda b: int(b["center_y"])):
            placed = False
            for group in all_groups:
                mean_y = float(np.mean([int(b["center_y"]) for b in group]))
                if abs(int(item["center_y"]) - mean_y) <= all_row_tol:
                    group.append(item)
                    placed = True
                    break
            if not placed:
                all_groups.append([item])

        part_i_center = None
        if parts["part_i"]:
            part_i_center = float(np.mean([cv2.boundingRect(b)[1] + cv2.boundingRect(b)[3] / 2.0 for b in parts["part_i"]]))
        part_iii_center = None
        if parts["part_iii"]:
            part_iii_center = float(np.mean([cv2.boundingRect(b)[1] + cv2.boundingRect(b)[3] / 2.0 for b in parts["part_iii"]]))

        ref_area = None
        ref_areas: List[float] = []
        if parts["part_i"]:
            ref_areas.append(float(np.mean([cv2.boundingRect(b)[2] * cv2.boundingRect(b)[3] for b in parts["part_i"]])))
        if parts["part_iii"]:
            ref_areas.append(float(np.mean([cv2.boundingRect(b)[2] * cv2.boundingRect(b)[3] for b in parts["part_iii"]])))
        if ref_areas:
            ref_area = min(ref_areas)

        best_subset: List[Dict[str, object]] = []
        best_score = float("inf")
        target_center = None
        if part_i_center is not None and part_iii_center is not None:
            target_center = (part_i_center + part_iii_center) / 2.0

        for group in all_groups:
            subset8 = _select_best_subset(group, expected_count=8)
            if not subset8:
                continue
            if not _is_uniform_size(subset8, size_tolerance_ratio * 3.0):
                continue

            subset_center = float(np.mean([int(b["center_y"]) for b in subset8]))
            if part_i_center is not None and subset_center <= part_i_center + all_row_tol:
                continue
            if part_iii_center is not None and subset_center >= part_iii_center - all_row_tol:
                continue

            subset_area = float(np.mean([int(b["area"]) for b in subset8]))
            if ref_area is not None:
                # Part II boxes are often smaller than Part I/III, but not tiny noise.
                if subset_area < ref_area * 0.08:
                    continue
                if subset_area > ref_area * 0.95:
                    continue

            xs = [int(b["x"]) for b in subset8]
            ws = [int(b["w"]) for b in subset8]
            centers = [x + w // 2 for x, w in zip(xs, ws)]
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            gap_mean = float(np.mean(gaps)) if gaps else 0.0
            gap_var = float(np.std(gaps) / gap_mean) if gap_mean > 0 else 1.0

            areas = [float(b["area"]) for b in subset8]
            area_mean = float(np.mean(areas))
            area_var = max(abs(a - area_mean) / area_mean for a in areas) if area_mean > 0 else 1.0

            center_penalty = 0.0
            if target_center is not None and estimated_page_height > 0:
                center_penalty = abs(subset_center - target_center) / float(estimated_page_height)

            score = area_var + 0.3 * gap_var + 0.8 * center_penalty
            if score < best_score:
                best_score = score
                best_subset = subset8

        if best_subset:
            parts["part_ii"] = [b["box"] for b in best_subset]

    parts["all_parts"] = parts["part_i"] + parts["part_ii"] + parts["part_iii"]
    return parts


def detect_sobao_danh_boxes(
    boxes: List[np.ndarray],
    boxes_per_row: int = 6,
    max_rows: int = 10,
    row_tolerance: int = 30,
    size_tolerance_ratio: float = 0.3,
    debug: bool = False,
) -> Dict[str, object]:
    # Phát hiện dữ liệu/đối tượng theo tiêu chí của bước này.
    """
    Phát hiện và gom nhóm các box của vùng Số báo danh.

    Đặc trưng vùng SoBaoDanh:
    - Mỗi hàng có 6 box.
    - Tối đa 10 hàng liên tiếp.
    - Kích thước box trong cùng một hàng tương đối đồng đều
      (cho phép dung sai để chịu được ảnh scan chất lượng khác nhau).

    Args:
        boxes: Danh sách box ứng viên vùng Số báo danh.
        boxes_per_row: Số box kỳ vọng trên mỗi hàng.
        max_rows: Số hàng tối đa cần giữ.
        row_tolerance: Dung sai gom nhóm theo trục Y.
        size_tolerance_ratio: Dung sai đồng đều kích thước trong hàng.
        debug: Bật/tắt log debug.

    Returns:
        Dictionary gồm:
        - 'sobao_danh': Danh sách toàn bộ box SoBaoDanh đã phát hiện.
        - 'sobao_danh_rows': Danh sách các hàng, mỗi hàng chứa 6 box.
        - 'row_count': Số hàng đã phát hiện.
    """
    if not boxes:
        return {
            "sobao_danh": [],
            "sobao_danh_rows": [],
            "row_count": 0,
        }
    
    box_info = _build_box_info(boxes)
    groups = _group_box_info_by_row(box_info, row_tolerance)
    
    if debug:
        logger.debug(f" Total groups: {len(groups)}")
        for idx, group in enumerate(groups):
            areas = [cv2.boundingRect(b["box"])[2] * cv2.boundingRect(b["box"])[3] for b in group]
            mean_area = np.mean(areas) if areas else 0
            logger.info(f"  Group {idx}: {len(group)} boxes, areas={[int(a) for a in areas]}, mean={int(mean_area)}")
    
    # Helper function to check size uniformity
    def is_uniform_size(group: List[Dict[str, object]]) -> bool:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        return _is_uniform_size_group(group, size_tolerance_ratio)

    def has_excessive_x_overlap(
        row_boxes: List[np.ndarray],
        max_overlap_pairs: int = 1,
        overlap_tol_px: int = 1,
    ) -> bool:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Một hàng SBD hợp lệ gần như không có box chồng nhau theo trục X.
        rects = sorted([cv2.boundingRect(b) for b in row_boxes], key=lambda r: r[0])
        overlap_pairs = 0
        for i in range(len(rects) - 1):
            x0, y0, w0, h0 = rects[i]
            x1, y1, w1, h1 = rects[i + 1]
            if x0 + w0 > x1 + overlap_tol_px:
                overlap_pairs += 1
                if overlap_pairs > max_overlap_pairs:
                    return True
        return False

    def _try_recover_merged_row(group: List[Dict[str, object]]) -> Optional[List[np.ndarray]]:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Recovery for rows where one merged contour combines two adjacent bubbles
        # and the row appears as 5 boxes instead of 6.
        if len(group) != boxes_per_row - 1:
            return None

        sorted_group = sorted(group, key=lambda b: b["x"])
        widths = [int(b["w"]) if "w" in b else cv2.boundingRect(b["box"])[2] for b in sorted_group]
        heights = [int(b["h"]) if "h" in b else cv2.boundingRect(b["box"])[3] for b in sorted_group]
        if not widths or not heights:
            return None

        median_w = float(np.median(widths))
        median_h = float(np.median(heights))
        if median_w <= 0 or median_h <= 0:
            return None

        merged_idx = int(np.argmax(widths))
        merged_item = sorted_group[merged_idx]
        merged_w = widths[merged_idx]
        merged_h = heights[merged_idx]

        # Require a clearly wider contour but with comparable height.
        if merged_w < median_w * 1.45:
            return None
        if not (0.75 * median_h <= merged_h <= 1.35 * median_h):
            return None

        mx = int(merged_item["x"])
        my = int(merged_item["y"])
        mh = int(merged_item["h"]) if "h" in merged_item else merged_h
        left_w = merged_w // 2
        right_w = merged_w - left_w
        if left_w <= 0 or right_w <= 0:
            return None

        left_poly = _rect_to_poly(mx, my, left_w, mh)
        right_poly = _rect_to_poly(mx + left_w, my, right_w, mh)

        candidate_boxes: List[np.ndarray] = []
        for i, item in enumerate(sorted_group):
            if i == merged_idx:
                candidate_boxes.extend([left_poly, right_poly])
            else:
                candidate_boxes.append(item["box"])

        if len(candidate_boxes) != boxes_per_row:
            return None

        candidate_boxes = sorted(candidate_boxes, key=lambda b: cv2.boundingRect(b)[0])
        temp_group = []
        for box in candidate_boxes:
            x, y, w, h = cv2.boundingRect(box)
            temp_group.append({"box": box, "x": x, "y": y, "area": w * h})

        if not is_uniform_size(temp_group):
            return None

        return candidate_boxes
    
    # Find all rows with exactly boxes_per_row (6) boxes
    # Also try to extract 6-box sub-rows from larger groups
    sobao_danh_rows: List[List[np.ndarray]] = []
    len_minus_one_groups: List[List[Dict[str, object]]] = []
    
    for idx, group in enumerate(groups):
        # Only accept rows with exactly boxes_per_row boxes and uniform size
        if len(group) == boxes_per_row and is_uniform_size(group):
            sorted_boxes = [b["box"] for b in sorted(group, key=lambda b: b["x"])]
            if has_excessive_x_overlap(sorted_boxes):
                if debug:
                    logger.info(f"  ✗ Group {idx} rejected: excessive X overlap in 6-box row")
            else:
                sobao_danh_rows.append(sorted_boxes)
                if debug:
                    logger.info(f"  ✓ Group {idx} matched as SoBaoDanh row {len(sobao_danh_rows)}")
        elif len(group) > boxes_per_row:
            # Try to extract non-overlapping 6-box rows from larger groups
            sorted_group = sorted(group, key=lambda b: b["x"])
            used_indices = set()
            
            for start_idx in range(len(sorted_group) - boxes_per_row + 1):
                # Skip if we've already used any of these boxes
                if any(i in used_indices for i in range(start_idx, start_idx + boxes_per_row)):
                    continue
                
                sub_group = sorted_group[start_idx:start_idx + boxes_per_row]
                
                if is_uniform_size(sub_group):
                    sorted_boxes = [b["box"] for b in sub_group]
                    if has_excessive_x_overlap(sorted_boxes):
                        if debug:
                            logger.info(
                                f"  ✗ Group {idx} sub-row {start_idx} rejected: excessive X overlap"
                            )
                        continue
                    sobao_danh_rows.append(sorted_boxes)
                    # Mark these boxes as used
                    for i in range(start_idx, start_idx + boxes_per_row):
                        used_indices.add(i)
                    if debug:
                        logger.info(f"  ✓ Group {idx} sub-row extracted as SoBaoDanh row {len(sobao_danh_rows)}")

            if debug and len(group) <= 15:
                logger.info(f"  ✗ Group {idx} rejected: len={len(group)} (need 6), no valid non-overlapping 6-box sub-rows")
        elif len(group) == boxes_per_row - 1:
            len_minus_one_groups.append(group)
            recovered = _try_recover_merged_row(group)
            if recovered is not None:
                sobao_danh_rows.append(recovered)
                if debug:
                    logger.info(f"  ✓ Group {idx} recovered as SoBaoDanh row {len(sobao_danh_rows)} (split merged box)")
        elif debug and len(group) <= 10:
            areas = [b["area"] for b in group]
            mean_area = np.mean(areas) if areas else 0
            variance = max([abs(a - mean_area) / mean_area for a in areas]) if mean_area > 0 else 0
            logger.info(f"  ✗ Group {idx} rejected: len={len(group)} (need 6), variance={variance:.2f} (max {size_tolerance_ratio})")
    
    sobao_danh_rows = _filter_rows_by_global_size_consistency(
        sobao_danh_rows,
        size_tolerance_ratio,
        debug=debug,
    )
    sobao_danh_rows = [
        row for row in sobao_danh_rows
        if not has_excessive_x_overlap(row)
    ]
    sobao_danh = [box for row in sobao_danh_rows for box in row]

    sobao_danh_rows = _trim_rows_to_consistent_window(sobao_danh_rows, max_rows)
    sobao_danh = [box for row in sobao_danh_rows for box in row]

    # Handle a common pattern: one unrelated header-like top row plus one valid
    # row detected as only 5 boxes due a missing/merged contour.
    if len(sobao_danh_rows) == max_rows and len_minus_one_groups:
        row_with_y = []
        for row in sobao_danh_rows:
            ys = [cv2.boundingRect(box)[1] for box in row]
            row_with_y.append((float(np.mean(ys)) if ys else 0.0, row))
        row_with_y.sort(key=lambda t: t[0])

        ys_only = [t[0] for t in row_with_y]
        if len(ys_only) >= 3:
            gaps = [ys_only[i + 1] - ys_only[i] for i in range(len(ys_only) - 1)]
            tail_gaps = gaps[1:] if len(gaps) > 1 else gaps
            median_tail_gap = float(np.median(tail_gaps)) if tail_gaps else 0.0

            top_gap_is_outlier = median_tail_gap > 1 and gaps[0] > (median_tail_gap * 1.6)
            if top_gap_is_outlier:
                if debug:
                    logger.info(
                        f"  • Top SBD row looks like outlier (first gap={gaps[0]:.1f}, "
                        f"median tail gap={median_tail_gap:.1f}); attempting replacement from len-5 row"
                    )

                kept_rows = [row for _, row in row_with_y[1:]]
                template_rows = kept_rows[:]

                if template_rows:
                    col_x_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
                    col_w_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
                    h_list: List[int] = []

                    for row in template_rows:
                        rects = sorted([cv2.boundingRect(b) for b in row], key=lambda r: r[0])
                        if len(rects) != boxes_per_row:
                            continue
                        for c, (rx, ry, rw, rh) in enumerate(rects):
                            col_x_lists[c].append(int(rx))
                            col_w_lists[c].append(int(rw))
                            h_list.append(int(rh))

                    if all(col_x_lists[c] for c in range(boxes_per_row)) and h_list:
                        col_x = [int(round(float(np.median(col_x_lists[c])))) for c in range(boxes_per_row)]
                        col_w = [int(round(float(np.median(col_w_lists[c])))) for c in range(boxes_per_row)]
                        row_h = max(1, int(round(float(np.median(h_list)))))

                        def rect_to_poly(x: int, y: int, w: int, h: int) -> np.ndarray:
                            # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
                            return np.array(
                                [
                                    [[x, y]],
                                    [[x + w, y]],
                                    [[x + w, y + h]],
                                    [[x, y + h]],
                                ],
                                dtype=np.int32,
                            )

                        expected_y = ys_only[1] - median_tail_gap
                        best_group = None
                        best_dist = float("inf")
                        for g in len_minus_one_groups:
                            gy = float(np.mean([int(it["y"]) for it in g])) if g else 0.0
                            dist = abs(gy - expected_y)
                            if dist < best_dist:
                                best_dist = dist
                                best_group = g

                        replacement_row: Optional[List[np.ndarray]] = None
                        if best_group is not None and best_dist <= max(35.0, median_tail_gap * 0.9):
                            g_sorted = sorted(best_group, key=lambda it: int(it["x"]))
                            assigned: List[Optional[np.ndarray]] = [None] * boxes_per_row
                            used_cols = set()

                            for it in g_sorted:
                                box = it["box"]
                                x, y, w, h = cv2.boundingRect(box)
                                best_col = None
                                best_col_dist = float("inf")
                                for c in range(boxes_per_row):
                                    if c in used_cols:
                                        continue
                                    dist = abs(x - col_x[c])
                                    if dist < best_col_dist:
                                        best_col_dist = dist
                                        best_col = c
                                if best_col is not None:
                                    assigned[best_col] = box
                                    used_cols.add(best_col)

                            row_y = int(round(float(np.mean([int(it["y"]) for it in g_sorted]))))
                            for c in range(boxes_per_row):
                                if assigned[c] is None:
                                    assigned[c] = _rect_to_poly(col_x[c], row_y, max(1, col_w[c]), row_h)

                            replacement_row = [box for box in assigned if box is not None]

                        if replacement_row is not None and len(replacement_row) == boxes_per_row:
                            kept_rows.append(replacement_row)
                            kept_rows.sort(key=lambda row: float(np.mean([cv2.boundingRect(b)[1] for b in row])))
                            sobao_danh_rows = kept_rows[:max_rows]
                            sobao_danh = [box for row in sobao_danh_rows for box in row]
                            if debug:
                                logger.info("  ✓ Replaced top outlier SBD row using len-5 recovery")
                        else:
                            sobao_danh_rows = kept_rows
                            sobao_danh = [box for row in sobao_danh_rows for box in row]
                            if debug:
                                logger.info("  • Dropped top outlier SBD row (no suitable len-5 replacement)")

    # Normalize malformed top row where one or more boxes are abnormally wide/shifted.
    # This appears on some scans (e.g. 0003) where first-row middle boxes merge visually.
    if len(sobao_danh_rows) >= 2:
        row_with_y = []
        for row in sobao_danh_rows:
            ys = [cv2.boundingRect(box)[1] for box in row]
            row_with_y.append((float(np.mean(ys)) if ys else 0.0, row))
        row_with_y.sort(key=lambda t: t[0])

        top_row = row_with_y[0][1]
        ref_rows = [r for _, r in row_with_y[1:] if len(r) == boxes_per_row]

        if len(top_row) == boxes_per_row and ref_rows:
            top_rects = sorted([cv2.boundingRect(b) for b in top_row], key=lambda r: r[0])

            col_x_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
            col_w_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
            h_list: List[int] = []

            for row in ref_rows:
                rects = sorted([cv2.boundingRect(b) for b in row], key=lambda r: r[0])
                if len(rects) != boxes_per_row:
                    continue
                for c, (rx, ry, rw, rh) in enumerate(rects):
                    col_x_lists[c].append(int(rx))
                    col_w_lists[c].append(int(rw))
                    h_list.append(int(rh))

            if all(col_x_lists[c] for c in range(boxes_per_row)) and h_list:
                col_x = [int(round(float(np.median(col_x_lists[c])))) for c in range(boxes_per_row)]
                col_w = [max(1, int(round(float(np.median(col_w_lists[c]))))) for c in range(boxes_per_row)]
                row_h = max(1, int(round(float(np.median(h_list)))))

                wide_outliers = 0
                shifted_outliers = 0
                overlap_count = 0
                for c, (rx, ry, rw, rh) in enumerate(top_rects):
                    width_tol_hi = col_w[c] * 1.35
                    width_tol_lo = col_w[c] * 0.65
                    if rw > width_tol_hi or rw < width_tol_lo:
                        wide_outliers += 1

                    shift_tol = max(6.0, col_w[c] * 0.35)
                    if abs(rx - col_x[c]) > shift_tol:
                        shifted_outliers += 1

                for c in range(boxes_per_row - 1):
                    x0, y0, w0, h0 = top_rects[c]
                    x1, y1, w1, h1 = top_rects[c + 1]
                    if x0 + w0 > x1:
                        overlap_count += 1

                top_row_is_malformed = (wide_outliers >= 1 and shifted_outliers >= 1) or overlap_count >= 1
                if top_row_is_malformed:
                    row_y = int(round(float(np.mean([r[1] for r in top_rects]))))
                    normalized_top = [
                        _rect_to_poly(col_x[c], row_y, col_w[c], row_h)
                        for c in range(boxes_per_row)
                    ]
                    row_with_y[0] = (row_with_y[0][0], normalized_top)
                    sobao_danh_rows = [row for _, row in row_with_y]
                    sobao_danh = [box for row in sobao_danh_rows for box in row]
                    if debug:
                        logger.info(
                            "  ✓ Normalized top SBD row geometry "
                            f"(wide={wide_outliers}, shifted={shifted_outliers}, overlap={overlap_count})"
                        )
    
    return {
        "sobao_danh": sobao_danh,
        "sobao_danh_rows": sobao_danh_rows,
        "row_count": len(sobao_danh_rows),
    }


def detect_ma_de_boxes(
    boxes: List[np.ndarray],
    boxes_per_row: int = 3,
    max_rows: int = 10,
    row_tolerance: int = 15,
    size_tolerance_ratio: float = 0.3,
    debug: bool = False,
) -> Dict[str, object]:
    # Phát hiện dữ liệu/đối tượng theo tiêu chí của bước này.
    """
    Phát hiện và gom nhóm các box của vùng Mã đề.

    Đặc trưng vùng Mã đề:
    - Mỗi hàng có 3 box.
    - Tối đa 10 hàng.
    - Kích thước box trong cùng hàng tương đối đồng đều.

    Args:
        boxes: Danh sách box ứng viên vùng Mã đề.
        boxes_per_row: Số box kỳ vọng trên mỗi hàng.
        max_rows: Số hàng tối đa cần giữ.
        row_tolerance: Dung sai gom nhóm theo trục Y.
        size_tolerance_ratio: Dung sai đồng đều kích thước trong hàng.
        debug: Bật/tắt log debug.

    Returns:
        Dictionary gồm:
        - 'ma_de': Danh sách toàn bộ box Mã đề đã phát hiện.
        - 'ma_de_rows': Danh sách các hàng, mỗi hàng chứa 3 box.
        - 'row_count': Số hàng đã phát hiện.
    """
    if not boxes:
        return {
            "ma_de": [],
            "ma_de_rows": [],
            "row_count": 0,
        }
    
    box_info = _build_box_info(boxes)
    groups = _group_box_info_by_row(box_info, row_tolerance)
    
    if debug:
        logger.debug(f" MaDe Detection - Total groups: {len(groups)}")
        for idx, group in enumerate(groups):
            areas = [cv2.boundingRect(b["box"])[2] * cv2.boundingRect(b["box"])[3] for b in group]
            mean_area = np.mean(areas) if areas else 0
            logger.info(f"  Group {idx}: {len(group)} boxes, areas={[int(a) for a in areas]}, mean={int(mean_area)}")
    
    # Helper function to check size uniformity
    def is_uniform_size(group: List[Dict[str, object]]) -> bool:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        return _is_uniform_size_group(group, size_tolerance_ratio)
    
    # Find all rows with exactly boxes_per_row (3) boxes
    # Also try to extract valid 3-box rows from larger groups (non-overlapping)
    ma_de_rows: List[List[np.ndarray]] = []
    
    for idx, group in enumerate(groups):
        # Only accept rows with exactly boxes_per_row boxes and uniform size
        if len(group) == boxes_per_row and is_uniform_size(group):
            sorted_boxes = [b["box"] for b in sorted(group, key=lambda b: b["x"])]
            ma_de_rows.append(sorted_boxes)
            if debug:
                logger.info(f"  ✓ Group {idx} matched as MaDe row {len(ma_de_rows)}")
        elif len(group) > boxes_per_row:
            # Try to extract non-overlapping 3-box rows from larger groups
            sorted_group = sorted(group, key=lambda b: b["x"])
            used_indices = set()
            
            for start_idx in range(len(sorted_group) - boxes_per_row + 1):
                # Skip if we've already used any of these boxes
                if any(i in used_indices for i in range(start_idx, start_idx + boxes_per_row)):
                    continue
                
                sub_group = sorted_group[start_idx:start_idx + boxes_per_row]
                
                if is_uniform_size(sub_group):
                    sorted_boxes = [b["box"] for b in sub_group]
                    ma_de_rows.append(sorted_boxes)
                    # Mark these boxes as used
                    for i in range(start_idx, start_idx + boxes_per_row):
                        used_indices.add(i)
                    if debug:
                        logger.info(f"  ✓ Group {idx} sub-row extracted as MaDe row {len(ma_de_rows)}")
            
            if debug and len(group) <= 15:
                logger.info(f"  ✗ Group {idx} rejected: len={len(group)} (need 3), no valid non-overlapping 3-box sub-rows")
        elif debug and len(group) <= 10:
            areas = [b["area"] for b in group]
            mean_area = np.mean(areas) if areas else 0
            variance = max([abs(a - mean_area) / mean_area for a in areas]) if mean_area > 0 else 0
            logger.info(f"  ✗ Group {idx} rejected: len={len(group)} (need 3), variance={variance:.2f} (max {size_tolerance_ratio})")
    
    ma_de_rows = _filter_rows_by_global_size_consistency(
        ma_de_rows,
        size_tolerance_ratio,
        debug=debug,
    )
    ma_de = [box for row in ma_de_rows for box in row]

    ma_de_rows = _trim_rows_to_consistent_window(ma_de_rows, max_rows)
    ma_de = [box for row in ma_de_rows for box in row]
    
    return {
        "ma_de": ma_de,
        "ma_de_rows": ma_de_rows,
        "row_count": len(ma_de_rows),
    }


def extrapolate_missing_rows(
    detection_results: Dict[str, object],
    target_rows: int = 10,
    debug: bool = False,
) -> Dict[str, object]:
    # Nội suy/ngoại suy dữ liệu thiếu dựa trên mốc tham chiếu.
    """
    Nội suy/ngoại suy các hàng còn thiếu cho SoBaoDanh và Mã đề dựa trên khoảng cách hàng.

    Giả định các hàng của hai vùng được căn thẳng theo trục dọc và có khoảng cách gần đều.
    Hàm sẽ điền các hàng còn thiếu để đạt đủ `target_rows`.

    Args:
        detection_results: Dictionary chứa 'sobao_danh_rows' và 'ma_de_rows'.
        target_rows: Số hàng mục tiêu (mặc định 10).
        debug: Bật in thông tin debug khi xử lý.

    Returns:
        Dictionary kết quả đã bổ sung các hàng nội suy/ngoại suy.
    """
    sobao_danh_rows = detection_results.get("sobao_danh_rows", [])
    ma_de_rows = detection_results.get("ma_de_rows", [])
    
    # Calculate row positions (average Y of first box in each row)
    sobao_y_positions = []
    for row in sobao_danh_rows:
        if row:
            y_sum = sum(cv2.boundingRect(box)[1] for box in row)
            avg_y = y_sum // len(row)
            sobao_y_positions.append(avg_y)
    
    ma_de_y_positions = []
    for row in ma_de_rows:
        if row:
            y_sum = sum(cv2.boundingRect(box)[1] for box in row)
            avg_y = y_sum // len(row)
            ma_de_y_positions.append(avg_y)
    
    if debug:
        logger.debug(f" SoBaoDanh Y positions: {sobao_y_positions}")
        logger.debug(f" MaDe Y positions: {ma_de_y_positions}")
    
    # Use SoBaoDanh Y positions as reference (since it has all 10 rows detected)
    # If SoBaoDanh has 10 rows, use them directly
    if len(sobao_y_positions) == target_rows:
        reference_positions = sobao_y_positions
        if debug:
            logger.debug(f" Using SoBaoDanh Y positions directly: {reference_positions}")
    else:
        # If SoBaoDanh doesn't have all rows, calculate expected positions
        # Calculate average row spacing from the section with more rows
        spacing_positions = sobao_y_positions if len(sobao_y_positions) >= len(ma_de_y_positions) else ma_de_y_positions
        
        if len(spacing_positions) >= 2:
            spacings = [spacing_positions[i + 1] - spacing_positions[i] for i in range(len(spacing_positions) - 1)]
            avg_spacing = int(np.mean(spacings)) if spacings else 0
        else:
            avg_spacing = 0
        
        if spacing_positions:
            first_row_y = spacing_positions[0]
            reference_positions = [first_row_y + i * avg_spacing for i in range(target_rows)]
            if debug:
                logger.debug(f" Calculated reference positions: {reference_positions}")
        else:
            # No reference rows available: return a complete empty-aligned structure
            # so downstream callers can still read summary keys safely.
            result = detection_results.copy()
            result["sobao_danh_rows_aligned"] = [None] * target_rows
            result["ma_de_rows_aligned"] = [None] * target_rows
            result["reference_positions"] = []
            result["sobao_y_positions"] = sobao_y_positions
            result["ma_de_y_positions"] = ma_de_y_positions
            result["sobao_missing_count"] = target_rows
            result["ma_de_missing_count"] = target_rows
            result["sobao_detected_count"] = 0
            result["ma_de_detected_count"] = 0
            return result
    
    # Map detected MaDe rows to SoBaoDanh Y positions
    def align_rows_to_reference_positions(detected_rows, detected_y_positions, reference_positions, name="", tolerance=20):
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        """Căn các hàng phát hiện được vào các vị trí tham chiếu theo trục Y."""
        aligned = [None] * len(reference_positions)
        
        for row_idx, (row, y_pos) in enumerate(zip(detected_rows, detected_y_positions)):
            # Prefer nearest available reference slot; avoid overwriting existing row.
            sorted_indices = sorted(
                range(len(reference_positions)),
                key=lambda i: abs(reference_positions[i] - y_pos),
            )

            assigned_idx = None
            for idx in sorted_indices:
                distance = abs(reference_positions[idx] - y_pos)
                if distance > tolerance:
                    break
                if aligned[idx] is None:
                    assigned_idx = idx
                    break

            # If all nearby slots are occupied, keep the closest one only when it is empty.
            if assigned_idx is not None:
                aligned[assigned_idx] = row
                if debug:
                    logger.info(f"  {name} row {row_idx} (Y={y_pos}) -> position {assigned_idx} (reference Y={reference_positions[assigned_idx]})")
            else:
                closest_idx = sorted_indices[0] if sorted_indices else -1
                distance = abs(reference_positions[closest_idx] - y_pos) if closest_idx >= 0 else 9999
                if debug:
                    logger.info(f"  {name} row {row_idx} (Y={y_pos}) -> UNALIGNED (distance={distance} > {tolerance} or slot occupied)")
        
        return aligned
    
    # Align both SoBaoDanh and MaDe rows to reference positions
    aligned_sobao = align_rows_to_reference_positions(
        sobao_danh_rows, sobao_y_positions, reference_positions, "SoBaoDanh", tolerance=30)
    aligned_ma_de = align_rows_to_reference_positions(
        ma_de_rows, ma_de_y_positions, reference_positions, "MaDe", tolerance=90)
    
    # Create result with aligned/extrapolated rows
    result = detection_results.copy()
    result["sobao_danh_rows_aligned"] = aligned_sobao
    result["ma_de_rows_aligned"] = aligned_ma_de
    result["reference_positions"] = reference_positions
    result["sobao_y_positions"] = sobao_y_positions
    result["ma_de_y_positions"] = ma_de_y_positions
    
    # Count actual and missing rows
    sobao_detected = sum(1 for r in aligned_sobao if r is not None)
    ma_de_detected = sum(1 for r in aligned_ma_de if r is not None)
    sobao_missing = target_rows - sobao_detected
    ma_de_missing = target_rows - ma_de_detected
    
    result["sobao_missing_count"] = sobao_missing
    result["ma_de_missing_count"] = ma_de_missing
    result["sobao_detected_count"] = sobao_detected
    result["ma_de_detected_count"] = ma_de_detected
    
    return result


def _build_synthetic_id_rows_from_part_i(
    image_shape: Tuple[int, int],
    part_i_boxes: List[np.ndarray],
    cols: int,
    rows: int,
    x_range_ratio: Tuple[float, float],
    distance_from_part_i_ratio: float,
    row_step_ratio: float,
) -> List[List[np.ndarray]]:
    # Hàm hỗ trợ dựng lưới ID tổng hợp khi detect quá ít hàng.
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    if h_img <= 0 or w_img <= 0 or cols <= 0 or rows <= 0:
        return []

    # Neo theo mép trên Part I; nếu thiếu Part I thì dùng mốc gần đúng theo layout.
    if part_i_boxes:
        part_i_top = min(cv2.boundingRect(b)[1] for b in part_i_boxes)
    else:
        part_i_top = int(round(h_img * 0.3))

    distance_ratio = float(np.clip(distance_from_part_i_ratio, 0.05, 0.60))
    row_step_ratio = float(np.clip(row_step_ratio, 0.010, 0.080))

    distance_px = max(1, int(round(distance_ratio * h_img)))
    step_y = max(6, int(round(row_step_ratio * h_img)))
    # Stack rows directly with no vertical gap.
    row_h = step_y

    # Tính Y hàng đầu từ khoảng cách tới Part I và chặn trong ảnh.
    y0 = part_i_top - distance_px
    grid_h = rows * row_h
    y0 = min(y0, h_img - grid_h - 1)
    y0 = max(0, y0)

    xr0 = float(np.clip(x_range_ratio[0], 0.0, 0.99))
    xr1 = float(np.clip(x_range_ratio[1], xr0 + 1e-4, 1.0))
    x0 = int(round(xr0 * w_img))
    x1 = int(round(xr1 * w_img))
    if x1 <= x0:
        x1 = min(w_img, x0 + cols)

    # Build contiguous column edges so boxes touch each other horizontally.
    x_edges = np.round(np.linspace(x0, x1, cols + 1)).astype(np.int32)
    x_edges[0] = x0
    x_edges[-1] = x1
    for i in range(1, len(x_edges)):
        min_allowed = x_edges[i - 1] + 1
        max_allowed = x1 - (cols - i)
        x_edges[i] = int(np.clip(x_edges[i], min_allowed, max_allowed))

    out_rows: List[List[np.ndarray]] = []
    for r in range(rows):
        y = y0 + (r * row_h)
        y_top = int(y)

        row_boxes: List[np.ndarray] = []
        for c in range(cols):
            x_left = int(x_edges[c])
            x_right = int(x_edges[c + 1])
            box_w = max(1, x_right - x_left)
            row_boxes.append(_rect_to_poly(x_left, y_top, box_w, row_h))

        out_rows.append(row_boxes)

    return out_rows


def _build_synthetic_id_rows_fixed_image_position(
    image_shape: Tuple[int, int],
    cols: int,
    rows: int,
    x_range_ratio: Tuple[float, float],
    top_y_ratio: float,
    row_step_ratio: float,
) -> List[List[np.ndarray]]:
    # Dựng lưới ID cố định theo vị trí ảnh (không phụ thuộc Part I).
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    if h_img <= 0 or w_img <= 0 or cols <= 0 or rows <= 0:
        return []

    top_ratio = float(np.clip(top_y_ratio, 0.0, 0.95))
    row_step_ratio = float(np.clip(row_step_ratio, 0.010, 0.080))

    y0 = int(round(top_ratio * h_img))
    row_h = max(6, int(round(row_step_ratio * h_img)))
    grid_h = rows * row_h
    y0 = min(y0, h_img - grid_h - 1)
    y0 = max(0, y0)

    xr0 = float(np.clip(x_range_ratio[0], 0.0, 0.99))
    xr1 = float(np.clip(x_range_ratio[1], xr0 + 1e-4, 1.0))
    x0 = int(round(xr0 * w_img))
    x1 = int(round(xr1 * w_img))
    if x1 <= x0:
        x1 = min(w_img, x0 + cols)

    x_edges = np.round(np.linspace(x0, x1, cols + 1)).astype(np.int32)
    x_edges[0] = x0
    x_edges[-1] = x1
    for i in range(1, len(x_edges)):
        min_allowed = x_edges[i - 1] + 1
        max_allowed = x1 - (cols - i)
        x_edges[i] = int(np.clip(x_edges[i], min_allowed, max_allowed))

    out_rows: List[List[np.ndarray]] = []
    for r in range(rows):
        y_top = int(y0 + (r * row_h))
        row_boxes: List[np.ndarray] = []
        for c in range(cols):
            x_left = int(x_edges[c])
            x_right = int(x_edges[c + 1])
            box_w = max(1, x_right - x_left)
            row_boxes.append(_rect_to_poly(x_left, y_top, box_w, row_h))
        out_rows.append(row_boxes)

    return out_rows


def _apply_affine_from_corner_markers(
    image: np.ndarray,
    corners: Dict[str, Optional[Tuple[int, int]]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    # Hiệu chỉnh topview từ 4 góc marker bằng perspective transform.
    if image is None or corners is None:
        return None, None

    tl = corners.get("top_left")
    tr = corners.get("top_right")
    br = corners.get("bottom_right")
    bl = corners.get("bottom_left")
    if tl is None or tr is None or br is None or bl is None:
        return None, None

    h_img, w_img = image.shape[:2]
    if h_img <= 0 or w_img <= 0:
        return None, None

    src = np.array(
        [
            [float(tl[0]), float(tl[1])],
            [float(tr[0]), float(tr[1])],
            [float(br[0]), float(br[1])],
            [float(bl[0]), float(bl[1])],
        ],
        dtype=np.float32,
    )

    width_top = float(np.hypot(src[1, 0] - src[0, 0], src[1, 1] - src[0, 1]))
    width_bottom = float(np.hypot(src[2, 0] - src[3, 0], src[2, 1] - src[3, 1]))
    height_left = float(np.hypot(src[3, 0] - src[0, 0], src[3, 1] - src[0, 1]))
    height_right = float(np.hypot(src[2, 0] - src[1, 0], src[2, 1] - src[1, 1]))

    if min(width_top, width_bottom, height_left, height_right) < 20.0:
        return None, None

    # Chuẩn hóa toàn trang về topview chiếm toàn khung ảnh đầu ra.
    dst = np.array(
        [
            [0.0, 0.0],
            [float(w_img - 1), 0.0],
            [float(w_img - 1), float(h_img - 1)],
            [0.0, float(h_img - 1)],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, matrix

