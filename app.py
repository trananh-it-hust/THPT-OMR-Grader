"""
app.py — Streamlit web interface for the THPT OMR pipeline.

Thin interface layer (ARCH_RULES L-2): all image processing is delegated to
``src.pipeline.process_image()``.  This module contains only:

- Streamlit UI configuration and layout.
- ``_to_rgb()`` — display helper (BGR → RGB for st.image).
- ``_build_structured_answers()`` — parse evals into answer dicts.
- ``_build_json_payload()`` / ``_build_batch_summary_row()`` — data helpers.
- ``_digit_eval_table()`` / ``_render_detailed_result()`` — UI rendering.
"""

from __future__ import annotations

import concurrent.futures
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from src.pipeline import process_image
from src.log_config import logger


# ---------------------------------------------------------------------------
#  Display helpers
# ---------------------------------------------------------------------------


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale image to RGB for ``st.image``.

    Args:
        image: Input image (BGR 3-channel or 2-channel grayscale).

    Returns:
        RGB ``np.ndarray`` suitable for Streamlit display.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
#  Answer extraction helpers
# ---------------------------------------------------------------------------


def _build_structured_answers(results: Dict[str, object], file_name: str = "") -> Dict[str, object]:
    """Extract structured per-question answers from raw cell evaluations.

    Parses ``part_i_evals``, ``part_ii_evals``, ``part_iii_evals``, and digit
    results from ``results`` (as returned by :func:`process_image`) into
    answer dicts keyed by question number.

    Args:
        results: Full result dict from ``process_image()``.

    Returns:
        Dict with keys:
        ``fc``, ``fc_invalid``, ``tf``, ``tf_invalid``,
        ``dg``, ``dg_invalid``, ``sbd``, ``sbd_invalid``, ``mdt``, ``mdt_invalid``.
    """
    part_i_evals:   List[Dict[str, object]] = results["part_i_evals"]
    part_ii_evals:  List[Dict[str, object]] = results["part_ii_evals"]
    part_iii_evals: List[Dict[str, object]] = results["part_iii_evals"]
    sbd_digits:     Dict[str, object]       = results["sbd_digits"]
    made_digits:    Dict[str, object]       = results["made_digits"]

    # --- Part I: Multiple Choice (40 questions, 4 options A-D) -------
    fc: Dict[str, List[int]] = {str(i): [] for i in range(1, 41)}
    fc_invalid: set = set()
    for item in part_i_evals:
        if not bool(item.get("filled", False)):
            continue
        q = (int(item.get("box_idx", -1)) * 10) + int(item.get("row", -1)) + 1
        if 1 <= q <= 40:
            fc[str(q)].append(int(item.get("col", -1)))

    prefix = f"[{file_name}] " if file_name else ""
    for q_str, choices in fc.items():
        if len(choices) > 1:
            logger.warning(f"{prefix}Phần I - Câu {q_str}: Đánh nhiều hơn 1 đáp án.")
            fc_invalid.add(q_str)
            fc[q_str] = [-2]
        elif not choices:
            logger.debug(f"{prefix}Phần I - Câu {q_str}: Bỏ trống.")

    # --- Part II: True/False (32 questions, 2 options) ---------------
    tf: Dict[str, List[int]] = {str(i): [] for i in range(1, 33)}
    tf_invalid: set = set()
    for item in part_ii_evals:
        if not bool(item.get("filled", False)):
            continue
        box_idx = int(item.get("box_idx", -1))
        row     = int(item.get("row",     -1))
        col     = int(item.get("col",     -1))
        q = (box_idx * 4) + row + 1
        if 1 <= q <= 32 and col in (0, 1):
            tf[str(q)].append(col)

    for q_str, answers in tf.items():
        if len(answers) > 1:
            logger.warning(f"{prefix}Phần II - Câu {q_str}: Đánh cả Đúng và Sai.")
            tf_invalid.add(q_str)
            tf[q_str] = [-2]
        elif not answers:
            logger.debug(f"{prefix}Phần II - Câu {q_str}: Bỏ trống.")

    # --- Part III: Numeric (6 questions, column-per-digit) -----------
    row_labels = ["-", ","] + [str(i) for i in range(10)]
    dg: Dict[str, str] = {str(i): "" for i in range(1, 7)}
    dg_invalid: set = set()

    for cau_num in range(1, 7):
        filled_by_col: Dict[int, List[str]] = {}
        for item in part_iii_evals:
            if not bool(item.get("filled", False)):
                continue
            if int(item.get("box_idx", -1)) + 1 != cau_num:
                continue
            col = int(item.get("col", -1))
            row = int(item.get("row", -1))
            if col < 0 or row < 0 or row >= len(row_labels):
                continue
            filled_by_col.setdefault(col, []).append(row_labels[row])

        if any(len(v) > 1 for v in filled_by_col.values()):
            logger.warning(f"{prefix}Phần III - Câu {cau_num}: Có cột đánh nhiều hơn 1 ô.")
            dg[str(cau_num)] = "X"
            dg_invalid.add(str(cau_num))
            continue

        digits = [filled_by_col[c][0] for c in sorted(filled_by_col) if filled_by_col[c]]
        dg[str(cau_num)] = "".join(digits)

    sbd = str(sbd_digits.get("decoded", ""))
    mdt = str(made_digits.get("decoded", ""))
    
    if "?" in sbd:
        logger.warning(f"{prefix}Số Báo Danh chưa hoàn chỉnh/có lỗi: {sbd}")
    if "?" in mdt:
        logger.warning(f"{prefix}Mã Đề chưa hoàn chỉnh/có lỗi: {mdt}")
    return {
        "fc":  fc,  "fc_invalid":  fc_invalid,
        "tf":  tf,  "tf_invalid":  tf_invalid,
        "dg":  dg,  "dg_invalid":  dg_invalid,
        "sbd": sbd, "sbd_invalid": "?" in sbd,
        "mdt": mdt, "mdt_invalid": "?" in mdt,
    }


def _digit_eval_table(result: Dict[str, object]) -> List[Dict[str, object]]:
    """Build a display table from a digit-decode result dict.

    Args:
        result: ``sbd_digits`` or ``made_digits`` from ``process_image()``.

    Returns:
        List of row dicts with keys ``Digit``, ``Column``, ``Filled``,
        ``Darkness`` — suitable for ``st.dataframe()``.
    """
    evaluations = result.get("evaluations", [])
    if not isinstance(evaluations, list):
        return []
    return [
        {
            "Digit":    str(item.get("row", "")),
            "Column":   int(item.get("col", -1)),
            "Filled":   "✓" if bool(item.get("filled", False)) else "✗",
            "Darkness": f"{float(item.get('mean_darkness', 0.0)):.1f}",
        }
        for item in sorted(evaluations, key=lambda x: (int(x.get("col", -1)), int(x.get("row", -1))))
    ]


def _build_json_payload(extracted: Dict[str, object]) -> Dict[str, object]:
    """Serialize structured answers to the canonical JSON output format.

    Args:
        extracted: Dict from ``_build_structured_answers()``.

    Returns:
        Payload dict ready for ``json.dumps()``.
    """
    return {
        "res": {
            "fc":          extracted["fc"],
            "fc_invalid":  sorted(extracted["fc_invalid"], key=int),
            "tf":          extracted["tf"],
            "tf_invalid":  sorted(extracted["tf_invalid"], key=int),
            "dg":          extracted["dg"],
            "dg_invalid":  sorted(extracted["dg_invalid"], key=int),
            "sbd":         extracted["sbd"],
            "sbd_invalid": extracted["sbd_invalid"],
            "mdt":         extracted["mdt"],
            "mdt_invalid": extracted["mdt_invalid"],
        },
        "sbd": extracted["sbd"],
        "mdt": extracted["mdt"],
    }


def _build_batch_summary_row(
    file_name: str,
    status: str,
    results: Optional[Dict[str, object]],
    extracted: Optional[Dict[str, object]],
    error_message: str = "",
) -> Dict[str, object]:
    """Build a single summary table row for the batch overview.

    Args:
        file_name: Original uploaded filename.
        status: ``"OK"`` or ``"ERROR"``.
        results: Result dict from ``process_image()`` (``None`` on error).
        extracted: Dict from ``_build_structured_answers()`` (``None`` on error).
        error_message: Human-readable error string (empty on success).

    Returns:
        Dict with display-ready column values.
    """
    if status != "OK" or results is None or extracted is None:
        return {
            "File": file_name, "Status": status, "Error": error_message,
            "SBD": "", "Mã đề": "", "Part I": "", "Part II": "",
            "Part III": "", "Boxes": "", "Preprocess": "",
        }

    fc_valid = sum(1 for v in extracted["fc"].values() if v and v[0] >= 0)
    tf_valid = sum(1 for v in extracted["tf"].values() if v and v[0] >= 0)
    dg_valid = sum(1 for v in extracted["dg"].values() if v and v != "X")
    return {
        "File":       file_name,
        "Status":     status,
        "SBD":        extracted["sbd"],
        "Mã đề":      extracted["mdt"],
        "Part I":     f"{fc_valid}/40",
        "Part II":    f"{tf_valid}/32",
        "Part III":   f"{dg_valid}/6",
        "Boxes":      len(results["data"]["boxes"]),
        "Preprocess": results["preprocess_mode"],
        "Error":      "",
    }

def _worker_process_single(
    name: str,
    payload_type: str,
    payload_data: object,
    f1: float,
    f2: float,
    f3: float
) -> Dict[str, object]:
    """Top-level worker function for ProcessPoolExecutor to avoid pickling closures/huge matrices."""
    import cv2
    import numpy as np

    try:
        # Decode image in the worker process individually
        if payload_type == "path":
            img = cv2.imread(str(payload_data))
            if img is None:
                raise ValueError(f"Không thể đọc ảnh từ {payload_data}")
        elif payload_type == "bytes":
            file_bytes = np.asarray(payload_data, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Không thể decode ảnh upload")
        elif payload_type == "error":
            raise ValueError(str(payload_data))
        else:
            raise ValueError(f"Loại dữ liệu worker không hỗ trợ: {payload_type}")

        results = process_image(img, fill_ratio_part1=f1, fill_ratio_part2=f2, fill_ratio_part3=f3, debug_prefix=None)
        extracted = _build_structured_answers(results, file_name=name)
        return {
            "file_name": name, "status": "OK", "error": "",
            "image": img, "results": results, "extracted": extracted,
            "summary": _build_batch_summary_row(name, "OK", results, extracted)
        }
    except Exception as e:
        return {
            "file_name": name, "status": "ERROR", "error": str(e),
            "image": None, "results": None, "extracted": None,
            "summary": _build_batch_summary_row(name, "ERROR", None, None, error_message=str(e))
        }

# ---------------------------------------------------------------------------
#  UI rendering
# ---------------------------------------------------------------------------


def _render_detailed_result(
    results: Dict[str, object],
    extracted: Dict[str, object],
    debug_mode: bool,
) -> None:
    """Render the full per-image result panel in Streamlit.

    Args:
        results: Result dict from ``process_image()``.
        extracted: Dict from ``_build_structured_answers()``.
        debug_mode: If ``True``, additional debug panels are shown.
    """
    st.markdown("---")
    st.subheader("📝 Câu Trả Lời Được Trích Xuất")

    tab1, tab2, tab3, tab4, tab5, tab_json = st.tabs(
        ["SBD (ID)", "Mã Đề", "Phần I", "Phần II", "Phần III", "JSON"]
    )

    with tab1:
        st.markdown("### Số Báo Danh")
        if extracted["sbd"]:
            if extracted["sbd_invalid"]:
                st.warning(f"SBD: {extracted['sbd']} (chưa chắc chắn hoàn toàn)")
            else:
                st.success(f"SBD: {extracted['sbd']}")
            st.dataframe(_digit_eval_table(results["sbd_digits"]), width="stretch")
        else:
            st.info("Không tìm thấy dữ liệu SBD")

    with tab2:
        st.markdown("### Mã Đề")
        if extracted["mdt"]:
            if extracted["mdt_invalid"]:
                st.warning(f"Mã Đề: {extracted['mdt']} (chưa chắc chắn hoàn toàn)")
            else:
                st.success(f"Mã Đề: {extracted['mdt']}")
            st.dataframe(_digit_eval_table(results["made_digits"]), width="stretch")
        else:
            st.info("Không tìm thấy dữ liệu Mã Đề")

    # --- Part I: Multiple Choice ---
    with tab3:
        st.markdown("### Phần I - Trắc Nghiệm (40 câu)")
        fc_data    = extracted["fc"]
        fc_invalid = extracted["fc_invalid"]
        choice_map = {0: "A", 1: "B", 2: "C", 3: "D", -2: "X"}

        if any(fc_data.values()):
            if fc_invalid:
                st.warning(
                    f"⚠️ {len(fc_invalid)} câu có nhiều đáp án: "
                    f"{', '.join(sorted(fc_invalid, key=int))}"
                )
            cols = st.columns(4)
            for q_num in range(1, 41):
                q_str       = str(q_num)
                answer_idx  = fc_data[q_str][0] if fc_data[q_str] else -1
                col_idx     = (q_num - 1) % 4
                with cols[col_idx]:
                    st.metric(f"Q{q_num}", choice_map.get(answer_idx, "-"))
            total_valid = sum(1 for v in fc_data.values() if v and v[0] >= 0)
            st.info(f"Tổng cộng câu trả lời hợp lệ: {total_valid}/40")
        else:
            st.info("Không phát hiện câu trả lời nào trong Phần I")

    # --- Part II: True/False ---
    with tab4:
        st.markdown("### Phần II - Đúng/Sai (32 câu)")
        tf_data    = extracted["tf"]
        tf_invalid = extracted["tf_invalid"]
        tf_map     = {0: "Sai", 1: "Đúng", -2: "X"}

        if any(tf_data.values()):
            if tf_invalid:
                st.warning(
                    f"⚠️ {len(tf_invalid)} câu có nhiều đáp án: "
                    f"{', '.join(sorted(tf_invalid, key=int))}"
                )
            cols = st.columns(4)
            for q_num in range(1, 33):
                q_str      = str(q_num)
                answer_idx = tf_data[q_str][0] if tf_data[q_str] else -1
                col_idx    = (q_num - 1) % 4
                with cols[col_idx]:
                    st.metric(f"Q{q_num}", tf_map.get(answer_idx, "-"))
            total_valid = sum(1 for v in tf_data.values() if v and v[0] >= 0)
            st.info(f"Tổng cộng câu trả lời hợp lệ: {total_valid}/32")
        else:
            st.info("Không phát hiện câu trả lời nào trong Phần II")

    # --- Part III: Numeric ---
    with tab5:
        st.markdown("### Phần III - Nhập Số (6 câu)")
        dg_data    = extracted["dg"]
        dg_invalid = extracted["dg_invalid"]

        if any(dg_data.values()):
            if dg_invalid:
                st.warning(
                    f"⚠️ {len(dg_invalid)} câu không hợp lệ: "
                    f"{', '.join(sorted(dg_invalid, key=int))}"
                )
            cols = st.columns(3)
            for cau_num in range(1, 7):
                cau_str = str(cau_num)
                answer  = dg_data[cau_str]
                col_idx = (cau_num - 1) % 3
                with cols[col_idx]:
                    st.metric(f"Câu {cau_num}", answer if answer else "-")
            total_valid = sum(1 for v in dg_data.values() if v and v != "X")
            st.info(f"Tổng cộng câu trả lời hợp lệ: {total_valid}/6")
        else:
            st.info("Không phát hiện câu trả lời nào trong Phần III")

    # --- JSON export ---
    with tab_json:
        st.markdown("### JSON Output")
        json_payload = _build_json_payload(extracted)
        st.json(json_payload)
        st.code(json.dumps([json_payload], indent=2, ensure_ascii=False), language="json")

    # --- Debug panel ---
    if debug_mode:
        st.markdown("---")
        st.subheader("🔍 Thông Tin Debug")

        debug_col1, debug_col2 = st.columns(2)
        with debug_col1:
            st.markdown("**Overlay Parts**")
            st.image(_to_rgb(results["parts_overlay"]), width="stretch")
        with debug_col2:
            if results["binary_threshold"] is not None:
                st.markdown("**Binary Threshold**")
                st.image(_to_rgb(results["binary_threshold"]), width="stretch")
            else:
                st.markdown("**Boxes Overlay**")
                st.image(_to_rgb(results["data"]["boxes_overlay"]), width="stretch")

        evals_all = results["part_i_evals"] + results["part_ii_evals"] + results["part_iii_evals"]
        total_cells  = len(evals_all)
        total_filled = sum(1 for e in evals_all if e.get("filled", False))

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Detected Boxes", len(results["data"]["boxes"]))
        with s2:
            st.metric("Tổng Cộng Ô", total_cells)
        with s3:
            pct = (total_filled / max(1, total_cells)) * 100.0
            st.metric("Tỷ Lệ Điền", f"{pct:.1f}%")

        st.caption(
            f"Preprocess: {results['preprocess_mode']} | "
            f"Part I/II/III: {len(results['parts']['part_i'])}/"
            f"{len(results['parts']['part_ii'])}/{len(results['parts']['part_iii'])} | "
            f"Upper split X: {float(results['split_x']):.1f}"
        )


# ---------------------------------------------------------------------------
#  Streamlit app entry point
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Nhận dạng phiếu trả lời",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Nhận dạng phiếu trả lời THPT Quốc gia")
st.markdown("Tải lên ảnh phiếu để nhận dạng và trích xuất đáp án tự động.")

# --- Sidebar configuration ---
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    debug_mode = st.checkbox("Chế Độ Debug", value=True)

    st.markdown("---")
    st.markdown("### Ngưỡng Fill Ratio")
    fill_ratio_phan1 = st.slider("PHẦN I",   0.30, 0.95, 0.65, 0.01)
    fill_ratio_phan2 = st.slider("PHẦN II",  0.30, 0.95, 0.65, 0.01)
    fill_ratio_phan3 = st.slider("PHẦN III", 0.30, 0.95, 0.65, 0.01)

    st.markdown("---")
    st.caption(
        "Luồng xử lý: detect box → group part → "
        "decode SBD/Mã đề → grid fill ratio → trích xuất đáp án"
    )

# --- Input method ---
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📤 Đầu Vào Hình Ảnh")
    tab_files, tab_folder = st.tabs(["Tải Lên File", "Nhập Thư Mục (Gợi ý)"])
    
    with tab_files:
        uploaded_files = st.file_uploader(
            "Chọn 1 hoặc nhiều ảnh",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )
        process_clicked_upload = st.button(
            "🚀 Bắt đầu xử lý (File)",
            type="primary",
            width="stretch",
            disabled=not uploaded_files,
            key="btn_upload"
        )
        
    with tab_folder:
        st.info("Khai báo thư mục chứa ảnh. Phương pháp này siêu nhanh vì bỏ qua bước upload của trình duyệt web.")
        folder_path = st.text_input("Đường dẫn (ví dụ: PhieuQG/):", value="PhieuQG/")
        process_clicked_folder = st.button(
            "🚀 Bắt đầu quét thư mục",
            type="primary",
            width="stretch",
            disabled=not folder_path,
            key="btn_folder"
        )

process_clicked = process_clicked_upload or process_clicked_folder
mode = "folder" if process_clicked_folder else "upload"

# --- Session state: cache results to avoid reprocessing on slider changes ---
_KEY_RESULTS   = "batch_results"
_KEY_SIGNATURE = "batch_signature"

if _KEY_RESULTS not in st.session_state:
    st.session_state[_KEY_RESULTS] = []
if _KEY_SIGNATURE not in st.session_state:
    st.session_state[_KEY_SIGNATURE] = ()

import time
if "batch_sig_val" not in st.session_state:
    st.session_state["batch_sig_val"] = ()

current_signature: Tuple[object, ...] = ()
if mode == "upload" and uploaded_files:
    current_signature = tuple((f.name, int(getattr(f, "size", 0))) for f in uploaded_files)
    st.session_state["batch_sig_val"] = current_signature
elif mode == "folder" and folder_path:
    if process_clicked_folder:
        st.session_state["batch_sig_val"] = (folder_path, time.time())
    current_signature = st.session_state["batch_sig_val"]

# --- Batch processing ---
if process_clicked:
    progress_bar = st.progress(0)
    status_text  = st.empty()

    batch_results: List[Dict[str, object]] = []
    tasks = []

    # 1. Gather tasks as lightweight payloads (avoid pickling gigantic active numpy arrays)
    if mode == "folder" and folder_path:
        folder = Path(folder_path)
        if folder.exists() and folder.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for p in folder.glob(ext):
                    # We pass the absolute Path object directly
                    tasks.append((p.name, "path", str(p)))
        else:
            st.error(f"Thư mục không tồn tại: {folder_path}")
            
    elif mode == "upload" and uploaded_files:
        for uploaded in uploaded_files:
            try:
                raw_bytes = bytearray(uploaded.getvalue())
                tasks.append((uploaded.name, "bytes", raw_bytes))
            except Exception as err:
                tasks.append((uploaded.name, "error", str(err)))
                
    total_files = len(tasks)

    # 2. Process concurrently avoiding the GIL via ProcessPoolExecutor
    processed_count = 0
    import multiprocessing
    max_workers = max(1, min(12, multiprocessing.cpu_count() - 1))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(_worker_process_single, t[0], t[1], t[2], fill_ratio_phan1, fill_ratio_phan2, fill_ratio_phan3): t[0]
            for t in tasks
        }
        for future in concurrent.futures.as_completed(future_to_name):
            processed_count += 1
            fname = future_to_name[future]
            status_text.text(f"Đang xử lý {processed_count}/{total_files}: {fname}")
            progress_bar.progress(int((processed_count / max(1, total_files)) * 100))
            try:
                batch_results.append(future.result())
            except Exception as e:
                pass # Handled internally by _worker_process_single

    if total_files > 0:
        progress_bar.progress(100)
        status_text.text(f"Hoàn thành xử lý {total_files} ảnh")
        
        if mode == "upload":
            original_order = {f.name: i for i, f in enumerate(uploaded_files)}
            batch_results.sort(key=lambda x: original_order.get(str(x["file_name"]), 0))
        else:
            batch_results.sort(key=lambda x: str(x["file_name"]))
            
        st.session_state[_KEY_RESULTS]   = batch_results
        st.session_state[_KEY_SIGNATURE] = current_signature

# --- Display batch results ---
has_batch_results = (
    (bool(uploaded_files) or (mode == "folder" and bool(folder_path)))
    and bool(st.session_state[_KEY_RESULTS])
    and st.session_state[_KEY_SIGNATURE] == current_signature
)

if has_batch_results:
    batch_results = st.session_state[_KEY_RESULTS]
    success_items = [item for item in batch_results if item["status"] == "OK"]
    error_items   = [item for item in batch_results if item["status"] != "OK"]

    st.markdown("---")
    st.subheader("📋 Kết Quả Tổng Hợp")

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Tổng số ảnh",  len(batch_results))
    with m2: st.metric("Thành công",   len(success_items))
    with m3: st.metric("Lỗi",          len(error_items))

    st.dataframe([item["summary"] for item in batch_results], width="stretch")

    # JSON download
    combined_payload = []
    for item in success_items:
        payload = _build_json_payload(item["extracted"])
        payload["file_name"] = item["file_name"]
        combined_payload.append(payload)

    st.download_button(
        label="⬇️ Tải JSON tổng hợp",
        data=json.dumps(combined_payload, indent=2, ensure_ascii=False),
        file_name="ket_qua_tong_hop.json",
        mime="application/json",
        width="stretch",
        disabled=not combined_payload,
    )

    # Per-image detail view
    if success_items:
        selected_file_name = st.selectbox(
            "Chọn ảnh để xem chi tiết",
            options=[item["file_name"] for item in success_items],
        )
        selected_item = next(item for item in success_items if item["file_name"] == selected_file_name)

        with col1:
            st.subheader("📸 Hình Ảnh Gốc")
            st.image(_to_rgb(selected_item["image"]), width="stretch")

        with col2:
            st.subheader("📊 Kết Quả Phân Tích")
            st.image(_to_rgb(selected_item["results"]["result_image"]), width="stretch")

        _render_detailed_result(
            selected_item["results"],
            selected_item["extracted"],
            debug_mode=debug_mode,
        )
    else:
        st.error("Không có ảnh nào xử lý thành công để hiển thị chi tiết.")

elif uploaded_files:
    st.info("Nhấn '🚀 Bắt đầu xử lý' để chạy batch và xem tiến độ/kết quả tổng hợp.")
else:
    st.info("👆 Tải lên một hình ảnh để bắt đầu")
    st.markdown("---")
    st.subheader("📖 Cách sử dụng")
    st.markdown(
        """
        1. Tải lên một hoặc nhiều ảnh phiếu trả lời.
        2. Chỉnh ngưỡng fill ratio ở thanh bên nếu cần.
        3. Nhấn nút xử lý để theo dõi tiến độ batch.
        4. Xem bảng tổng hợp và chọn từng ảnh để xem chi tiết.
        5. Bật Debug để xem thêm thông tin nội bộ của pipeline.
        """
    )
