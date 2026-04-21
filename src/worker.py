"""
worker.py — Top-level worker functions for ProcessPoolExecutor.

Must be a separate module (not defined inside app.py) so that Windows
``spawn``-based multiprocessing does NOT re-import ``app.py`` when forking a
new worker process.  Re-importing ``app.py`` in a worker would trigger all
top-level Streamlit calls (``st.set_page_config``, ``st.title``, …) without
a Streamlit runtime, flooding the console with
``missing ScriptRunContext`` warnings.

Only import lightweight, pure-Python / OpenCV modules here — never ``streamlit``.
"""

from __future__ import annotations

from typing import Dict


def _suppress_streamlit_logging() -> None:
    """Remove Streamlit's logging handlers from the root logger in this process.

    When Streamlit is imported in the main process it attaches its own handler
    to the root ``logging`` logger.  On Windows, ``ProcessPoolExecutor`` spawns
    workers by importing the same modules, so the Streamlit handler leaks into
    every worker and emits ``missing ScriptRunContext`` on every log call.

    Calling this once at the start of a worker process removes all such handlers
    silently, so the pipeline can log normally to stdout without any warnings.
    """
    import logging

    root = logging.getLogger()
    for handler in list(root.handlers):
        # Streamlit handlers live in streamlit.* namespaces
        module = type(handler).__module__ or ""
        if module.startswith("streamlit"):
            root.removeHandler(handler)

    # Also strip Streamlit handlers from every named logger that exists
    for name in list(logging.Logger.manager.loggerDict.keys()):
        lgr = logging.getLogger(name)
        for h in list(lgr.handlers):
            if (type(h).__module__ or "").startswith("streamlit"):
                lgr.removeHandler(h)


def detect_single(
    name: str,
    payload_type: str,
    payload_data: object,
    max_dim: int,
) -> Dict[str, object]:
    """Worker entry-point: decode one image and run detect_image().

    This function is called by ``concurrent.futures.ProcessPoolExecutor``
    in a **separate OS process** on Windows.  It must be importable from the
    top level of a module that does **not** import streamlit.

    Args:
        name: Original filename (for error reporting / ordering).
        payload_type: One of ``"path"``, ``"bytes"``, ``"error"``.
        payload_data: File path string, raw bytes array, or error message.
        max_dim: Passed directly to :func:`src.pipeline.detect_image`.

    Returns:
        Dict with keys:
        - ``file_name``  — echoed *name*
        - ``status``     — ``"OK"`` or ``"ERROR"``
        - ``error``      — empty string on success, exception message on failure
        - ``detection``  — result of ``detect_image()`` (``None`` on error)
        - ``image``      — decoded BGR ``np.ndarray`` (``None`` on error)
    """
    _suppress_streamlit_logging()  # Must be first — removes Streamlit log handlers

    import cv2
    import numpy as np

    try:
        if payload_type == "path":
            img = cv2.imread(str(payload_data))
            if img is None:
                raise ValueError(f"Không thể đọc ảnh: {payload_data}")
        elif payload_type == "bytes":
            arr = np.asarray(bytearray(payload_data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Không thể decode ảnh upload")
        elif payload_type == "error":
            raise ValueError(str(payload_data))
        else:
            raise ValueError(f"payload_type không hỗ trợ: {payload_type!r}")

        from src.pipeline import detect_image  # noqa: PLC0415
        # Detect logic (phase 1)
        # Always use max_dim=0 because morphology parameters are hardcoded for 2000px+
        detection = detect_image(img, max_dim=0)
        return {
            "file_name": name,
            "status":    "OK",
            "error":     "",
            "detection": detection,
            "image":     img,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "file_name": name,
            "status":    "ERROR",
            "error":     str(exc),
            "detection": None,
            "image":     None,
        }
