"""
helmet.py — TrafficSentinel Helmet Detection Module (with Violation Data)
══════════════════════════════════════════════════════════════════════════
ONLY detects "no helmet" violations.
- Red box drawn on frame
- Returns structured violation dict for challan processing
- Evidence capture supported (confidence, bbox, frame)
- No IoU matching, no vehicle assignment, no green boxes
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("TrafficSentinel.Helmet")

# ── Constants ─────────────────────────────────────────────────────────────────
HELMET_CONF         = 0.50
HELMET_MIN_HEIGHT   = 30
HELMET_FRAME_SKIP   = 3
COOLDOWN_FRAMES     = 300        # ~12s at 25fps

NO_HELMET_KEYWORDS  = {"no", "without", "no_helmet", "without_helmet", "nohelmet"}


# ── Model path search ─────────────────────────────────────────────────────────
def _find_model(name: str = "best.pt") -> str:
    search_dirs = [
        "",
        "/content",
        "/content/drive/MyDrive",
        "/content/drive/MyDrive/YOLOv8",
        "/content/drive/MyDrive/models",
        "/content/models",
        "/content/weights",
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for d in search_dirs:
        candidate = os.path.join(d, name) if d else name
        if os.path.isfile(candidate):
            log.info(f"[Helmet] Found model: {candidate}")
            return candidate
    return name


def _is_no_helmet(label: str) -> bool:
    label_lower = label.lower().replace(" ", "_").replace("-", "_")
    for kw in NO_HELMET_KEYWORDS:
        if kw in label_lower:
            return True
    return False


# ── Spatial deduplication grid ────────────────────────────────────────────────
class _ZoneDedup:
    """
    Prevents same spatial zone from firing repeated violations
    within COOLDOWN_FRAMES. Grid-based — no track ID needed.
    """
    GRID_COLS = 8
    GRID_ROWS = 5

    def __init__(self) -> None:
        self._last: Dict[Tuple[int, int], int] = {}

    def _cell(self, cx: int, cy: int, w: int, h: int) -> Tuple[int, int]:
        col = min(int(cx / max(w, 1) * self.GRID_COLS), self.GRID_COLS - 1)
        row = min(int(cy / max(h, 1) * self.GRID_ROWS), self.GRID_ROWS - 1)
        return col, row

    def is_duplicate(self, cx: int, cy: int, w: int, h: int, frame_no: int) -> bool:
        cell = self._cell(cx, cy, w, h)
        last = self._last.get(cell, -COOLDOWN_FRAMES - 1)
        if frame_no - last < COOLDOWN_FRAMES:
            return True
        self._last[cell] = frame_no
        return False


# ── Main Detector ─────────────────────────────────────────────────────────────

class HelmetDetector:
    """
    Pure direct YOLO helmet violation detector.

    draw_violations(frame, stream_id, frame_no) returns:
        (annotated_frame, [violation_dict, ...])

    Each violation_dict is fully populated for:
        - DB insertion  (violations table)
        - Evidence save (AsyncDBWriter)
        - Challan flow  (/api/challan/create)
    """

    def __init__(self, model_path: str = "best.pt") -> None:
        self._model_path: str = _find_model(model_path)
        self._model: Optional[Any] = None
        self._ready: bool = False
        self._device: str = "cpu"
        self._skip_ctr: int = 0
        self._lock = threading.Lock()
        self._dedup = _ZoneDedup()

        if os.path.exists(self._model_path):
            threading.Thread(target=self._load, daemon=True, name="Helmet-Load").start()
        else:
            log.warning(
                f"[Helmet] Model not found at '{self._model_path}'. "
                "Place best.pt in the working directory."
            )

    # ── Load ───────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            from ultralytics import YOLO as _YOLO
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"

            log.info(f"[Helmet] Loading {self._model_path} on {self._device} …")
            model = _YOLO(self._model_path)
            if self._device == "cuda":
                model.to("cuda")

            dummy = np.zeros((360, 640, 3), dtype=np.uint8)
            model(dummy, conf=HELMET_CONF, verbose=False)

            with self._lock:
                self._model = model
                self._ready = True

            log.info("[Helmet] Ready — NO-HELMET only, violation data enabled.")
            if hasattr(model, "names"):
                log.info(f"[Helmet] Classes: {model.names}")
        except Exception as exc:
            log.error(f"[Helmet] Load failed: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_path(self) -> str:
        return self._model_path

    def draw_violations(
        self,
        frame: np.ndarray,
        stream_id: str = "default",
        frame_no: int = 0,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run YOLO on frame.
        Returns (annotated_frame, violations_list).

        Violations list is EMPTY on skipped frames or duplicate zones
        to prevent duplicate DB entries. Box is still drawn on frame
        even for duplicates (it's a real detection, just not re-logged).
        """
        if not self._ready or self._model is None:
            return frame, []

        self._skip_ctr += 1
        if self._skip_ctr < HELMET_FRAME_SKIP:
            return frame, []
        self._skip_ctr = 0

        h, w = frame.shape[:2]
        violations: List[Dict] = []
        ts = datetime.now().isoformat()

        try:
            with self._lock:
                results = self._model(frame, conf=HELMET_CONF, verbose=False)[0]

            if results.boxes is None or len(results.boxes) == 0:
                return frame, []

            for box, cls_t, conf_t in zip(
                results.boxes.xyxy, results.boxes.cls, results.boxes.conf
            ):
                x1, y1, x2, y2 = map(int, box.tolist())
                cls_id     = int(cls_t)
                confidence = float(conf_t)
                label      = results.names.get(cls_id, str(cls_id))

                # ── Filters ────────────────────────────────────────────────────
                if confidence < HELMET_CONF:
                    continue
                if (y2 - y1) < HELMET_MIN_HEIGHT:
                    continue
                if not _is_no_helmet(label):
                    continue  # has helmet → silently skip

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Always draw the box (valid detection)
                _draw_no_helmet_box(frame, x1, y1, x2, y2, confidence)

                # Deduplicate: don't re-fire DB event for same zone
                if self._dedup.is_duplicate(cx, cy, w, h, frame_no):
                    continue

                # ── Full violation dict ────────────────────────────────────────
                vid = str(uuid.uuid4())[:8]
                v: Dict = {
                    # Identity
                    "id":             vid,
                    "type":           "no_helmet",
                    "label":          "No Helmet",
                    "severity":       "MEDIUM",
                    # Spatial
                    "bbox":           [x1, y1, x2, y2],
                    "centroid":       [cx, cy],
                    "orig_w":         w,
                    "orig_h":         h,
                    # Detection
                    "confidence":     round(confidence, 3),
                    "track_id":       f"H_{vid}",   # pseudo track — no vehicle link
                    "vehicle":        "rider",
                    "vehicle_number": None,          # set later via challan UI
                    # Context
                    "frame_no":       frame_no,
                    "stream_id":      stream_id,
                    "timestamp":      ts,
                    # Speed / lane (N/A for helmet)
                    "speed_kmh":      0.0,
                    "lane_id":        -1,
                    "challan_id":     None,
                    "evidence_file":  "",
                }
                violations.append(v)
                log.info(
                    f"[Helmet] VIOLATION  conf={confidence:.2f}  "
                    f"bbox=[{x1},{y1},{x2},{y2}]  frame={frame_no}  stream={stream_id}"
                )

        except Exception as exc:
            log.error(f"[Helmet] Inference error: {exc}")

        return frame, violations

    def status(self) -> Dict:
        return {
            "model_path":  self._model_path,
            "file_exists": os.path.exists(self._model_path),
            "model_ready": self._ready,
            "device":      self._device,
            "mode":        "NO_HELMET_ONLY" if self._ready else "disabled",
            "conf_thresh": HELMET_CONF,
            "min_box_h":   HELMET_MIN_HEIGHT,
        }


# ── Drawing ────────────────────────────────────────────────────────────────────

def _draw_no_helmet_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    confidence: float,
) -> None:
    color      = (0, 0, 255)
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Corner accent marks
    cl = max(8, min(16, min(x2 - x1, y2 - y1) // 5))
    for (cx_, cy_, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx_, cy_), (cx_ + dx * cl, cy_), color, 2)
        cv2.line(frame, (cx_, cy_), (cx_, cy_ + dy * cl), color, 2)

    tag = f"NO HELMET  {confidence*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(tag, font, font_scale, thickness)
    ty = max(y1 - 4, th + 6)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
    cv2.putText(frame, tag, (x1 + 3, ty - 1), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)