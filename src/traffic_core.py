"""
traffic_core.py
===============================================================================
TrafficSentinel — Enhancing Road Traffic Violation Detection
using Mask2Former and Real-Time Video Stream Analytics

Supports two camera modes (auto-detected per video):
  OVERHEAD  — static mounted camera, vehicles move toward camera (y increases)
  DASHCAM   — vehicle-mounted, tracked vehicles move away (y decreases)

Classes
-------
  AppConfig              centralised configuration & directory management
  DetectionManager       YOLOv8 / synthetic fallback
  SegmentationManager    Mask2Former / polygon fallback
  TrackedObject          per-object state container
  TrackingManager        ByteTrack-compatible IoU tracker
  OCRManager             EasyOCR number plate extraction
  EvidenceManager        evidence images + CSV logging
  ViolationEngine        temporal FSM violation detection
  SignalDetector         traffic light state (HSV + YOLO)
  OverlayRenderer        frame annotation & HUD
  HistoryManager         session JSON persistence
  TrafficViolationSystem top-level orchestrator
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("traffic_core")

# ── Optional heavy-weight imports ─────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
    logger.info("Ultralytics YOLOv8 available.")
except ImportError:
    _YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — synthetic detections active.")

try:
    import torch
    _TORCH_AVAILABLE = True
    logger.info("PyTorch %s available.", torch.__version__)
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch not installed — polygon segmentation fallback active.")

try:
    import easyocr
    _OCR_AVAILABLE = True
    logger.info("EasyOCR available.")
except ImportError:
    _OCR_AVAILABLE = False
    logger.warning("easyocr not installed — OCR returns empty strings.")


# =============================================================================
# 1. AppConfig
# =============================================================================

class AppConfig:
    """Centralised configuration for the entire TrafficSentinel pipeline."""

    # ── Directory paths ───────────────────────────────────────────────────────
    STATIC_DIR       = "static"
    UPLOAD_DIR       = os.path.join(STATIC_DIR, "uploads")
    PROCESSED_DIR    = os.path.join(STATIC_DIR, "processed")
    EVIDENCE_DIR     = os.path.join(STATIC_DIR, "evidence")
    LOG_DIR          = os.path.join(STATIC_DIR, "logs")
    EVIDENCE_SUBDIRS = ["red_light", "helmet", "lane", "speed", "plates"]

    # ── Video processing ──────────────────────────────────────────────────────
    PROC_WIDTH   = 960
    PROC_HEIGHT  = 540
    MAX_FRAMES   = 1800

    # ── YOLO detection ────────────────────────────────────────────────────────
    CONF_THRESHOLD      = 0.40
    NMS_IOU             = 0.45
    YOLO_MODEL_PATH     = os.environ.get("YOLO_MODEL", "yolov8n.pt")
    VEHICLE_CLASSES     = {"car", "motorcycle", "bus", "truck", "bicycle"}
    PERSON_CLASSES      = {"person"}
    PLATE_CLASS         = "license_plate"
    TRAFFIC_LIGHT_CLASS = "traffic light"

    # ── FSM windows ───────────────────────────────────────────────────────────
    RED_LIGHT_MIN_AGE   = 4    # min track age (frames) before red-light check
    HELMET_FRAMES       = 6
    LANE_FRAMES         = 8
    LANE_CONFIRM_FRAMES = 16
    SPEED_FRAMES        = 5
    SPEED_MIN_HISTORY   = 12   # min centroid history frames before speed is trusted
    OCR_BEST_FRAMES     = 7

    # ── Confidence gate ───────────────────────────────────────────────────────
    VIOLATION_MIN_SCORE = 0.75

    # ── Speed calibration ─────────────────────────────────────────────────────
    PIXEL_PER_METRE  = 14.0   # approximate; replace with homography
    FPS_DEFAULT      = 25.0
    SPEED_LIMIT_KMH  = 60.0
    SPEED_MAX_KMH    = 180.0

    # ── Track cooldown ────────────────────────────────────────────────────────
    VIOLATION_COOLDOWN_FRAMES = 90

    # ── Signal HSV ranges ────────────────────────────────────────────────────
    # Real traffic light LEDs appear warm orange-red (H≈5-25) not pure red.
    # We use a wide red range covering both warm-red and deep-red LEDs.
    # Saturation/value lowered to 80 to catch overexposed or distant lights.
    RED_HUE_LO1   = (0,   80, 80)    # deep red + orange-red LED (wider sat/val)
    RED_HUE_HI1   = (25,  255, 255)  # covers warm/amber-red bulbs
    RED_HUE_LO2   = (155, 80, 80)    # deep red wrapping at 180
    RED_HUE_HI2   = (180, 255, 255)
    # Green LED: slightly relaxed sat+val to catch dim/distant lights
    GREEN_HUE_LO  = (38,  100, 100)
    GREEN_HUE_HI  = (88,  255, 255)
    # Yellow/amber LED
    YELLOW_HUE_LO = (15,  100, 100)
    YELLOW_HUE_HI = (38,  255, 255)

    @classmethod
    def ensure_directories(cls) -> None:
        dirs = [cls.UPLOAD_DIR, cls.PROCESSED_DIR, cls.LOG_DIR, cls.EVIDENCE_DIR]
        for sub in cls.EVIDENCE_SUBDIRS:
            dirs.append(os.path.join(cls.EVIDENCE_DIR, sub))
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# =============================================================================
# 2. DetectionManager
# =============================================================================

class DetectionManager:
    """YOLOv8 object detection with synthetic fallback."""

    COCO_KEEP = {"car", "motorcycle", "bus", "truck", "bicycle", "person", "traffic light"}

    def __init__(self) -> None:
        self._model: Any  = None
        self._syn_counter = 0
        self._load_model()

    def _load_model(self) -> None:
        if not _YOLO_AVAILABLE:
            return
        try:
            self._model = _YOLO(AppConfig.YOLO_MODEL_PATH)
            logger.info("YOLOv8 loaded: %s", AppConfig.YOLO_MODEL_PATH)
        except Exception as exc:
            logger.warning("YOLOv8 load failed (%s) — synthetic mode.", exc)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self._model is not None:
            return self._yolo_inference(frame)
        return self._synthetic_detections(frame)

    def _yolo_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self._model.predict(frame, conf=AppConfig.CONF_THRESHOLD,
                                      iou=AppConfig.NMS_IOU, verbose=False)
        detections: List[Dict[str, Any]] = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])].lower()
                if cls_name not in self.COCO_KEEP:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({"bbox": [x1, y1, x2, y2], "class_name": cls_name,
                                   "confidence": float(box.conf[0]), "mask": None,
                                   "track_id": None, "source": "yolo"})
        return detections

    def set_camera_mode(self, mode: str) -> None:
        """Called by TrafficViolationSystem after camera mode detection."""
        self._camera_mode = mode

    def _synthetic_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Deterministic synthetic detections for demo / no-YOLO fallback.

        Camera-mode-aware behaviour
        ---------------------------
        DASHCAM mode:
          Generates one "crossing vehicle" that starts below the crosswalk
          (y > stop_y) and moves progressively upward (y decreasing) through
          the stop line over ~60 frames.  This correctly exercises the dashcam
          red-light FSM.  Other vehicles are placed as static background.

        OVERHEAD mode:
          Vehicles are placed below the stop line and held there (the slow
          8-frame seed keeps positions stable, preventing false FSM triggers
          from random teleportation).

        NOTE: This mode produces plausible behaviour for pipeline testing.
              Install ultralytics + yolov8n.pt for real detection.
        """
        h, w = frame.shape[:2]
        stop_y = int(h * 0.62)   # mirrors polygon_fallback stop line

        camera_mode = getattr(self, '_camera_mode', 'overhead')

        if camera_mode == "dashcam":
            # ── DASHCAM: one crossing vehicle + background cars ───────────────
            detections: List[Dict[str, Any]] = []
            fc = self._syn_counter
            self._syn_counter += 1

            # Crossing vehicle: starts at y≈stop_y+120 (below line),
            # moves up by 3px/frame so it crosses stop_y around frame 40
            # and is well past it by frame 80
            car_start_y = stop_y + 120
            car_y = max(80, car_start_y - fc * 3)
            car_x = w // 2 - 80
            car_w, car_h_box = 160, 100
            # Shrink as it moves away (perspective)
            scale = max(0.4, 1.0 - fc * 0.004)
            car_w2  = int(car_w  * scale)
            car_h2  = int(car_h_box * scale)
            detections.append({
                "bbox":       [car_x, car_y, car_x + car_w2, car_y + car_h2],
                "class_name": "car",
                "confidence": 0.92,
                "mask":       None,
                "track_id":   None,
                "source":     "synthetic",
            })

            # A few background vehicles with slow seed
            rng = random.Random((fc // 8) * 7919 + 99991)
            for _ in range(rng.randint(1, 3)):
                cls = rng.choice(["car", "car", "truck"])
                x1  = rng.randint(int(w*0.05), int(w*0.75))
                y1  = rng.randint(stop_y + 40, h - 70)
                x2  = min(x1 + rng.randint(60, 160), w)
                y2  = min(y1 + rng.randint(40, 100), h)
                detections.append({
                    "bbox":       [x1, y1, x2, y2],
                    "class_name": cls,
                    "confidence": round(rng.uniform(0.60, 0.90), 2),
                    "mask":       None,
                    "track_id":   None,
                    "source":     "synthetic",
                })
            return detections

        else:
            # ── OVERHEAD: persistent objects below stop line ──────────────────
            slow_seed = (self._syn_counter // 8) * 7919 + 12345
            rng       = random.Random(slow_seed)
            self._syn_counter += 1
            detections = []
            for _ in range(rng.randint(2, 5)):
                cls = rng.choice(["car", "car", "car", "motorcycle", "person"])
                x1  = rng.randint(int(w * 0.05), int(w * 0.80))
                y1  = rng.randint(stop_y + 40, h - 80)
                x2  = min(x1 + rng.randint(60, 200), w)
                y2  = min(y1 + rng.randint(40, 130), h)
                detections.append({
                    "bbox":       [x1, y1, x2, y2],
                    "class_name": cls,
                    "confidence": round(rng.uniform(0.60, 0.95), 2),
                    "mask":       None,
                    "track_id":   None,
                    "source":     "synthetic",
                })
            return detections


# =============================================================================
# 3. SegmentationManager
# =============================================================================

class SegmentationManager:
    """
    Mask2Former semantic segmentation with polygon fallback.

    Cityscapes labels used:
      0  road        8  lane markings (derived via Hough)
      1  sidewalk    9  terrain (restricted)
    """

    ROAD_IDS       = {0}
    RESTRICTED_IDS = {1, 9}
    SEG_STRIDE     = 15
    DEFAULT_MODEL  = "facebook/mask2former-swin-small-cityscapes-semantic"

    def __init__(self, frame_shape: Tuple[int, int]) -> None:
        self.height, self.width = frame_shape
        self._m2f_model:     Any = None
        self._m2f_processor: Any = None
        self._device:        str = "cpu"
        self._model_id:      str = ""
        self._last_result:   Optional[Dict[str, Any]] = None
        self._last_frame_no: int = -9999
        self._try_load_mask2former()

    def _try_load_mask2former(self) -> None:
        if not _TORCH_AVAILABLE:
            return
        try:
            import torch
            from transformers import (Mask2FormerForUniversalSegmentation,
                                      AutoImageProcessor)
            self._device = ("cuda" if torch.cuda.is_available() else
                           ("mps" if hasattr(torch.backends, "mps") and
                            torch.backends.mps.is_available() else "cpu"))
            model_id = os.environ.get("MASK2FORMER_MODEL", self.DEFAULT_MODEL)
            self._model_id = model_id
            self._m2f_processor = AutoImageProcessor.from_pretrained(
                model_id, do_resize=True, size={"height": 512, "width": 512})
            self._m2f_model = (Mask2FormerForUniversalSegmentation
                               .from_pretrained(model_id)
                               .to(self._device).eval())
            logger.info("Mask2Former ready on %s.", self._device)
        except Exception as exc:
            logger.warning("Mask2Former load failed (%s) — polygon fallback.", exc)
            self._m2f_model = None

    @property
    def ready(self) -> bool:
        return self._m2f_model is not None

    @property
    def backend(self) -> str:
        return (f"Mask2Former ({self._model_id})" if self.ready
                else "polygon-fallback")

    def segment(self, frame: np.ndarray, frame_number: int = 0) -> Dict[str, Any]:
        if (self._last_result is not None and
                (frame_number - self._last_frame_no) < self.SEG_STRIDE):
            return self._last_result
        result = (self._mask2former_inference(frame) if self._m2f_model is not None
                  else self._polygon_fallback(frame))
        result["backend"]    = self.backend
        self._last_result    = result
        self._last_frame_no  = frame_number
        return result

    def _mask2former_inference(self, frame: np.ndarray) -> Dict[str, Any]:
        import torch
        from PIL import Image
        h, w    = frame.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs  = {k: v.to(self._device) for k, v in
                   self._m2f_processor(images=pil_img, return_tensors="pt").items()}
        with torch.no_grad():
            outputs = self._m2f_model(**inputs)
        seg_map = (self._m2f_processor
                   .post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]
                   .cpu().numpy().astype(np.int32))
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        road_raw  = np.isin(seg_map, list(self.ROAD_IDS)).astype(np.uint8) * 255
        road_mask = cv2.morphologyEx(cv2.morphologyEx(road_raw, cv2.MORPH_CLOSE, kernel),
                                     cv2.MORPH_OPEN, kernel)
        lane_masks        = self._derive_lane_masks(frame, road_mask)
        stop_line         = self._derive_stop_line(road_mask, h, w)
        restr_raw         = np.isin(seg_map, list(self.RESTRICTED_IDS)).astype(np.uint8) * 255
        restricted_regions = self._mask_to_polygons(
            cv2.dilate(restr_raw, kernel, iterations=1), min_area=500)
        intersection_region = self._derive_intersection(road_mask, h, w)
        seg_overlay         = self._build_seg_overlay(seg_map, frame)
        return {"road_mask": road_mask, "lane_masks": lane_masks,
                "stop_line_polygon": stop_line, "restricted_regions": restricted_regions,
                "intersection_region": intersection_region,
                "seg_map": seg_map, "seg_overlay": seg_overlay}

    def _derive_lane_masks(self, frame: np.ndarray,
                           road_mask: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        h, w       = road_mask.shape[:2]
        grey_road  = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                     mask=road_mask)
        _, bright  = cv2.threshold(grey_road, 160, 255, cv2.THRESH_BINARY)
        lines      = cv2.HoughLinesP(cv2.Canny(bright, 50, 150), rho=1,
                                     theta=np.pi/180, threshold=40,
                                     minLineLength=h//5, maxLineGap=30)
        divider_xs: List[int] = []
        if lines is not None:
            for line in lines:
                x1l, y1l, x2l, y2l = line[0]
                angle = abs(math.degrees(math.atan2(abs(y2l-y1l), abs(x2l-x1l)+1e-6)))
                if angle > 60:
                    divider_xs.append((x1l+x2l)//2)
        clustered: List[int] = []
        for x in sorted(set(divider_xs)):
            if not clustered or abs(x - clustered[-1]) > 40:
                clustered.append(x)
        boundaries = [0] + clustered + [w]
        lane_masks: List[Tuple[np.ndarray, int]] = []
        for lid, (xl, xr) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            strip = np.zeros((h, w), dtype=np.uint8)
            strip[:, xl:xr] = 255
            lane_m = cv2.bitwise_and(strip, road_mask)
            if cv2.countNonZero(lane_m) > 200:
                lane_masks.append((lane_m, lid))
        if not lane_masks:
            band = w // 3
            for i in range(3):
                strip = np.zeros((h, w), dtype=np.uint8)
                strip[:, i*band:(i+1)*band] = 255
                lane_masks.append((cv2.bitwise_and(strip, road_mask), i+1))
        return lane_masks

    def _derive_stop_line(self, road_mask: np.ndarray,
                          h: int, w: int) -> List[Tuple[int, int]]:
        col_sum      = road_mask.sum(axis=1).astype(np.float32)
        search_start = int(h * 0.35)
        search_end   = int(h * 0.70)
        region       = col_sum[search_start:search_end]
        if region.max() > 0:
            grad     = np.gradient(region)
            stop_rel = int(np.argmin(grad))
            stop_y   = max(search_start, min(search_start + stop_rel, search_end))
        else:
            stop_y = int(h * 0.60)
        return [(0, stop_y), (w, stop_y)]

    def _derive_intersection(self, road_mask: np.ndarray,
                             h: int, w: int) -> List[Tuple[int, int]]:
        upper = road_mask[:int(h * 0.45), :]
        pts   = cv2.findNonZero(upper)
        if pts is None or len(pts) < 4:
            return []
        hull = cv2.convexHull(pts).reshape(-1, 2)
        return [(int(p[0]), int(p[1])) for p in hull]

    @staticmethod
    def _mask_to_polygons(mask: np.ndarray,
                          min_area: int = 500) -> List[List[Tuple[int, int]]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [[(int(p[0][0]), int(p[0][1])) for p in cv2.approxPolyDP(c, 3, True)]
                for c in contours if cv2.contourArea(c) >= min_area]

    @staticmethod
    def _build_seg_overlay(seg_map: np.ndarray, frame: np.ndarray) -> np.ndarray:
        PALETTE = np.array([
            [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
            [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
            [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
            [0,60,100],[0,80,100],[0,0,230],[119,11,32]], dtype=np.uint8)
        h, w  = seg_map.shape
        cmap  = np.zeros((h, w, 3), dtype=np.uint8)
        for lid, colour in enumerate(PALETTE):
            cmap[seg_map == lid] = colour
        return cv2.addWeighted(frame, 0.55, cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR), 0.45, 0)

    def _polygon_fallback(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Geometric fallback when Mask2Former is unavailable.
        Restricted regions are intentionally empty — they require real
        segmentation to avoid false positives on road markings.
        """
        h, w = self.height, self.width
        road_pts = np.array([[0,h],[w,h],[int(w*0.80),int(h*0.40)],
                              [int(w*0.20),int(h*0.40)]], dtype=np.int32)
        road_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(road_mask, [road_pts], 255)
        lane_masks: List[Tuple[np.ndarray, int]] = []
        x_top_start = int(w * 0.20)
        x_top_width = int(w * 0.80) - x_top_start
        y_top       = int(h * 0.40)
        for i in range(3):
            xl_top = x_top_start + i*(x_top_width//3)
            xr_top = x_top_start + (i+1)*(x_top_width//3)
            pts = np.array([[i*(w//3),h],[(i+1)*(w//3),h],
                             [xr_top,y_top],[xl_top,y_top]], dtype=np.int32)
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 255)
            lane_masks.append((cv2.bitwise_and(m, road_mask), i+1))
        stop_y  = int(h * 0.62)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [road_pts], (80, 50, 10))
        return {"road_mask": road_mask, "lane_masks": lane_masks,
                "stop_line_polygon": [(0, stop_y), (w, stop_y)],
                "restricted_regions": [],
                "intersection_region": [],
                "seg_map": np.zeros((h, w), dtype=np.int32),
                "seg_overlay": cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)}


# =============================================================================
# 4a. TrackedObject
# =============================================================================

class TrackedObject:
    """Per-object state container for tracking across frames."""

    HISTORY_LEN = 30

    def __init__(self, track_id: int, class_name: str, bbox: List[int]) -> None:
        self.track_id   = track_id
        self.class_name = class_name
        self.bbox       = bbox
        self.bbox_history:     deque = deque(maxlen=self.HISTORY_LEN)
        self.centroid_history: deque = deque(maxlen=self.HISTORY_LEN)
        self.velocity_history: deque = deque(maxlen=10)
        self.lane_history:     deque = deque(maxlen=10)
        self.centroid:          Tuple[int, int]      = (0, 0)
        self.bottom_center:     Tuple[int, int]      = (0, 0)
        self.direction_vec:     Tuple[float, float]  = (0.0, 0.0)
        self.speed_kmh:         float                = 0.0
        self.assigned_lane:     int                  = -1
        self.missed_frames:     int                  = 0

        # FSM state — starts as "new" not "before_line"
        # The FSM will set the real zone on the first evaluation frame;
        # only objects that are observed approaching AND crossing the line fire.
        self.crossing_state       = "new"     # new | before_line | on_line | crossed_line
        self.spawn_bottom_y:      int   = 0   # bottom_center.y at first detection
        self.spawn_frame:         int   = 0   # frame number when track was created

        self.helmet_decisions:    deque = deque(maxlen=10)
        self.lane_decisions:      deque = deque(maxlen=8)
        self.speed_decisions:     deque = deque(maxlen=8)
        self.violation_cooldown:  Dict[str, int] = {}
        self.violations_fired:    set             = set()
        # OCR
        self.plate_crops: List[np.ndarray] = []
        self.plate_texts: List[str]        = []
        self.best_plate_text:  str         = ""
        self._update(bbox)

    def _update(self, bbox: List[int]) -> None:
        self.bbox = bbox
        self.bbox_history.append(list(bbox))
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        self.centroid      = (cx, cy)
        self.bottom_center = (cx, bbox[3])
        self.centroid_history.append((cx, cy))
        self._estimate_velocity()

    def init_spawn(self, frame_number: int) -> None:
        """Record spawn position — called once after first update."""
        self.spawn_bottom_y = self.bottom_center[1]
        self.spawn_frame    = frame_number

    def _update(self, bbox: List[int]) -> None:
        self.bbox = bbox
        self.bbox_history.append(list(bbox))
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        self.centroid      = (cx, cy)
        self.bottom_center = (cx, bbox[3])
        self.centroid_history.append((cx, cy))
        self._estimate_velocity()

    def update(self, bbox: List[int]) -> None:
        self._update(bbox)
        self.missed_frames = 0

    def _estimate_velocity(self) -> None:
        hist = list(self.centroid_history)
        if len(hist) < 3:
            return
        tail = hist[-5:] if len(hist) >= 5 else hist
        dx   = tail[-1][0] - tail[0][0]
        dy   = tail[-1][1] - tail[0][1]
        n    = max(len(tail) - 1, 1)
        self.direction_vec = (dx / n, dy / n)
        pix_per_frame      = math.hypot(dx / n, dy / n)
        raw_kmh            = (pix_per_frame / AppConfig.PIXEL_PER_METRE *
                              AppConfig.FPS_DEFAULT * 3.6)
        self.speed_kmh     = round(min(raw_kmh, AppConfig.SPEED_MAX_KMH), 1)
        self.velocity_history.append(self.speed_kmh)

    def in_cooldown(self, vtype: str, current_frame: int) -> bool:
        return (current_frame - self.violation_cooldown.get(vtype, -9999)
                < AppConfig.VIOLATION_COOLDOWN_FRAMES)

    def set_cooldown(self, vtype: str, current_frame: int) -> None:
        self.violation_cooldown[vtype] = current_frame


# =============================================================================
# 4b. TrackingManager
# =============================================================================

class TrackingManager:
    """IoU centroid tracker with ByteTrack-compatible interface."""

    IOU_THRESHOLD = 0.35
    MAX_MISSED    = 10

    def __init__(self) -> None:
        self._tracks:  Dict[int, TrackedObject] = {}
        self._next_id: int                       = 1

    def update(self, detections: List[Dict[str, Any]],
               frame_number: int = 0) -> List[TrackedObject]:
        self._centroid_track(detections, frame_number)
        lost = [tid for tid, t in self._tracks.items()
                if t.missed_frames > self.MAX_MISSED]
        for tid in lost:
            del self._tracks[tid]
        return list(self._tracks.values())

    def _centroid_track(self, detections: List[Dict[str, Any]],
                        frame_number: int = 0) -> None:
        existing_ids   = list(self._tracks.keys())
        matched_exist: set = set()
        matched_det:   set = set()
        for di, det in enumerate(detections):
            best_iou, best_tid = self.IOU_THRESHOLD, None
            for tid in existing_ids:
                if tid in matched_exist:
                    continue
                iou = self._iou(det["bbox"], self._tracks[tid].bbox)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                self._tracks[best_tid].update(det["bbox"])
                det["track_id"] = best_tid
                matched_exist.add(best_tid)
                matched_det.add(di)
        for di, det in enumerate(detections):
            if di not in matched_det:
                nid = self._next_id; self._next_id += 1
                t   = TrackedObject(nid, det["class_name"], det["bbox"])
                t.init_spawn(frame_number)       # record where this track was born
                self._tracks[nid] = t
                det["track_id"]   = nid
        for tid in existing_ids:
            if tid not in matched_exist:
                self._tracks[tid].missed_frames += 1

    @staticmethod
    def _iou(b1: List[int], b2: List[int]) -> float:
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        iw, ih   = max(0, ix2-ix1), max(0, iy2-iy1)
        inter    = iw * ih
        union    = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / union if union > 0 else 0.0


# =============================================================================
# 5. OCRManager
# =============================================================================

class OCRManager:
    """EasyOCR number plate extraction with multi-frame voting."""

    def __init__(self) -> None:
        self._reader: Any = None
        if _OCR_AVAILABLE:
            try:
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                logger.info("EasyOCR ready.")
            except Exception as exc:
                logger.warning("EasyOCR init failed: %s", exc)

    @property
    def ready(self) -> bool:
        return self._reader is not None

    def extract_text(self, crop: np.ndarray) -> Tuple[str, float]:
        if crop is None or crop.size == 0:
            return "", 0.0
        if self._reader is not None:
            try:
                results = self._reader.readtext(crop, detail=1)
                if results:
                    best = max(results, key=lambda r: r[2])
                    return "".join(best[1].upper().split()), round(float(best[2]), 2)
            except Exception:
                pass
        return "", 0.0

    @staticmethod
    def vote(texts: List[str]) -> str:
        counts: Dict[str, int] = defaultdict(int)
        for t in texts:
            if t:
                counts[t] += 1
        return max(counts, key=lambda k: counts[k]) if counts else ""


# =============================================================================
# 6. EvidenceManager
# =============================================================================

class EvidenceManager:
    """Saves annotated evidence frames and CSV violation logs."""

    VIOLATION_DIR_MAP = {
        "red_light": os.path.join(AppConfig.EVIDENCE_DIR, "red_light"),
        "helmet":    os.path.join(AppConfig.EVIDENCE_DIR, "helmet"),
        "lane":      os.path.join(AppConfig.EVIDENCE_DIR, "lane"),
        "speed":     os.path.join(AppConfig.EVIDENCE_DIR, "speed"),
    }
    PLATE_DIR = os.path.join(AppConfig.EVIDENCE_DIR, "plates")
    CSV_PATH  = os.path.join(AppConfig.LOG_DIR, "violations.csv")
    CSV_COLUMNS = ["timestamp","frame_number","object_id","object_class",
                   "violation_type","signal_state","lane_id","speed_value",
                   "plate_number","confidence_or_rule_score","evidence_path"]

    def __init__(self) -> None:
        if not os.path.isfile(self.CSV_PATH):
            with open(self.CSV_PATH, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.CSV_COLUMNS).writeheader()
        self.records: List[Dict[str, Any]] = []

    def save_evidence(self, vtype: str, frame: np.ndarray, track: TrackedObject,
                      frame_number: int, signal_state: str, lane_id: int = -1,
                      speed: float = 0.0, plate_text: str = "", score: float = 1.0,
                      plate_crop: Optional[np.ndarray] = None) -> str:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_name = f"{vtype}_id{track.track_id}_f{frame_number}_{ts}.jpg"
        target_dir = self.VIOLATION_DIR_MAP.get(vtype, AppConfig.EVIDENCE_DIR)
        img_path   = os.path.join(target_dir, img_name)

        ev = frame.copy()
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(ev, (x1-3,y1-3), (x2+3,y2+3), (0,0,255), 4)
        ol = ev.copy()
        cv2.rectangle(ol, (x1,y1), (x2,y2), (0,0,200), -1)
        cv2.addWeighted(ol, 0.15, ev, 0.85, 0, ev)
        label = f"{vtype.upper().replace('_',' ')} | ID{track.track_id}"
        if speed > 0: label += f" | {speed:.1f}km/h"
        if plate_text: label += f" | {plate_text}"
        lw = max(len(label)*9, 120)
        cv2.rectangle(ev, (x1-3, max(0,y1-32)), (x1+lw, y1-3), (0,0,200), -1)
        cv2.putText(ev, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(ev, f"CONF {int(score*100)}%", (x2-80, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,255,180), 1, cv2.LINE_AA)
        hf = ev.shape[0]
        ts_label = (f"Frame:{frame_number}  "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
                    f"Signal:{signal_state.upper()}")
        cv2.rectangle(ev, (0,hf-26), (len(ts_label)*8+12,hf), (10,10,25), -1)
        cv2.putText(ev, ts_label, (5,hf-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (160,200,200), 1, cv2.LINE_AA)
        cv2.imwrite(img_path, ev)

        if plate_crop is not None and plate_crop.size > 0:
            cv2.imwrite(os.path.join(self.PLATE_DIR,
                        f"plate_id{track.track_id}_f{frame_number}_{ts}.jpg"), plate_crop)

        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(), "frame_number": frame_number,
            "object_id": track.track_id, "object_class": track.class_name,
            "violation_type": vtype, "signal_state": signal_state, "lane_id": lane_id,
            "speed_value": round(speed,1), "plate_number": plate_text,
            "confidence_or_rule_score": round(score,3), "evidence_path": img_path}
        with open(self.CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.CSV_COLUMNS).writerow(row)
        self.records.append(row)
        return img_path


# =============================================================================
# 7. ViolationEngine
# =============================================================================

class ViolationEngine:
    """
    Temporal FSM violation detection.

    Camera mode awareness
    ---------------------
    OVERHEAD camera: vehicles approach the camera → y INCREASES over time.
                     Violation: vehicle y goes from below stop line to above it
                     (in image coords, further from camera = smaller y).
                     Direction of violation movement: dy > 0 (downward in image
                     = toward intersection from ego camera above).

    DASHCAM camera:  camera is mounted in a vehicle approaching the intersection.
                     Tracked vehicles ahead move AWAY → y DECREASES over time
                     (they get smaller and higher in the frame).
                     Stop line / crosswalk is in the LOWER MIDDLE of the frame
                     (roughly 55–65% down).
                     Violation: vehicle crosses from below crosswalk (y > stop_y)
                     to above it (y < stop_y) while signal is RED.
                     Direction of violation movement: dy < 0 (upward in image
                     = moving away from camera through intersection).

    The mode is AUTO-DETECTED each run by analysing the first 20 frames:
    if tracked objects consistently move with decreasing y (upward) AND
    the scene has a dashcam-style perspective (hood visible, wide lower road),
    DASHCAM mode is selected; otherwise OVERHEAD.

    Stop line position
    ------------------
    OVERHEAD: derived from road mask or default 62% down.
    DASHCAM:  detected from white crosswalk stripes, typically 55–65% down.
              Falls back to 62% if stripes are not found.
    """

    def __init__(self, seg_info: Dict[str, Any], evidence_mgr: EvidenceManager,
                 ocr_mgr: OCRManager, camera_mode: str = "overhead") -> None:
        self._seg         = seg_info
        self._ev          = evidence_mgr
        self._ocr         = ocr_mgr
        self.camera_mode  = camera_mode   # "overhead" | "dashcam"
        self._stop_y      = self._compute_stop_y()

    def _compute_stop_y(self) -> int:
        pts = self._seg.get("stop_line_polygon", [])
        if pts:
            return int(pts[0][1])
        return int(AppConfig.PROC_HEIGHT * 0.62)

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, frame: np.ndarray, tracks: List[TrackedObject],
                 detections: List[Dict[str, Any]], signal_state: str,
                 frame_number: int) -> List[Dict[str, Any]]:
        """
        Run all violation checks for all vehicle tracks.
        Only violations scoring >= VIOLATION_MIN_SCORE are returned/saved.
        """
        violations: List[Dict[str, Any]] = []

        for track in tracks:
            if track.class_name not in AppConfig.VEHICLE_CLASSES:
                continue

            lane_id = self._assign_lane(track)
            track.assigned_lane = lane_id
            track.lane_history.append(lane_id)

            # Red-light
            v = self._check_red_light(track, signal_state, frame_number)
            if v and v["score"] >= AppConfig.VIOLATION_MIN_SCORE:
                plate_crop = self._find_plate_crop(frame, track, detections)
                plate_text = self._run_ocr(track, plate_crop)
                path = self._ev.save_evidence("red_light", frame, track, frame_number,
                    signal_state, lane_id=lane_id, plate_text=plate_text,
                    score=v["score"], plate_crop=plate_crop)
                v["evidence_path"] = path; v["plate_number"] = plate_text
                violations.append(v)
            elif v:
                track.violations_fired.discard("red_light")
                # In dashcam mode reset crossing state to before_line so
                # the FSM can re-evaluate without getting stuck at on_line
                if self.camera_mode == "dashcam":
                    track.crossing_state = "before_line"
                else:
                    track.crossing_state = "on_line"

            # Helmet (motorcycles only)
            if track.class_name == "motorcycle":
                v = self._check_helmet(frame, track, tracks, detections,
                                       frame_number, signal_state)
                if v and v["score"] >= AppConfig.VIOLATION_MIN_SCORE:
                    plate_crop = self._find_plate_crop(frame, track, detections)
                    plate_text = self._run_ocr(track, plate_crop)
                    path = self._ev.save_evidence("helmet", frame, track, frame_number,
                        signal_state, plate_text=plate_text, score=v["score"],
                        plate_crop=plate_crop)
                    v["evidence_path"] = path; v["plate_number"] = plate_text
                    violations.append(v)
                elif v:
                    track.violations_fired.discard("helmet")

            # Lane
            v = self._check_lane(track, frame_number, signal_state)
            if v and v["score"] >= AppConfig.VIOLATION_MIN_SCORE:
                path = self._ev.save_evidence("lane", frame, track, frame_number,
                    signal_state, lane_id=lane_id, score=v["score"])
                v["evidence_path"] = path
                violations.append(v)
            elif v:
                track.violations_fired.discard("lane")

            # Speed
            v = self._check_speed(track, frame_number, signal_state)
            if v and v["score"] >= AppConfig.VIOLATION_MIN_SCORE:
                plate_crop = self._find_plate_crop(frame, track, detections)
                plate_text = self._run_ocr(track, plate_crop)
                path = self._ev.save_evidence("speed", frame, track, frame_number,
                    signal_state, speed=track.speed_kmh, plate_text=plate_text,
                    score=v["score"], plate_crop=plate_crop)
                v["evidence_path"] = path; v["plate_number"] = plate_text
                violations.append(v)
            elif v:
                track.violations_fired.discard("speed")

        return violations

    # ── Red-light FSM ─────────────────────────────────────────────────────────

    def _check_red_light(self, track: TrackedObject, signal_state: str,
                         frame_number: int) -> Optional[Dict[str, Any]]:
        """
        Red-light violation: fire ONLY when a vehicle is observed both
        APPROACHING the stop line AND CROSSING it within the same tracking session.

        Core principle
        --------------
        Vehicles already past the stop line when first detected are NEVER flagged
        (crossing_state = "already_past"). This eliminates all frame-3 mass
        false positives from vehicles that appear anywhere on screen.

        Signal coupling (score, not hard gate)
        ----------------------------------------
        "red"     -> score 1.00
        "yellow"  -> score 0.82
        "unknown" -> score 0.78  (still passes >=75%)
        "green"   -> skip + reset

        FSM zones
        ---------
        OVERHEAD (vehicles move DOWN, y increases):
          approach side = by < stop_y - band
          crossing side = by > stop_y + band

        DASHCAM (vehicles move UP, y decreases):
          approach side = by > stop_y + band
          crossing side = by < stop_y - band
        """
        if "red_light" in track.violations_fired:
            return None
        if track.in_cooldown("red_light", frame_number):
            return None
        if len(track.centroid_history) < AppConfig.RED_LIGHT_MIN_AGE:
            return None
        if signal_state == "green":
            track.crossing_state = "before_line"
            return None

        score_map  = {"red": 1.00, "yellow": 0.82, "unknown": 0.78}
        base_score = score_map.get(signal_state, 0.78)

        by   = track.bottom_center[1]
        band = 30

        if self.camera_mode == "dashcam":
            on_approach_side = by > self._stop_y + band
            in_band          = abs(by - self._stop_y) <= band
            on_crossing_side = by < self._stop_y - band
        else:
            on_approach_side = by < self._stop_y - band
            in_band          = abs(by - self._stop_y) <= band
            on_crossing_side = by > self._stop_y + band

        # First evaluation: determine starting zone
        if track.crossing_state == "new":
            if on_approach_side:
                track.crossing_state = "before_line"
            else:
                # Already at or past line when track started — cannot determine legality
                track.crossing_state = "already_past"
            return None

        # Permanently skip vehicles that were already past the line
        if track.crossing_state == "already_past":
            return None

        # Standard FSM
        if on_approach_side:
            track.crossing_state = "before_line"
        elif in_band:
            if track.crossing_state == "before_line":
                track.crossing_state = "on_line"
        elif on_crossing_side:
            if track.crossing_state in ("before_line", "on_line"):
                track.crossing_state = "crossed_line"
                track.violations_fired.add("red_light")
                track.set_cooldown("red_light", frame_number)
                return {
                    "violation_type": "red_light",
                    "track_id":       track.track_id,
                    "class_name":     track.class_name,
                    "frame_number":   frame_number,
                    "signal_state":   signal_state,
                    "score":          round(base_score, 3),
                    "speed":          track.speed_kmh,
                    "lane_id":        track.assigned_lane,
                }

        return None


    # ── Helmet ────────────────────────────────────────────────────────────────

    def _check_helmet(self, frame: np.ndarray, track: TrackedObject,
                      all_tracks: List[TrackedObject], detections: List[Dict[str, Any]],
                      frame_number: int, signal_state: str) -> Optional[Dict[str, Any]]:
        if "helmet" in track.violations_fired:
            return None
        if track.in_cooldown("helmet", frame_number):
            return None
        rider = self._associate_rider(track, all_tracks)
        if rider is None:
            return None
        head_crop = self._localise_head(frame, rider)
        track.helmet_decisions.append(self._classify_helmet(head_crop))
        if len(track.helmet_decisions) < AppConfig.HELMET_FRAMES:
            return None
        no_helmet_votes = sum(1 for d in track.helmet_decisions if d == "no_helmet")
        if no_helmet_votes < AppConfig.HELMET_FRAMES * 0.6:
            return None
        track.violations_fired.add("helmet")
        track.set_cooldown("helmet", frame_number)
        # Normalise score: the rule fires at 60 % votes (maps to 0.75 on our scale).
        # Scale so 60 % → 0.75 and 100 % → 1.0, giving honest confidence output.
        raw   = no_helmet_votes / max(len(track.helmet_decisions), 1)
        score = round(0.75 + (raw - 0.60) * (0.25 / 0.40), 3)
        score = min(max(score, 0.75), 1.0)
        return {"violation_type": "helmet", "track_id": track.track_id,
                "class_name": track.class_name, "frame_number": frame_number,
                "signal_state": signal_state,
                "score": score,
                "speed": track.speed_kmh, "lane_id": track.assigned_lane,
                "rider_id": rider.track_id}

    def _associate_rider(self, bike: TrackedObject,
                         all_tracks: List[TrackedObject]) -> Optional[TrackedObject]:
        bx1, by1, bx2, by2 = bike.bbox
        bw, bh = bx2-bx1, by2-by1
        margin_x = max(int(bw*0.5), 50)
        margin_y = max(int(bh*0.5), 50)
        max_dist  = math.hypot(bw, bh) * 1.2
        best_iou, best_dist = 0.0, float("inf")
        best_by_iou: Optional[TrackedObject]  = None
        best_by_dist: Optional[TrackedObject] = None
        for t in all_tracks:
            if t.class_name not in AppConfig.PERSON_CLASSES:
                continue
            px, py = t.centroid
            if not (bx1-margin_x <= px <= bx2+margin_x and
                    by1-margin_y <= py <= by2+margin_y):
                continue
            iou  = TrackingManager._iou(bike.bbox, t.bbox)
            dist = math.hypot(px-(bx1+bx2)//2, py-(by1+by2)//2)
            if iou > best_iou:
                best_iou, best_by_iou = iou, t
            if dist < best_dist:
                best_dist, best_by_dist = dist, t
        if best_by_iou is not None:
            return best_by_iou
        if best_by_dist is not None and best_dist < max_dist:
            return best_by_dist
        return None

    def _localise_head(self, frame: np.ndarray,
                       rider: TrackedObject) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = rider.bbox
        pad_x  = int((x2-x1) * 0.15)
        head_h = int((y2-y1) * 0.38)
        hx1, hx2 = max(0, x1-pad_x), min(frame.shape[1], x2+pad_x)
        hy2      = y1 + head_h
        crop     = frame[y1:hy2, hx1:hx2]
        return crop if crop.size >= 64 else None

    @staticmethod
    def _classify_helmet(head_crop: Optional[np.ndarray]) -> str:
        """
        Multi-cue helmet classifier.
        Scores bare-head signals (skin tone, hair texture, tonal variation)
        vs helmet signals (uniform colour blob, low-variance dark/light region).
        """
        if head_crop is None or head_crop.size == 0:
            return "uncertain"
        if head_crop.shape[0] < 8 or head_crop.shape[1] < 8:
            return "uncertain"
        bgr  = head_crop if len(head_crop.shape) == 3 else cv2.cvtColor(head_crop, cv2.COLOR_GRAY2BGR)
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        total_px   = float(grey.size)
        mean_bright = float(np.mean(grey))
        std_bright  = float(np.std(grey))
        mean_sat    = float(np.mean(hsv[:,:,1]))
        std_sat     = float(np.std(hsv[:,:,1]))
        # Skin tone
        skin_mask  = (cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                                  np.array([0,20,80],np.uint8),
                                  np.array([25,200,255],np.uint8)) |
                      cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                                  np.array([0,10,60],np.uint8),
                                  np.array([20,220,255],np.uint8)))
        skin_frac  = float(np.sum(skin_mask > 0)) / max(total_px, 1)
        # Texture
        texture    = float(np.mean(np.abs(grey - cv2.GaussianBlur(grey,(3,3),0))))
        # Dominant hue blob
        hue_vals   = hsv[:,:,0].flatten().astype(np.int32)
        sat_vals   = hsv[:,:,1].flatten()
        sat_mask   = sat_vals > 30
        hue_hist, _ = np.histogram(hue_vals[sat_mask], bins=12, range=(0,180))
        dominant_frac = (float(hue_hist.max()) / max(sat_mask.sum(),1)
                         if sat_mask.sum() > 10 else 0.0)
        no_h = 0.0; hel = 0.0
        if skin_frac > 0.12:   no_h += 2.5
        elif skin_frac > 0.05: no_h += 1.2
        if texture > 8.0:      no_h += 1.5
        elif texture > 5.0:    no_h += 0.8
        if std_bright > 30:    no_h += 1.0
        if mean_bright > 140 and std_bright > 20: no_h += 1.0
        if dominant_frac > 0.55:               hel += 2.0
        if std_bright < 20 and mean_bright < 80: hel += 2.5
        if std_bright < 15 and mean_bright > 80: hel += 1.5
        if mean_sat < 20 and std_sat < 15:      hel += 1.0
        gap = no_h - hel
        if gap > 1.5: return "no_helmet"
        if gap < -1.0: return "helmet"
        return "uncertain"

    # ── Lane ──────────────────────────────────────────────────────────────────

    def _check_lane(self, track: TrackedObject, frame_number: int,
                    signal_state: str) -> Optional[Dict[str, Any]]:
        if "lane" in track.violations_fired:
            return None
        if track.in_cooldown("lane", frame_number):
            return None
        history = list(track.lane_history)
        if len(history) < AppConfig.LANE_CONFIRM_FRAMES:
            return None
        in_restricted = self._in_restricted_zone(track)
        track.lane_decisions.append("restricted" if in_restricted else "ok")
        decisions = list(track.lane_decisions)
        consec_restricted = sum(1 for _ in (d for d in reversed(decisions)
                                            if d == "restricted") for _2 in [None])
        consec_restricted = 0
        for d in reversed(decisions):
            if d == "restricted": consec_restricted += 1
            else: break
        recent   = history[-AppConfig.LANE_FRAMES:]
        baseline = history[:AppConfig.LANE_FRAMES]
        rc = Counter(x for x in recent   if x > 0)
        bc = Counter(x for x in baseline if x > 0)
        rd = rc.most_common(1)[0][0] if rc else -1
        bd = bc.most_common(1)[0][0] if bc else -1
        rc_frac = rc.get(rd, 0) / max(len(recent), 1)
        if rd > 0 and bd > 0 and rd != bd and rc_frac > 0.75:
            track.lane_decisions.append("lane_change")
        restricted_votes = sum(1 for d in decisions if d == "restricted")
        change_votes     = sum(1 for d in decisions if d == "lane_change")
        total            = max(len(decisions), 1)
        vtype: Optional[str] = None
        if consec_restricted >= 5 and restricted_votes >= max(AppConfig.LANE_CONFIRM_FRAMES*0.7, 6):
            vtype = "restricted_lane"
        elif change_votes >= max(AppConfig.LANE_FRAMES*0.8, 4):
            vtype = "illegal_lane_change"
        if vtype is None:
            return None
        track.violations_fired.add("lane")
        track.set_cooldown("lane", frame_number)
        # Use only the recent decision window (not full unbounded history) so
        # the score stays representative and does not decay as the deque grows.
        window      = AppConfig.LANE_CONFIRM_FRAMES
        recent_decs = decisions[-window:]
        win_total   = max(len(recent_decs), 1)
        win_votes   = (sum(1 for d in recent_decs if d == "restricted")
                       if vtype == "restricted_lane"
                       else sum(1 for d in recent_decs if d == "lane_change"))
        # Fire threshold is ~70 % of window → maps to 0.75 on output scale.
        raw   = win_votes / win_total
        score = round(0.75 + (raw - 0.70) * (0.25 / 0.30), 3)
        score = min(max(score, 0.75), 1.0)
        return {"violation_type": "lane", "lane_sub_type": vtype,
                "track_id": track.track_id, "class_name": track.class_name,
                "frame_number": frame_number, "signal_state": signal_state,
                "score": score,
                "speed": track.speed_kmh, "lane_id": track.assigned_lane}

    # ── Speed ─────────────────────────────────────────────────────────────────

    def _check_speed(self, track: TrackedObject, frame_number: int,
                     signal_state: str) -> Optional[Dict[str, Any]]:
        """
        Speed violation requires:
        1. Track must have >= SPEED_MIN_HISTORY frames of centroid history
           so the displacement estimate is stable (not a single-frame jump).
        2. Speed reading must be below SPEED_MAX_KMH (sanity cap for pixel errors).
        3. Over-threshold consistently for >= 80% of last SPEED_FRAMES decisions.

        Without condition 1, a newly-created track with only 2-3 centroid
        points will produce wildly wrong pixel displacement values.
        """
        if "speed" in track.violations_fired:
            return None
        if track.in_cooldown("speed", frame_number):
            return None

        # Require stable history before trusting speed estimate
        if len(track.centroid_history) < AppConfig.SPEED_MIN_HISTORY:
            return None

        over = AppConfig.SPEED_LIMIT_KMH < track.speed_kmh < AppConfig.SPEED_MAX_KMH
        track.speed_decisions.append("over" if over else "ok")
        if len(track.speed_decisions) < AppConfig.SPEED_FRAMES:
            return None
        over_votes = sum(1 for d in track.speed_decisions if d == "over")
        if over_votes < AppConfig.SPEED_FRAMES * 0.8:
            return None

        # Additional stability check: all recent velocity readings must be
        # consistent (std dev < 30 km/h) to reject noisy pixel-jump artefacts
        recent_v = list(track.velocity_history)
        if len(recent_v) >= 3:
            mean_v = sum(recent_v) / len(recent_v)
            std_v  = (sum((v - mean_v)**2 for v in recent_v) / len(recent_v)) ** 0.5
            if std_v > 30.0:
                # High variance = noisy track, not real speeding
                track.speed_decisions.clear()
                return None

        track.violations_fired.add("speed")
        track.set_cooldown("speed", frame_number)
        avg_speed = sum(track.velocity_history) / max(len(track.velocity_history), 1)
        raw   = over_votes / max(len(track.speed_decisions), 1)
        score = round(0.75 + (raw - 0.80) * (0.25 / 0.20), 3)
        score = min(max(score, 0.75), 1.0)
        return {"violation_type": "speed", "track_id": track.track_id,
                "class_name": track.class_name, "frame_number": frame_number,
                "signal_state": signal_state,
                "score": score,
                "speed": round(avg_speed, 1), "lane_id": track.assigned_lane}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assign_lane(self, track: TrackedObject) -> int:
        bx, by = track.bottom_center
        for mask_arr, lid in self._seg.get("lane_masks", []):
            h_m, w_m = mask_arr.shape[:2]
            if 0 <= by < h_m and 0 <= bx < w_m and mask_arr[by, bx] > 0:
                return lid
        return -1

    def _in_restricted_zone(self, track: TrackedObject) -> bool:
        bx, by = track.bottom_center
        for poly in self._seg.get("restricted_regions", []):
            pts = np.array(poly, dtype=np.int32)
            if cv2.pointPolygonTest(pts, (float(bx), float(by)), False) >= 0:
                return True
        return False

    def _find_plate_crop(self, frame: np.ndarray, track: TrackedObject,
                         detections: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = track.bbox
        for det in detections:
            if det["class_name"] in {AppConfig.PLATE_CLASS, "license plate"}:
                px1, py1, px2, py2 = det["bbox"]
                if x1 <= px1 and px2 <= x2 and y1 <= py1 and py2 <= y2:
                    crop = frame[py1:py2, px1:px2]
                    if crop.size > 0:
                        return crop
        return None

    def _run_ocr(self, track: TrackedObject,
                 plate_crop: Optional[np.ndarray]) -> str:
        if plate_crop is None:
            return track.best_plate_text
        if len(track.plate_texts) < AppConfig.OCR_BEST_FRAMES:
            text, _ = self._ocr.extract_text(plate_crop)
            if text:
                track.plate_texts.append(text)
                track.plate_crops.append(plate_crop)
        voted = OCRManager.vote(track.plate_texts)
        track.best_plate_text = voted
        return voted


# =============================================================================
# 8. SignalDetector
# =============================================================================

class SignalDetector:
    """
    Traffic light state estimation.

    Strategy (priority order)
    -------------------------
    1. YOLO "traffic light" bbox → classify colour inside that box.
    2. No YOLO bbox → scan well-defined candidate ROIs in the upper frame.
       For each ROI, look for compact bright LED blobs of the right colour
       (excludes vegetation false positives via connected-component size filter).
    3. Stability: 10-frame majority vote prevents single-frame flicker.

    Signal ROI auto-calibration
    ---------------------------
    On the first call the detector scans the upper 40% of the frame
    for the most likely traffic light locations using a brightness-blob
    heuristic, and registers those as candidate ROIs for subsequent frames.
    This adapts automatically to different camera angles and positions.
    """

    STABLE_WINDOW = 5          # reduced from 10 — faster response to signal changes
    # For pure-heuristic fallback: scan only the top 35% of the frame
    # (lights hang from gantries above the road level)
    SCAN_HEIGHT_FRAC = 0.35

    def __init__(self) -> None:
        self._votes:     deque          = deque(maxlen=self.STABLE_WINDOW)
        self.state:      str            = "unknown"
        self._rois:      List[Tuple[int,int,int,int]] = []  # calibrated (x1,y1,x2,y2)
        self._calibrated: bool          = False

    def update(self, frame: np.ndarray,
               detections: List[Dict[str, Any]]) -> str:
        """Update state and return stable majority-voted signal colour."""
        # One-time ROI calibration from first informative frame
        if not self._calibrated:
            self._calibrate_rois(frame)

        raw = self._detect_raw(frame, detections)
        self._votes.append(raw)
        counts: Dict[str, int] = defaultdict(int)
        for v in self._votes:
            counts[v] += 1
        self.state = max(counts, key=lambda k: counts[k])
        return self.state

    def _calibrate_rois(self, frame: np.ndarray) -> None:
        """
        Find traffic light locations by scanning for compact warm-red, green,
        or amber blobs in the upper 45% of the frame.

        Uses color-filtered blobs rather than raw brightness blobs.
        This avoids false positives from bright sky, windows, and reflections,
        and correctly finds warm orange-red LEDs (H≈5-25) that brightness
        thresholding alone would miss.

        Each found blob is padded to capture the full light housing as an ROI.
        ROIs are used in subsequent frames for fast classification.
        """
        h, w      = frame.shape[:2]
        scan_h    = int(h * 0.60)   # extended to 60% — dashcam lights can be lower
        upper     = frame[:scan_h, :]
        hsv       = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

        # Build a combined mask of any traffic-light colour
        r1   = cv2.inRange(hsv, np.array(AppConfig.RED_HUE_LO1),
                                np.array(AppConfig.RED_HUE_HI1))
        r2   = cv2.inRange(hsv, np.array(AppConfig.RED_HUE_LO2),
                                np.array(AppConfig.RED_HUE_HI2))
        gm   = cv2.inRange(hsv, np.array(AppConfig.GREEN_HUE_LO),
                                np.array(AppConfig.GREEN_HUE_HI))
        ym   = cv2.inRange(hsv, np.array(AppConfig.YELLOW_HUE_LO),
                                np.array(AppConfig.YELLOW_HUE_HI))
        combined = r1 | r2 | gm | ym

        n, _, stats, cents = cv2.connectedComponentsWithStats(combined, connectivity=8)
        self._rois = []
        for lbl in range(1, n):
            area = stats[lbl, cv2.CC_STAT_AREA]
            bw2  = stats[lbl, cv2.CC_STAT_WIDTH]
            bh2  = stats[lbl, cv2.CC_STAT_HEIGHT]
            # LED: small-ish, roughly circular
            if 8 <= area <= 800 and 0.3 <= bw2 / max(bh2, 1) <= 3.0:
                cx = int(cents[lbl][0])
                cy = int(cents[lbl][1])
                # Pad generously to capture housing
                pad = max(bw2, bh2) * 3
                x1  = max(0,      cx - pad)
                y1  = max(0,      cy - pad)
                x2  = min(w,      cx + pad)
                y2  = min(scan_h, cy + pad)
                self._rois.append((x1, y1, x2, y2))

        self._calibrated = True
        logger.info("SignalDetector: calibrated %d color ROIs from first frame.",
                    len(self._rois))

    def _detect_raw(self, frame: np.ndarray,
                    detections: List[Dict[str, Any]]) -> str:
        """
        Single-frame signal state detection (un-smoothed).

        Priority order
        --------------
        1. YOLO "traffic light" bbox → classify colour inside that box.
        2. Calibrated ROIs from first frame → classify each ROI, majority vote.
        3. Full upper-frame colour blob scan (fallback, runs every frame).
           Scans upper 45% for compact warm-red/green/yellow blobs.
           For each candidate blob region, classifies the colour inside it.
        """
        h = frame.shape[0]
        scan_h = int(h * 0.60)   # match calibration scan height

        # Priority 1: YOLO bboxes
        tl_boxes = [d["bbox"] for d in detections
                    if d["class_name"] == AppConfig.TRAFFIC_LIGHT_CLASS]
        if tl_boxes:
            for bbox in tl_boxes:
                x1, y1, x2, y2 = bbox
                region = frame[y1:y2, x1:x2]
                if region.size > 0:
                    s = self._classify_region(region)
                    if s != "unknown":
                        return s

        # Priority 2+3: scan upper frame for coloured LED blobs
        # This approach directly scans every frame rather than relying on
        # first-frame ROI positions (handles moving dashcam scenes where
        # light positions change every frame)
        upper = frame[:scan_h, :]
        hsv   = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

        # Red (warm + deep)
        r1   = cv2.inRange(hsv, np.array(AppConfig.RED_HUE_LO1),
                                np.array(AppConfig.RED_HUE_HI1))
        r2   = cv2.inRange(hsv, np.array(AppConfig.RED_HUE_LO2),
                                np.array(AppConfig.RED_HUE_HI2))
        # Green (high sat+val LEDs only)
        gm   = cv2.inRange(hsv, np.array(AppConfig.GREEN_HUE_LO),
                                np.array(AppConfig.GREEN_HUE_HI))
        # Yellow/amber
        ym   = cv2.inRange(hsv, np.array(AppConfig.YELLOW_HUE_LO),
                                np.array(AppConfig.YELLOW_HUE_HI))

        # Count LED-like blobs per colour
        def led_blob_px(mask: np.ndarray) -> int:
            """Count pixels belonging to compact blobs (LED size 8-600 px)."""
            total = int(np.sum(mask > 0))
            if total < 5:
                return 0
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            count = 0
            for lbl in range(1, n):
                area = stats[lbl, cv2.CC_STAT_AREA]
                bw_  = stats[lbl, cv2.CC_STAT_WIDTH]
                bh_  = stats[lbl, cv2.CC_STAT_HEIGHT]
                # Accept compact blobs of LED-appropriate size
                if 8 <= area <= 600 and 0.3 <= bw_ / max(bh_, 1) <= 3.0:
                    count += int(area)
            return count

        red_px    = led_blob_px(r1 | r2)
        green_px  = led_blob_px(gm)
        yellow_px = led_blob_px(ym)

        best = max(red_px, green_px, yellow_px)
        if best < 3:
            return "unknown"
        if best == red_px    and red_px    >= 3:  return "red"
        if best == green_px  and green_px  >= 3:  return "green"
        if best == yellow_px and yellow_px >= 3:  return "yellow"
        return "unknown"

    @staticmethod
    def _classify_region(region: np.ndarray) -> str:
        """
        Classify a BGR region as red / green / yellow / unknown.

        Uses connected-component blob detection to filter out large diffuse
        regions (vegetation, buildings) and count only compact bright blobs
        consistent with LED traffic light heads.
        """
        if region.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h_r, w_r = region.shape[:2]
        max_blob_area = max(h_r * w_r * 0.20, 30)  # LED blobs are compact

        def count_led_px(mask: np.ndarray) -> int:
            total = int(np.sum(mask > 0))
            if total < 5:
                return 0
            # If the region is already tiny (cropped light bbox), trust pixel count
            if h_r * w_r < 400:
                return total
            # For larger regions, require compact blobs
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            led_px = 0
            for lbl in range(1, n):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area <= max_blob_area:
                    led_px += int(area)
            return led_px

        # Red — two HSV ranges (hue wraps)
        r1 = cv2.inRange(hsv, np.array(AppConfig.RED_HUE_LO1),
                              np.array(AppConfig.RED_HUE_HI1))
        r2 = cv2.inRange(hsv, np.array(AppConfig.RED_HUE_LO2),
                              np.array(AppConfig.RED_HUE_HI2))
        red_px    = count_led_px(r1 | r2)
        green_px  = count_led_px(cv2.inRange(hsv, np.array(AppConfig.GREEN_HUE_LO),
                                              np.array(AppConfig.GREEN_HUE_HI)))
        yellow_px = count_led_px(cv2.inRange(hsv, np.array(AppConfig.YELLOW_HUE_LO),
                                              np.array(AppConfig.YELLOW_HUE_HI)))

        best = max(red_px, green_px, yellow_px)
        if best < 3:
            return "unknown"
        if best == red_px    and red_px    > 3:  return "red"
        if best == green_px  and green_px  > 3:  return "green"
        if best == yellow_px and yellow_px > 3:  return "yellow"
        return "unknown"


# =============================================================================
# 9. OverlayRenderer
# =============================================================================

class OverlayRenderer:
    """Draws analytics overlays on processed frames."""

    SIGNAL_COLORS   = {"red":(0,0,220),"green":(0,200,0),
                       "yellow":(0,200,220),"unknown":(128,128,128)}
    VIOLATION_COLORS = {"red_light":(0,0,255),"helmet":(0,140,255),
                        "lane":(255,165,0),"speed":(255,0,200)}
    CLASS_COLORS    = {"car":(50,200,50),"motorcycle":(255,120,0),
                       "bus":(0,200,200),"truck":(180,0,180),
                       "person":(255,200,0),"bicycle":(100,255,100),
                       "default":(180,180,180)}
    LANE_COLOURS    = [(0,220,255),(0,255,180),(255,200,0)]

    def render(self, frame: np.ndarray, tracks: List[TrackedObject],
               seg_info: Dict[str, Any], signal_state: str,
               frame_number: int, active_violations: List[Dict[str, Any]],
               fps: float) -> np.ndarray:
        out = frame.copy()
        self._draw_seg_overlays(out, seg_info)
        self._draw_tracks(out, tracks)
        self._draw_violation_banners(out, active_violations)
        self._draw_hud(out, frame_number, signal_state, fps, len(tracks))
        return out

    def _draw_seg_overlays(self, frame: np.ndarray,
                           seg_info: Dict[str, Any]) -> None:
        h, w = frame.shape[:2]
        seg_overlay = seg_info.get("seg_overlay")
        if seg_overlay is not None and seg_overlay.shape == frame.shape:
            cv2.addWeighted(seg_overlay, 0.40, frame, 0.60, 0, frame)
        else:
            for i, (mask, _) in enumerate(seg_info.get("lane_masks", [])):
                colour    = self.LANE_COLOURS[i % len(self.LANE_COLOURS)]
                mask_bool = mask > 0
                frame[mask_bool] = (
                    0.22*np.array(colour,dtype=np.float32)
                    + 0.78*frame[mask_bool].astype(np.float32)).astype(np.uint8)
        for i, (mask, lid) in enumerate(seg_info.get("lane_masks", [])):
            colour    = self.LANE_COLOURS[i % len(self.LANE_COLOURS)]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, colour, 1)
            ys, xs = np.where(mask > 0)
            if ys.size > 0:
                cv2.putText(frame, f"L{lid}",
                            (int(np.mean(xs))-10, int(np.mean(ys))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        pts = seg_info.get("stop_line_polygon", [])
        if len(pts) >= 2:
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0,0,255), 2)
            cv2.putText(frame, "STOP LINE",
                        (pts[0][0]+5, pts[0][1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,0,255), 1, cv2.LINE_AA)
        for poly in seg_info.get("restricted_regions", []):
            arr = np.array(poly, dtype=np.int32)
            cv2.polylines(frame, [arr], True, (0,0,200), 2)
            if len(arr):
                cv2.putText(frame, "RESTRICTED", (arr[0][0], arr[0][1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0,0,200), 1, cv2.LINE_AA)
        inter = seg_info.get("intersection_region", [])
        if len(inter) >= 3:
            arr = np.array(inter, dtype=np.int32)
            ol  = frame.copy()
            cv2.fillPoly(ol, [arr], (30,180,255))
            cv2.addWeighted(ol, 0.10, frame, 0.90, 0, frame)
            cv2.polylines(frame, [arr], True, (30,180,255), 1)
        backend = seg_info.get("backend","")
        if backend:
            cv2.putText(frame, f"SEG:{backend}", (6, h-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80,180,80), 1, cv2.LINE_AA)

    def _draw_tracks(self, frame: np.ndarray,
                     tracks: List[TrackedObject]) -> None:
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            colour = (self.CLASS_COLORS.get(t.class_name, self.CLASS_COLORS["default"])
                      if not t.violations_fired else (0,0,255))
            thick  = 3 if t.violations_fired else 2
            if t.violations_fired:
                cv2.rectangle(frame, (x1-2,y1-2), (x2+2,y2+2), colour, thick)
            else:
                cv2.rectangle(frame, (x1,y1), (x2,y2), colour, thick)
            label = f"ID{t.track_id} {t.class_name[:3]} {t.speed_kmh:.0f}km/h"
            if t.assigned_lane > 0: label += f" L{t.assigned_lane}"
            if t.best_plate_text:   label += f" [{t.best_plate_text}]"
            lw = max(len(label)*7, 60)
            cv2.rectangle(frame, (x1, max(0,y1-18)), (x1+lw, y1), colour, -1)
            cv2.putText(frame, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255,255,255), 1, cv2.LINE_AA)
            hist = list(t.centroid_history)[-12:]
            for i in range(1, len(hist)):
                cv2.line(frame, hist[i-1], hist[i], colour, 1)

    def _draw_violation_banners(self, frame: np.ndarray,
                                violations: List[Dict[str, Any]]) -> None:
        for i, v in enumerate(violations[:4]):
            vtype  = v["violation_type"]
            colour = self.VIOLATION_COLORS.get(vtype, (255,255,255))
            pct    = int(v.get("score",0)*100)
            msg    = (f"VIOLATION: {vtype.upper()} | "
                      f"ID{v['track_id']} | {v['class_name']} | {pct}%")
            y = 80 + i*28
            cv2.rectangle(frame, (0,y-20), (len(msg)*8+12,y+6), colour, -1)
            cv2.putText(frame, msg, (5,y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255,255,255), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, frame_no: int, signal_state: str,
                  fps: float, track_count: int) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w,40), (8,12,20), -1)
        sig_col = self.SIGNAL_COLORS.get(signal_state, (128,128,128))
        cv2.circle(frame, (20,20), 11, sig_col, -1)
        cv2.putText(frame, signal_state.upper(), (36,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, sig_col, 1, cv2.LINE_AA)
        right = f"Frame:{frame_no}  FPS:{fps:.1f}  Tracks:{track_count}"
        cv2.putText(frame, right, (w-len(right)*7-8,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190,200,210), 1, cv2.LINE_AA)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (w//2-len(ts)*3,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,170,180), 1, cv2.LINE_AA)


# =============================================================================
# 10. HistoryManager
# =============================================================================

class HistoryManager:
    """Persists session JSON files and a lightweight index."""

    HISTORY_DIR = os.path.join(AppConfig.LOG_DIR, "history")
    INDEX_FILE  = os.path.join(AppConfig.LOG_DIR, "history", "index.json")

    def __init__(self) -> None:
        os.makedirs(self.HISTORY_DIR, exist_ok=True)
        if not os.path.isfile(self.INDEX_FILE):
            self._write_index([])

    def save_session(self, session_id: str, summary: Dict[str, Any]) -> None:
        clean = self._sanitise(dict(summary))
        clean["session_id"]  = session_id
        clean["analysed_at"] = datetime.now().isoformat()
        with open(os.path.join(self.HISTORY_DIR, f"{session_id}.json"), "w") as f:
            json.dump(clean, f, indent=2)
        index = self.load_index()
        index.insert(0, {
            "session_id":          session_id,
            "analysed_at":         clean["analysed_at"],
            "total_frames":        clean.get("total_frames",       0),
            "total_violations":    clean.get("total_violations",   0),
            "red_light_count":     clean.get("red_light_count",   0),
            "helmet_count":        clean.get("helmet_count",       0),
            "lane_count":          clean.get("lane_count",         0),
            "speed_count":         clean.get("speed_count",        0),
            "upload_video_url":    clean.get("upload_video_url",   ""),
            "processed_video_url": clean.get("processed_video_url",""),
            "original_filename":   clean.get("original_filename",  ""),
            "seg_backend":         clean.get("seg_backend",        ""),
        })
        self._write_index(index)

    def load_index(self) -> List[Dict[str, Any]]:
        try:
            with open(self.INDEX_FILE) as f:
                return json.load(f)
        except Exception:
            return []

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.HISTORY_DIR, f"{session_id}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def delete_session(self, session_id: str) -> bool:
        path = os.path.join(self.HISTORY_DIR, f"{session_id}.json")
        removed = False
        if os.path.isfile(path):
            os.remove(path); removed = True
        self._write_index([e for e in self.load_index()
                           if e["session_id"] != session_id])
        return removed

    def _write_index(self, index: List[Dict[str, Any]]) -> None:
        with open(self.INDEX_FILE, "w") as f:
            json.dump(index, f, indent=2)

    def _sanitise(self, obj: Any) -> Any:
        if isinstance(obj, dict):   return {k: self._sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [self._sanitise(v) for v in obj]
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return obj


# =============================================================================
# 11. TrafficViolationSystem
# =============================================================================

class TrafficViolationSystem:
    """
    Top-level orchestrator.

    Camera mode auto-detection
    --------------------------
    Analyses optical flow across the first 20 frames to determine whether
    the video is from a static overhead camera or a moving dashcam:

    DASHCAM indicators
    - Dominant optical flow direction is UPWARD (tracked points move up = away)
    - Dashboard / hood visible in bottom 15% of frame (uniform dark horizontal band)
    - Perspective: road occupies the central vertical strip, not the full frame

    OVERHEAD indicators
    - Objects approach camera (downward optical flow)
    - Road occupies the lower 2/3 of frame in a trapezoid
    - No dashboard structure

    After detection, the mode is passed to ViolationEngine and SignalDetector.
    The stop-line position is recalibrated for the detected mode.
    """

    def __init__(self) -> None:
        logger.info("Initialising TrafficViolationSystem...")
        self._detector = DetectionManager()
        self._ocr      = OCRManager()
        self._renderer = OverlayRenderer()
        self.history   = HistoryManager()
        # Per-run subsystems (reset in process_video)
        self._tracker  = TrackingManager()
        self._evidence = EvidenceManager()
        self._signal   = SignalDetector()
        self._seg_mgr  = SegmentationManager(
            (AppConfig.PROC_HEIGHT, AppConfig.PROC_WIDTH))
        logger.info("TVS ready. YOLO=%s  OCR=%s",
                    self._detector.ready, self._ocr.ready)

    def models_ready(self) -> Dict[str, bool]:
        return {"yolo": self._detector.ready,
                "mask2former": self._seg_mgr.ready,
                "ocr": self._ocr.ready}

    # ── Camera mode detection ─────────────────────────────────────────────────

    @staticmethod
    def _detect_camera_mode(cap: cv2.VideoCapture,
                            proc_w: int, proc_h: int) -> str:
        """
        Auto-detect whether video is dashcam or overhead camera.

        Method
        ------
        1. Read 20 evenly-spaced frames.
        2. Check for dashboard signature: the bottom 12% of the frame
           should be a dark, low-texture horizontal band (car hood/dash).
        3. Measure dominant vertical motion using simple frame differencing:
           if bright regions consistently move UPWARD (y decreases) across
           consecutive frames, it is a dashcam.

        Returns "dashcam" or "overhead".
        """
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, total // 20)
        frames: List[np.ndarray] = []
        for i in range(0, min(total, 20*step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.resize(frame, (proc_w, proc_h)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
        if not frames:
            return "overhead"

        # ── Dashboard check ───────────────────────────────────────────────────
        # In dashcam footage the bottom ~12% shows the car hood — dark, uniform,
        # with low edge density (no road markings / moving objects).
        dash_scores = []
        for f in frames:
            h = f.shape[0]
            hood = cv2.cvtColor(f[int(h*0.88):, :], cv2.COLOR_BGR2GRAY)
            # Low mean brightness AND low std = dark uniform band
            mean_bright = float(np.mean(hood))
            std_bright  = float(np.std(hood))
            edge_density = float(np.mean(cv2.Canny(hood, 30, 90) > 0))
            # Score: 1.0 = clearly a dashboard
            score = (
                (1.0 if mean_bright < 80  else 0.5 if mean_bright < 130 else 0.0) +
                (0.5 if std_bright  < 25  else 0.0) +
                (0.5 if edge_density < 0.05 else 0.0)
            )
            dash_scores.append(score)
        dash_evidence = float(np.mean(dash_scores))

        # ── Upward motion check ───────────────────────────────────────────────
        # Compare consecutive frames: if bright blobs move upward, it is dashcam.
        up_votes = 0
        for i in range(1, len(frames)):
            prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            curr = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.abs(curr - prev)
            # Find centroid of change in upper vs lower half
            h = diff.shape[0]
            upper_change = float(np.sum(diff[:h//2]))
            lower_change = float(np.sum(diff[h//2:]))
            # In dashcam: tracked objects move upward = more change in upper half
            if upper_change > lower_change * 1.2:
                up_votes += 1
        up_frac = up_votes / max(len(frames)-1, 1)

        # ── Decision ──────────────────────────────────────────────────────────
        is_dashcam = (dash_evidence >= 1.0 or
                      (dash_evidence >= 0.5 and up_frac > 0.4))
        mode = "dashcam" if is_dashcam else "overhead"
        logger.info("Camera mode detected: %s  (dash_evidence=%.2f  up_frac=%.2f)",
                    mode, dash_evidence, up_frac)
        return mode

    # ── Stop-line calibration for dashcam ─────────────────────────────────────

    @staticmethod
    def _find_dashcam_stop_line(frame: np.ndarray,
                                proc_h: int, proc_w: int) -> int:
        """
        Locate the crosswalk / stop line in dashcam footage by finding
        the dense horizontal band of white stripes in the lower-middle area.

        Returns the Y coordinate of the TOP edge of the crosswalk.
        Falls back to 62% of frame height if not found.
        """
        h, w   = frame.shape[:2]
        # Search between 40% and 80% of frame height
        search = frame[int(h*0.40):int(h*0.80), :]
        grey   = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        # White stripe: very bright pixels
        _, white = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
        # Count white pixels per row
        row_sum  = white.sum(axis=1)
        # Smooth
        kernel   = np.ones(5) / 5.0
        smooth   = np.convolve(row_sum.astype(np.float32), kernel, mode="same")
        if smooth.max() > 0:
            # Find the start of the highest-density white band
            peak_row = int(np.argmax(smooth))
            stop_y   = int(h * 0.40) + peak_row
            # Clamp to reasonable range
            stop_y   = max(int(h*0.45), min(stop_y, int(h*0.75)))
            return stop_y
        return int(h * 0.62)

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def process_video(self, input_path: str, output_path: str,
                      session_id: str = "") -> Dict[str, Any]:
        """
        Full violation-detection pipeline.

        1. Reset per-run subsystems.
        2. Auto-detect camera mode (dashcam / overhead).
        3. Calibrate stop-line position for dashcam.
        4. Run per-frame loop: segment → detect → track → signal → evaluate → render.
        5. Persist to history and return summary.
        """
        # ── Reset per-run state ───────────────────────────────────────────────
        self._tracker  = TrackingManager()
        self._evidence = EvidenceManager()
        self._signal   = SignalDetector()
        self._seg_mgr  = SegmentationManager(
            (AppConfig.PROC_HEIGHT, AppConfig.PROC_WIDTH))

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        src_fps  = cap.get(cv2.CAP_PROP_FPS) or AppConfig.FPS_DEFAULT
        src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        proc_w, proc_h = AppConfig.PROC_WIDTH, AppConfig.PROC_HEIGHT

        # ── Auto-detect camera mode ───────────────────────────────────────────
        camera_mode = self._detect_camera_mode(cap, proc_w, proc_h)
        # Pass mode to detector so synthetic fallback behaves correctly
        self._detector.set_camera_mode(camera_mode)

        # ── Initial segmentation on first frame ───────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_raw = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Video has no readable frames.")
        first_frame = cv2.resize(first_raw, (proc_w, proc_h))
        seg_info    = self._seg_mgr.segment(first_frame, frame_number=0)

        # ── Calibrate stop line for dashcam ──────────────────────────────────
        if camera_mode == "dashcam":
            stop_y = self._find_dashcam_stop_line(first_frame, proc_h, proc_w)
            seg_info["stop_line_polygon"] = [(0, stop_y), (proc_w, stop_y)]
            logger.info("Dashcam stop-line calibrated at y=%d (%.1f%% height)",
                        stop_y, stop_y / proc_h * 100)

        logger.info("Video: %dx%d @ %.1f fps (%d frames) | mode=%s | seg=%s",
                    src_w, src_h, src_fps, total_in, camera_mode,
                    self._seg_mgr.backend)

        v_engine = ViolationEngine(seg_info, self._evidence, self._ocr,
                                   camera_mode=camera_mode)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (proc_w, proc_h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_no       = 0
        total_det      = 0
        all_violations: List[Dict[str, Any]] = []
        fps_times: deque = deque(maxlen=30)

        while frame_no < AppConfig.MAX_FRAMES:
            ret, raw = cap.read()
            if not ret:
                break
            frame = cv2.resize(raw, (proc_w, proc_h))
            t0    = time.perf_counter()

            # Segmentation (stride-cached)
            seg_info      = self._seg_mgr.segment(frame, frame_number=frame_no)
            v_engine._seg = seg_info

            detections   = self._detector.detect(frame)
            total_det   += len(detections)
            tracks       = self._tracker.update(detections, frame_number=frame_no)
            signal_state = self._signal.update(frame, detections)

            frame_violations = v_engine.evaluate(
                frame, tracks, detections, signal_state, frame_no)
            all_violations.extend(frame_violations)

            elapsed  = time.perf_counter() - t0
            fps_times.append(elapsed)
            proc_fps = 1.0 / (sum(fps_times)/len(fps_times)) if fps_times else 0.0

            annotated = self._renderer.render(
                frame, tracks, seg_info, signal_state,
                frame_no, frame_violations, proc_fps)
            writer.write(annotated)
            frame_no += 1

        cap.release()
        writer.release()
        logger.info("Pipeline done — %d frames, %d violations (mode=%s)",
                    frame_no, len(all_violations), camera_mode)

        summary = self._build_summary(frame_no, total_det, all_violations,
                                      fps_times, camera_mode)
        sid = session_id or uuid.uuid4().hex
        self.history.save_session(sid, summary)
        summary["session_id"] = sid
        return summary

    def _build_summary(self, total_frames: int, total_det: int,
                       violations: List[Dict[str, Any]],
                       fps_times: deque, camera_mode: str = "overhead") -> Dict[str, Any]:
        v_by_type: Dict[str, int] = defaultdict(int)
        for v in violations:
            v_by_type[v["violation_type"]] += 1
        avg_fps = (1.0/(sum(fps_times)/len(fps_times)) if fps_times else 0.0)

        evidence_items = []
        for rec in self._evidence.records[-50:]:
            img_rel = rec["evidence_path"].replace("\\", "/")
            if not img_rel.startswith("/"):
                img_rel = "/" + img_rel
            evidence_items.append({
                "path": img_rel, "violation_type": rec["violation_type"],
                "object_id": rec["object_id"], "plate_number": rec["plate_number"],
                "speed_value": rec["speed_value"], "frame_number": rec["frame_number"]})

        return {
            "total_frames":           total_frames,
            "total_detections":       total_det,
            "total_tracks":           len(self._tracker._tracks),
            "total_violations":       len(violations),
            "red_light_count":        v_by_type.get("red_light", 0),
            "helmet_count":           v_by_type.get("helmet",    0),
            "lane_count":             v_by_type.get("lane",      0),
            "speed_count":            v_by_type.get("speed",     0),
            "average_processing_fps": round(avg_fps, 1),
            "seg_backend":            self._seg_mgr.backend,
            "camera_mode":            camera_mode,
            "evidence_items":         evidence_items,
            "recent_violations":      [
                {**v, "timestamp": datetime.now().isoformat()}
                for v in violations[-100:]],
            "extracted_plate_records": [
                r for r in self._evidence.records if r["plate_number"]],
        }