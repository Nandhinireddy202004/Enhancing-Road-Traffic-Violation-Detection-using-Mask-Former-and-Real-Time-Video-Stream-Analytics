"""
app.py — TrafficSentinel v10 (Modular — Multiple Violations Per Vehicle Enabled)
══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import csv as _csv_module
import json
import logging
import os
import queue
import smtplib as _smtplib
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from email import encoders as _encoders
from email.mime.base import MIMEBase as _MIMEBase
from email.mime.multipart import MIMEMultipart as _MIMEMultipart
from email.mime.text import MIMEText as _MIMEText
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

# Import modular components
from helmet_detection import HelmetDetector
from overspeed import SpeedEstimator, OverSpeedDetector
from wrong_lane import LaneDetector
from red_light_jump import RedLightViolationEngine, SignalController

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("TrafficSentinel")

# ── optional deps ─────────────────────────────────────────────────────────────
YOLO_AVAILABLE = False
M2F_AVAILABLE = False
TORCH_AVAILABLE = False
TURBO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from ultralytics import YOLO as _YOLO_CLS
    YOLO_AVAILABLE = True
except ImportError:
    log.warning("ultralytics not installed — YOLO in demo mode.")

try:
    if TORCH_AVAILABLE:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        from PIL import Image as _PILImage
        M2F_AVAILABLE = True
except ImportError:
    pass

try:
    from turbojpeg import TurboJPEG as _TurboJPEG
    _turbo = _TurboJPEG()
    TURBO_AVAILABLE = True
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
EVIDENCE_FOLDER = "evidence"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EVIDENCE_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
INTERNAL_WIDTH = 1280
INTERNAL_HEIGHT = 720
FRAME_SKIP = 1
SEG_INTERVAL = 20
STREAM_FPS = 25
JPEG_QUALITY = 65
Q_MAX_SIZE = 2

M2F_MODEL_ID = "facebook/mask2former-swin-tiny-cityscapes-semantic"
M2F_W, M2F_H = 384, 216
CITYSCAPES_ROAD_IDS = frozenset({0, 1})
MARKING_V_THRESH = 180
ROAD_DILATE_K = 25
SHOW_ROAD_MASK = True

YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF = 0.35
YOLO_IOU = 0.45
YOLO_VEHICLE_IDS: Dict[int, str] = {
    1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
}

SPEED_LIMIT_KMH = 50.0
SCALE_MPP = 0.1  # Calibrated for slow moving vehicles

IOU_WEIGHT = 0.60
DIST_WEIGHT = 0.40
MAX_LOST_FRAMES = 20
TRAJ_LEN = 30
VEL_WINDOW = 6

VIOLATION_META: Dict[str, Dict] = {
    "red_light": {"label": "Red Light Jump", "severity": "HIGH", "color": (0, 0, 255)},
    "no_helmet": {"label": "No Helmet", "severity": "MEDIUM", "color": (0, 165, 255)},
    "over_speed": {"label": "Over Speed", "severity": "HIGH", "color": (0, 80, 255)},
    "wrong_direction": {"label": "Wrong Direction", "severity": "HIGH", "color": (255, 0, 0)},
    "lane_crossing": {"label": "Lane Crossing", "severity": "MEDIUM", "color": (255, 165, 0)},
}
_SIGNAL_BGR: Dict[str, Tuple[int, int, int]] = {
    "GREEN": (30, 200, 30),
    "YELLOW": (0, 200, 220),
    "RED": (0, 0, 220),
}
COOLDOWN_FRAMES = 300

DB_PATH = "traffic_history.db"
DB_BATCH_SIZE = 20
DB_FLUSH_SECS = 2.0
EVIDENCE_JPEG_Q = 72

# ── EMAIL CONFIGURATION (HARDCODED FOR DEMO) ─────────────────────────────────
# IMPORTANT: Replace these with your actual Gmail credentials
# For Gmail, you need to use an App Password (not your regular password)
# Steps to get App Password:
# 1. Enable 2-Factor Authentication on your Google account
# 2. Go to Security → App Passwords
# 3. Select "Mail" and "Other", generate password
# 4. Use that 16-character password here
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "nandhinireddysane20@gmail.com"
SMTP_PASS = "xtflnvxtonzqcpme"
SMTP_FROM = "nandhinireddysane20@gmail.com"
EMAIL_ENABLED = True
# Alternative: Use environment variables (uncomment to use)
# SMTP_USER = os.environ.get("SMTP_USER", "")
# SMTP_PASS = os.environ.get("SMTP_PASS", "")
# SMTP_FROM = os.environ.get("SMTP_USER", "")
# EMAIL_ENABLED = bool(SMTP_USER and SMTP_PASS)

# Test email address for sending test emails
TEST_EMAIL = "test-recipient@example.com"  # ← REPLACE WITH TEST EMAIL FOR TESTING

BLACKLIST_THR = 10
VEHICLE_CSV = "vehicles.csv"
DEDUP_WINDOW_SEC = 30  # Different violation types allowed immediately

FINE_TABLE: Dict[str, float] = {
    "no_helmet": 500.0,
    "red_light": 1000.0,
    "over_speed": 2000.0,
    "wrong_direction": 750.0,
    "lane_crossing": 500.0,
    "no_license": 1500.0,
}

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════
violation_log: List[Dict] = []
frame_stats: Dict[str, Any] = {
    "fps": 0.0,
    "total_vehicles": 0,
    "violations_today": 0,
    "active_streams": 0,
}
processors: Dict[str, "StreamProcessor"] = {}
signal_ctrl = SignalController()

# Deduplication cache - per violation type
_dedup_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
_dedup_lock = threading.Lock()


def _is_duplicate(track_id: str, vtype: str) -> bool:
    """Check if same violation type was recently logged for this track."""
    now = time.time()
    with _dedup_lock:
        last = _dedup_cache.get(track_id, {}).get(vtype, 0.0)
        if now - last < DEDUP_WINDOW_SEC:
            return True
        _dedup_cache.setdefault(track_id, {})[vtype] = now
        return False


def _clear_duplicate_cache(track_id: str) -> None:
    """Clear dedup cache for a track."""
    with _dedup_lock:
        _dedup_cache.pop(track_id, None)


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _init_db() -> None:
    os.makedirs(EVIDENCE_FOLDER, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA cache_size=8000;")
        c.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id             TEXT PRIMARY KEY,
                timestamp      TEXT,
                type           TEXT,
                label          TEXT,
                severity       TEXT,
                track_id       TEXT,
                vehicle_number TEXT,
                vehicle        TEXT,
                confidence     REAL,
                frame_no       INTEGER,
                stream_id      TEXT,
                evidence_file  TEXT,
                speed_kmh      REAL,
                lane_id        INTEGER,
                challan_id     TEXT,
                extra_data     TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_number  TEXT PRIMARY KEY,
                owner_name      TEXT DEFAULT '',
                address         TEXT DEFAULT '',
                email           TEXT DEFAULT '',
                phone           TEXT DEFAULT '',
                total_challans  INTEGER DEFAULT 0,
                active_challans INTEGER DEFAULT 0,
                blacklisted     INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now')),
                updated_at      TEXT DEFAULT (datetime('now'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS challans (
                challan_id      TEXT PRIMARY KEY,
                vehicle_number  TEXT,
                violation_id    TEXT,
                violation_type  TEXT,
                fine_amount     REAL DEFAULT 0,
                status          TEXT DEFAULT 'unpaid',
                notes           TEXT DEFAULT '',
                issued_by       TEXT DEFAULT 'TrafficSentinel',
                email_sent      INTEGER DEFAULT 0,
                timestamp       TEXT DEFAULT (datetime('now')),
                paid_at         TEXT,
                FOREIGN KEY (vehicle_number) REFERENCES vehicles(vehicle_number)
            )
        """)
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_type ON violations(type)",
            "CREATE INDEX IF NOT EXISTS idx_severity ON violations(severity)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON violations(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_stream ON violations(stream_id)",
            "CREATE INDEX IF NOT EXISTS idx_vnum ON violations(vehicle_number)",
            "CREATE INDEX IF NOT EXISTS idx_ch_vnum ON challans(vehicle_number)",
            "CREATE INDEX IF NOT EXISTS idx_ch_status ON challans(status)",
        ]:
            c.execute(stmt)
        c.commit()
    log.info("[DB] Initialized.")


@contextmanager
def _db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _db_row(v: Dict) -> tuple:
    _known = {
        "id", "timestamp", "type", "label", "severity", "track_id", "vehicle_number",
        "vehicle", "confidence", "frame_no", "stream_id", "evidence_file",
        "speed_kmh", "lane_id", "challan_id", "bbox", "centroid", "orig_w", "orig_h",
    }
    extra = {k: val for k, val in v.items() if k not in _known}
    return (
        v.get("id", str(uuid.uuid4())[:8]),
        v.get("timestamp", datetime.now().isoformat()),
        v.get("type", "unknown"),
        v.get("label", "Unknown"),
        v.get("severity", "LOW"),
        v.get("track_id", ""),
        v.get("vehicle_number", None),
        v.get("vehicle", "unknown"),
        float(v.get("confidence", 0.0)),
        int(v.get("frame_no", 0)),
        v.get("stream_id", "default"),
        v.get("evidence_file", ""),
        float(v.get("speed_kmh", 0.0)),
        int(v.get("lane_id", -1)),
        v.get("challan_id", None),
        json.dumps(extra) if extra else None,
    )


class AsyncDBWriter:
    def __init__(self) -> None:
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="DBWriter")
        self._thread.start()

    def enqueue(self, v: Dict, frame: Optional[np.ndarray] = None) -> None:
        self._q.put_nowait((v, frame))

    def flush(self) -> None:
        self._q.join()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        batch: List[Dict] = []
        last_flush = time.monotonic()
        while not self._stop.is_set() or not self._q.empty():
            try:
                item = self._q.get(timeout=DB_FLUSH_SECS)
                v, frame = item
                if frame is not None:
                    self._save_evidence(v, frame)
                batch.append(v)
                self._q.task_done()
            except queue.Empty:
                pass
            now = time.monotonic()
            if len(batch) >= DB_BATCH_SIZE or (batch and now - last_flush >= DB_FLUSH_SECS):
                self._flush_batch(batch)
                batch.clear()
                last_flush = now
        if batch:
            self._flush_batch(batch)

    @staticmethod
    def _flush_batch(batch: List[Dict]) -> None:
        if not batch:
            return
        try:
            with _db_conn() as c:
                c.executemany("""
                    INSERT OR REPLACE INTO violations
                    (id,timestamp,type,label,severity,track_id,vehicle_number,vehicle,
                     confidence,frame_no,stream_id,evidence_file,speed_kmh,lane_id,challan_id,extra_data)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, [_db_row(v) for v in batch])
        except Exception as exc:
            log.error(f"[DB] Batch flush error: {exc}")

    @staticmethod
    def _save_evidence(v: Dict, frame: np.ndarray) -> None:
        try:
            conf = float(v.get("confidence") or 0.0)
            if conf < 0.50:
                return

            os.makedirs(EVIDENCE_FOLDER, exist_ok=True)
            fname = f"{v['type']}_{v['id']}_{v.get('frame_no', 0)}.jpg"
            path = os.path.join(EVIDENCE_FOLDER, fname)

            orig_h, orig_w = frame.shape[:2]
            pipeline_w = int(v.get("orig_w") or orig_w)
            pipeline_h = int(v.get("orig_h") or orig_h)

            if orig_w > 640:
                scale = 640.0 / orig_w
                ev = cv2.resize(frame, (640, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
            else:
                ev = frame.copy()

            disp_h, disp_w = ev.shape[:2]
            raw_bbox = v.get("bbox")

            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                sx = disp_w / max(pipeline_w, 1)
                sy = disp_h / max(pipeline_h, 1)
                x1 = max(0, int(raw_bbox[0] * sx))
                y1 = max(0, int(raw_bbox[1] * sy))
                x2 = min(disp_w - 1, int(raw_bbox[2] * sx))
                y2 = min(disp_h - 1, int(raw_bbox[3] * sy))

                if x2 - x1 >= 4 and y2 - y1 >= 4:
                    sev_col = {
                        "HIGH": (0, 0, 255),
                        "MEDIUM": (0, 130, 255),
                        "LOW": (150, 0, 220),
                    }.get(v.get("severity", "MEDIUM"), (0, 130, 255))

                    overlay = ev.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), sev_col, -1)
                    cv2.addWeighted(overlay, 0.20, ev, 0.80, 0, ev)
                    cv2.rectangle(ev, (x1, y1), (x2, y2), sev_col, 3)

                    tag = f"{v.get('label','').upper()}  {conf*100:.0f}%"
                    cv2.putText(ev, tag, (x1, max(y1 - 6, 16)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            bar = np.zeros((48, disp_w, 3), dtype=np.uint8)
            bar[:] = (18, 18, 35)
            sev_col2 = {"HIGH": (0, 0, 200), "MEDIUM": (0, 100, 200), "LOW": (100, 0, 180)}.get(
                v.get("severity", "MEDIUM"), (0, 100, 200)
            )
            cv2.rectangle(bar, (0, 0), (5, 48), sev_col2, -1)
            cv2.circle(bar, (22, 24), 13, sev_col2, -1)
            cv2.putText(bar, "!", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            label_txt = v.get("label", v.get("type", "")).upper()
            cv2.putText(bar, label_txt, (42, 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            meta_txt = f"{v.get('vehicle','rider')}  |  {v.get('track_id','')}  |  {conf*100:.0f}% conf"
            cv2.putText(bar, meta_txt, (42, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 200, 220), 1, cv2.LINE_AA)

            bot = np.zeros((20, disp_w, 3), dtype=np.uint8)
            bot[:] = (12, 12, 28)
            ts_str = (v.get("timestamp") or "")[:19].replace("T", " ")
            cv2.putText(bot, f"{ts_str}   {v.get('stream_id','')}", (10, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 160, 180), 1, cv2.LINE_AA)

            ev_final = np.vstack([bar, ev, bot])
            cv2.imwrite(path, ev_final, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_Q])
            v["evidence_file"] = fname
            log.debug(f"[Evidence] Saved {fname}")

        except Exception as exc:
            log.error(f"[DB] Evidence save error: {exc}")


def db_get_history(limit=50, vtype=None, severity=None) -> List[Dict]:
    try:
        params: list = []
        conds: list = []
        if vtype:
            conds.append("type = ?")
            params.append(vtype)
        if severity:
            conds.append("severity = ?")
            params.append(severity)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        params.append(limit)
        with _db_conn() as c:
            rows = c.execute(
                f"SELECT * FROM violations {where} ORDER BY timestamp DESC LIMIT ?", params
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("extra_data"):
                try:
                    d.update(json.loads(d["extra_data"]))
                except Exception:
                    pass
            d.pop("extra_data", None)
            result.append(d)
        return result
    except Exception as exc:
        log.error(f"[DB] get_history error: {exc}")
        return []


def db_get_stats() -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "total": 0, "red_light": 0, "no_helmet": 0,
        "wrong_direction": 0, "lane_crossing": 0, "over_speed": 0, "other": 0,
    }
    try:
        with _db_conn() as c:
            row = c.execute("SELECT COUNT(*) FROM violations").fetchone()
            base["total"] = row[0] if row else 0
            for r in c.execute("SELECT type, COUNT(*) FROM violations GROUP BY type").fetchall():
                if r[0] in base:
                    base[r[0]] = r[1]
                else:
                    base["other"] += r[1]
    except Exception as exc:
        log.error(f"[DB] get_stats error: {exc}")
    return base


def db_clear() -> bool:
    try:
        with _db_conn() as c:
            c.execute("DELETE FROM violations")
        return True
    except Exception as exc:
        log.error(f"[DB] clear error: {exc}")
        return False


_db_writer: Optional[AsyncDBWriter] = None


def get_db_writer() -> AsyncDBWriter:
    global _db_writer
    if _db_writer is None:
        _db_writer = AsyncDBWriter()
    return _db_writer


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class TrackedObject:
    __slots__ = (
        "track_id", "label", "bbox", "confidence",
        "velocity_px", "trajectory", "lane_id",
        "lost_frames", "age", "stream_id", "_vel_buf",
        "violation_counts"
    )

    def __init__(self, track_id: str, det: Dict) -> None:
        self.track_id = track_id
        self.label = det["label"]
        self.bbox = det["bbox"]
        self.confidence = det["confidence"]
        self.velocity_px = 0.0
        self.trajectory: deque = deque(maxlen=TRAJ_LEN)
        self.lane_id = -1
        self.lost_frames = 0
        self.age = 0
        self.stream_id = det.get("stream_id", "default")
        self._vel_buf: deque = deque(maxlen=VEL_WINDOW)
        self.violation_counts: Dict[str, int] = defaultdict(int)

    def update(self, det: Dict) -> None:
        prev_cx, prev_cy = self.centroid
        self.bbox = det["bbox"]
        self.confidence = det["confidence"]
        self.label = det["label"]
        self.lost_frames = 0
        self.age += 1
        cx, cy = self.centroid
        self._vel_buf.append(float(np.hypot(cx - prev_cx, cy - prev_cy)))
        self.velocity_px = float(np.mean(self._vel_buf)) if self._vel_buf else 0.0
        self.trajectory.append((cx, cy))

    @property
    def centroid(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (max(1, (a[2] - a[0]) * (a[3] - a[1])) + max(1, (b[2] - b[0]) * (b[3] - b[1])) - inter)


class ByteTracker:
    def __init__(self) -> None:
        self._tracks: Dict[str, TrackedObject] = {}
        self._counter: int = 0
        self._fw: int = 1280
        self._fh: int = 720
        self._lane_det = None

    def set_lane_det(self, lane_det) -> None:
        self._lane_det = lane_det

    def update(self, detections: List[Dict], frame_hw: Tuple[int, int]) -> List[TrackedObject]:
        self._fh, self._fw = frame_hw
        high = [d for d in detections if d["confidence"] >= 0.50]
        low = [d for d in detections if d["confidence"] < 0.50]
        unmatched_d, unmatched_t = self._associate(high, list(self._tracks.values()))
        low_unmatched, _ = self._associate(low, unmatched_t, iou_thresh=0.30)
        for det in unmatched_d + low_unmatched:
            self._new_track(det)
        dead = [tid for tid, t in self._tracks.items() if t.lost_frames > MAX_LOST_FRAMES]
        for tid in dead:
            del self._tracks[tid]
            if self._lane_det is not None:
                self._lane_det.remove_track(tid)
            _clear_duplicate_cache(tid)
        matched_ids = {d.get("track_id") for d in detections if "track_id" in d}
        for tid, t in self._tracks.items():
            if tid not in matched_ids:
                t.lost_frames += 1
        return list(self._tracks.values())

    def _associate(self, dets, tracks, iou_thresh=0.20):
        if not dets or not tracks:
            return dets, tracks
        cost = np.full((len(dets), len(tracks)), 1e9)
        diag = float(np.hypot(self._fw, self._fh)) + 1e-6
        for i, det in enumerate(dets):
            for j, trk in enumerate(tracks):
                iou = _bbox_iou(det["bbox"], trk.bbox)
                d_cx = (det["bbox"][0] + det["bbox"][2]) // 2
                d_cy = (det["bbox"][1] + det["bbox"][3]) // 2
                t_cx, t_cy = trk.centroid
                dist_n = np.hypot(d_cx - t_cx, d_cy - t_cy) / diag
                if iou > iou_thresh:
                    cost[i, j] = 1.0 - (IOU_WEIGHT * iou + DIST_WEIGHT * (1 - dist_n))
        matched_d: set = set()
        matched_t: set = set()
        if cost.min() < 1e8:
            for i in np.argsort(cost.min(axis=1)):
                if i in matched_d:
                    continue
                j = int(np.argmin(cost[i]))
                if j in matched_t or cost[i, j] >= 1e8:
                    continue
                tracks[j].update(dets[i])
                dets[i]["track_id"] = tracks[j].track_id
                matched_d.add(i)
                matched_t.add(j)
        return (
            [dets[i] for i in range(len(dets)) if i not in matched_d],
            [tracks[j] for j in range(len(tracks)) if j not in matched_t],
        )

    def _new_track(self, det: Dict) -> TrackedObject:
        self._counter += 1
        tid = f"T{self._counter:04d}"
        t = TrackedObject(tid, det)
        t.trajectory.append(t.centroid)
        self._tracks[tid] = t
        det["track_id"] = tid
        return t


# ══════════════════════════════════════════════════════════════════════════════
#  ROAD SEGMENTOR (simplified)
# ══════════════════════════════════════════════════════════════════════════════

class RoadSegmentor:
    def __init__(self) -> None:
        self._proc = None
        self._model = None
        self._device = "cpu"
        self._ready = False
        self._cache_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._cached_road: Optional[np.ndarray] = None
        self._cached_marking: Optional[np.ndarray] = None
        self._cache_hw: Tuple[int, int] = (0, 0)
        self._frames_since: int = SEG_INTERVAL
        self._infer_pending: bool = False
        if M2F_AVAILABLE:
            threading.Thread(target=self._load, daemon=True, name="M2F-Load").start()

    def _load(self) -> None:
        try:
            if TORCH_AVAILABLE:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._proc = AutoImageProcessor.from_pretrained(M2F_MODEL_ID)
            self._model = (
                Mask2FormerForUniversalSegmentation
                .from_pretrained(M2F_MODEL_ID).to(self._device).eval()
            )
            self._ready = True
            log.info("[M2F] Ready.")
        except Exception as exc:
            log.error(f"[M2F] Load failed: {exc}")

    def get_both(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        self._frames_since += 1
        need = (self._frames_since >= SEG_INTERVAL or self._cached_road is None or self._cache_hw != (h, w))
        if need and not self._infer_pending:
            self._frames_since = 0
            self._infer_pending = True
            threading.Thread(target=self._async_refresh, args=(frame.copy(), h, w),
                             daemon=True, name="M2F-Infer").start()
        with self._cache_lock:
            if self._cached_road is None:
                return _fallback_road(h, w), np.zeros((h, w), dtype=np.uint8)
            return self._cached_road, self._cached_marking

    def _async_refresh(self, frame: np.ndarray, h: int, w: int) -> None:
        with self._infer_lock:
            try:
                seg = self._infer(frame, h, w) if self._ready else None
                if seg is not None:
                    road = np.zeros((h, w), dtype=np.uint8)
                    for lid in CITYSCAPES_ROAD_IDS:
                        road[seg == lid] = 255
                else:
                    road = _fallback_road(h, w)
                marking = (self._extract_markings(frame, seg, road)
                           if seg is not None else np.zeros((h, w), dtype=np.uint8))
                with self._cache_lock:
                    self._cached_road = road
                    self._cached_marking = marking
                    self._cache_hw = (h, w)
            except Exception as exc:
                log.error(f"[M2F] Async error: {exc}")
            finally:
                self._infer_pending = False

    def _infer(self, frame: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        try:
            small = cv2.resize(frame, (M2F_W, M2F_H), interpolation=cv2.INTER_LINEAR)
            pil = _PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            inp = self._proc(images=pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model(**inp)
            pred = self._proc.post_process_semantic_segmentation(
                out, target_sizes=[(M2F_H, M2F_W)])[0]
            return cv2.resize(pred.cpu().numpy().astype(np.int32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        except Exception as exc:
            log.error(f"[M2F] Infer error: {exc}")
            return None

    @staticmethod
    def _extract_markings(frame, seg, road) -> np.ndarray:
        road_px = (seg == 0).astype(np.uint8) * 255
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bright = cv2.bitwise_and(
            (hsv[:, :, 2] >= MARKING_V_THRESH).astype(np.uint8) * 255, road_px)
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        k_op = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_h), cv2.MORPH_OPEN, k_op)

    @staticmethod
    def dilate(mask: np.ndarray, ksize: int = ROAD_DILATE_K) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.dilate(mask, k)


def _fallback_road(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[int(h * 0.35):, :] = 255
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  VEHICLE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class VehicleDetector:
    YOLO_W = 640
    YOLO_H = 360

    def __init__(self) -> None:
        self._model = None
        self._ready = False
        self._device = "cpu"
        self._skip_ctr = 0
        self._cache: List[Dict] = []
        self._cache_lock = threading.Lock()
        self._dilated_road: Optional[np.ndarray] = None
        self._dilated_hw: Tuple[int, int] = (0, 0)
        if YOLO_AVAILABLE:
            threading.Thread(target=self._load, daemon=True, name="YOLO-Load").start()

    def _load(self) -> None:
        try:
            if TORCH_AVAILABLE:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = _YOLO_CLS(YOLO_MODEL_NAME)
            if self._device == "cuda":
                self._model.to("cuda")
            self._model(np.zeros((self.YOLO_H, self.YOLO_W, 3), dtype=np.uint8), verbose=False)
            self._ready = True
            log.info("[YOLO] Ready.")
        except Exception as exc:
            log.error(f"[YOLO] Load failed: {exc}")

    def detect(self, frame: np.ndarray, road_mask: np.ndarray) -> List[Dict]:
        with self._cache_lock:
            self._skip_ctr += 1
            do_infer = self._skip_ctr >= FRAME_SKIP
            if do_infer:
                self._skip_ctr = 0
        if not do_infer:
            with self._cache_lock:
                return list(self._cache)

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.YOLO_W, self.YOLO_H), interpolation=cv2.INTER_LINEAR)
        raw = self._infer(small) if self._ready else self._demo(frame)
        sx, sy = w / self.YOLO_W, h / self.YOLO_H
        for d in raw:
            x1, y1, x2, y2 = d["bbox"]
            d["bbox"] = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]

        if self._dilated_road is None or self._dilated_hw != (h, w):
            self._dilated_road = RoadSegmentor.dilate(road_mask)
            self._dilated_hw = (h, w)

        result = [d for d in raw if self._on_road(d["bbox"], self._dilated_road, h, w)]
        with self._cache_lock:
            self._cache = result
        return list(result)

    def _infer(self, frame: np.ndarray) -> List[Dict]:
        try:
            res = self._model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                              classes=list(YOLO_VEHICLE_IDS.keys()), verbose=False)[0]
            return [
                {"label": YOLO_VEHICLE_IDS[int(cls_t)],
                 "confidence": float(conf_t),
                 "bbox": list(map(int, box.tolist()))}
                for box, cls_t, conf_t in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf)
                if int(cls_t) in YOLO_VEHICLE_IDS
            ]
        except Exception as exc:
            log.error(f"[YOLO] Infer error: {exc}")
            return []

    @staticmethod
    def _on_road(bbox, dil, h, w) -> bool:
        x1, y1, x2, y2 = bbox
        cx = max(0, min((x1 + x2) // 2, w - 1))
        cy = max(0, min((y1 + y2) // 2, h - 1))
        by = max(0, min(y2, h - 1))
        return dil[cy, cx] > 0 or dil[by, cx] > 0

    @staticmethod
    def _demo(frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        rng = np.random.default_rng(int(time.time() * 10) % 99999)
        return [
            {"label": str(rng.choice(["car", "motorcycle", "bus", "truck"])),
             "bbox": [int(rng.integers(0, w - 130)), int(rng.integers(h // 2, h - 90)),
                      int(rng.integers(0, w - 130)) + int(rng.integers(70, 130)),
                      int(rng.integers(h // 2, h - 90)) + int(rng.integers(50, 90))],
             "confidence": float(rng.uniform(0.55, 0.97))}
            for _ in range(int(rng.integers(2, 5)))
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  VIOLATION ENGINE - WITH PROPER WRONG LANE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class ViolationEngine:
    def __init__(
        self,
        speed_est: SpeedEstimator,
        lane_det: LaneDetector,
        rl_engine: RedLightViolationEngine,
    ) -> None:
        self._speed = speed_est
        self._lane = lane_det
        self._rl = rl_engine
        self._cooldown: Dict[str, Dict[str, int]] = defaultdict(dict)

    def process(
        self,
        tracks: List[TrackedObject],
        phase: str,
        meta: Dict,
        fps: float,
    ) -> List[Dict]:
        fn = meta["frame_no"]
        fw = meta["frame_width"]
        fired: List[Dict] = []

        # Update lane tracking for all tracks
        for t in tracks:
            if t.lost_frames > 0:
                continue
            cx, cy = t.centroid
            self._lane.update_track(t.track_id, (cx, cy))

        # Update scene direction (required for wrong lane detection)
        self._lane.tick_scene_direction()

        # Process each track
        for t in tracks:
            if t.lost_frames > 0:
                continue

            cx, cy = t.centroid
            kmh = self._speed.update(t.track_id, (cx, cy), fps)
            t.lane_id = self._lane.assign_lane((cx, cy))

            if t.confidence < 0.45:
                continue

            # ── 1. OVER SPEED VIOLATION ──
            if self._speed.is_speeding(t.track_id):
                if not self._on_cd(t.track_id, "over_speed", fn):
                    self._set_cd(t.track_id, "over_speed", fn)
                    v = self._make_v("over_speed", t, meta)
                    v.update({
                        "speed_kmh": kmh,
                        "limit_kmh": self._speed.speed_limit,
                        "excess_kmh": round(kmh - self._speed.speed_limit, 1),
                    })
                    fired.append(v)
                    t.violation_counts["over_speed"] += 1
                    log.info(f"[Violation] {t.track_id}: OVER SPEED #{t.violation_counts['over_speed']} at {kmh:.1f}km/h")

            # ── 2. RED LIGHT VIOLATION ──
            rl = self._rl.process(
                track_id=t.track_id, bbox=list(t.bbox),
                velocity_px=t.velocity_px, signal_phase=phase,
                frame_no=fn, frame_w=fw,
            )
            if rl and not self._on_cd(t.track_id, "red_light", fn):
                self._set_cd(t.track_id, "red_light", fn)
                v = self._make_v("red_light", t, meta)
                v.update(rl)
                fired.append(v)
                t.violation_counts["red_light"] += 1
                log.info(f"[Violation] {t.track_id}: RED LIGHT #{t.violation_counts['red_light']}")

            # ── 3. WRONG LANE / WRONG DIRECTION VIOLATION ──
            # This uses the LaneDetector.check_violation() which returns a dict
            lane_result = self._lane.check_violation(
                t.track_id, (cx, cy), bbox=list(t.bbox), frame_no=fn
            )
            if lane_result:
                violation_type = lane_result.get("type", "wrong_direction")
                if not self._on_cd(t.track_id, violation_type, fn):
                    self._set_cd(t.track_id, violation_type, fn)
                    v = self._make_v(violation_type, t, meta)
                    v.update({
                        "reason": lane_result.get("reason", violation_type),
                        "lane_id": lane_result.get("lane_id", t.lane_id),
                        "direction": lane_result.get("direction", "UNKNOWN"),
                        "confidence": max(v["confidence"], lane_result.get("confidence", 0.0)),
                    })
                    fired.append(v)
                    t.violation_counts[violation_type] += 1
                    log.info(f"[Violation] {t.track_id}: {violation_type.upper()} #{t.violation_counts[violation_type]}")

        return fired

    def _on_cd(self, tid: str, vtype: str, fn: int) -> bool:
        last = self._cooldown[tid].get(vtype)
        return last is not None and (fn - last) < COOLDOWN_FRAMES

    def _set_cd(self, tid: str, vtype: str, fn: int) -> None:
        self._cooldown[tid][vtype] = fn

    @staticmethod
    def _make_v(vtype: str, t: TrackedObject, meta: Dict) -> Dict:
        info = VIOLATION_META.get(vtype, {"label": vtype.replace("_", " ").title(), "severity": "MEDIUM", "color": (128, 128, 0)})
        return {
            "id": str(uuid.uuid4())[:8],
            "type": vtype,
            "label": info["label"],
            "severity": info["severity"],
            "track_id": t.track_id,
            "vehicle": t.label,
            "bbox": list(t.bbox),
            "confidence": round(t.confidence, 3),
            "timestamp": meta.get("ts", datetime.now().isoformat()),
            "frame_no": meta.get("frame_no", 0),
            "stream_id": meta.get("stream_id", "default"),
            "speed_kmh": 0.0,
            "lane_id": t.lane_id,
            "orig_w": meta.get("frame_width", 1280),
            "orig_h": meta.get("frame_height", 720),
            "evidence_file": "",
            "violation_count": t.violation_counts.get(vtype, 0) + 1,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME ANNOTATOR
# ══════════════════════════════════════════════════════════════════════════════

class FrameAnnotator:
    NORMAL_COLOR = (40, 200, 40)

    @staticmethod
    def annotate(
        frame: np.ndarray,
        road_mask: Optional[np.ndarray],
        tracks: List[TrackedObject],
        violations: List[Dict],
        phase: str,
        frame_no: int,
        fps: float,
        n_violations: int,
        rl_engine: RedLightViolationEngine,
        lane_det: LaneDetector,
        speed_est: SpeedEstimator,
    ) -> np.ndarray:
        try:
            out = frame.copy()

            if SHOW_ROAD_MASK and road_mask is not None:
                mask_bool = road_mask > 0
                if mask_bool.any():
                    out[mask_bool] = (out[mask_bool] * 0.85 + 255 * 0.15).astype(np.uint8)

            # Draw lanes and stop line
            out = lane_det.draw_lanes(out)
            out = rl_engine.draw_stop_line(out, phase)

            # Draw speed limit HUD
            out = speed_est.draw_hud_speed_limit(out)

            # Group violations by track
            viol_by_tid = {v["track_id"]: v for v in violations}
            track_violations: Dict[str, List[Dict]] = defaultdict(list)
            for v in violations:
                track_violations[v["track_id"]].append(v)

            for t in tracks:
                if t.lost_frames > 1:
                    continue
                x1, y1, x2, y2 = t.bbox
                tid = t.track_id
                
                # Get speed
                speed = speed_est.get_speed(tid)
                is_speeding = speed > speed_est.get_speed_limit()
                
                # Draw speed on frame
                if speed > 0:
                    out = speed_est.draw_speed_on_frame(out, tid, t.bbox, speed, is_speeding)
                else:
                    # Draw bounding box
                    color = FrameAnnotator.NORMAL_COLOR
                    thick = 1
                    text = f"{t.label} {t.confidence:.2f} [{tid}]"
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
                    cv2.putText(out, text, (x1 + 2, max(y1 - 4, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                # Draw multiple violation indicators
                if tid in track_violations:
                    viol_list = track_violations[tid]
                    y_offset = y1 - 25
                    for i, v in enumerate(viol_list[:3]):
                        viol_color = VIOLATION_META.get(v["type"], {}).get("color", (0, 0, 255))
                        viol_text = v["label"][:12]
                        cv2.putText(out, f"⚠ {viol_text}", (x1, y_offset - (i * 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, viol_color, 1, cv2.LINE_AA)

                # Draw trajectory
                traj = list(t.trajectory)
                n = len(traj)
                if n >= 2:
                    for i in range(max(0, n - 15) + 1, n):
                        cv2.circle(out, traj[i], 2, (80, 80, 255), -1)

            sig_col = _SIGNAL_BGR.get(phase, (180, 180, 180))
            hud = [
                (f"FPS:{fps:.0f}  F:{frame_no}", (220, 220, 0)),
                (f"Viols:{n_violations}", (0, 80, 255)),
                (f"Sig:{phase}", sig_col),
                (f"Limit:{speed_est.get_speed_limit():.0f}km/h", (0, 200, 100)),
                (f"Scale:{speed_est.get_scale():.4f}", (0, 150, 150)),
                (f"Multi-Viol: ENABLED", (100, 200, 100)),
            ]
            for i, (txt, col) in enumerate(hud):
                cv2.putText(out, txt, (8, 18 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
            return out
        except Exception as exc:
            log.error(f"[Annotator] Error: {exc}")
            return frame


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_shared_road_seg: Optional[RoadSegmentor] = None
_shared_detector: Optional[VehicleDetector] = None
_shared_helmet: Optional[HelmetDetector] = None
_singleton_lock = threading.Lock()


def _get_road_seg() -> RoadSegmentor:
    global _shared_road_seg
    if _shared_road_seg is None:
        with _singleton_lock:
            if _shared_road_seg is None:
                _shared_road_seg = RoadSegmentor()
    return _shared_road_seg


def _get_detector() -> VehicleDetector:
    global _shared_detector
    if _shared_detector is None:
        with _singleton_lock:
            if _shared_detector is None:
                _shared_detector = VehicleDetector()
    return _shared_detector


def _get_helmet() -> HelmetDetector:
    global _shared_helmet
    if _shared_helmet is None:
        with _singleton_lock:
            if _shared_helmet is None:
                _shared_helmet = HelmetDetector("best.pt")
    return _shared_helmet


# ══════════════════════════════════════════════════════════════════════════════
#  STREAM PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class StreamProcessor:

    def __init__(self, stream_id: str, source) -> None:
        self.stream_id = stream_id
        self.source = source
        self.running = False
        self.video_finished = False

        self._preproc_q: queue.Queue = queue.Queue(maxsize=Q_MAX_SIZE)
        self._detect_q: queue.Queue = queue.Queue(maxsize=Q_MAX_SIZE)

        self._buf = [None, None]
        self._write_idx = 0
        self._has_frame = False

        self.frame_no: int = 0
        self._fps_buf: deque = deque(maxlen=20)
        self._source_fps: float = 25.0

        self._road_seg = _get_road_seg()
        self._detector = _get_detector()
        self._tracker = ByteTracker()
        self._helmet = _get_helmet()
        self._speed_est = SpeedEstimator(SPEED_LIMIT_KMH, SCALE_MPP, FRAME_SKIP)
        self._lane_det = LaneDetector()
        self._rl_engine = RedLightViolationEngine()
        self._viol_eng = ViolationEngine(self._speed_est, self._lane_det, self._rl_engine)
        self._tracker.set_lane_det(self._lane_det)

        self._cap: Optional[cv2.VideoCapture] = None
        self._t_capture = self._t_detect = self._t_annotate = None

    def start(self) -> None:
        log.info(f"[{self.stream_id}] Opening: {self.source!r}")
        src = self.source

        cam_idx = None
        if isinstance(src, int):
            cam_idx = src
        elif isinstance(src, str) and src.strip().lstrip("-").isdigit():
            cam_idx = int(src.strip())

        if cam_idx is not None:
            self._cap = cv2.VideoCapture(cam_idx)
            if self._cap and self._cap.isOpened():
                try:
                    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self._cap.set(cv2.CAP_PROP_FPS, 30)
                except Exception:
                    pass
            else:
                raise RuntimeError(
                    f"Cannot open camera {cam_idx}. "
                    "Upload a video file or provide an RTSP/HTTP URL."
                )
        elif isinstance(src, str) and os.path.isfile(src):
            self._cap = cv2.VideoCapture(os.path.abspath(src))
        else:
            self._cap = cv2.VideoCapture(src)

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {src!r}")

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps > 0 else 25.0

        self.running = True
        self._t_capture = threading.Thread(target=self._capture_loop, daemon=True, name=f"Cap-{self.stream_id}")
        self._t_detect = threading.Thread(target=self._detect_loop, daemon=True, name=f"Det-{self.stream_id}")
        self._t_annotate = threading.Thread(target=self._annotate_loop, daemon=True, name=f"Ann-{self.stream_id}")
        self._t_capture.start()
        self._t_detect.start()
        self._t_annotate.start()
        log.info(f"[{self.stream_id}] All threads started.")

    def stop(self) -> None:
        self.running = False
        if self._cap:
            self._cap.release()
        for q in (self._preproc_q, self._detect_q):
            for _ in range(3):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
        log.info(f"[{self.stream_id}] stopped.")

    def _capture_loop(self) -> None:
        source_is_file = isinstance(self.source, str) and os.path.isfile(str(self.source))
        frame_interval = 1.0 / max(self._source_fps, 1.0) if source_is_file else 0.0
        last_read = time.monotonic()

        while self.running:
            if source_is_file and frame_interval > 0:
                now = time.monotonic()
                wait = frame_interval - (now - last_read)
                if wait > 0.002:
                    time.sleep(wait)
            last_read = time.monotonic()

            ret, raw = self._cap.read()
            if not ret:
                if not self.video_finished:
                    self.video_finished = True
                    if self._cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        self.running = False
                        break
                time.sleep(0.05)
                continue

            frame = _resize_720p(raw)
            if self._preproc_q.full():
                try:
                    self._preproc_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._preproc_q.put_nowait(frame)
            except queue.Full:
                pass

        try:
            self._preproc_q.put_nowait(None)
        except Exception:
            pass

    def _detect_loop(self) -> None:
        while self.running:
            item = self._preproc_q.get()
            if item is None:
                break
            while not self._preproc_q.empty():
                try:
                    newer = self._preproc_q.get_nowait()
                    if newer is None:
                        item = None
                        break
                    item = newer
                except queue.Empty:
                    break
            if item is None:
                break

            t0 = time.perf_counter()
            frame = item
            h, w = frame.shape[:2]
            self.frame_no += 1
            fn = self.frame_no

            signal_ctrl.tick(frame_stats["fps"] or 25.0)
            phase = signal_ctrl.phase
            fps = frame_stats["fps"] or 25.0
            ts = datetime.now().isoformat()

            tracks: List[TrackedObject] = []
            all_violations: List[Dict] = []
            road_mask = None

            try:
                road_mask, _ = self._road_seg.get_both(frame)
                self._lane_det.update_lanes(frame, road_mask)
                self._rl_engine.update_stop_line(frame, road_mask)

                detections = self._detector.detect(frame, road_mask)

                # Helmet detection
                frame, helmet_violations = self._helmet.draw_violations(
                    frame, stream_id=self.stream_id, frame_no=fn
                )

                tracks = self._tracker.update(detections, (h, w))

                # Update speed for all tracks
                for track in tracks:
                    if track.lost_frames == 0:
                        speed = self._speed_est.update(track.track_id, track.centroid, fps)
                        if speed > 1:
                            log.debug(f"[SPEED] Track {track.track_id}: {speed:.1f} km/h")

                meta = {
                    "ts": ts,
                    "frame_no": fn,
                    "stream_id": self.stream_id,
                    "frame_width": w,
                    "frame_height": h,
                }

                # Vehicle violations
                vehicle_violations = self._viol_eng.process(tracks, phase, meta, fps)

                # Combine all violations
                all_violations = helmet_violations + vehicle_violations

                # Save evidence
                evidence_frame = frame.copy()
                for v in all_violations:
                    v["orig_w"] = w
                    v["orig_h"] = h
                    violation_log.append(v)
                    get_db_writer().enqueue(v, evidence_frame)
                    frame_stats["violations_today"] += 1
                    log.warning(
                        f"[{self.stream_id}] VIOLATION [{v['label']}] "
                        f"track={v['track_id']} sev={v['severity']} "
                        f"conf={v.get('confidence',0):.2f} "
                        f"count={v.get('violation_count', 1)}"
                    )

            except Exception as exc:
                log.error(f"[{self.stream_id}] Detect error: {exc}")

            payload = {
                "frame": frame,
                "road_mask": road_mask,
                "tracks": tracks,
                "violations": all_violations,
                "phase": phase,
                "frame_no": fn,
                "fps": fps,
                "n_violations": len(violation_log),
                "rl_engine": self._rl_engine,
                "lane_det": self._lane_det,
                "speed_est": self._speed_est,
            }
            if self._detect_q.full():
                try:
                    self._detect_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._detect_q.put_nowait(payload)
            except queue.Full:
                pass

            elapsed = time.perf_counter() - t0
            self._fps_buf.append(1.0 / max(elapsed, 1e-6))
            if len(self._fps_buf) >= 3:
                frame_stats["fps"] = round(float(np.mean(self._fps_buf)), 1)
            frame_stats["total_vehicles"] = len(tracks)

        try:
            self._detect_q.put_nowait(None)
        except Exception:
            pass

    def _annotate_loop(self) -> None:
        while self.running:
            item = self._detect_q.get()
            if item is None:
                break
            while not self._detect_q.empty():
                try:
                    newer = self._detect_q.get_nowait()
                    if newer is None:
                        item = None
                        break
                    item = newer
                except queue.Empty:
                    break
            if item is None:
                break
            try:
                annotated = FrameAnnotator.annotate(
                    frame=item["frame"],
                    road_mask=item.get("road_mask"),
                    tracks=item.get("tracks", []),
                    violations=item.get("violations", []),
                    phase=item.get("phase", "GREEN"),
                    frame_no=item.get("frame_no", 0),
                    fps=item.get("fps", 0.0),
                    n_violations=item.get("n_violations", 0),
                    rl_engine=item["rl_engine"],
                    lane_det=item["lane_det"],
                    speed_est=item["speed_est"],
                )
            except Exception as exc:
                log.error(f"[{self.stream_id}] Annotate error: {exc}")
                annotated = item.get("frame")
                if annotated is None:
                    continue

            write_slot = 1 - self._write_idx
            self._buf[write_slot] = annotated
            self._write_idx = write_slot
            self._has_frame = True

    def generate_mjpeg(self):
        target_interval = 1.0 / STREAM_FPS
        enc_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        errors = 0
        last_sent = time.monotonic()

        while self.running or self._has_frame:
            now = time.monotonic()
            wait = target_interval - (now - last_sent)
            if wait > 0.001:
                time.sleep(wait)
            last_sent = time.monotonic()

            if not self._has_frame:
                continue
            frame = self._buf[self._write_idx]
            if frame is None:
                continue

            try:
                if TURBO_AVAILABLE:
                    buf = _turbo.encode(frame, quality=JPEG_QUALITY)
                else:
                    ok, buf_arr = cv2.imencode(".jpg", frame, enc_params)
                    if not ok:
                        continue
                    buf = buf_arr.tobytes()
            except Exception:
                errors += 1
                if errors > 20:
                    break
                time.sleep(0.01)
                continue

            try:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + (buf if isinstance(buf, bytes) else buf.tobytes())
                    + b"\r\n"
                )
                errors = 0
            except GeneratorExit:
                break
            except Exception:
                errors += 1
                if errors > 20:
                    break


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _resize_720p(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= INTERNAL_WIDTH and h <= INTERNAL_HEIGHT:
        return frame
    scale = min(INTERNAL_WIDTH / w, INTERNAL_HEIGHT / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════════════════════════
#  CSV / VEHICLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_csv_cache: Dict[str, Dict] = {}
_csv_loaded: bool = False
_csv_lock = threading.Lock()


def _load_vehicle_csv(path: str = VEHICLE_CSV) -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in _csv_module.DictReader(f):
                vn = row.get("vehicle_number", "").strip().upper()
                if vn:
                    data[vn] = {
                        "owner_name": row.get("owner_name", ""),
                        "address": row.get("address", ""),
                        "email": row.get("email", ""),
                        "phone": row.get("phone", ""),
                        "old_challans": int(row.get("old_challans", "0") or 0),
                    }
    except Exception as exc:
        log.error(f"[CSV] Load error: {exc}")
    return data


def get_csv_cache() -> Dict[str, Dict]:
    global _csv_cache, _csv_loaded
    with _csv_lock:
        if not _csv_loaded:
            _csv_cache = _load_vehicle_csv()
            _csv_loaded = True
        return _csv_cache


def _upsert_vehicle(vehicle_number: str, data: Dict) -> None:
    try:
        with _db_conn() as c:
            c.execute("""
                INSERT INTO vehicles (vehicle_number, owner_name, address, email, phone)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(vehicle_number) DO UPDATE SET
                    owner_name = CASE WHEN excluded.owner_name != '' THEN excluded.owner_name ELSE owner_name END,
                    address    = CASE WHEN excluded.address    != '' THEN excluded.address    ELSE address    END,
                    email      = CASE WHEN excluded.email      != '' THEN excluded.email      ELSE email      END,
                    phone      = CASE WHEN excluded.phone      != '' THEN excluded.phone      ELSE phone      END,
                    updated_at = datetime('now')
            """, (vehicle_number, data.get("owner_name", ""),
                  data.get("address", ""), data.get("email", ""), data.get("phone", "")))
    except Exception as exc:
        log.error(f"[Vehicle] upsert error: {exc}")


def _check_and_apply_blacklist(vehicle_number: str) -> bool:
    try:
        with _db_conn() as c:
            row = c.execute(
                "SELECT active_challans FROM vehicles WHERE vehicle_number=?", (vehicle_number,)
            ).fetchone()
            if row and row["active_challans"] >= BLACKLIST_THR:
                c.execute(
                    "UPDATE vehicles SET blacklisted=1, updated_at=datetime('now') WHERE vehicle_number=?",
                    (vehicle_number,)
                )
                return True
    except Exception as exc:
        log.error(f"[Blacklist] error: {exc}")
    return False


def _send_challan_email(
    to_email, vehicle_number, violation_type, fine, challan_id,
    timestamp, owner_name="", evidence_path=None
) -> bool:
    if not EMAIL_ENABLED:
        log.info(f"[Email] Email disabled. Would send to {to_email} for {challan_id}")
        return True
    
    if not SMTP_USER or not SMTP_PASS:
        log.warning("[Email] SMTP credentials missing.")
        return False
    
    if not to_email:
        log.warning(f"[Email] No email address for {vehicle_number}")
        return False
    
    try:
        msg = _MIMEMultipart("mixed")
        msg["Subject"] = f"Traffic Challan Notice – {challan_id}"
        msg["From"] = SMTP_FROM
        msg["To"] = to_email

        vtype_label = violation_type.replace("_", " ").title()
        greeting = f"Dear {owner_name}," if owner_name else "Dear Vehicle Owner,"
        badge_color = {
            "no_helmet": "#e67e22",
            "red_light": "#c0392b",
            "over_speed": "#c0392b",
            "wrong_direction": "#2980b9",
            "lane_crossing": "#f39c12",
        }.get(violation_type, "#7f8c8d")

        html = f"""<!DOCTYPE html>
<html><body style="font-family:Arial;background:#f4f6f9;padding:20px;margin:0">
<div style="max-width:620px;margin:auto;background:#fff;border-radius:14px;
            overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.13)">
  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:30px;text-align:center">
    <div style="font-size:32px">🚔</div>
    <h1 style="color:#fff;margin:8px 0 4px;font-size:22px;letter-spacing:1px">
      TRAFFIC CHALLAN NOTICE</h1>
    <p style="color:#a0aec0;margin:0;font-size:12px">TrafficSentinel Automated Enforcement</p>
  </div>
  <div style="padding:30px">
    <p style="color:#2d3748;font-size:15px;margin-bottom:20px">{greeting}</p>
    <div style="background:#f8f9fa;border-radius:10px;overflow:hidden;margin-bottom:20px">
      <div style="background:{badge_color};color:#fff;padding:10px 16px;font-weight:700;
                  font-size:13px;letter-spacing:0.5px">{vtype_label.upper()} VIOLATION</div>
      <table style="width:100%;border-collapse:collapse">
             <tr><td style="padding:12px 16px;color:#718096;font-size:13px;border-bottom:1px solid #e2e8f0">Challan ID</td>
               <td style="padding:12px 16px;font-family:monospace;font-weight:700;border-bottom:1px solid #e2e8f0">{challan_id}</td></tr>
             <tr><td style="padding:12px 16px;color:#718096;font-size:13px;border-bottom:1px solid #e2e8f0">Vehicle No.</td>
               <td style="padding:12px 16px;font-weight:700;border-bottom:1px solid #e2e8f0">{vehicle_number}</td></tr>
             <tr><td style="padding:12px 16px;color:#718096;font-size:13px;border-bottom:1px solid #e2e8f0">Violation</td>
               <td style="padding:12px 16px;color:{badge_color};font-weight:700;border-bottom:1px solid #e2e8f0">{vtype_label}</td></tr>
             <tr><td style="padding:12px 16px;color:#718096;font-size:13px;border-bottom:1px solid #e2e8f0">Fine Amount</td>
               <td style="padding:12px 16px;color:#c0392b;font-size:20px;font-weight:700;border-bottom:1px solid #e2e8f0">₹{fine:.2f}</td></tr>
             <tr><td style="padding:12px 16px;color:#718096;font-size:13px">Date &amp; Time</td>
               <td style="padding:12px 16px">{timestamp[:19].replace('T',' ')}</td></tr>
       </table>
    </div>
    <div style="background:#fff5f5;border-left:4px solid {badge_color};
                padding:14px 16px;border-radius:6px;margin-bottom:16px">
      <p style="margin:0;color:#c0392b;font-size:13px">
        ⚠ <strong>Please pay within 30 days</strong> to avoid additional penalties.
        Evidence image is attached for your reference.
      </p>
    </div>
    <p style="color:#a0aec0;font-size:11px;text-align:center;margin:0">
      This is an automated notice from TrafficSentinel. Do not reply to this email.
    </p>
  </div>
</div></body></html>"""

        msg.attach(_MIMEText(html, "html"))

        if evidence_path and os.path.exists(evidence_path):
            with open(evidence_path, "rb") as ef:
                part = _MIMEBase("application", "octet-stream")
                part.set_payload(ef.read())
            _encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            f'attachment; filename="evidence_{challan_id}.jpg"')
            msg.attach(part)

        with _smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)

        log.info(f"[Email] Sent {challan_id} → {to_email}")
        return True
    except Exception as exc:
        log.error(f"[Email] Failed: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/streams/start", methods=["POST"])
def start_stream():
    data = request.get_json(force=True)
    source = data.get("source", 0)
    sid = data.get("stream_id", f"stream_{len(processors)+1}")
    if sid in processors:
        return jsonify({"error": "Stream already running"}), 400
    try:
        p = StreamProcessor(sid, source)
        p.start()
        processors[sid] = p
        frame_stats["active_streams"] = len(processors)
        return jsonify({"stream_id": sid, "status": "started"})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/streams/<sid>/stop", methods=["POST"])
def stop_stream(sid: str):
    p = processors.pop(sid, None)
    if p is None:
        return jsonify({"error": "Stream not found"}), 404
    p.stop()
    frame_stats["active_streams"] = len(processors)
    return jsonify({"status": "stopped"})


@app.route("/api/streams/<sid>/feed")
def video_feed(sid: str):
    p = processors.get(sid)
    if p is None:
        return jsonify({"error": "Stream not found"}), 404
    return Response(p.generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/streams")
def list_streams():
    return jsonify({"streams": list(processors.keys())})


@app.route("/api/signal/status")
def signal_status():
    return jsonify(signal_ctrl.status())


@app.route("/api/signal/set", methods=["POST"])
def signal_set():
    data = request.get_json(force=True)
    try:
        signal_ctrl.set_phase(data.get("phase", "GREEN"))
        return jsonify(signal_ctrl.status())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/signal/config", methods=["POST"])
def signal_config():
    data = request.get_json(force=True)
    mode = data.get("mode", "AUTO").upper()
    try:
        if mode == "AUTO":
            signal_ctrl.set_auto(
                float(data.get("green", 30)),
                float(data.get("yellow", 5)),
                float(data.get("red", 20)),
            )
        else:
            signal_ctrl.set_phase(data.get("phase", "GREEN"))
        return jsonify(signal_ctrl.status())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/violations")
def get_violations():
    limit = int(request.args.get("limit", 50))
    vtype = request.args.get("type")
    severity = request.args.get("severity")
    data = violation_log
    if vtype:
        data = [v for v in data if v["type"] == vtype]
    if severity:
        data = [v for v in data if v["severity"] == severity]
    return jsonify({"violations": data[-limit:][::-1], "total": len(data)})


@app.route("/api/history")
def get_history():
    limit = int(request.args.get("limit", 50))
    vtype = request.args.get("type")
    severity = request.args.get("severity")
    data = db_get_history(limit=limit, vtype=vtype, severity=severity)
    return jsonify({"violations": data, "total": len(data),
                    "filters": {"type": vtype, "severity": severity}})


@app.route("/api/history/stats")
def history_stats():
    return jsonify(db_get_stats())


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    ok = db_clear()
    violation_log.clear()
    return jsonify({"status": "success" if ok else "error"})


@app.route("/api/stats")
def get_stats():
    tc: Dict[str, int] = defaultdict(int)
    sc: Dict[str, int] = defaultdict(int)
    for v in violation_log:
        tc[v["type"]] += 1
        sc[v["severity"]] += 1
    return jsonify({
        **frame_stats,
        "violation_type_counts": dict(tc),
        "severity_counts": dict(sc),
        "total_violations": len(violation_log),
        "database_stats": db_get_stats(),
        "signal": signal_ctrl.status(),
    })


@app.route("/api/speed/config", methods=["POST"])
def speed_config():
    data = request.get_json(force=True)
    for sid, p in processors.items():
        if hasattr(p, "_speed_est"):
            if "limit_kmh" in data:
                p._speed_est.set_limit(float(data["limit_kmh"]))
                log.warning(f"[API] Speed limit set to {data['limit_kmh']} km/h for {sid}")
            if "scale_mpp" in data:
                p._speed_est.set_scale(float(data["scale_mpp"]))
                log.warning(f"[API] Scale set to {data['scale_mpp']:.6f} m/px for {sid}")
    return jsonify({
        "status": "updated",
        "limit_kmh": data.get("limit_kmh", SPEED_LIMIT_KMH),
        "scale_mpp": data.get("scale_mpp", SCALE_MPP)
    })


@app.route("/api/speed/status")
def speed_status():
    if processors:
        first_processor = next(iter(processors.values()))
        if hasattr(first_processor, "_speed_est"):
            return jsonify({
                "speed_limit": first_processor._speed_est.get_speed_limit(),
                "scale_mpp": first_processor._speed_est.get_scale(),
                "active_tracks": len(first_processor._speed_est.get_all_speeds()),
                "speeds": first_processor._speed_est.get_all_speeds()
            })
    return jsonify({"status": "no_active_streams"})


@app.route("/api/evidence/<path:fname>")
def serve_evidence(fname: str):
    try:
        return send_from_directory(EVIDENCE_FOLDER, os.path.basename(fname))
    except Exception:
        return jsonify({"error": "Not found"}), 404


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    safe_name = os.path.basename(f.filename)
    path = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], safe_name))
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    f.save(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return jsonify({"error": "File save failed"}), 500
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        os.remove(path)
        return jsonify({"error": "Not a valid video file"}), 400
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    sid = f"upload_{uuid.uuid4().hex[:6]}"
    try:
        p = StreamProcessor(sid, path)
        p.start()
        processors[sid] = p
        frame_stats["active_streams"] = len(processors)
        return jsonify({
            "stream_id": sid, "status": "processing", "file": safe_name,
            "video_info": {"fps": round(fps, 2), "frames": int(fc),
                           "width": int(vw), "height": int(vh)},
        })
    except RuntimeError as exc:
        try:
            os.remove(path)
        except Exception:
            pass
        return jsonify({"error": str(exc)}), 500


@app.route("/api/violations/export")
def export_violations():
    limit = int(request.args.get("limit", 1000))
    data = db_get_history(limit=limit)
    r = app.response_class(
        response=json.dumps(data, indent=2), status=200, mimetype="application/json",
    )
    r.headers["Content-Disposition"] = (
        f"attachment; filename=violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    return r


@app.route("/api/helmet/status")
def helmet_status():
    return jsonify(_get_helmet().status())


@app.route("/api/helmet/upload", methods=["POST"])
def helmet_upload():
    f = request.files.get("model")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400
    save_path = "best.pt"
    try:
        f.save(save_path)
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        global _shared_helmet
        _shared_helmet = HelmetDetector(save_path)
        return jsonify({"ok": True, "path": save_path, "size_mb": round(size_mb, 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vehicle/lookup")
def vehicle_lookup():
    vn = request.args.get("vehicle_number", "").strip().upper()
    if not vn:
        return jsonify({"error": "vehicle_number required"}), 400
    csv_data = get_csv_cache().get(vn, {})
    try:
        with _db_conn() as c:
            row = c.execute("SELECT * FROM vehicles WHERE vehicle_number=?", (vn,)).fetchone()
    except Exception:
        row = None
    result = dict(row) if row else {
        "vehicle_number": vn, **csv_data,
        "total_challans": 0, "active_challans": 0, "blacklisted": 0,
    }
    result["csv_data"] = csv_data
    return jsonify(result)


@app.route("/api/vehicle/search")
def vehicle_search():
    q = request.args.get("q", "").strip().upper()
    if not q:
        return jsonify({"vehicles": []}), 200
    try:
        with _db_conn() as c:
            rows = c.execute(
                "SELECT * FROM vehicles WHERE vehicle_number LIKE ? LIMIT 20", (f"%{q}%",)
            ).fetchall()
        return jsonify({"vehicles": [dict(r) for r in rows]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/vehicle/<vehicle_number>")
def get_vehicle(vehicle_number: str):
    vn = vehicle_number.strip().upper()
    try:
        with _db_conn() as c:
            vrow = c.execute("SELECT * FROM vehicles WHERE vehicle_number=?", (vn,)).fetchone()
            challans = c.execute(
                "SELECT * FROM challans WHERE vehicle_number=? ORDER BY timestamp DESC", (vn,)
            ).fetchall()
            viols = c.execute(
                "SELECT * FROM violations WHERE vehicle_number=? ORDER BY timestamp DESC LIMIT 30", (vn,)
            ).fetchall()
        return jsonify({
            "vehicle": dict(vrow) if vrow else None,
            "challans": [dict(r) for r in challans],
            "violations": [dict(r) for r in viols],
            "csv_data": get_csv_cache().get(vn, {}),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/challan/create", methods=["POST"])
def create_challan():
    data = request.get_json(force=True)
    vehicle_number = data.get("vehicle_number", "").strip().upper()
    violation_type = data.get("violation_type", "")
    fine_amount = float(data.get("fine_amount", FINE_TABLE.get(violation_type, 500.0)))
    violation_id = data.get("violation_id", "")
    notes = data.get("notes", "")

    if not vehicle_number:
        return jsonify({"error": "vehicle_number required"}), 400

    csv_info = get_csv_cache().get(vehicle_number, {})
    veh_data = {
        "owner_name": data.get("owner_name") or csv_info.get("owner_name", ""),
        "address": data.get("address") or csv_info.get("address", ""),
        "email": data.get("email") or csv_info.get("email", ""),
        "phone": data.get("phone") or csv_info.get("phone", ""),
    }
    _upsert_vehicle(vehicle_number, veh_data)

    challan_id = f"CH{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:4].upper()}"
    ts = datetime.now().isoformat()

    try:
        with _db_conn() as c:
            c.execute("""
                INSERT INTO challans
                  (challan_id, vehicle_number, violation_id, violation_type,
                   fine_amount, status, notes, timestamp)
                VALUES (?, ?, ?, ?, ?, 'unpaid', ?, datetime('now'))
            """, (challan_id, vehicle_number, violation_id, violation_type, fine_amount, notes))

            c.execute("""
                UPDATE vehicles SET
                    total_challans  = total_challans  + 1,
                    active_challans = active_challans + 1,
                    updated_at      = datetime('now')
                WHERE vehicle_number = ?
            """, (vehicle_number,))

            if violation_id:
                c.execute(
                    "UPDATE violations SET challan_id=?, vehicle_number=? WHERE id=?",
                    (challan_id, vehicle_number, violation_id),
                )
    except Exception as exc:
        return jsonify({"error": f"DB error: {exc}"}), 500

    blacklisted = _check_and_apply_blacklist(vehicle_number)

    evidence_path = None
    if violation_id:
        try:
            with _db_conn() as c:
                vrow = c.execute(
                    "SELECT evidence_file FROM violations WHERE id=?", (violation_id,)
                ).fetchone()
            if vrow and vrow["evidence_file"]:
                evidence_path = os.path.join(EVIDENCE_FOLDER, vrow["evidence_file"])
                if not os.path.exists(evidence_path):
                    evidence_path = None
        except Exception:
            pass

    to_email = veh_data.get("email", "")
    email_status = "no_email"

    if to_email:
        sent = _send_challan_email(
            to_email=to_email,
            vehicle_number=vehicle_number,
            violation_type=violation_type,
            fine=fine_amount,
            challan_id=challan_id,
            timestamp=ts,
            owner_name=veh_data.get("owner_name", ""),
            evidence_path=evidence_path,
        )
        email_status = "sent" if sent else "failed"
        if sent:
            try:
                with _db_conn() as c:
                    c.execute("UPDATE challans SET email_sent=1 WHERE challan_id=?", (challan_id,))
            except Exception:
                pass
    else:
        log.warning(f"[Challan] No email for {vehicle_number} — skipping email for {challan_id}")

    return jsonify({
        "status": "ok",
        "challan_id": challan_id,
        "vehicle_number": vehicle_number,
        "violation_type": violation_type,
        "fine": fine_amount,
        "challan_status": "unpaid",
        "blacklisted": blacklisted,
        "email_status": email_status,
        "evidence_path": evidence_path,
        "timestamp": ts,
    })


@app.route("/api/challan/list")
def list_challans():
    limit = int(request.args.get("limit", 50))
    status = request.args.get("status", "")
    vn = request.args.get("vehicle_number", "").strip().upper()
    vtype = request.args.get("violation_type", "")
    params: list = []
    conds: list = []
    if status:
        conds.append("c.status=?")
        params.append(status)
    if vn:
        conds.append("c.vehicle_number=?")
        params.append(vn)
    if vtype:
        conds.append("c.violation_type=?")
        params.append(vtype)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    params.append(limit)
    try:
        with _db_conn() as c:
            rows = c.execute(f"""
                SELECT c.*, v.owner_name, v.email
                FROM challans c LEFT JOIN vehicles v ON c.vehicle_number=v.vehicle_number
                {where} ORDER BY c.timestamp DESC LIMIT ?
            """, params).fetchall()
        return jsonify({"challans": [dict(r) for r in rows], "total": len(rows)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/challan/<challan_id>")
def get_challan(challan_id: str):
    try:
        with _db_conn() as c:
            row = c.execute("""
                SELECT c.*, v.owner_name, v.address, v.email, v.phone
                FROM challans c LEFT JOIN vehicles v ON c.vehicle_number=v.vehicle_number
                WHERE c.challan_id=?
            """, (challan_id,)).fetchone()
        if not row:
            return jsonify({"error": "Challan not found"}), 404
        return jsonify(dict(row))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/challan/<challan_id>/pay", methods=["POST"])
def pay_challan(challan_id: str):
    try:
        with _db_conn() as c:
            row = c.execute("SELECT * FROM challans WHERE challan_id=?", (challan_id,)).fetchone()
            if not row:
                return jsonify({"error": "Challan not found"}), 404
            if row["status"] == "paid":
                return jsonify({"error": "Already paid"}), 400
            c.execute("UPDATE challans SET status='paid', paid_at=datetime('now') WHERE challan_id=?",
                      (challan_id,))
            c.execute("""
                UPDATE vehicles SET
                    active_challans = MAX(0, active_challans - 1),
                    updated_at      = datetime('now')
                WHERE vehicle_number = ?
            """, (row["vehicle_number"],))
        return jsonify({"status": "paid", "challan_id": challan_id,
                        "vehicle_number": row["vehicle_number"]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/challan/<challan_id>/cancel", methods=["POST"])
def cancel_challan(challan_id: str):
    try:
        with _db_conn() as c:
            row = c.execute("SELECT * FROM challans WHERE challan_id=?", (challan_id,)).fetchone()
            if not row:
                return jsonify({"error": "Challan not found"}), 404
            c.execute("UPDATE challans SET status='cancelled' WHERE challan_id=?", (challan_id,))
            if row["status"] == "unpaid":
                c.execute("""
                    UPDATE vehicles SET active_challans=MAX(0,active_challans-1),
                    updated_at=datetime('now') WHERE vehicle_number=?
                """, (row["vehicle_number"],))
        return jsonify({"status": "cancelled", "challan_id": challan_id})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/blacklist")
def get_blacklist():
    try:
        with _db_conn() as c:
            rows = c.execute(
                "SELECT * FROM vehicles WHERE blacklisted=1 ORDER BY active_challans DESC"
            ).fetchall()
        return jsonify({"vehicles": [dict(r) for r in rows], "total": len(rows)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/fines")
def get_fine_table():
    return jsonify(FINE_TABLE)


@app.route("/api/email/send", methods=["POST"])
def manual_send_email():
    data = request.get_json(force=True)
    evidence_file = data.get("evidence_file", "").strip()
    evidence_path = (
        os.path.join(EVIDENCE_FOLDER, os.path.basename(evidence_file))
        if evidence_file else None
    )
    ok = _send_challan_email(
        data.get("to_email", ""),
        data.get("vehicle_number", ""),
        data.get("violation_type", ""),
        float(data.get("fine_amount", 0)),
        data.get("challan_id", ""),
        data.get("timestamp", datetime.now().isoformat()),
        data.get("owner_name", ""),
        evidence_path=evidence_path,
    )
    return jsonify({"sent": ok, "evidence_path": evidence_path})


@app.route("/api/csv/reload", methods=["POST"])
def reload_csv():
    global _csv_loaded
    with _csv_lock:
        _csv_loaded = False
    data = get_csv_cache()
    return jsonify({"status": "reloaded", "count": len(data)})


@app.route("/api/debug/road_mask", methods=["POST"])
def toggle_road_mask():
    global SHOW_ROAD_MASK
    data = request.get_json(force=True)
    SHOW_ROAD_MASK = bool(data.get("enabled", SHOW_ROAD_MASK))
    return jsonify({"show_road_mask": SHOW_ROAD_MASK})


@app.route("/api/history/maintain", methods=["POST"])
def maintain_db():
    try:
        c = sqlite3.connect(DB_PATH)
        c.execute("VACUUM")
        c.close()
        return jsonify({"status": "success", "message": "VACUUM completed"})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/webcam/frame", methods=["POST"])
def webcam_frame():
    global _wc_frame
    data = request.get_data()
    if not data:
        return jsonify({"error": "no data"}), 400
    with _wc_lock:
        _wc_frame = data
    return jsonify({"ok": True})


@app.route("/api/webcam/start", methods=["POST"])
def webcam_start():
    sid = _wc_sid
    if sid in processors:
        return jsonify({"stream_id": sid, "status": "already_running"})

    class _BrowserCapture:
        def __init__(self):
            self._open = True

        def isOpened(self):
            return self._open

        def read(self):
            for _ in range(40):
                with _wc_lock:
                    f = _wc_frame
                if f is not None:
                    break
                time.sleep(0.05)
            with _wc_lock:
                f = _wc_frame
            if f is None:
                return False, None
            arr = np.frombuffer(f, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return (False, None) if img is None else (True, img)

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 15.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return -1
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            return 0

        def set(self, *_):
            return True

        def release(self):
            self._open = False

    p = StreamProcessor(sid, sid)
    p._cap = _BrowserCapture()
    p.running = True
    p._source_fps = 15.0
    p._t_capture = threading.Thread(target=p._capture_loop, daemon=True, name=f"Cap-{sid}")
    p._t_detect = threading.Thread(target=p._detect_loop, daemon=True, name=f"Det-{sid}")
    p._t_annotate = threading.Thread(target=p._annotate_loop, daemon=True, name=f"Ann-{sid}")
    p._t_capture.start()
    p._t_detect.start()
    p._t_annotate.start()
    processors[sid] = p
    frame_stats["active_streams"] = len(processors)
    return jsonify({"stream_id": sid, "status": "started"})


@app.route("/api/webcam/stop", methods=["POST"])
def webcam_stop():
    global _wc_frame
    p = processors.pop(_wc_sid, None)
    if p:
        p.stop()
    with _wc_lock:
        _wc_frame = None
    frame_stats["active_streams"] = len(processors)
    return jsonify({"status": "stopped"})


@app.route("/api/test/email", methods=["POST"])
def test_email():
    """Test endpoint to verify email configuration"""
    if not EMAIL_ENABLED:
        return jsonify({"error": "Email is disabled", "status": "disabled"}), 400
    
    if not SMTP_USER or not SMTP_PASS:
        return jsonify({"error": "SMTP credentials not configured", "status": "unconfigured"}), 400
    
    to_email = request.json.get("to_email", TEST_EMAIL) if request.json else TEST_EMAIL
    
    try:
        msg = _MIMEMultipart()
        msg["Subject"] = "TrafficSentinel - Test Email"
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        
        html = """
        <html>
        <body>
            <h2>TrafficSentinel Test Email</h2>
            <p>This is a test email from your TrafficSentinel system.</p>
            <p>Your email configuration is working correctly!</p>
            <hr>
            <small>TrafficSentinel Automated System</small>
        </body>
        </html>
        """
        msg.attach(_MIMEText(html, "html"))
        
        with _smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        
        return jsonify({
            "status": "success",
            "message": f"Test email sent to {to_email}",
            "config": {
                "smtp_host": SMTP_HOST,
                "smtp_port": SMTP_PORT,
                "from_email": SMTP_FROM,
                "email_enabled": EMAIL_ENABLED
            }
        })
    except Exception as exc:
        return jsonify({
            "status": "failed",
            "error": str(exc),
            "message": "Failed to send test email. Check your credentials and network."
        }), 500


@app.route("/api/health")
def health():
    helmet = _get_helmet()
    speed_status = {}
    if processors:
        first_processor = next(iter(processors.values()))
        if hasattr(first_processor, "_speed_est"):
            speed_status = {
                "speed_limit": first_processor._speed_est.get_speed_limit(),
                "scale_mpp": first_processor._speed_est.get_scale(),
                "active_tracks": len(first_processor._speed_est.get_all_speeds())
            }
    
    return jsonify({
        "status": "healthy",
        "version": "v10-modular-multiple-violations",
        "timestamp": datetime.now().isoformat(),
        "active_streams": len(processors),
        "total_violations": len(violation_log),
        "yolo_vehicle": YOLO_AVAILABLE,
        "helmet": helmet.status(),
        "mask2former": M2F_AVAILABLE,
        "turbojpeg": TURBO_AVAILABLE,
        "gpu": TORCH_AVAILABLE and (torch.cuda.is_available() if TORCH_AVAILABLE else False),
        "smtp_configured": bool(SMTP_USER and SMTP_PASS),
        "email_enabled": EMAIL_ENABLED,
        "fine_table": FINE_TABLE,
        "speed_detection": speed_status,
        "multiple_violations": True,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

_wc_frame: Optional[bytes] = None
_wc_lock = threading.Lock()
_wc_sid = "browser_webcam"

if __name__ == "__main__":
    _init_db()

    log.info("=" * 72)
    log.info("  TrafficSentinel v10  — MULTIPLE VIOLATIONS PER VEHICLE ENABLED")
    log.info("=" * 72)
    log.info(f"  YOLO vehicle  : {YOLO_AVAILABLE}  ({YOLO_MODEL_NAME})")
    log.info(f"  Speed limit   : {SPEED_LIMIT_KMH} km/h")
    log.info(f"  Scale factor  : {SCALE_MPP:.6f} m/px")
    log.info(f"  Frame skip    : {FRAME_SKIP}")
    log.info(f"  DB            : {DB_PATH}")
    log.info(f"  Fine (helmet) : ₹{FINE_TABLE['no_helmet']:.0f}")
    log.info(f"  Email         : {'ENABLED' if EMAIL_ENABLED else 'DISABLED'}")
    if SMTP_USER and SMTP_PASS:
        log.info(f"  SMTP          : Configured ({SMTP_USER})")
        log.info(f"  Test Endpoint : POST /api/test/email")
    else:
        log.warning(f"  SMTP          : NOT configured — set credentials in code")
    log.info(f"  Listening     : http://0.0.0.0:5000")
    log.info("=" * 72)
    log.info("  Multiple Violations Features:")
    log.info("    • Same vehicle can have overspeed, wrong direction, lane crossing")
    log.info("    • Per-violation-type cooldown (not global)")
    log.info("    • Violation counts tracked per vehicle")
    log.info("    • Separate evidence for each violation")
    log.info("=" * 72)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(EVIDENCE_FOLDER, exist_ok=True)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
