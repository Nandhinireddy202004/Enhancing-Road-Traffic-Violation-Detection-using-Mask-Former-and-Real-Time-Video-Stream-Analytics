"""
red_light_jump.py — TrafficSentinel Red Light Violation Module
══════════════════════════════════════════════════════════════════
- Detects stop line via Hough transform + temporal smoothing
- Fires violation ONLY when signal is RED AND vehicle crosses line AND is moving
- Per-track state machine: FREE → WATCHING → VIOLATED
- Cooldown to prevent duplicate events
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("TrafficSentinel.RedLight")

# ── Constants ─────────────────────────────────────────────────────────────────
STOP_BAND_PX      = 30        # zone around stop line (px)
STOP_SPEED_PX     = 3.5       # velocity_px threshold: above = "moving"
MIN_MOVE_FRAMES   = 3         # frames of motion before triggering
COOLDOWN_FRAMES   = 300       # frames between repeated violations per track

# Hough params for stop-line detection
SL_HOUGH_THRESH   = 80
SL_MIN_LEN_FRAC   = 0.40      # fraction of frame width
SL_MAX_GAP        = 20
SL_ANGLE_MAX_DEG  = 10.0      # lines must be near-horizontal
SL_WIDTH_MIN_FRAC = 0.50
SL_CLUSTER_BAND   = 20
SL_XWALK_PROX     = 55
SL_FIT_BAND       = 15
TEMPORAL_BUF      = 10
TEMPORAL_GATE_PX  = 40

# Signal colours for display
_SIGNAL_BGR: Dict[str, Tuple[int, int, int]] = {
    "GREEN":  (30,  200,  30),
    "YELLOW": (0,   200, 220),
    "RED":    (0,     0, 220),
}


# ── Stop Line ─────────────────────────────────────────────────────────────────

class StopLine:
    __slots__ = ("x1", "y1", "x2", "y2", "valid")

    def __init__(self, x1: int, y1: int, x2: int, y2: int, valid: bool = True) -> None:
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.valid = valid

    @property
    def y_mid(self) -> int:
        return (self.y1 + self.y2) // 2

    def y_at(self, x: int, frame_w: int) -> int:
        span = self.x2 - self.x1
        if span == 0:
            return self.y_mid
        t = max(0.0, min(1.0, float(x - self.x1) / float(span)))
        return int(round(self.y1 + t * (self.y2 - self.y1)))

    def draw(self, frame: np.ndarray, phase: str) -> np.ndarray:
        color = _SIGNAL_BGR.get(phase, (0, 200, 255))
        cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), color, 2, cv2.LINE_AA)
        # Draw stop zone band
        poly = np.array([
            [self.x1, self.y1 - STOP_BAND_PX],
            [self.x2, self.y2 - STOP_BAND_PX],
            [self.x2, self.y2 + STOP_BAND_PX],
            [self.x1, self.y1 + STOP_BAND_PX],
        ], dtype=np.int32)
        cv2.polylines(frame, [poly.reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)
        return frame


class StopLineDetector:
    """
    Detects the stop/white line closest to traffic using Hough transforms
    with temporal smoothing to avoid jitter.
    """

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=TEMPORAL_BUF)
        self._lock = threading.Lock()

    def get(self, frame: np.ndarray, road_mask: np.ndarray) -> Optional[StopLine]:
        with self._lock:
            candidate = self._detect(frame, road_mask)
            if candidate is not None:
                mid_y   = candidate.y_mid
                gate_ok = True
                if self._history:
                    prev_mids = [(e[1] + e[3]) // 2 for e in self._history]
                    gate_ok   = abs(mid_y - int(np.median(prev_mids))) <= TEMPORAL_GATE_PX
                if gate_ok:
                    self._history.append(
                        (candidate.x1, candidate.y1, candidate.x2, candidate.y2)
                    )

            if not self._history:
                return self._fallback(frame)

            arr = np.array(self._history, dtype=np.float32)
            return StopLine(
                int(np.median(arr[:, 0])),
                int(np.median(arr[:, 1])),
                int(np.median(arr[:, 2])),
                int(np.median(arr[:, 3])),
                valid=(candidate is not None),
            )

    # ── Detection pipeline ─────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray, road_mask: np.ndarray) -> Optional[StopLine]:
        H, W = frame.shape[:2]
        markings = self._extract_markings(frame, road_mask)
        if markings is None or markings.sum() == 0:
            return None

        min_len = int(W * SL_MIN_LEN_FRAC)
        raw     = cv2.HoughLinesP(
            markings, 1, np.pi / 180.0, SL_HOUGH_THRESH,
            minLineLength=min_len, maxLineGap=SL_MAX_GAP,
        )
        if raw is None:
            return None

        segs = [tuple(r[0]) for r in raw]
        segs = [s for s in segs if _angle_ok(s)]
        segs = [s for s in segs if _seg_len(s) >= W * SL_WIDTH_MIN_FRAC]
        if not segs:
            return None

        segs = _filter_position(segs, road_mask, H)
        if not segs:
            return None

        winner = _select_best(segs)
        if winner is None:
            return None

        return _fit_line(winner, markings, H, W)

    @staticmethod
    def _extract_markings(frame: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_blur = cv2.GaussianBlur(hsv[:, :, 2], (7, 7), 0)
        bright = cv2.adaptiveThreshold(
            v_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -10
        )
        k_rd   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bright = cv2.bitwise_and(bright, cv2.dilate(road_mask, k_rd))
        k_h    = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        return cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_h)

    @staticmethod
    def _fallback(frame: np.ndarray) -> StopLine:
        h, w = frame.shape[:2]
        fy   = int(h * 0.60)
        return StopLine(0, fy, w, fy, valid=False)


# ── Per-track violation state ─────────────────────────────────────────────────

class _RLState:
    __slots__ = ("phase", "move_count", "crossed", "last_viol_frame")

    def __init__(self) -> None:
        self.phase           = "FREE"   # FREE | WATCHING | VIOLATED
        self.move_count      = 0
        self.crossed         = False
        self.last_viol_frame = -COOLDOWN_FRAMES


# ── Red Light Violation Engine ────────────────────────────────────────────────

class RedLightViolationEngine:
    """
    State-machine based red light violation detector.

    Call process() once per frame for each tracked vehicle.
    """

    def __init__(self) -> None:
        self._states:   Dict[str, _RLState] = defaultdict(_RLState)
        self._stop_det: StopLineDetector    = StopLineDetector()
        self._stop_line: Optional[StopLine] = None

    # ── Public ─────────────────────────────────────────────────────────────────

    def update_stop_line(self, frame: np.ndarray, road_mask: np.ndarray) -> None:
        """Refresh stop-line detection (call once per frame)."""
        self._stop_line = self._stop_det.get(frame, road_mask)

    def get_stop_line(self) -> Optional[StopLine]:
        return self._stop_line

    def process(
        self,
        track_id:     str,
        bbox:         List[int],           # [x1, y1, x2, y2]
        velocity_px:  float,
        signal_phase: str,
        frame_no:     int,
        frame_w:      int,
    ) -> Optional[Dict]:
        """
        Check one vehicle for red-light violation.
        Returns violation dict or None.
        """
        if self._stop_line is None:
            return None

        st = self._states[track_id]
        x1, y1, x2, y2 = bbox
        cx      = (x1 + x2) // 2
        stop_y  = self._stop_line.y_at(cx, frame_w)
        at_line = y2 >= stop_y - STOP_BAND_PX

        # Reset if phase is not RED
        if signal_phase != "RED":
            st.phase    = "FREE"
            st.move_count = 0
            st.crossed  = False
            return None

        # Enter WATCHING state when RED starts
        if st.phase == "FREE":
            st.phase    = "WATCHING"
            st.move_count = 0
            st.crossed  = at_line

        if st.phase == "VIOLATED":
            return None  # already fired for this red phase

        # Track crossing
        if at_line and not st.crossed:
            st.crossed = True

        # Accumulate movement frames while past line
        if st.crossed and velocity_px > STOP_SPEED_PX:
            st.move_count += 1
        else:
            st.move_count = max(0, st.move_count - 1)

        # Cooldown check
        if (frame_no - st.last_viol_frame) < COOLDOWN_FRAMES:
            return None

        # Fire violation
        if st.move_count >= MIN_MOVE_FRAMES:
            st.phase           = "VIOLATED"
            st.last_viol_frame = frame_no
            return {
                "type":         "red_light",
                "label":        "Red Light Jump",
                "severity":     "HIGH",
                "track_id":     track_id,
                "signal":       "RED",
                "stop_line_y":  self._stop_line.y_mid,
                "stop_y_at_cx": stop_y,
                "velocity_px":  round(velocity_px, 1),
                "move_frames":  st.move_count,
            }

        return None

    def reset_phase(self, track_id: str) -> None:
        """Call when a track is lost/removed."""
        self._states.pop(track_id, None)

    def draw_stop_line(self, frame: np.ndarray, phase: str) -> np.ndarray:
        if self._stop_line is not None:
            self._stop_line.draw(frame, phase)
        return frame


# ── Signal Controller ─────────────────────────────────────────────────────────

class SignalController:
    CYCLE = ("GREEN", "YELLOW", "RED")

    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self.mode      = "AUTO"
        self.phase     = "GREEN"
        self.durations = {"GREEN": 30.0, "YELLOW": 5.0, "RED": 20.0}
        self._fps      = 25.0
        self._tick     = 0

    def tick(self, fps: float = 25.0) -> None:
        with self._lock:
            self._fps = max(fps, 1.0)
            if self.mode != "AUTO":
                return
            limit      = int(self.durations[self.phase] * self._fps)
            self._tick += 1
            if self._tick >= limit:
                self._tick = 0
                idx        = self.CYCLE.index(self.phase)
                self.phase = self.CYCLE[(idx + 1) % 3]

    def set_phase(self, phase: str) -> None:
        phase = phase.upper()
        if phase not in self.CYCLE:
            raise ValueError(f"Unknown phase: {phase!r}")
        with self._lock:
            self.mode, self.phase, self._tick = "MANUAL", phase, 0

    def set_auto(self, green: float, yellow: float, red: float) -> None:
        with self._lock:
            self.durations = {
                "GREEN":  max(float(green),  1.0),
                "YELLOW": max(float(yellow), 1.0),
                "RED":    max(float(red),    1.0),
            }
            self.mode, self._tick = "AUTO", 0

    def status(self) -> Dict:
        with self._lock:
            limit     = int(self.durations[self.phase] * self._fps)
            remaining = max(0.0, (limit - self._tick) / self._fps)
            return {
                "mode":       self.mode,
                "phase":      self.phase,
                "durations":  dict(self.durations),
                "remaining_s": round(remaining, 1),
            }


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _angle_ok(s: tuple) -> bool:
    x1, y1, x2, y2 = s
    dx = x2 - x1
    if dx == 0:
        return False
    return abs(np.degrees(np.arctan2(abs(y2 - y1), abs(dx)))) <= SL_ANGLE_MAX_DEG


def _seg_len(s: tuple) -> float:
    return float(np.hypot(s[2] - s[0], s[3] - s[1]))


def _filter_position(segs: List, road_mask: np.ndarray, H: int) -> List:
    rows  = np.where(road_mask.any(axis=1))[0]
    r_top = int(rows[0]) if rows.size > 0 else int(H * 0.35)
    r_bot = int(rows[-1]) if rows.size > 0 else int(H * 0.85)
    lower = r_top + int((r_bot - r_top) * 0.35)
    return [s for s in segs if lower <= (s[1] + s[3]) / 2 <= r_bot]


def _select_best(segs: List) -> Optional[tuple]:
    if not segs:
        return None
    cy_segs  = sorted(segs, key=lambda s: (s[1] + s[3]) / 2.0)
    clusters: List[Tuple[float, list]] = []
    anchor   = (cy_segs[0][1] + cy_segs[0][3]) / 2.0
    cur      = [cy_segs[0]]
    for seg in cy_segs[1:]:
        cy = (seg[1] + seg[3]) / 2.0
        if cy - anchor <= SL_CLUSTER_BAND:
            cur.append(seg)
        else:
            clusters.append((float(np.median([(s[1] + s[3]) / 2.0 for s in cur])), cur))
            anchor, cur = cy, [seg]
    clusters.append((float(np.median([(s[1] + s[3]) / 2.0 for s in cur])), cur))

    meds = [c[0] for c in clusters]
    n    = len(clusters)
    xwalk: set = set()
    for i in range(n):
        nbrs = sum(
            1 for j in range(n) if j != i and abs(meds[j] - meds[i]) <= SL_XWALK_PROX
        )
        if nbrs >= 1:
            xwalk.add(i)

    non_xwalk = [(m, cs) for idx, (m, cs) in enumerate(clusters) if idx not in xwalk]
    if non_xwalk:
        _, best = max(non_xwalk, key=lambda t: t[0])
    else:
        _, best = min(clusters, key=lambda t: t[0])
    return max(best, key=_seg_len)


def _fit_line(winner: tuple, markings: np.ndarray, H: int, W: int) -> StopLine:
    cy   = (winner[1] + winner[3]) / 2.0
    y_lo = max(0, int(cy) - SL_FIT_BAND)
    y_hi = min(H - 1, int(cy) + SL_FIT_BAND)
    pts: List[Tuple[float, float]] = []
    for ry in range(y_lo, y_hi + 1):
        for x in np.where(markings[ry, :] > 0)[0]:
            pts.append((float(x), float(ry)))

    if len(pts) < 10:
        x1, y1, x2, y2 = winner
        return StopLine(
            max(0, min(W - 1, x1)), max(0, min(H - 1, y1)),
            max(0, min(W - 1, x2)), max(0, min(H - 1, y2)),
        )

    arr            = np.array(pts, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    if abs(vx) < 1e-6:
        x1, y1, x2, y2 = winner
        return StopLine(
            max(0, min(W - 1, x1)), max(0, min(H - 1, y1)),
            max(0, min(W - 1, x2)), max(0, min(H - 1, y2)),
        )

    xs = arr[:, 0]
    xl = int(np.percentile(xs, 2))
    xr = int(np.percentile(xs, 98))
    yl = int(round(y0 + vy * (xl - x0) / vx))
    yr = int(round(y0 + vy * (xr - x0) / vx))
    return StopLine(
        max(0, min(W - 1, xl)), max(0, min(H - 1, yl)),
        max(0, min(W - 1, xr)), max(0, min(H - 1, yr)),
    )