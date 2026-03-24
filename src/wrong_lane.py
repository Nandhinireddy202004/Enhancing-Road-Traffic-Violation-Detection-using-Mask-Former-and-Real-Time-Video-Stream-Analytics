"""
wrong_lane.py — TrafficSentinel Advanced Wrong Lane Detection (v8 — Production Fixed)
══════════════════════════════════════════════════════════════════════

ADVANCES OVER v6
─────────────────
1. MOVEMENT VECTOR ANGLE  (handles all camera orientations)
   v6 only checked cy displacement → blind to forward/side-facing cameras.
   v7 computes the full 2D movement vector (dx, dy) and maps it to a
   compass sector: DOWN / UP / RIGHT / LEFT.
   Works correctly for overhead, forward-facing, and angled cameras.

2. PER-LANE DIRECTION LEARNING  (two-way road support)
   Learns the expected direction for each lane independently.
   Example: left lane vehicles go DOWN, right lane vehicles go UP.
   A vehicle going DOWN in the right lane = wrong lane even though DOWN
   is the "majority" direction globally.
   Falls back to global majority when per-lane data is insufficient.

3. VELOCITY-WEIGHTED MAJORITY VOTE
   Slow/stationary vehicles are noise — they shouldn't influence majority.
   Each vehicle's vote is weighted by its movement magnitude.
   A vehicle moving 150px contributes weight=1.0; 20px → weight=0.13.

4. TEMPORAL CONSISTENCY SCORE (replaces consecutive-frame counter)
   v6: wrong_count resets to 0 on any single correct-direction frame.
   v7: rolling ratio window — wrong_frames / total_frames in last N.
   Fires when ratio > WRONG_RATIO_THRESHOLD for at least WRONG_MIN_SAMPLES.
   Much more robust to momentary jitter and tracking noise.

5. EMA TRAJECTORY SMOOTHING
   Exponential Moving Average applied to raw centroids before direction
   analysis. Removes YOLO bounding box jitter (5-15px noise) without
   introducing the lag of a simple moving average.
   α=0.35 → fast response; α=0.15 → more stable.

6. ADAPTIVE MINIMUM VEHICLES
   When only 1 vehicle is on screen, MIN_VEHICLES_FOR_VOTE=2 blocks all
   detection. v7 uses adaptive logic:
     • ≥3 vehicles: use global majority (reliable)
     • 2 vehicles:  use global majority
     • 1 vehicle:   use per-lane expected direction if learned, else skip
   This allows detection of lone wrong-way drivers without false positives.

7. REAL-POSITION DEDUPLICATION (FRAME_SKIP immune)
   Filters consecutive identical centroids before storing in pos_buf.
   Ensures pos_buf contains only real YOLO detections, not cached repeats.
   Direction analysis is exact regardless of FRAME_SKIP value.

8. MEANINGFUL CONFIDENCE SCORE
   conf = wrong_ratio in consistency window (0.0 – 1.0).
   0.6 = borderline (just crossed threshold)
   0.85+ = highly confident wrong-way driver
   Used for evidence capture priority and challan confidence.

FIXES IN v8 (production validation)
─────────────────────────────────────
F1. RAW positions in pos_buf: EMA lag was understating displacement by 1.6×
    causing slow vehicles to appear stationary. EMA kept only for lane assignment.

F2. MAJORITY HYSTERESIS: majority_sector now requires 3 consecutive frames
    of agreement before committing. Prevents single-frame weight fluctuations
    from flipping majority mid-accumulation and corrupting consistency buffers.

F3. TWO-WAY ROAD PER-LANE SUPPORT: _expected_for_track now uses per-lane
    direction even when global majority is set, if per-lane direction differs.
    Enables correct detection on bidirectional roads (left lane ↓, right lane ↑).

F4. EMA-SMOOTHED LANE ASSIGNMENT: assign_lane_smooth() uses ema_cx (not raw cx)
    for lane boundary lookups. Prevents cx jitter near lane boundaries from
    alternating lane IDs and contaminating per-lane direction learner.

F5. NO CONSISTENCY BUF CLEAR ON FIRE: consistency_buf is kept after violation.
    Next detection of same vehicle is fast (ratio already high). Only cooldown
    governs re-fire timing.

F6. TUNED CONSTANTS:
    MIN_REAL_POSITIONS: 6→5, MIN_DISPLACEMENT_PX: 12→10, SECTOR_DEG: 55→65
    WRONG_MIN_SAMPLES: 8→6, LANE_DIR_MIN_VOTES: 5→3, LANE_DIR_CERTAINTY: 0.65→0.60
    VIOLATION_COOLDOWN: 180→120

F7. MINIMUM CONFIDENCE GATE: violation only fires when conf > 0.60 (explicit check).

F8. COMPREHENSIVE DEBUG LOGGING:
    - update_track: raw/ema position, buf_len, displacement
    - tick_scene_direction: majority, candidate, hold_count, sector_weights
    - _check_wrong_direction: actual_dir, expected_dir, wrong_ratio, lane_id
    - violation fire: full context for post-mortem analysis

CALL ORDER (every frame):
  1. lane_det.update_track(tid, centroid)   — all active tracks
  2. lane_det.tick_scene_direction()        — once per frame
  3. lane_det.check_violation(tid, ...)    — each track after tick
"""
from __future__ import annotations

import logging
import math
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("TrafficSentinel.WrongLane")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NUM_LANES               = 3      # fallback equal-width lane count

# ── Position buffer ───────────────────────────────────────────────────────────
POS_WINDOW              = 24     # real (non-duplicate) positions kept per track
MIN_REAL_POSITIONS      = 5      # minimum real positions before direction analysis

# ── EMA smoothing for centroid (jitter reduction) ─────────────────────────────
EMA_ALPHA               = 0.30   # 0.0=max smooth, 1.0=no smooth

# ── Movement vector direction ─────────────────────────────────────────────────
MIN_DISPLACEMENT_PX     = 10     # pixels — minimum movement to be considered moving
SECTOR_DEG              = 65     # degrees — sector half-width

# ── Majority vote ─────────────────────────────────────────────────────────────
MIN_VEHICLES_GLOBAL     = 2      # vehicles needed for global majority vote
MIN_VEHICLE_WEIGHT      = 0.15   # minimum weight to participate in majority

# ── Per-lane direction learning ───────────────────────────────────────────────
LANE_DIR_HISTORY        = 60     # frames of directional votes kept per lane
LANE_DIR_MIN_VOTES      = 3      # minimum votes needed before learning
LANE_DIR_CERTAINTY      = 0.60   # certainty threshold for locking direction

# ── Temporal consistency ──────────────────────────────────────────────────────
CONSISTENCY_WINDOW      = 20     # rolling window of direction-check results
WRONG_MIN_SAMPLES       = 6      # minimum samples needed for violation
WRONG_RATIO_THRESHOLD   = 0.60   # wrong_frames/total_frames → violation

# ── Violation cooldown ────────────────────────────────────────────────────────
VIOLATION_COOLDOWN      = 120    # frames (~4.8s @25fps) between repeat violations

# ── Lane crossing detection ───────────────────────────────────────────────────
LANE_SMOOTH_WIN         = 4
LANE_CROSS_LIMIT        = 5
LANE_CROSS_WINDOW       = 45
LANE_CROSS_MIN_PX       = 14

# ── Hough lane-line detection ─────────────────────────────────────────────────
HOUGH_THRESH            = 50
HOUGH_MIN_LEN_FRAC      = 0.10
HOUGH_MAX_GAP           = 35
ANGLE_MIN_DEG           = 15
ANGLE_MAX_DEG           = 85
LANE_UPDATE_INTERVAL    = 20
LANE_CLUSTER_BW_FRAC    = 0.07


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECTION SECTOR  (handles all camera orientations)
# ══════════════════════════════════════════════════════════════════════════════

# Sectors: DOWN(+1), UP(-1), RIGHT(+2), LEFT(-2), UNKNOWN(0)
_SECTOR_AXES = (
    (+1,   90.0),   # DOWN  — y increases
    (-1,  270.0),   # UP    — y decreases
    (+2,    0.0),   # RIGHT — x increases
    (-2,  180.0),   # LEFT  — x decreases
)


def _classify_vector(dx: float, dy: float) -> int:
    """
    Classify a 2D movement vector into the NEAREST direction sector.
    Returns: +1=DOWN, -1=UP, +2=RIGHT, -2=LEFT, 0=UNKNOWN
    """
    mag = math.hypot(dx, dy)
    if mag < 1e-3:
        return 0

    angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0

    # Pick sector whose axis centre is angularly CLOSEST to this vector
    best_sector, best_dist = 0, 360.0
    for sector, centre in _SECTOR_AXES:
        dist = abs((angle_deg - centre + 180.0) % 360.0 - 180.0)
        if dist < best_dist:
            best_dist = dist
            best_sector = sector

    return best_sector


def _opposite(sector: int) -> int:
    """Return the opposite direction sector."""
    return -sector  # DOWN↔UP, RIGHT↔LEFT


def _sector_label(sector: int) -> str:
    return {1: "DOWN", -1: "UP", 2: "RIGHT", -2: "LEFT", 0: "UNKNOWN"}[sector]


# ══════════════════════════════════════════════════════════════════════════════
#  PER-TRACK STATE
# ══════════════════════════════════════════════════════════════════════════════

class _Track:
    """All per-track state for one vehicle."""
    
    __slots__ = (
        "pos_buf", "prev_raw", "ema_cx", "ema_cy",
        "direction", "weight", "consistency_buf", "cooldown_until",
        "lane_id", "lane_buf", "lane_hist", "cross_events", "prev_cx_lane"
    )

    def __init__(self) -> None:
        self.pos_buf: deque = deque(maxlen=POS_WINDOW)
        self.prev_raw: Optional[Tuple[int, int]] = None
        self.ema_cx: float = 0.0
        self.ema_cy: float = 0.0
        self.direction: int = 0
        self.weight: float = 0.0
        self.consistency_buf: deque = deque(maxlen=CONSISTENCY_WINDOW)
        self.cooldown_until: int = 0
        self.lane_id: int = -1
        self.lane_buf: deque = deque(maxlen=LANE_SMOOTH_WIN)
        self.lane_hist: deque = deque(maxlen=LANE_CROSS_WINDOW)
        self.cross_events: deque = deque(maxlen=LANE_CROSS_WINDOW)
        self.prev_cx_lane: Optional[int] = None

    def push(self, cx: int, cy: int) -> bool:
        """Record new centroid. Returns True if this was a real (non-duplicate) position."""
        # Update EMA for stable lane assignment
        if self.prev_raw is None:
            self.ema_cx = float(cx)
            self.ema_cy = float(cy)
        else:
            self.ema_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * self.ema_cx
            self.ema_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * self.ema_cy

        # Deduplication: skip if raw centroid unchanged (cached YOLO frame)
        if self.prev_raw == (cx, cy):
            return False
        
        self.prev_raw = (cx, cy)
        self.pos_buf.append((float(cx), float(cy)))
        return True

    def compute_direction_and_weight(self) -> Tuple[int, float]:
        """Compute movement sector and weight from position buffer."""
        if len(self.pos_buf) < MIN_REAL_POSITIONS:
            return 0, 0.0

        # Total displacement vector: first → last position
        x0, y0 = self.pos_buf[0]
        xn, yn = self.pos_buf[-1]
        dx = xn - x0
        dy = yn - y0
        mag = math.hypot(dx, dy)

        if mag < MIN_DISPLACEMENT_PX:
            return 0, 0.0

        sector = _classify_vector(dx, dy)
        # Weight proportional to displacement (normalized to ~0-1)
        weight = min(mag / 150.0, 1.0)
        return sector, weight

    def wrong_ratio(self) -> float:
        """Fraction of consistency_buf samples that are 'wrong'."""
        buf = list(self.consistency_buf)
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    def record_consistency(self, is_wrong: bool) -> None:
        """Record whether this frame was wrong direction."""
        self.consistency_buf.append(1 if is_wrong else 0)

    def smooth_lane(self, raw_lane: int) -> int:
        """Smooth lane assignment with mode filter."""
        self.lane_buf.append(raw_lane)
        counts: Dict[int, int] = {}
        for v in self.lane_buf:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def record_crossing(self, smooth_lane: int, raw_cx: int) -> bool:
        """Record lane crossing event."""
        prev_lane = self.lane_hist[-1] if self.lane_hist else None
        prev_cx = self.prev_cx_lane
        crossed = (
            prev_lane is not None
            and smooth_lane != prev_lane
            and prev_cx is not None
            and abs(raw_cx - prev_cx) >= LANE_CROSS_MIN_PX
        )
        self.cross_events.append(1 if crossed else 0)
        self.lane_hist.append(smooth_lane)
        self.prev_cx_lane = raw_cx
        return crossed

    def crossing_count(self) -> int:
        """Get number of lane crossings in window."""
        return int(sum(self.cross_events))

    def reset_crossings(self) -> None:
        """Reset crossing counter."""
        self.cross_events.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  PER-LANE DIRECTION LEARNER
# ══════════════════════════════════════════════════════════════════════════════

class _LaneDirectionLearner:
    """Learns expected traffic direction for each lane independently."""
    
    MIN_UNIQUE = 2  # Minimum unique vehicles needed before per-lane direction is trusted

    def __init__(self) -> None:
        self._votes: Dict[int, deque] = {}   # lane → deque of (sector, weight)
        self._unique: Dict[int, set] = {}    # lane → set of vehicle IDs that voted
        self._expected: Dict[int, int] = {}  # locked direction per lane

    def vote(self, lane_id: int, sector: int, weight: float, vehicle_id: str = "") -> None:
        """Record a direction vote for a lane."""
        if lane_id < 0 or sector == 0 or weight < MIN_VEHICLE_WEIGHT:
            return
            
        if lane_id not in self._votes:
            self._votes[lane_id] = deque(maxlen=LANE_DIR_HISTORY)
            self._unique[lane_id] = set()
            
        self._votes[lane_id].append((sector, weight))
        if vehicle_id:
            self._unique[lane_id].add(vehicle_id)
        self._recompute(lane_id)

    def expected_direction(self, lane_id: int) -> int:
        """Return expected direction for lane, or 0 if not yet reliably learned."""
        return self._expected.get(lane_id, 0)

    def unique_count(self, lane_id: int) -> int:
        """Number of unique vehicles that have voted for this lane."""
        return len(self._unique.get(lane_id, set()))

    def _recompute(self, lane_id: int) -> None:
        """Recompute expected direction for a lane."""
        votes = list(self._votes.get(lane_id, []))
        
        if len(votes) < LANE_DIR_MIN_VOTES:
            return
            
        if self.unique_count(lane_id) < self.MIN_UNIQUE:
            return
            
        tally: Dict[int, float] = {}
        for sector, weight in votes:
            tally[sector] = tally.get(sector, 0.0) + weight
            
        total = sum(tally.values())
        if total <= 0:
            return
            
        best_sector = max(tally, key=lambda s: tally[s])
        best_frac = tally[best_sector] / total
        
        if best_frac >= LANE_DIR_CERTAINTY:
            prev = self._expected.get(lane_id, 0)
            if prev != best_sector:
                log.info(
                    f"[LaneDir] Lane {lane_id} direction locked → "
                    f"{_sector_label(best_sector)} "
                    f"({best_frac:.0%} of {len(votes)} votes, "
                    f"{self.unique_count(lane_id)} unique vehicles)"
                )
                self._expected[lane_id] = best_sector

    def reset(self) -> None:
        """Reset all learned data."""
        self._votes.clear()
        self._unique.clear()
        self._expected.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LANE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """Advanced wrong-lane violation detector."""
    
    def __init__(self, num_lanes: int = NUM_LANES) -> None:
        self._num_lanes = num_lanes
        self._lock = threading.Lock()
        self._tracks: Dict[str, _Track] = {}
        self._lane_dir = _LaneDirectionLearner()
        self._lane_lines: List[Dict] = []
        self._frame_hw: Tuple[int, int] = (720, 1280)
        self._frames_since_hough: int = LANE_UPDATE_INTERVAL
        
        # Global majority state
        self._majority_sector: int = 0
        self._total_weight: float = 0.0
        self._sector_weights: Dict[int, float] = {}
        self._active_count: int = 0
        
        # Majority hysteresis
        self._majority_candidate: int = 0
        self._majority_hold_count: int = 0
        self._majority_hysteresis_frames = 3

    # ══════════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def update_lanes(self, frame: np.ndarray, road_mask: np.ndarray) -> None:
        """Refresh Hough lane lines periodically."""
        with self._lock:
            self._frame_hw = frame.shape[:2]
            self._frames_since_hough += 1
            if self._frames_since_hough >= LANE_UPDATE_INTERVAL:
                self._frames_since_hough = 0
                self._run_hough(frame, road_mask)

    def update_track(self, track_id: str, centroid: Tuple[int, int]) -> None:
        """STEP 1 — Record centroid for this track."""
        if track_id not in self._tracks:
            self._tracks[track_id] = _Track()

        trk = self._tracks[track_id]
        cx, cy = centroid
        is_real = trk.push(cx, cy)

        if is_real and len(trk.pos_buf) >= 2:
            x0, y0 = trk.pos_buf[0]
            xn, yn = trk.pos_buf[-1]
            disp = math.hypot(xn - x0, yn - y0)
        else:
            disp = 0.0
            
        log.debug(
            f"[Lane:update] track={track_id} "
            f"raw=({cx},{cy}) ema=({trk.ema_cx:.0f},{trk.ema_cy:.0f}) "
            f"real_update={is_real} buf_len={len(trk.pos_buf)} "
            f"displacement={disp:.0f}px"
        )

    def tick_scene_direction(self) -> None:
        """STEP 2 — Compute all directions, build weighted majority."""
        sector_weights: Dict[int, float] = {}
        active = 0

        for trk in self._tracks.values():
            sector, weight = trk.compute_direction_and_weight()
            trk.direction = sector
            trk.weight = weight

            if sector != 0 and weight >= MIN_VEHICLE_WEIGHT:
                sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
                active += 1

            # Feed vote into per-lane learner
            if sector != 0 and trk.lane_id >= 0:
                tid_key = str(id(trk))
                self._lane_dir.vote(trk.lane_id, sector, weight, tid_key)

        self._sector_weights = sector_weights
        self._total_weight = sum(sector_weights.values())
        self._active_count = active

        # Weighted majority with hysteresis
        if sector_weights:
            best = max(sector_weights, key=lambda s: sector_weights[s])
            best_w = sector_weights[best]
            new_candidate = best if (self._total_weight > 0 and 
                                     best_w / self._total_weight >= 0.55) else 0
        else:
            new_candidate = 0

        # Hysteresis: accumulate hold count for candidate
        if new_candidate == self._majority_candidate and new_candidate != 0:
            self._majority_hold_count = min(
                self._majority_hold_count + 1, self._majority_hysteresis_frames + 1
            )
        else:
            self._majority_candidate = new_candidate
            self._majority_hold_count = 1

        # Commit candidate after holding long enough
        if self._majority_hold_count >= self._majority_hysteresis_frames:
            committed = self._majority_candidate
        else:
            committed = self._majority_sector

        if committed != self._majority_sector:
            log.info(
                f"[Lane:tick] Majority COMMITTED → {_sector_label(committed)} "
                f"(held {self._majority_hold_count} frames) "
                f"weights={sector_weights} n={active}"
            )
        self._majority_sector = committed

        log.debug(
            f"[Lane:tick] majority={_sector_label(self._majority_sector)} "
            f"candidate={_sector_label(self._majority_candidate)}({self._majority_hold_count}/{3}) "
            f"sector_weights={ {_sector_label(k): round(v,2) for k,v in sector_weights.items()} } "
            f"active_vehicles={active}"
        )

    def check_violation(
        self,
        track_id: str,
        centroid: Tuple[int, int],
        bbox: Optional[List[int]] = None,
        frame_no: int = 0,
    ) -> Optional[Dict]:
        """STEP 3 — Check if this track is violating."""
        trk = self._tracks.get(track_id)
        if trk is None:
            return None

        # Cooldown guard
        if frame_no > 0 and frame_no < trk.cooldown_until:
            return None

        # Update lane assignment using EMA-smoothed position
        cx, cy = centroid
        raw_lane = self._assign_lane_cx(int(trk.ema_cx), cy)
        smooth_lane = trk.smooth_lane(raw_lane)
        trk.lane_id = smooth_lane

        # Check wrong direction
        wd = self._check_wrong_direction(track_id, trk, frame_no, centroid, bbox)
        if wd is not None:
            return wd

        # Check frequent lane crossing
        trk.record_crossing(smooth_lane, cx)
        if trk.crossing_count() >= LANE_CROSS_LIMIT:
            conf = min(trk.crossing_count() / LANE_CROSS_LIMIT, 1.0)
            trk.reset_crossings()
            trk.cooldown_until = frame_no + VIOLATION_COOLDOWN
            log.info(
                f"[Lane] ⚠ LANE CROSSING track={track_id} "
                f"crosses={trk.crossing_count()} conf={conf:.2f} lane={smooth_lane}"
            )
            return {
                "type": "lane_crossing",
                "confidence": round(conf, 3),
                "bbox": bbox,
                "direction": _sector_label(self._majority_sector),
                "lane_id": smooth_lane,
                "track_id": track_id,
            }

        return None

    def assign_lane(self, centroid: Tuple[int, int]) -> int:
        """Assign lane using raw centroid."""
        return self._assign_lane_cx(centroid[0], centroid[1])

    def _assign_lane_cx(self, cx: int, cy: int) -> int:
        """Core lane lookup."""
        with self._lock:
            h, w = self._frame_hw
            if self._lane_lines:
                xs = sorted(self._line_x_at(cy, h))
                bounds = [0.0] + [x / max(w, 1) for x in xs] + [1.0]
                xn = cx / max(w, 1)
                for i in range(len(bounds) - 1):
                    if bounds[i] <= xn < bounds[i + 1]:
                        return i
                return len(bounds) - 2
            lw = w / self._num_lanes
            return min(int(cx / lw), self._num_lanes - 1)

    def draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Draw lane lines and direction indicators."""
        with self._lock:
            h, w = self._frame_hw
            if not self._lane_lines:
                for i in range(1, self._num_lanes):
                    x = int(i * w / self._num_lanes)
                    cv2.line(frame, (x, int(h * 0.30)), (x, h),
                            (255, 200, 0), 1, cv2.LINE_AA)
                    expected = self._lane_dir.expected_direction(i - 1)
                    if expected != 0:
                        lbl = _sector_label(expected)
                        cv2.putText(frame, lbl,
                                   (int((i - 0.5) * w / self._num_lanes) - 15, int(h * 0.32)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1, cv2.LINE_AA)
            else:
                for ln in self._lane_lines:
                    cv2.line(frame,
                            (int(ln["x_top"]), 0),
                            (int(ln["x_bot"]), h),
                            (255, 200, 0), 1, cv2.LINE_AA)

        # Global direction HUD
        fy = frame.shape[0] - 10
        maj = self._majority_sector
        if maj != 0 and self._total_weight > 0:
            label = f"FLOW: {_sector_label(maj)} {self._sector_weights.get(maj, 0)/self._total_weight:.0%} n={self._active_count}"
            color = (0, 220, 30)
        else:
            label = f"FLOW: learning ({self._active_count} vehicles)"
            color = (120, 120, 120)
        cv2.putText(frame, label, (8, fy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        return frame

    def get_lane_lines(self) -> List[Dict]:
        """Get detected lane lines."""
        with self._lock:
            return list(self._lane_lines)

    def remove_track(self, track_id: str) -> None:
        """Remove a track from memory."""
        self._tracks.pop(track_id, None)

    def scene_stats(self) -> Dict:
        """Return current scene statistics."""
        return {
            "majority_sector": self._majority_sector,
            "majority_label": _sector_label(self._majority_sector),
            "sector_weights": dict(self._sector_weights),
            "total_weight": round(self._total_weight, 2),
            "active_count": self._active_count,
            "lane_directions": dict(self._lane_dir._expected),
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  INTERNAL METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def _check_wrong_direction(
        self,
        track_id: str,
        trk: _Track,
        frame_no: int,
        centroid: Tuple[int, int],
        bbox: Optional[List[int]],
    ) -> Optional[Dict]:
        """Check if vehicle is going wrong direction."""
        this_dir = trk.direction

        # Fast path: vehicle agrees with global majority
        if this_dir != 0 and self._majority_sector != 0 and this_dir == self._majority_sector:
            lane_exp = self._lane_dir.expected_direction(trk.lane_id)
            lane_unique = self._lane_dir.unique_count(trk.lane_id)
            trk_key = str(id(trk))
            this_taught = trk_key in self._lane_dir._unique.get(trk.lane_id, set())
            
            unique_others = lane_unique - (1 if this_taught else 0)
            per_lane_trusted = (lane_exp != 0 and unique_others >= _LaneDirectionLearner.MIN_UNIQUE)
            per_lane_opposes = per_lane_trusted and (lane_exp == _opposite(this_dir))

            if not per_lane_opposes:
                trk.record_consistency(False)
                return None

        # Determine expected direction
        expected = self._expected_for_track(trk)

        if expected == 0:
            trk.record_consistency(False)
            return None

        if this_dir == 0:
            trk.record_consistency(False)
            return None

        is_wrong = (this_dir == _opposite(expected))

        # Single-vehicle adaptive logic
        if self._active_count < MIN_VEHICLES_GLOBAL:
            lane_expected = self._lane_dir.expected_direction(trk.lane_id)
            if lane_expected == 0:
                trk.record_consistency(False)
                return None

        trk.record_consistency(is_wrong)

        # Temporal consistency check
        n_total = len(trk.consistency_buf)
        if n_total < WRONG_MIN_SAMPLES:
            return None

        wrong_ratio = sum(trk.consistency_buf) / n_total

        log.debug(
            f"[Lane:wd] track={track_id} "
            f"actual_dir={_sector_label(this_dir)} "
            f"expected_dir={_sector_label(expected)} "
            f"wrong_ratio={wrong_ratio:.2f} "
            f"threshold={WRONG_RATIO_THRESHOLD} "
            f"samples={n_total}/{WRONG_MIN_SAMPLES} "
            f"weight={trk.weight:.2f} "
            f"lane={trk.lane_id} "
            f"majority={_sector_label(self._majority_sector)} "
            f"active_vehicles={self._active_count}"
        )

        if wrong_ratio < WRONG_RATIO_THRESHOLD:
            return None

        # Minimum confidence gate
        conf = round(min(wrong_ratio, 1.0), 3)
        if conf < 0.60:
            return None

        # Set cooldown
        trk.cooldown_until = frame_no + VIOLATION_COOLDOWN

        lane_id = self.assign_lane(centroid)
        x0, y0 = trk.pos_buf[0]
        xn, yn = trk.pos_buf[-1]
        dx, dy = xn - x0, yn - y0
        mag = math.hypot(dx, dy)

        log.info(
            f"[Lane] ⚠ WRONG DIRECTION "
            f"track={track_id} "
            f"actual={_sector_label(this_dir)} "
            f"expected={_sector_label(expected)} "
            f"wrong_ratio={wrong_ratio:.2f} "
            f"conf={conf} "
            f"lane={lane_id} "
            f"frame={frame_no} "
            f"displacement={mag:.0f}px "
            f"scene_vehicles={self._active_count} "
            f"majority={_sector_label(self._majority_sector)}"
        )

        return {
            "type": "wrong_direction",
            "confidence": conf,
            "bbox": bbox,
            "direction": _sector_label(expected),
            "actual_dir": _sector_label(this_dir),
            "lane_id": lane_id,
            "track_id": track_id,
        }

    def _expected_for_track(self, trk: _Track) -> int:
        """Determine expected direction for this track."""
        global_maj = self._majority_sector
        lane_exp = self._lane_dir.expected_direction(trk.lane_id) if trk.lane_id >= 0 else 0

        if lane_exp != 0:
            if global_maj == 0 or global_maj == lane_exp:
                return lane_exp
            else:
                return lane_exp

        if global_maj != 0 and self._active_count >= MIN_VEHICLES_GLOBAL:
            return global_maj

        return 0

    def _run_hough(self, frame: np.ndarray, road_mask: np.ndarray) -> None:
        """Run Hough transform to detect lane lines."""
        h, w = frame.shape[:2]
        try:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey = cv2.bitwise_and(grey, grey, mask=road_mask)
            blur = cv2.GaussianBlur(grey, (5, 5), 0)
            edges = cv2.Canny(blur, 40, 150)
            roi = np.zeros_like(edges)
            roi[int(h * 0.30):, :] = edges[int(h * 0.30):, :]
            raw = cv2.HoughLinesP(
                roi, 1, np.pi / 180.0, HOUGH_THRESH,
                minLineLength=int(w * HOUGH_MIN_LEN_FRAC),
                maxLineGap=HOUGH_MAX_GAP,
            )
            if raw is None:
                return
            segs = [tuple(r[0]) for r in raw if self._angle_ok(r[0])]
            extended = [e for e in (self._extend_line(s, h) for s in segs) if e is not None]
            if extended:
                lines = self._cluster_lines(extended, w)
                if lines:
                    self._lane_lines = lines
                    log.debug(f"[Lane:Hough] {len(lines)} lines")
        except Exception as exc:
            log.error(f"[Lane:Hough] {exc}")

    def _line_x_at(self, cy: int, h: int) -> List[float]:
        """Get x coordinate of lane lines at given y."""
        return [
            ln["x_top"] + (ln["x_bot"] - ln["x_top"]) * (cy / max(h, 1))
            for ln in self._lane_lines
        ]

    @staticmethod
    def _angle_ok(seg: tuple) -> bool:
        """Check if line segment angle is within acceptable range."""
        x1, y1, x2, y2 = seg
        angle = float(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-6)))
        return ANGLE_MIN_DEG <= angle <= ANGLE_MAX_DEG

    @staticmethod
    def _extend_line(seg: tuple, h: int) -> Optional[Dict]:
        """Extend line segment to full frame height."""
        x1, y1, x2, y2 = seg
        dy = y2 - y1
        if abs(dy) < 1:
            return None
        slope = (x2 - x1) / dy
        return {
            "x_top": x1 + slope * (0 - y1),
            "x_bot": x1 + slope * (h - y1),
            "slope": slope,
        }

    @staticmethod
    def _cluster_lines(lines: List[Dict], w: int) -> List[Dict]:
        """Cluster nearby lines."""
        bw = w * LANE_CLUSTER_BW_FRAC
        clusters: List[List[Dict]] = []
        for ln in lines:
            placed = False
            for cl in clusters:
                avg = float(np.mean([l["x_bot"] for l in cl]))
                if abs(ln["x_bot"] - avg) < bw:
                    cl.append(ln)
                    placed = True
                    break
            if not placed:
                clusters.append([ln])
        return [
            {
                "x_top": float(np.median([l["x_top"] for l in cl])),
                "x_bot": float(np.median([l["x_bot"] for l in cl])),
                "slope": float(np.median([l["slope"] for l in cl])),
            }
            for cl in clusters
        ]