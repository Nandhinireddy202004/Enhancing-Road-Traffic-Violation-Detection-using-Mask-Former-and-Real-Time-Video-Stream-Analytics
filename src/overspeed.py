"""
overspeed.py — TrafficSentinel Over Speed Detection Module
══════════════════════════════════════════════════════════════════════════════

Features:
- Real-time speed calculation with pixel displacement
- Speed display on annotated video frames
- User-configurable speed limit via API
- Violation detection with confidence scoring
- Visual speed indicators on bounding boxes
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

log = logging.getLogger("TrafficSentinel.Speed")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SPEED_LIMIT_KMH = 60.0
SCALE_MPP = 0.03  # Default scale - adjust based on camera
FRAME_SKIP = 4
REAL_WINDOW_LONG = 10
MIN_REAL_FRAMES = 3
MIN_DISP_PX = 3
MAX_DISP_PX = 500
MAX_SPEED_KMH = 200.0
SPEED_HISTORY_WINDOW = 5
SUSTAINED_FRAMES = 2
VIOLATION_COOLDOWN = 60
MIN_CONFIDENCE = 0.40


class OverSpeedDetector:
    """
    Over speed violation detector with visual speed display.
    """
    
    def __init__(
        self,
        speed_limit: float = SPEED_LIMIT_KMH,
        scale_mpp: float = SCALE_MPP,
        frame_skip: int = FRAME_SKIP,
    ):
        self.speed_limit = max(1.0, float(speed_limit))
        self._scale = max(1e-9, float(scale_mpp))
        self._frame_skip = max(1, int(frame_skip))
        
        # Per-track state
        self._real_pos: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=REAL_WINDOW_LONG)
        )
        self._prev_pos: Dict[str, Optional[Tuple[int, int]]] = {}
        self._speed_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=SPEED_HISTORY_WINDOW)
        )
        self._speed_smoothed: Dict[str, float] = {}
        self._confidence: Dict[str, float] = {}
        self._violation_cooldown: Dict[str, int] = {}
        self._frame_counter = 0
        
        log.info(f"[OverSpeed] Initialized: limit={self.speed_limit}km/h, scale={self._scale:.6f}m/px")

    # ══════════════════════════════════════════════════════════════════════════
    #  CONFIGURATION (User settable via API)
    # ══════════════════════════════════════════════════════════════════════════
    
    def set_speed_limit(self, kmh: float) -> None:
        """Set speed limit - callable from frontend"""
        self.speed_limit = max(1.0, float(kmh))
        log.warning(f"[OverSpeed] Speed limit changed to {self.speed_limit} km/h")
    
    def set_scale(self, mpp: float) -> None:
        """Set calibration scale - callable from frontend"""
        self._scale = max(1e-9, float(mpp))
        log.info(f"[OverSpeed] Scale changed to {self._scale:.6f} m/px")
    
    def get_speed_limit(self) -> float:
        """Get current speed limit"""
        return self.speed_limit
    
    def get_scale(self) -> float:
        """Get current scale"""
        return self._scale

    # ══════════════════════════════════════════════════════════════════════════
    #  SPEED CALCULATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def update_speed(
        self,
        track_id: str,
        centroid: Tuple[int, int],
        fps: float = 25.0,
        frame_no: int = 0,
    ) -> Tuple[float, float]:
        """
        Update speed for a track.
        
        Returns:
            Tuple of (speed_kmh, confidence)
        """
        fps = max(1.0, float(fps))
        
        # Check cooldown
        if track_id in self._violation_cooldown:
            if frame_no > 0 and frame_no < self._violation_cooldown[track_id]:
                return self._speed_smoothed.get(track_id, 0.0), self._confidence.get(track_id, 0.0)
        
        # Filter: only record REAL position changes
        prev = self._prev_pos.get(track_id)
        if prev is not None and prev == centroid:
            return self._speed_smoothed.get(track_id, 0.0), self._confidence.get(track_id, 0.0)
        
        # Store new position
        self._prev_pos[track_id] = centroid
        real_pos = self._real_pos[track_id]
        real_pos.append(centroid)
        
        # Need minimum real detections
        if len(real_pos) < MIN_REAL_FRAMES:
            self._speed_smoothed[track_id] = 0.0
            self._confidence[track_id] = 0.0
            return 0.0, 0.0
        
        # Calculate speed
        speed_kmh, confidence = self._calculate_speed(real_pos, fps)
        
        # Apply caps
        speed_kmh = min(max(speed_kmh, 0.0), MAX_SPEED_KMH)
        
        # Smooth over recent readings
        self._speed_history[track_id].append(speed_kmh)
        smoothed = np.mean(list(self._speed_history[track_id]))
        self._speed_smoothed[track_id] = round(smoothed, 1)
        self._confidence[track_id] = confidence
        
        return self._speed_smoothed[track_id], self._confidence[track_id]
    
    def _calculate_speed(self, positions: deque, fps: float) -> Tuple[float, float]:
        """Calculate speed from positions"""
        if len(positions) < 2:
            return 0.0, 0.0
        
        oldest = positions[0]
        newest = positions[-1]
        disp_px = self._euclidean(oldest[0], oldest[1], newest[0], newest[1])
        
        if disp_px < MIN_DISP_PX:
            return 0.0, 0.0
        
        n_intervals = len(positions) - 1
        secs_per_det = self._frame_skip / fps
        time_s = n_intervals * secs_per_det
        
        if time_s <= 0:
            return 0.0, 0.0
        
        speed_mps = (disp_px * self._scale) / time_s
        speed_kmh = speed_mps * 3.6
        
        # Confidence based on displacement and frame count
        confidence = min(1.0, (disp_px / 80.0) * (len(positions) / 8.0))
        confidence = max(0.3, min(0.95, confidence))
        
        return speed_kmh, confidence
    
    @staticmethod
    def _euclidean(x0: float, y0: float, x1: float, y1: float) -> float:
        dx = x1 - x0
        dy = y1 - y0
        return math.sqrt(dx * dx + dy * dy)
    
    # ══════════════════════════════════════════════════════════════════════════
    #  SPEED DISPLAY ON VIDEO FRAMES
    # ══════════════════════════════════════════════════════════════════════════
    
    def draw_speed_on_frame(
        self,
        frame: np.ndarray,
        track_id: str,
        bbox: List[int],
        speed: float,
        is_speeding: bool,
    ) -> np.ndarray:
        """
        Draw speed information on the vehicle bounding box.
        This will show speed on the video frame.
        
        Args:
            frame: Video frame to draw on
            track_id: Track identifier
            bbox: Bounding box [x1, y1, x2, y2]
            speed: Speed in km/h
            is_speeding: Whether vehicle is exceeding limit
            
        Returns:
            Frame with speed drawn
        """
        x1, y1, x2, y2 = bbox
        
        # Color based on speeding status
        if is_speeding:
            color = (0, 0, 255)  # Red for speeding
            speed_text = f"⚠️ {speed:.0f} km/h ⚠️"
        else:
            color = (0, 255, 0)  # Green for normal
            speed_text = f"{speed:.0f} km/h"
        
        # Draw speed label above bounding box
        label = f"{track_id} | {speed_text}"
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw background rectangle for text
        text_x = x1
        text_y = y1 - 10
        cv2.rectangle(
            frame,
            (text_x - 2, text_y - text_h - 2),
            (text_x + text_w + 2, text_y + 2),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        return frame
    
    def draw_hud_speed_limit(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw speed limit HUD on top of frame.
        
        Args:
            frame: Video frame to draw on
            
        Returns:
            Frame with speed limit displayed
        """
        # Background for HUD
        cv2.rectangle(frame, (10, 10), (250, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 70), (255, 255, 255), 1)
        
        # Speed limit text
        cv2.putText(
            frame, f"SPEED LIMIT: {self.speed_limit:.0f} km/h",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Scale info (for debugging)
        cv2.putText(
            frame, f"Scale: {self._scale:.5f} m/px",
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA
        )
        
        return frame

    # ══════════════════════════════════════════════════════════════════════════
    #  VIOLATION DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    
    def check_violation(
        self,
        track_id: str,
        frame_no: int,
        bbox: Optional[List[int]] = None,
        vehicle_label: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Check if track is violating speed limit."""
        speed = self._speed_smoothed.get(track_id, 0.0)
        confidence = self._confidence.get(track_id, 0.0)
        
        # Must exceed limit
        if speed <= self.speed_limit:
            return None
        
        # Must have sufficient confidence
        if confidence < MIN_CONFIDENCE:
            return None
        
        # Check sustained speeding
        if track_id not in self._speed_history:
            return None
        
        history = list(self._speed_history[track_id])
        if len(history) < SUSTAINED_FRAMES:
            return None
        
        recent_speeds = history[-SUSTAINED_FRAMES:]
        if not all(s > self.speed_limit for s in recent_speeds):
            return None
        
        # Create violation record
        violation = {
            "type": "over_speed",
            "label": "Over Speed",
            "severity": "HIGH",
            "speed_kmh": round(speed, 1),
            "limit_kmh": self.speed_limit,
            "excess_kmh": round(speed - self.speed_limit, 1),
            "confidence": confidence,
            "bbox": bbox,
            "vehicle": vehicle_label,
            "track_id": track_id,
            "frame_no": frame_no,
            "sustained_frames": SUSTAINED_FRAMES,
        }
        
        # Set cooldown
        self._violation_cooldown[track_id] = frame_no + VIOLATION_COOLDOWN
        
        log.warning(f"[OverSpeed] 🚨 VIOLATION: {track_id} speed={speed:.1f}km/h (limit={self.speed_limit}km/h)")
        
        return violation
    
    # ══════════════════════════════════════════════════════════════════════════
    #  QUERIES
    # ══════════════════════════════════════════════════════════════════════════
    
    def get_speed(self, track_id: str) -> float:
        """Get current speed"""
        return self._speed_smoothed.get(track_id, 0.0)
    
    def get_confidence(self, track_id: str) -> float:
        """Get current confidence"""
        return self._confidence.get(track_id, 0.0)
    
    def is_speeding(self, track_id: str) -> bool:
        """Check if track is speeding"""
        return self.get_speed(track_id) > self.speed_limit
    
    def get_all_speeds(self) -> Dict[str, float]:
        """Get all speeds"""
        return dict(self._speed_smoothed)
    
    def remove_track(self, track_id: str) -> None:
        """Remove track"""
        for d in [self._real_pos, self._prev_pos, self._speed_history,
                  self._speed_smoothed, self._confidence, self._violation_cooldown]:
            d.pop(track_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        speeds = list(self._speed_smoothed.values())
        return {
            "active_tracks": len(self._speed_smoothed),
            "avg_speed": np.mean(speeds) if speeds else 0.0,
            "max_speed": max(speeds) if speeds else 0.0,
            "speeding_count": sum(1 for s in speeds if s > self.speed_limit),
            "scale_mpp": self._scale,
            "speed_limit": self.speed_limit,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class SpeedEstimator:
    """Legacy wrapper for backward compatibility."""
    
    def __init__(
        self,
        limit_kmh: float = SPEED_LIMIT_KMH,
        scale_mpp: float = SCALE_MPP,
        frame_skip: int = FRAME_SKIP,
    ):
        self.detector = OverSpeedDetector(limit_kmh, scale_mpp, frame_skip)
        self.speed_limit = limit_kmh
        self._frame_counter = 0
    
    def set_limit(self, kmh: float) -> None:
        self.speed_limit = kmh
        self.detector.set_speed_limit(kmh)
    
    def set_scale(self, mpp: float) -> None:
        self.detector.set_scale(mpp)
    
    def update(self, track_id: str, centroid: Tuple[int, int], fps: float = 25.0) -> float:
        self._frame_counter += 1
        speed, _ = self.detector.update_speed(track_id, centroid, fps, self._frame_counter)
        return speed
    
    def get_speed(self, track_id: str) -> float:
        return self.detector.get_speed(track_id)
    
    def get_confidence(self, track_id: str) -> float:
        return self.detector.get_confidence(track_id)
    
    def is_speeding(self, track_id: str) -> bool:
        return self.detector.is_speeding(track_id)
    
    def check_violation(self, track_id: str, bbox: Optional[List[int]] = None, 
                        vehicle_label: str = "") -> Optional[Dict[str, Any]]:
        return self.detector.check_violation(track_id, self._frame_counter, bbox, vehicle_label)
    
    def remove_track(self, track_id: str) -> None:
        self.detector.remove_track(track_id)
    
    def get_stats(self) -> Dict[str, Any]:
        return self.detector.get_stats()
    
    def get_speed_limit(self) -> float:
        return self.detector.get_speed_limit()
    
    def get_scale(self) -> float:
        return self.detector.get_scale()
    
    def draw_speed_on_frame(self, frame: np.ndarray, track_id: str, bbox: List[int], 
                            speed: float, is_speeding: bool) -> np.ndarray:
        return self.detector.draw_speed_on_frame(frame, track_id, bbox, speed, is_speeding)
    
    def draw_hud_speed_limit(self, frame: np.ndarray) -> np.ndarray:
        return self.detector.draw_hud_speed_limit(frame)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'OverSpeedDetector',
    'SpeedEstimator',
    'SPEED_LIMIT_KMH',
    'SCALE_MPP',
    'FRAME_SKIP'
]