"""
SpeakAI — Public Speaking Coach Processor

A custom VideoProcessorPublisher that processes webcam frames to extract
real-time public speaking metrics:
  - Eye contact estimation (face landmark + gaze direction)
  - Head stability (frame-to-frame landmark delta)
  - Posture score (shoulder alignment via pose landmarks)
  - Facial engagement (mouth openness, brow movement proxy)
  - Emotion/attention intensity (composite score)

These metrics are pushed to connected frontend clients via WebSocket.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric Types
# ---------------------------------------------------------------------------

@dataclass
class SpeakingMetrics:
    """Structured metrics extracted per frame batch."""
    eye_contact: float = 0.0          # 0–100 %
    head_stability: float = 0.0       # 0–100 %
    posture_score: float = 0.0        # 0–100 %
    facial_engagement: float = 0.0    # 0–100 %
    attention_intensity: float = 0.0  # 0–100 %
    filler_words: int = 0             # count (updated from STT side)
    words_per_minute: int = 0         # WPM (updated from STT side)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# WebSocket Metrics Broadcaster
# ---------------------------------------------------------------------------

class MetricsBroadcaster:
    """Manages connected WebSocket clients and broadcasts metrics."""

    def __init__(self):
        self._clients: Set[Any] = set()
        self._latest_metrics: Optional[Dict[str, Any]] = None

    def register(self, ws) -> None:
        self._clients.add(ws)
        logger.info(f"📡 Client connected. Total: {len(self._clients)}")

    def unregister(self, ws) -> None:
        self._clients.discard(ws)
        logger.info(f"📡 Client disconnected. Total: {len(self._clients)}")

    @property
    def latest_metrics(self) -> Optional[Dict[str, Any]]:
        return self._latest_metrics

    async def broadcast(self, metrics: Dict[str, Any]) -> None:
        """Send metrics to all connected clients."""
        self._latest_metrics = metrics
        if not self._clients:
            return
        message = json.dumps({"type": "metrics", "data": metrics})
        disconnected = set()
        for ws in self._clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.unregister(ws)

    async def broadcast_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send coaching feedback to all connected clients."""
        if not self._clients:
            return
        message = json.dumps({"type": "feedback", "data": feedback})
        disconnected = set()
        for ws in self._clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.unregister(ws)


# Global broadcaster instance
broadcaster = MetricsBroadcaster()


# ---------------------------------------------------------------------------
# Vision Processor — Frame Analysis Engine
# ---------------------------------------------------------------------------

class SpeakingCoachAnalyzer:
    """
    Analyzes video frames using MediaPipe Face Mesh and Pose to extract
    public speaking metrics. Runs in a thread pool for non-blocking inference.
    """

    def __init__(self, fps: int = 5):
        self.fps = fps
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="speaking_analyzer"
        )
        self._prev_landmarks = None
        self._metrics_history: List[SpeakingMetrics] = []
        self._initialized = False

        # MediaPipe models
        self._face_mesh = None
        self._pose = None

    def initialize(self):
        """Lazy-init MediaPipe models."""
        if self._initialized:
            return
        if mp is None:
            logger.warning("⚠️ MediaPipe not installed. Using simulated metrics.")
            self._initialized = True
            return

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._initialized = True
        logger.info("✅ MediaPipe models initialized")

    def analyze_frame_sync(self, frame_rgb: np.ndarray) -> SpeakingMetrics:
        """Synchronous frame analysis (runs in thread pool)."""
        self.initialize()

        if self._face_mesh is None or self._pose is None:
            return self._simulated_metrics()

        h, w, _ = frame_rgb.shape
        metrics = SpeakingMetrics(timestamp=time.time())

        # --- Face Mesh Analysis ---
        face_results = self._face_mesh.process(frame_rgb)
        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0].landmark

            # Eye contact: estimate gaze via iris position relative to eye bounds
            metrics.eye_contact = self._estimate_eye_contact(face_lm, w, h)

            # Head stability: compare landmark positions to previous frame
            metrics.head_stability = self._estimate_head_stability(face_lm, w, h)

            # Facial engagement: mouth openness + brow raise proxy
            metrics.facial_engagement = self._estimate_facial_engagement(face_lm, h)

        # --- Pose Analysis ---
        pose_results = self._pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            pose_lm = pose_results.pose_landmarks.landmark
            metrics.posture_score = self._estimate_posture(pose_lm)

        # Composite attention intensity
        metrics.attention_intensity = round(
            0.3 * metrics.eye_contact
            + 0.2 * metrics.head_stability
            + 0.3 * metrics.facial_engagement
            + 0.2 * metrics.posture_score,
            1,
        )

        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 300:  # ~60s at 5fps
            self._metrics_history = self._metrics_history[-300:]

        return metrics

    # --- Estimation helpers ---

    def _estimate_eye_contact(self, landmarks, w: int, h: int) -> float:
        """Iris position relative to eye socket center → gaze score."""
        try:
            # Left iris center (index 468) vs left eye corners (33, 133)
            iris_l = landmarks[468]
            eye_l_inner = landmarks[133]
            eye_l_outer = landmarks[33]
            eye_center_x = (eye_l_inner.x + eye_l_outer.x) / 2
            eye_center_y = (eye_l_inner.y + eye_l_outer.y) / 2

            dx = abs(iris_l.x - eye_center_x) * w
            dy = abs(iris_l.y - eye_center_y) * h
            deviation = (dx ** 2 + dy ** 2) ** 0.5

            # Normalize: < 5px deviation = 100%, > 25px = 0%
            score = max(0.0, min(100.0, 100.0 - (deviation - 5) * (100 / 20)))
            return round(score, 1)
        except (IndexError, AttributeError):
            return 50.0

    def _estimate_head_stability(self, landmarks, w: int, h: int) -> float:
        """Frame-to-frame nose tip displacement → stability."""
        try:
            nose = landmarks[1]  # Nose tip
            current = np.array([nose.x * w, nose.y * h])

            if self._prev_landmarks is not None:
                prev_nose = self._prev_landmarks[1]
                prev = np.array([prev_nose.x * w, prev_nose.y * h])
                displacement = np.linalg.norm(current - prev)
                # < 2px movement = 100%, > 20px = 0%
                score = max(0.0, min(100.0, 100.0 - (displacement - 2) * (100 / 18)))
            else:
                score = 90.0

            self._prev_landmarks = landmarks
            return round(score, 1)
        except (IndexError, AttributeError):
            return 80.0

    def _estimate_facial_engagement(self, landmarks, h: int) -> float:
        """Mouth openness + brow raise as engagement proxy."""
        try:
            # Mouth openness: upper lip (13) to lower lip (14) distance
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            mouth_open = abs(upper_lip.y - lower_lip.y) * h

            # Brow raise: eyebrow (70) to eye (159) distance
            brow = landmarks[70]
            eye = landmarks[159]
            brow_raise = abs(brow.y - eye.y) * h

            # Normalize
            mouth_score = min(100.0, mouth_open * 8)  # Speaking → higher
            brow_score = min(100.0, brow_raise * 6)    # Animated expression

            engagement = 0.6 * mouth_score + 0.4 * brow_score
            return round(max(20.0, min(100.0, engagement)), 1)
        except (IndexError, AttributeError):
            return 60.0

    def _estimate_posture(self, pose_landmarks) -> float:
        """Shoulder alignment + vertical spine alignment."""
        try:
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            nose = pose_landmarks[0]

            # Shoulder level alignment (y-axis difference)
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            shoulder_score = max(0.0, min(100.0, 100.0 - shoulder_diff * 500))

            # Spine alignment: nose should be centered between shoulders
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            spine_offset = abs(nose.x - shoulder_center_x)
            spine_score = max(0.0, min(100.0, 100.0 - spine_offset * 300))

            posture = 0.5 * shoulder_score + 0.5 * spine_score
            return round(posture, 1)
        except (IndexError, AttributeError):
            return 75.0

    def _simulated_metrics(self) -> SpeakingMetrics:
        """Fallback simulated metrics when MediaPipe is unavailable."""
        import random
        base_t = time.time()
        return SpeakingMetrics(
            eye_contact=round(75 + 20 * np.sin(base_t * 0.5) + random.uniform(-3, 3), 1),
            head_stability=round(85 + 10 * np.sin(base_t * 0.3) + random.uniform(-2, 2), 1),
            posture_score=round(88 + 8 * np.sin(base_t * 0.2) + random.uniform(-2, 2), 1),
            facial_engagement=round(70 + 15 * np.sin(base_t * 0.4) + random.uniform(-3, 3), 1),
            attention_intensity=round(80 + 12 * np.sin(base_t * 0.35) + random.uniform(-2, 2), 1),
            words_per_minute=int(130 + 20 * np.sin(base_t * 0.1) + random.uniform(-5, 5)),
            filler_words=max(0, int(3 + 2 * np.sin(base_t * 0.2) + random.uniform(-1, 1))),
            timestamp=base_t,
        )

    async def analyze_frame(self, frame_rgb: np.ndarray) -> SpeakingMetrics:
        """Async wrapper for thread-pool analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.analyze_frame_sync,
            frame_rgb,
        )

    def close(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
        if self._face_mesh:
            self._face_mesh.close()
        if self._pose:
            self._pose.close()
        logger.info("🛑 SpeakingCoachAnalyzer closed")


# ---------------------------------------------------------------------------
# Feedback Engine
# ---------------------------------------------------------------------------

class FeedbackEngine:
    """
    Accepts structured metrics and generates short coaching feedback messages.
    Uses threshold-based rules for instant feedback and optionally calls
    a Vision Agent reasoning API for deeper analysis.
    """

    THRESHOLDS = {
        "eye_contact_low": 50.0,
        "eye_contact_good": 80.0,
        "stability_low": 60.0,
        "posture_low": 70.0,
        "engagement_low": 50.0,
        "wpm_fast": 170,
        "wpm_slow": 100,
    }

    def __init__(self, cooldown_seconds: float = 8.0):
        self._last_feedback_time: float = 0
        self._cooldown = cooldown_seconds
        self._feedback_history: List[Dict[str, Any]] = []
        self._feedback_id = 0

    def evaluate(self, metrics: SpeakingMetrics) -> Optional[Dict[str, Any]]:
        """Generate feedback if thresholds breached and cooldown elapsed."""
        now = time.time()
        if now - self._last_feedback_time < self._cooldown:
            return None

        feedback = self._check_thresholds(metrics)
        if feedback:
            self._last_feedback_time = now
            self._feedback_id += 1
            feedback["id"] = self._feedback_id
            feedback["timestamp"] = now
            self._feedback_history.append(feedback)
            return feedback
        return None

    def _check_thresholds(self, m: SpeakingMetrics) -> Optional[Dict[str, Any]]:
        T = self.THRESHOLDS

        if m.eye_contact < T["eye_contact_low"]:
            return {
                "severity": "warning",
                "headline": "Low Eye Contact",
                "explanation": f"Eye contact dropped to {m.eye_contact:.0f}%. Look towards the camera.",
                "tip": "Imagine you're speaking to one person right behind the lens.",
            }

        if m.posture_score < T["posture_low"]:
            return {
                "severity": "warning",
                "headline": "Check Your Posture",
                "explanation": f"Posture score is {m.posture_score:.0f}%. Shoulders may be uneven.",
                "tip": "Roll your shoulders back and stand tall.",
            }

        if m.head_stability < T["stability_low"]:
            return {
                "severity": "info",
                "headline": "Head Movement Detected",
                "explanation": f"Head stability at {m.head_stability:.0f}%. Excessive movement can be distracting.",
                "tip": "Try to keep your head steady while making your key points.",
            }

        if m.facial_engagement < T["engagement_low"]:
            return {
                "severity": "info",
                "headline": "Increase Expression",
                "explanation": f"Facial engagement at {m.facial_engagement:.0f}%. You appear flat.",
                "tip": "Smile and vary your expressions to connect with your audience.",
            }

        if m.words_per_minute > T["wpm_fast"]:
            return {
                "severity": "warning",
                "headline": "Speaking Too Fast",
                "explanation": f"Pace at {m.words_per_minute} WPM — above ideal range.",
                "tip": "Pause between key points. Let your ideas breathe.",
            }

        if 0 < m.words_per_minute < T["wpm_slow"]:
            return {
                "severity": "info",
                "headline": "Speaking Slowly",
                "explanation": f"Pace at {m.words_per_minute} WPM — below typical range.",
                "tip": "Try to maintain a steady rhythm around 130–150 WPM.",
            }

        # Positive feedback
        if m.eye_contact > T["eye_contact_good"] and m.posture_score > 85:
            return {
                "severity": "info",
                "headline": "Great Presence!",
                "explanation": f"Eye contact {m.eye_contact:.0f}%, posture {m.posture_score:.0f}%. Excellent form.",
                "tip": "Keep this up — you're commanding the room.",
            }

        return None
