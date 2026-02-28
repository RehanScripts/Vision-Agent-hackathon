"""
SpeakAI — Health Checks & Freshness Validation (Policy Layer)

Implements the PolicyLayer protocol.
All gating decisions live here — the service layer never decides
what mode it's in; it asks this module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .models import SpeakingMetrics
from .state_machine import SessionState, SessionMode

logger = logging.getLogger("speakai.health")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Max age (seconds) for metrics to be considered "fresh"
FRESHNESS_MAX_AGE_S: float = 5.0

# Max age (seconds) before video frames are considered stale
VIDEO_STALE_TIMEOUT_S: float = 8.0

# Minimum confidence to trust a transcript
TRANSCRIPT_CONFIDENCE_THRESHOLD: float = 0.3

# Sources that represent real visual data (not fallback)
_REAL_VISUAL_SOURCES = frozenset({"sdk", "mediapipe"})

# Sources that must NOT be treated as live data
# Note: "audio_only" is NOT synthetic — it represents real live audio-derived data
_SYNTHETIC_SOURCES = frozenset({
    "simulated", "live_estimation", "no_mediapipe", "unknown",
})


# ---------------------------------------------------------------------------
# Health Check Result
# ---------------------------------------------------------------------------

class HealthStatus:
    """Snapshot of system health at a point in time."""

    __slots__ = ("video", "audio", "model", "checked_at")

    def __init__(
        self,
        video: bool = False,
        audio: bool = False,
        model: bool = False,
    ) -> None:
        self.video = video
        self.audio = audio
        self.model = model
        self.checked_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": self.video,
            "audio": self.audio,
            "model": self.model,
            "checked_at": self.checked_at,
        }

    @property
    def all_healthy(self) -> bool:
        return self.video and self.audio and self.model

    @property
    def any_healthy(self) -> bool:
        return self.video or self.audio or self.model


# ---------------------------------------------------------------------------
# Session Policy (implements PolicyLayer)
# ---------------------------------------------------------------------------

class SessionPolicy:
    """
    Centralised policy engine.  Answers:
      • What mode are we in?
      • Are metrics fresh enough to act on?
      • Should we forward metrics to the frontend?
      • What state should the session be in?

    The service layer supplies raw signals; this module makes decisions.
    """

    def __init__(
        self,
        freshness_max_age: float = FRESHNESS_MAX_AGE_S,
        video_stale_timeout: float = VIDEO_STALE_TIMEOUT_S,
    ) -> None:
        self._freshness_max_age = freshness_max_age
        self._video_stale_timeout = video_stale_timeout

        # Raw signals — set by the service layer
        self._frames_processed: int = 0
        self._last_frame_time: float = 0.0
        self._audio_active: bool = False
        self._model_active: bool = False
        self._agent_joined: bool = False

        self._last_health: Optional[HealthStatus] = None

    # ── Signal setters (called by service layer) ────────────────────────

    def report_frame(self, timestamp: float | None = None) -> None:
        """Called every time a real video frame is processed."""
        self._frames_processed += 1
        self._last_frame_time = timestamp or time.time()

    def report_audio_state(self, active: bool) -> None:
        self._audio_active = active

    def report_model_state(self, active: bool) -> None:
        self._model_active = active

    def report_agent_joined(self, joined: bool) -> None:
        self._agent_joined = joined

    # ── Health checks ───────────────────────────────────────────────────

    def check_health(self) -> HealthStatus:
        """
        Run all health checks.  Returns a HealthStatus snapshot.
        """
        now = time.time()

        video_ok = (
            self._frames_processed > 0
            and (now - self._last_frame_time) < self._video_stale_timeout
        )

        health = HealthStatus(
            video=video_ok,
            audio=self._audio_active,
            model=self._model_active and self._agent_joined,
        )
        self._last_health = health
        return health

    # ── Mode determination ──────────────────────────────────────────────

    def determine_mode(self) -> SessionMode:
        """
        Derive operating mode from latest health check.
        NEVER silently upgrades to multimodal — all three checks must pass.
        """
        health = self.check_health()

        if health.all_healthy:
            return SessionMode.MULTIMODAL

        if health.audio and health.model:
            return SessionMode.AUDIO_ONLY

        return SessionMode.UNAVAILABLE

    # ── Freshness validation ────────────────────────────────────────────

    def validate_freshness(self, metrics: SpeakingMetrics) -> bool:
        """
        True only when metrics are:
          1. Recent (within freshness window)
          2. From a real source (not simulated/fallback)
        """
        if metrics.timestamp <= 0:
            return False

        age = time.time() - metrics.timestamp
        if age > self._freshness_max_age:
            return False

        source = (metrics.source or "").lower()
        if source in _SYNTHETIC_SOURCES:
            return False

        return True

    def validate_visual_freshness(self, metrics: SpeakingMetrics) -> bool:
        """
        Stricter check: metrics must be from a real visual source.
        Used before making any claim about the user's body language.
        """
        if not self.validate_freshness(metrics):
            return False

        source = (metrics.source or "").lower()
        return source in _REAL_VISUAL_SOURCES

    # ── Emit gating ────────────────────────────────────────────────────

    def should_emit_metrics(self, metrics: SpeakingMetrics) -> bool:
        """
        Metrics are forwarded to the frontend ONLY when:
          • They have a valid timestamp
          • The session health supports the mode of data they represent

        Audio-only metrics (WPM, fillers) are allowed in audio-only mode.
        Visual metrics require multimodal mode.
        """
        if metrics.timestamp <= 0:
            return False

        mode = self.determine_mode()

        source = (metrics.source or "").lower()

        # In multimodal mode, all real metrics are valid
        if mode == SessionMode.MULTIMODAL and source in _REAL_VISUAL_SOURCES:
            return True

        # In audio-only mode, only audio-derived metrics are valid
        if mode == SessionMode.AUDIO_ONLY:
            # We allow metrics that explicitly carry audio-derived data
            # but visual fields must be zero (enforced at emission)
            return True

        return False

    # ── State determination ─────────────────────────────────────────────

    def determine_state(self) -> SessionState:
        """
        Recommend session state based on current health.
        The state machine enforces legality of transitions.
        """
        health = self.check_health()

        if health.all_healthy:
            return SessionState.ACTIVE

        if health.any_healthy:
            return SessionState.DEGRADED

        if self._agent_joined:
            return SessionState.READY

        return SessionState.INIT

    # ── Diagnostics ─────────────────────────────────────────────────────

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def last_health(self) -> Optional[HealthStatus]:
        return self._last_health

    def diagnostics(self) -> Dict[str, Any]:
        """Full diagnostic snapshot for system_status messages."""
        health = self.check_health()
        return {
            "health": health.to_dict(),
            "mode": self.determine_mode().value,
            "recommended_state": self.determine_state().value,
            "frames_processed": self._frames_processed,
            "last_frame_age_s": round(time.time() - self._last_frame_time, 2) if self._last_frame_time > 0 else None,
            "audio_active": self._audio_active,
            "model_active": self._model_active,
            "agent_joined": self._agent_joined,
        }
