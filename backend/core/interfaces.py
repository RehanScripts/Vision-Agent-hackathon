"""
SpeakAI — Layer Interfaces

Protocol definitions for the three isolated layers:
  1. Transport  — media connectivity (agent, tracks, TTS publishing)
  2. Inference  — AI processing (Gemini, MediaPipe, reasoning)
  3. Policy     — gating decisions (health, freshness, mode)

Each layer communicates through these protocols — never by reaching
into another layer's internals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .models import SpeakingMetrics, CoachingFeedback
from .state_machine import SessionState, SessionMode


# ═══════════════════════════════════════════════════════════════════════════
# Transport Layer — media connectivity
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class TransportLayer(Protocol):
    """Owns agent lifecycle, track subscriptions, and TTS output."""

    @property
    def agent(self) -> Any:
        """The underlying SDK agent (or None)."""
        ...

    @property
    def has_video_track(self) -> bool:
        """True if a video track is currently subscribed."""
        ...

    @property
    def has_audio_track(self) -> bool:
        """True if an audio track is currently active."""
        ...

    async def say(self, text: str) -> None:
        """Publish spoken audio into the call via TTS."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Inference Layer — AI processing
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class InferenceLayer(Protocol):
    """Owns all AI processing: video analysis, reasoning, response generation."""

    @property
    def latest_metrics(self) -> SpeakingMetrics:
        """Most recent metrics from video or audio pipeline."""
        ...

    @property
    def frames_processed(self) -> int:
        """Total video frames analysed by MediaPipe."""
        ...

    async def evaluate(self, metrics: SpeakingMetrics) -> Optional[CoachingFeedback]:
        """Run reasoning engine on current metrics."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Policy Layer — gating decisions
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class PolicyLayer(Protocol):
    """Owns health checks, freshness validation, and mode determination."""

    def check_health(self) -> Dict[str, bool]:
        """
        Run health checks for all channels.
        Returns {"video": bool, "audio": bool, "model": bool}.
        """
        ...

    def determine_mode(self) -> SessionMode:
        """Determine current operating mode from health status."""
        ...

    def validate_freshness(self, metrics: SpeakingMetrics) -> bool:
        """True if metrics are fresh enough for AI response."""
        ...

    def should_emit_metrics(self, metrics: SpeakingMetrics) -> bool:
        """True if metrics should be forwarded to frontend."""
        ...

    def determine_state(self) -> SessionState:
        """Determine what session state health supports."""
        ...
