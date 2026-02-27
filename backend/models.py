"""
SpeakAI — Data Models

All dataclasses / typed dictionaries used across the backend.
Single source of truth for shapes of data flowing through the system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Metrics extracted per frame
# ---------------------------------------------------------------------------

@dataclass
class SpeakingMetrics:
    """
    Structured metrics produced by the vision pipeline.

    When the Vision Agents SDK is active these values come from SDK inference.
    When it is not available they come from the MediaPipe CV fallback.
    The `source` field makes the origin explicit in every message.
    """
    eye_contact: float = 0.0
    head_stability: float = 0.0
    posture_score: float = 0.0
    facial_engagement: float = 0.0
    attention_intensity: float = 0.0
    filler_words: int = 0
    words_per_minute: int = 0
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"  # "sdk" | "mediapipe_fallback" | "simulated"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Coaching feedback
# ---------------------------------------------------------------------------

@dataclass
class CoachingFeedback:
    id: int = 0
    severity: str = "info"   # "info" | "warning" | "critical"
    headline: str = ""
    explanation: str = ""
    tip: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    source: str = "rules"    # "rules" | "llm" | "sdk"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-session telemetry / structured log
# ---------------------------------------------------------------------------

@dataclass
class SessionTelemetry:
    """Counters & latencies tracked per session — never crashes the session."""
    session_id: str = ""
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    sdk_inferences: int = 0
    reasoning_calls: int = 0
    last_frame_latency_ms: float = 0.0
    last_sdk_latency_ms: float = 0.0
    last_reasoning_latency_ms: float = 0.0
    sdk_active: bool = False
    multimodal_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
