"""
SpeakAI — Data Models

Dataclasses for every piece of data flowing through the system.
The `source` field on each model lets judges verify which path is active.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

@dataclass
class SpeakingMetrics:
    """
    Structured metrics produced per video frame.

    source values:
      • "sdk"                — Vision Agent SDK (primary path)
      • "mediapipe_fallback" — local CV fallback
      • "simulated"          — demo mode
    """
    eye_contact: float = 0.0
    head_stability: float = 0.0
    posture_score: float = 0.0
    facial_engagement: float = 0.0
    attention_intensity: float = 0.0
    filler_words: int = 0
    words_per_minute: int = 0
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Coaching feedback
# ---------------------------------------------------------------------------

@dataclass
class CoachingFeedback:
    id: int = 0
    severity: str = "info"      # "info" | "warning" | "critical"
    headline: str = ""
    explanation: str = ""
    tip: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    source: str = "rules"       # "rules" | "llm" | "sdk"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Session telemetry
# ---------------------------------------------------------------------------

@dataclass
class SessionTelemetry:
    """Per-session counters — never crashes the session."""
    session_id: str = ""
    call_id: str = ""
    frames_processed: int = 0
    sdk_inferences: int = 0
    reasoning_calls: int = 0
    last_frame_latency_ms: float = 0.0
    last_reasoning_latency_ms: float = 0.0
    sdk_active: bool = False
    multimodal_active: bool = False
    agent_joined: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
