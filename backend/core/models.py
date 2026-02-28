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
# Chat / Conversation models
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single message in the conversation between user and AI agent."""
    id: str = ""
    role: str = "assistant"      # "user" | "assistant" | "system"
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    source: str = "text"          # "text" | "voice" | "vision"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptEntry:
    """A speech-to-text transcript segment from the call audio."""
    speaker: str = ""             # participant id or "user" / "agent"
    text: str = ""
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    is_final: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationState:
    """Tracks the live conversation state."""
    is_user_speaking: bool = False
    is_agent_speaking: bool = False
    turn_count: int = 0
    last_user_speech: str = ""
    last_agent_response: str = ""
    transcript_length: int = 0
    context_summary: str = ""

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
    audio_active: bool = False
    transcript_entries: int = 0
    chat_messages: int = 0
    conversation_turns: int = 0
    # ── Production integrity fields ─────────────────────────────────────
    session_state: str = "init"        # SessionState enum value
    session_mode: str = "unavailable"  # SessionMode enum value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
