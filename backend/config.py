"""
SpeakAI — Configuration

Centralised settings from environment variables.
All tuneable constants live here — zero magic numbers in other files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    )


# ---------------------------------------------------------------------------
# Stream + SDK keys
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SDKConfig:
    """API keys and SDK tuning knobs."""
    stream_api_key: str = os.getenv("STREAM_API_KEY", "")
    stream_api_secret: str = os.getenv("STREAM_API_SECRET", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")

    # FPS sent to Gemini Realtime for multimodal video understanding
    llm_fps: int = 2
    # CV fallback FPS (only if SDK unavailable)
    fallback_cv_fps: float = 5.0

    @property
    def has_all_keys(self) -> bool:
        return all([
            self.stream_api_key,
            self.stream_api_secret,
            self.gemini_api_key,
        ])


# ---------------------------------------------------------------------------
# Processing tunables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessingConfig:
    # Coaching reasoning cooldown (seconds between LLM reasoning calls)
    reasoning_cooldown: float = 3.0
    # Hard timeout for a single LLM reasoning call
    reasoning_timeout: float = 5.0
    # Threshold-feedback cooldown
    feedback_cooldown: float = 5.0
    # Status broadcast interval
    status_broadcast_interval: float = 5.0
    # Frame processing FPS for the SpeakingCoachProcessor
    processor_fps: int = 5


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

server_cfg = ServerConfig()
sdk_cfg = SDKConfig()
processing_cfg = ProcessingConfig()
