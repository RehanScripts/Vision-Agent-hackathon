"""
SpeakAI — Configuration

Centralised settings loaded from environment variables.
All tuneable constants live here — no magic numbers scattered in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ServerConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    )


@dataclass(frozen=True)
class SDKConfig:
    """Vision Agents SDK keys & tuning knobs."""
    stream_api_key: str = os.getenv("STREAM_API_KEY", "")
    stream_api_secret: str = os.getenv("STREAM_API_SECRET", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")

    # Vision Agent LLM FPS — frames sent to Gemini Realtime per second
    llm_fps: int = 2
    # Local CV fallback FPS (only used when SDK unavailable)
    fallback_cv_fps: float = 5.0


@dataclass(frozen=True)
class ProcessingConfig:
    """Frame queue & reasoning tunables."""
    # Bounded frame queue between WS ingest → processing worker
    frame_queue_max: int = 3
    # Reasoning engine cooldown (seconds between LLM calls)
    reasoning_cooldown: float = 3.0
    # Hard timeout for a single LLM reasoning call
    reasoning_timeout: float = 5.0
    # Threshold-based feedback cooldown
    feedback_cooldown: float = 5.0
    # System status broadcast interval (seconds)
    status_broadcast_interval: float = 5.0


# Singleton instances
server_cfg = ServerConfig()
sdk_cfg = SDKConfig()
processing_cfg = ProcessingConfig()
