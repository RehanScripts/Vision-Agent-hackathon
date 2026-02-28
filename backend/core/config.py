"""
SpeakAI — Configuration

Centralised settings from environment variables.
All tuneable constants live here — zero magic numbers in other files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

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
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")

    # FPS sent to Gemini Realtime for multimodal video understanding
    llm_fps: int = 5
    # CV fallback FPS (only if SDK unavailable)
    fallback_cv_fps: float = 10.0

    @property
    def has_all_keys(self) -> bool:
        return all([
            self.stream_api_key,
            self.stream_api_secret,
            self.gemini_api_key,
            self.elevenlabs_api_key,
        ])


# ---------------------------------------------------------------------------
# Processing tunables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessingConfig:
    # Coaching reasoning cooldown (seconds between LLM reasoning calls)
    reasoning_cooldown: float = 0.8
    # Hard timeout for a single LLM reasoning call
    reasoning_timeout: float = 1.5
    # Threshold-feedback cooldown
    feedback_cooldown: float = 1.0
    # Status broadcast interval
    status_broadcast_interval: float = 1.0
    # Frame processing FPS for the SpeakingCoachProcessor
    processor_fps: int = 10


# ---------------------------------------------------------------------------
# Conversation / Communication tunables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversationConfig:
    # Max conversation history entries to keep in context
    max_context_messages: int = 30
    # Silence threshold (seconds) before agent proactively speaks
    silence_prompt_timeout: float = 6.0
    # Min silence (seconds) before agent responds (avoids interruptions)
    response_delay: float = 0.15
    # Max tokens for conversation context sent to LLM
    max_context_tokens: int = 1024
    # Audio chunk size for speech processing (ms)
    audio_chunk_ms: int = 60
    # VAD (Voice Activity Detection) sensitivity 0.0–1.0
    vad_sensitivity: float = 0.6
    # Enable proactive agent (agent speaks without being asked)
    proactive_mode: bool = True
    # System prompt for conversation mode
    system_prompt: str = (
        "You are SpeakAI, a real-time communication coach in a live video call. "
        "You observe the speaker's body language, facial expressions, posture, and speech. "
        "Provide instant coaching, answer questions, and converse naturally. "
        "Be brief (≤20 words per response), encouraging, and actionable. "
        "If the user asks a question, answer immediately. "
        "If you notice something about their presentation, offer a quick tip."
    )


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

server_cfg = ServerConfig()
sdk_cfg = SDKConfig()
processing_cfg = ProcessingConfig()
conversation_cfg = ConversationConfig()
