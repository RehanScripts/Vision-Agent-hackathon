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

# ── Bridge env-var naming: Gemini SDK reads GOOGLE_API_KEY ──────────────
_gemini_key = os.getenv("GEMINI_API_KEY", "")
if _gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = _gemini_key


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
    reasoning_cooldown: float = 0.6  # ⚡ Was 0.8 — faster coaching cadence
    # Hard timeout for a single LLM reasoning call
    reasoning_timeout: float = 1.2  # ⚡ Was 1.5 — fail fast, fallback fast
    # Threshold-feedback cooldown
    feedback_cooldown: float = 0.8  # ⚡ Was 1.0
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
    max_context_messages: int = 20  # ⚡ Was 30 — smaller context = faster prompt
    # Silence threshold (seconds) before agent proactively speaks
    silence_prompt_timeout: float = 6.0
    # Min silence (seconds) before agent responds (avoids interruptions)
    response_delay: float = 0.05  # ⚡ Was 0.15 — near-instant response start
    # Voice debounce (seconds) — how long to wait for speech to stabilize
    voice_debounce_seconds: float = 0.15  # ⚡ Fast turn detection
    # Max tokens for conversation context sent to LLM
    max_context_tokens: int = 800  # ⚡ Was 1024 — shorter prompt = faster inference
    # Audio chunk size for speech processing (ms)
    audio_chunk_ms: int = 60
    # VAD (Voice Activity Detection) sensitivity 0.0–1.0
    vad_sensitivity: float = 0.6
    # Enable proactive agent (agent speaks without being asked)
    proactive_mode: bool = True
    # Hard timeout for a single LLM conversation call (seconds)
    llm_timeout: float = 5.0  # Realtime LLM can take longer than non-streaming
    # Context window: how many recent messages to include in LLM prompt
    context_window: int = 4  # ⚡ Fewer messages = faster prompt
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
# Performance / Latency Optimization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerformanceConfig:
    """Latency optimization settings."""
    # Pre-warm LLM connection on agent join (eliminates cold-start)
    pre_warm_llm: bool = True
    # Pre-warm TTS connection on agent join
    pre_warm_tts: bool = True
    # ElevenLabs streaming latency optimization level (0-4, 4=max)
    tts_optimize_streaming_latency: int = 4
    # Use ElevenLabs turbo model
    tts_model: str = "eleven_turbo_v2_5"
    # ElevenLabs output format — raw PCM avoids decode overhead
    tts_output_format: str = "pcm_16000"
    # Gemini model — flash-lite is fastest available
    llm_model: str = "gemini-2.0-flash-live-001"
    # Gemini temperature — lower = faster, more deterministic
    llm_temperature: float = 0.3
    # Target latencies (for dashboard display)
    join_target_ms: int = 500
    transport_target_ms: int = 30
    e2e_response_target_ms: int = 1500


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

server_cfg = ServerConfig()
sdk_cfg = SDKConfig()
processing_cfg = ProcessingConfig()
conversation_cfg = ConversationConfig()
performance_cfg = PerformanceConfig()
