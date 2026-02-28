"""
SpeakAI — Audio Processor

================================================================================
AUDIO PROCESSING PIPELINE — LISTENS TO LIVE CALL AUDIO
================================================================================

Processes incoming audio tracks from the Stream Video call:

  1. The Agent subscribes to the participant's audio track.
  2. Audio frames arrive as PCM data via the SDK's audio pipeline.
  3. We run Voice Activity Detection (VAD) to detect speech boundaries.
  4. Speech segments are buffered and sent for transcription.
  5. Transcription results are published as TranscriptEntry events.
  6. Speech metrics (WPM, filler words, volume) are extracted.

This integrates with the Vision Agent's audio subsystem — no manual
WebSocket audio streaming. The Agent handles all media transport.
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
import re
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from ..core.models import TranscriptEntry, SpeakingMetrics
from ..core.config import conversation_cfg

logger = logging.getLogger("speakai.audio")

# Common filler words to detect
FILLER_WORDS = {
    "um", "uh", "erm", "ah", "like", "you know", "basically",
    "actually", "literally", "right", "so", "well", "kind of",
    "sort of", "i mean", "you see",
}

FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in FILLER_WORDS) + r")\b",
    re.IGNORECASE,
)


class AudioAnalyzer:
    """
    Analyses audio characteristics from raw PCM data.
    Runs in-process (no external API calls for basic metrics).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._energy_window: Deque[float] = deque(maxlen=50)  # ~5s at 100ms chunks
        self._speaking_start: Optional[float] = None
        self._word_timestamps: List[float] = []
        self._is_speaking = False

    def process_chunk(self, pcm_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyse a chunk of PCM audio data.
        Returns dict with: is_speaking, volume_rms, energy
        """
        if pcm_data.size == 0:
            return {"is_speaking": False, "volume_rms": 0.0, "energy": 0.0}

        # Normalise to float
        audio = pcm_data.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0  # int16 range

        # RMS volume
        rms = float(np.sqrt(np.mean(audio ** 2)))
        energy = float(np.sum(audio ** 2))

        self._energy_window.append(rms)

        # Simple VAD based on energy threshold
        avg_energy = np.mean(list(self._energy_window)) if self._energy_window else 0
        threshold = max(0.01, avg_energy * conversation_cfg.vad_sensitivity + 0.005)
        is_speaking = rms > threshold

        # Track speaking state transitions
        now = time.time()
        if is_speaking and not self._is_speaking:
            self._speaking_start = now
        elif not is_speaking and self._is_speaking:
            # Speaking just ended
            if self._speaking_start:
                duration = now - self._speaking_start
                # Rough word estimate: ~2.5 words/sec for English
                estimated_words = int(duration * 2.5)
                self._word_timestamps.extend(
                    [now] * estimated_words
                )

        self._is_speaking = is_speaking

        return {
            "is_speaking": is_speaking,
            "volume_rms": round(rms, 4),
            "energy": round(energy, 6),
        }

    def estimate_wpm(self, window_seconds: float = 60.0) -> int:
        """Estimate words per minute from recent word timestamps."""
        now = time.time()
        cutoff = now - window_seconds
        recent = [t for t in self._word_timestamps if t > cutoff]
        if len(recent) < 2:
            return 0
        elapsed = now - recent[0]
        if elapsed < 5:
            return 0
        return int(len(recent) / (elapsed / 60.0))

    @staticmethod
    def count_fillers(text: str) -> int:
        """Count filler words in a transcript string."""
        return len(FILLER_PATTERN.findall(text))

    def reset(self) -> None:
        self._energy_window.clear()
        self._word_timestamps.clear()
        self._speaking_start = None
        self._is_speaking = False


class AudioProcessor:
    """
    Audio processing pipeline that plugs into the Vision Agent.

    Lifecycle:
      attach_agent(agent) → process_audio(track, pid) → stop() → close()

    Publishes:
      - TranscriptEntry events (speech-to-text results)
      - Audio-derived metrics (WPM, filler words, volume)
    """

    def __init__(
        self,
        on_transcript: Optional[Callable[[TranscriptEntry], Any]] = None,
        on_speech_state: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self._on_transcript = on_transcript
        self._on_speech_state = on_speech_state

        self._analyzer = AudioAnalyzer()
        self._agent: Any = None
        self._active = False
        self._processing_task: Optional[asyncio.Task] = None

        # Transcript buffer
        self._transcript_buffer: List[TranscriptEntry] = []
        self._full_transcript: List[TranscriptEntry] = []
        self._current_speech_text = ""

        # Metrics
        self.total_words: int = 0
        self.total_fillers: int = 0
        self.current_wpm: int = 0

        logger.info("AudioProcessor initialised")

    @property
    def is_user_speaking(self) -> bool:
        return self._analyzer._is_speaking

    @property
    def transcript(self) -> List[TranscriptEntry]:
        return list(self._full_transcript)

    @property
    def last_transcript_text(self) -> str:
        if self._full_transcript:
            return self._full_transcript[-1].text
        return ""

    def attach_agent(self, agent: Any) -> None:
        """Called when the Agent is created — gives access to agent + LLM."""
        self._agent = agent
        logger.info("AudioProcessor attached to agent")

    async def process_audio(
        self,
        track: Any,
        participant_id: Optional[str] = None,
    ) -> None:
        """
        Called by the Agent when a participant's audio track appears.
        The Vision Agent SDK handles audio routing — we receive callbacks.

        For SDK-integrated mode: the Agent's Gemini Realtime plugin
        already processes audio for understanding. We additionally
        extract speech metrics and maintain a transcript.
        """
        self._active = True
        logger.info(
            f"AudioProcessor: processing audio from "
            f"participant {participant_id or 'unknown'}"
        )

        # The Agent's Gemini Realtime plugin handles audio transcription.
        # We monitor the agent's transcript events if available.
        if self._agent and hasattr(self._agent, "on"):
            try:
                # Subscribe to transcript events from the agent
                self._agent.on("transcript", self._handle_agent_transcript)
                self._agent.on("user_speech_started", self._handle_speech_start)
                self._agent.on("user_speech_ended", self._handle_speech_end)
                logger.info("AudioProcessor: subscribed to agent transcript events")
            except Exception as e:
                logger.debug(f"AudioProcessor: agent event subscription: {e}")

        # Start a background poll for the agent's transcript state
        self._processing_task = asyncio.create_task(
            self._audio_monitor(), name="audio-monitor"
        )

    async def _audio_monitor(self) -> None:
        """
        Monitors the agent's audio state and extracts speech metrics.
        Gemini Realtime handles actual transcription — we augment with metrics.
        """
        while self._active:
            try:
                await asyncio.sleep(0.5)

                if not self._active:
                    break

                # Update WPM estimate
                self.current_wpm = self._analyzer.estimate_wpm()

                # Broadcast speech state changes
                if self._on_speech_state:
                    state = {
                        "is_user_speaking": self._analyzer._is_speaking,
                        "volume_rms": self._analyzer._energy_window[-1]
                        if self._analyzer._energy_window else 0.0,
                        "current_wpm": self.current_wpm,
                        "total_words": self.total_words,
                        "total_fillers": self.total_fillers,
                    }
                    cb = self._on_speech_state(state)
                    if asyncio.iscoroutine(cb):
                        await cb

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"AudioProcessor monitor error: {e}")

    async def _handle_agent_transcript(self, event: Any) -> None:
        """Handle transcript events emitted by the Agent's LLM plugin."""
        try:
            text = getattr(event, "text", str(event))
            speaker = getattr(event, "speaker", "user")
            is_final = getattr(event, "is_final", True)
            confidence = getattr(event, "confidence", 1.0)

            if not text or not text.strip():
                return

            entry = TranscriptEntry(
                speaker=speaker,
                text=text.strip(),
                timestamp=time.time(),
                confidence=confidence,
                is_final=is_final,
            )

            if is_final:
                self._full_transcript.append(entry)

                # Count words and fillers
                words = len(text.split())
                self.total_words += words
                fillers = self._analyzer.count_fillers(text)
                self.total_fillers += fillers

                # Add word timestamps for WPM calculation
                now = time.time()
                self._analyzer._word_timestamps.extend([now] * words)

            if self._on_transcript:
                cb = self._on_transcript(entry)
                if asyncio.iscoroutine(cb):
                    await cb

        except Exception as e:
            logger.debug(f"AudioProcessor transcript handler error: {e}")

    async def _handle_speech_start(self, event: Any = None) -> None:
        """User started speaking."""
        self._analyzer._is_speaking = True
        self._analyzer._speaking_start = time.time()

    async def _handle_speech_end(self, event: Any = None) -> None:
        """User stopped speaking."""
        self._analyzer._is_speaking = False

    def update_metrics(self, metrics: SpeakingMetrics) -> SpeakingMetrics:
        """Augment video-derived metrics with audio-derived data."""
        metrics.words_per_minute = self.current_wpm or metrics.words_per_minute
        metrics.filler_words = self.total_fillers
        return metrics

    async def stop(self) -> None:
        """Stop audio processing."""
        self._active = False
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except (asyncio.CancelledError, Exception):
                pass

        # Unsubscribe from agent events
        if self._agent and hasattr(self._agent, "off"):
            try:
                self._agent.off("transcript", self._handle_agent_transcript)
                self._agent.off("user_speech_started", self._handle_speech_start)
                self._agent.off("user_speech_ended", self._handle_speech_end)
            except Exception:
                pass

        logger.info("AudioProcessor stopped")

    async def close(self) -> None:
        """Clean up all resources."""
        await self.stop()
        self._analyzer.reset()
        self._full_transcript.clear()
        logger.info("AudioProcessor closed")
