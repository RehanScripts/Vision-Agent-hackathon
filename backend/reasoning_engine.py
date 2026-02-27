"""
SpeakAI — Reasoning Engine

Separated from the frame-processing hot path.
Runs at a controlled cadence (every ~3 s), never per-frame.

Two layers:
  1. Threshold rules  — instant, zero-latency, always available.
  2. SDK LLM reasoning — richer insight via Vision Agent's Gemini Realtime,
     with hard async timeout + automatic fallback to rules on failure.

The reasoning worker in session.py calls `engine.evaluate()` which handles
cooldowns, timeouts, and fallback transparently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from models import SpeakingMetrics, CoachingFeedback
from config import processing_cfg

logger = logging.getLogger("speakai.reasoning")


# ---------------------------------------------------------------------------
# Rule-based feedback (threshold checks)
# ---------------------------------------------------------------------------

class ThresholdFeedback:
    """Fast rule-based coaching feedback with cooldown."""

    THRESHOLDS = dict(
        eye_low=50.0, eye_good=80.0,
        stability_low=60.0,
        posture_low=70.0,
        engagement_low=50.0,
        wpm_fast=170, wpm_slow=100,
    )

    def __init__(self, cooldown: float = processing_cfg.feedback_cooldown):
        self._cooldown = cooldown
        self._last_t: float = 0
        self._id = 0

    def evaluate(self, m: SpeakingMetrics) -> Optional[CoachingFeedback]:
        now = time.time()
        if now - self._last_t < self._cooldown:
            return None
        result = self._check(m)
        if result:
            self._last_t = now
        return result

    def _check(self, m: SpeakingMetrics) -> Optional[CoachingFeedback]:
        T = self.THRESHOLDS
        self._id += 1

        if m.eye_contact < T["eye_low"]:
            return CoachingFeedback(
                id=self._id, severity="warning", source="rules",
                headline="Low Eye Contact",
                explanation=f"Eye contact at {m.eye_contact:.0f}%.",
                tip="Look towards the camera lens.",
            )
        if m.posture_score < T["posture_low"]:
            return CoachingFeedback(
                id=self._id, severity="warning", source="rules",
                headline="Check Your Posture",
                explanation=f"Posture score {m.posture_score:.0f}%.",
                tip="Roll shoulders back and stand tall.",
            )
        if m.head_stability < T["stability_low"]:
            return CoachingFeedback(
                id=self._id, severity="info", source="rules",
                headline="Head Movement",
                explanation=f"Stability at {m.head_stability:.0f}%.",
                tip="Keep your head steady during key points.",
            )
        if m.facial_engagement < T["engagement_low"]:
            return CoachingFeedback(
                id=self._id, severity="info", source="rules",
                headline="Increase Expression",
                explanation=f"Engagement at {m.facial_engagement:.0f}%.",
                tip="Smile and vary your expressions.",
            )
        if m.words_per_minute > T["wpm_fast"]:
            return CoachingFeedback(
                id=self._id, severity="warning", source="rules",
                headline="Speaking Too Fast",
                explanation=f"Pace at {m.words_per_minute} WPM.",
                tip="Pause between key points.",
            )
        if 0 < m.words_per_minute < T["wpm_slow"]:
            return CoachingFeedback(
                id=self._id, severity="info", source="rules",
                headline="Speaking Slowly",
                explanation=f"Pace at {m.words_per_minute} WPM.",
                tip="Aim for 130–150 WPM for natural flow.",
            )
        if m.eye_contact > T["eye_good"] and m.posture_score > 85:
            return CoachingFeedback(
                id=self._id, severity="info", source="rules",
                headline="Great Presence!",
                explanation=f"Eye {m.eye_contact:.0f}%, posture {m.posture_score:.0f}%.",
                tip="Keep it up!",
            )
        self._id -= 1  # undo — nothing to report
        return None


# ---------------------------------------------------------------------------
# Combined reasoning engine (LLM + fallback)
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """
    Orchestrates coaching feedback at a controlled cadence.

    Priority:
      1. Try SDK LLM reasoning (if processor supports it).
      2. Fall back to threshold rules on timeout / failure / SDK unavailable.

    NEVER blocks the frame processing loop — the session.py reasoning worker
    calls `evaluate()` from its own asyncio.Task.
    """

    def __init__(
        self,
        cooldown: float = processing_cfg.reasoning_cooldown,
        timeout: float = processing_cfg.reasoning_timeout,
    ):
        self._cooldown = cooldown
        self._timeout = timeout
        self._last_call: float = 0
        self._threshold_fb = ThresholdFeedback()

    async def evaluate(
        self,
        metrics: SpeakingMetrics,
        sdk_processor: Any,  # SDKVisionProcessor (if active)
    ) -> Optional[CoachingFeedback]:
        """
        Produce coaching feedback.  Rate-limited by cooldown.
        Returns None if cooldown hasn't elapsed.
        """
        now = time.time()
        if now - self._last_call < self._cooldown:
            return None
        self._last_call = now

        # Attempt LLM reasoning via SDK
        if sdk_processor is not None and sdk_processor.is_active:
            try:
                text = await asyncio.wait_for(
                    sdk_processor.get_llm_insight(metrics),
                    timeout=self._timeout,
                )
                if text:
                    return CoachingFeedback(
                        id=int(time.time() * 1000) % 100_000,
                        severity="info",
                        headline="AI Coach Insight",
                        explanation=text,
                        tip=None,
                        source="llm",
                    )
            except asyncio.TimeoutError:
                logger.warning("LLM reasoning timed out — falling back to rules")
            except Exception as e:
                logger.warning(f"LLM reasoning failed: {e} — falling back to rules")

        # Fallback to threshold rules
        return self._threshold_fb.evaluate(metrics)
