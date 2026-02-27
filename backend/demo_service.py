"""
SpeakAI — Demo Service

Fallback service for when SDK is unavailable (missing API keys / import errors).
Streams simulated metrics at ~5 FPS so the frontend always works.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from models import SpeakingMetrics, CoachingFeedback, SessionTelemetry
from reasoning_engine import ThresholdFeedback
from config import processing_cfg

logger = logging.getLogger("speakai.demo")


class DemoService:
    """
    Lightweight fallback that streams simulated metrics.
    No SDK, no Stream call, no API keys needed.
    """

    def __init__(
        self,
        session_id: str,
        on_metrics: Optional[Callable] = None,
        on_feedback: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
    ) -> None:
        self.session_id = session_id
        self.telemetry = SessionTelemetry(session_id=session_id)

        self._on_metrics = on_metrics
        self._on_feedback = on_feedback
        self._on_status = on_status

        self._active = False
        self._demo_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None
        self._started_at: Optional[float] = None
        self._latest_metrics = SpeakingMetrics(source="simulated")

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def latest_metrics(self) -> SpeakingMetrics:
        return self._latest_metrics

    async def start(self, call_id: str = "demo") -> Dict[str, Any]:
        self._active = True
        self._started_at = time.time()
        self.telemetry.call_id = call_id

        self._demo_task = asyncio.create_task(
            self._demo_worker(), name=f"demo-{self.session_id}"
        )
        self._status_task = asyncio.create_task(
            self._status_worker(), name=f"demo-status-{self.session_id}"
        )

        logger.info(f"[{self.session_id}] Demo service started")
        return {
            "session_id": self.session_id,
            "call_id": call_id,
            "sdk_active": False,
            "multimodal_active": False,
            "mode": "demo",
        }

    async def stop(self) -> Dict[str, Any]:
        self._active = False
        duration = time.time() - (self._started_at or time.time())

        for task in [self._demo_task, self._status_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        summary = {"duration_seconds": round(duration, 1), **self.telemetry.to_dict()}
        logger.info(f"[{self.session_id}] Demo service stopped")
        return summary

    async def _demo_worker(self) -> None:
        fb_engine = ThresholdFeedback()
        import random

        while self._active:
            try:
                t = time.time()
                metrics = SpeakingMetrics(
                    eye_contact=round(75 + 20 * np.sin(t * 0.5) + random.uniform(-3, 3), 1),
                    head_stability=round(85 + 10 * np.sin(t * 0.3) + random.uniform(-2, 2), 1),
                    posture_score=round(88 + 8 * np.sin(t * 0.2) + random.uniform(-2, 2), 1),
                    facial_engagement=round(70 + 15 * np.sin(t * 0.4) + random.uniform(-3, 3), 1),
                    attention_intensity=round(80 + 12 * np.sin(t * 0.35) + random.uniform(-2, 2), 1),
                    words_per_minute=int(130 + 20 * np.sin(t * 0.1) + random.uniform(-5, 5)),
                    filler_words=max(0, int(3 + 2 * np.sin(t * 0.2) + random.uniform(-1, 1))),
                    timestamp=t,
                    source="simulated",
                )
                self._latest_metrics = metrics
                self.telemetry.frames_processed += 1

                if self._on_metrics:
                    cb = self._on_metrics(metrics)
                    if asyncio.iscoroutine(cb):
                        await cb

                fb = fb_engine.evaluate(metrics)
                if fb and self._on_feedback:
                    cb = self._on_feedback(fb)
                    if asyncio.iscoroutine(cb):
                        await cb

                await asyncio.sleep(0.2)  # ~5 FPS
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.session_id}] Demo error: {e}")

    async def _status_worker(self) -> None:
        while self._active:
            try:
                await asyncio.sleep(processing_cfg.status_broadcast_interval)
                if not self._active:
                    break

                status = {
                    "session_id": self.session_id,
                    "sdk_active": False,
                    "multimodal_active": False,
                    "agent_joined": False,
                    "frames_processed": self.telemetry.frames_processed,
                    "source": "simulated",
                }

                if self._on_status:
                    cb = self._on_status(status)
                    if asyncio.iscoroutine(cb):
                        await cb

            except asyncio.CancelledError:
                break
            except Exception:
                pass
