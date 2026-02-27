"""
SpeakAI — Session

================================================================================
PER-CONNECTION SESSION OBJECT — ZERO GLOBAL MUTABLE STATE
================================================================================

Each WebSocket connection gets exactly one `CoachSession`.
A session owns:
  • One vision processor (SDK primary, MediaPipe fallback)
  • One bounded frame queue (backpressure: drop-oldest, maxsize=3)
  • One frame-processing asyncio.Task
  • One reasoning asyncio.Task (decoupled cadence, 2-3 s)
  • One status-broadcast asyncio.Task
  • Per-session telemetry counters

SessionManager is a thin dict-based registry — it stores sessions by ID,
creates them on connect, and removes + cleans them on disconnect.

Frame flow:
  WebSocket → enqueue_frame() → BoundedQueue → _frame_worker() → VisionProcessor
                                                       ↓
                                              metrics → WS "metrics"
                                                       ↓ (cached)
                                 _reasoning_worker() → ReasoningEngine → WS "feedback"

All CPU-heavy work (SDK inference or MediaPipe CV) runs inside the
processor — either via SDK async pipeline or via ThreadPoolExecutor.
The event loop is NEVER blocked.
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from models import SpeakingMetrics, SessionTelemetry
from config import processing_cfg
from vision_processor import SDKVisionProcessor, FallbackCVProcessor
from reasoning_engine import ReasoningEngine

logger = logging.getLogger("speakai.session")


# ---------------------------------------------------------------------------
# Per-WebSocket Session
# ---------------------------------------------------------------------------

class CoachSession:
    """
    Encapsulates everything for a single coaching session.
    No global mutable state — each WS connection gets one of these.
    """

    def __init__(self, session_id: str, ws: Any) -> None:
        self.session_id = session_id
        self.ws = ws
        self.telemetry = SessionTelemetry(session_id=session_id)

        # Processors — selected at start()
        self._sdk_processor: Optional[SDKVisionProcessor] = None
        self._fallback_processor: Optional[FallbackCVProcessor] = None

        # Reasoning (decoupled from frame rate)
        self._reasoning = ReasoningEngine()

        # Bounded frame queue (backpressure)
        self._frame_queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=processing_cfg.frame_queue_max
        )

        # Background tasks
        self._frame_task: Optional[asyncio.Task] = None
        self._reasoning_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None
        self._demo_task: Optional[asyncio.Task] = None

        # Latest metrics (for reasoning worker to read)
        self._latest_metrics = SpeakingMetrics()
        self._active = False
        self._mode: str = "idle"
        self._started_at: Optional[float] = None

        logger.info(f"[{session_id}] Session created")

    # ── Session lifecycle ───────────────────────────────────────────────

    async def start(self, mode: str = "live") -> Dict[str, Any]:
        """
        Initialise processors and start background workers.

        Tries SDK first.  If it fails → falls back to MediaPipe CV.
        Returns a dict with session metadata.
        """
        self._active = True
        self._mode = mode
        self._started_at = time.time()

        if mode == "demo":
            # Demo mode: simulated metrics, no SDK needed
            self._fallback_processor = FallbackCVProcessor(self.session_id)
            self._fallback_processor.initialize()
            self._demo_task = asyncio.create_task(
                self._demo_worker(), name=f"demo-{self.session_id}"
            )
            self._status_task = asyncio.create_task(
                self._status_worker(), name=f"status-{self.session_id}"
            )
            logger.info(f"[{self.session_id}] Demo mode started")
            return {"session_id": self.session_id, "mode": "demo", "sdk_active": False}

        # Live mode — try SDK first
        sdk_ok = False
        try:
            self._sdk_processor = SDKVisionProcessor(self.session_id)
            await self._sdk_processor.start()
            sdk_ok = True
            self.telemetry.sdk_active = True
            self.telemetry.multimodal_active = True
            logger.info(f"[{self.session_id}] SDK processor active (primary path)")
        except Exception as e:
            logger.warning(
                f"[{self.session_id}] SDK init failed ({e}) — activating MediaPipe fallback"
            )
            self._sdk_processor = None
            self._fallback_processor = FallbackCVProcessor(self.session_id)
            self._fallback_processor.initialize()
            self.telemetry.sdk_active = False
            self.telemetry.multimodal_active = False

        # Start workers
        self._frame_task = asyncio.create_task(
            self._frame_worker(), name=f"frames-{self.session_id}"
        )
        self._reasoning_task = asyncio.create_task(
            self._reasoning_worker(), name=f"reason-{self.session_id}"
        )
        self._status_task = asyncio.create_task(
            self._status_worker(), name=f"status-{self.session_id}"
        )

        return {
            "session_id": self.session_id,
            "mode": "live",
            "sdk_active": sdk_ok,
        }

    async def stop(self) -> Dict[str, Any]:
        """Stop all workers and return session summary."""
        self._active = False
        duration = time.time() - (self._started_at or time.time())

        # Cancel all tasks
        for task in [self._frame_task, self._reasoning_task, self._status_task, self._demo_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        summary = {
            "duration_seconds": round(duration, 1),
            **self.telemetry.to_dict(),
        }
        logger.info(f"[{self.session_id}] Session stopped — {self.telemetry.to_dict()}")
        return summary

    async def close(self) -> None:
        """Full teardown: stop + close processors."""
        if self._active:
            await self.stop()

        if self._sdk_processor:
            await self._sdk_processor.close()
            self._sdk_processor = None

        if self._fallback_processor:
            self._fallback_processor.close()
            self._fallback_processor = None

        logger.info(f"[{self.session_id}] Session closed and cleaned up")

    # ── Frame ingest (WS handler calls this — NEVER blocks) ────────────

    async def enqueue_frame(self, base64_jpeg: str) -> None:
        """Push a frame into the bounded queue. Drop oldest if full."""
        self.telemetry.frames_received += 1

        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
                self.telemetry.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass

        try:
            self._frame_queue.put_nowait(base64_jpeg)
        except asyncio.QueueFull:
            self.telemetry.frames_dropped += 1

    # ── Frame processing worker ────────────────────────────────────────

    async def _frame_worker(self) -> None:
        """
        Drains the frame queue and runs vision analysis.
        This is the HOT PATH — frames flow here continuously.
        """
        logger.info(f"[{self.session_id}] Frame worker started")

        while self._active:
            try:
                # Wait for frame with timeout so we can check _active
                try:
                    b64_frame = await asyncio.wait_for(
                        self._frame_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                t0 = time.perf_counter()

                # ── PRIMARY PATH: SDK ──
                if self._sdk_processor and self._sdk_processor.is_active:
                    metrics = await self._sdk_processor.ingest_frame(b64_frame)
                    if metrics is None:
                        # SDK returned nothing — use latest or skip
                        metrics = self._sdk_processor.latest
                    self.telemetry.sdk_inferences += 1
                    self.telemetry.last_sdk_latency_ms = self._sdk_processor.last_inference_ms

                # ── FALLBACK PATH: MediaPipe ──
                elif self._fallback_processor:
                    metrics = await self._fallback_processor.analyze_frame(b64_frame)
                    if metrics is None:
                        continue
                else:
                    continue

                elapsed_ms = (time.perf_counter() - t0) * 1000
                self.telemetry.frames_processed += 1
                self.telemetry.last_frame_latency_ms = round(elapsed_ms, 1)

                self._latest_metrics = metrics

                # Send metrics to frontend
                await self._send({
                    "type": "metrics",
                    "data": metrics.to_dict(),
                })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.session_id}] Frame worker error: {e}", exc_info=True)

        logger.info(f"[{self.session_id}] Frame worker stopped")

    # ── Reasoning worker (decoupled cadence) ───────────────────────────

    async def _reasoning_worker(self) -> None:
        """
        Runs at ~3 s intervals. Calls LLM (with timeout) or falls back to rules.
        NEVER runs per-frame.
        """
        logger.info(f"[{self.session_id}] Reasoning worker started")

        while self._active:
            try:
                await asyncio.sleep(processing_cfg.reasoning_cooldown)
                if not self._active:
                    break

                t0 = time.perf_counter()
                fb = await self._reasoning.evaluate(
                    self._latest_metrics, self._sdk_processor
                )
                elapsed = (time.perf_counter() - t0) * 1000
                self.telemetry.last_reasoning_latency_ms = round(elapsed, 1)

                if fb:
                    self.telemetry.reasoning_calls += 1
                    await self._send({
                        "type": "feedback",
                        "data": fb.to_dict(),
                    })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.session_id}] Reasoning error: {e}")

        logger.info(f"[{self.session_id}] Reasoning worker stopped")

    # ── System status broadcaster ──────────────────────────────────────

    async def _status_worker(self) -> None:
        """Periodically sends system_status debug message to frontend."""
        while self._active:
            try:
                await asyncio.sleep(processing_cfg.status_broadcast_interval)
                if not self._active:
                    break

                processor = self._sdk_processor or self._fallback_processor
                await self._send({
                    "type": "system_status",
                    "payload": {
                        "session_id": self.session_id,
                        "sdk_active": self.telemetry.sdk_active,
                        "multimodal_active": self.telemetry.multimodal_active,
                        "inference_latency_ms": self.telemetry.last_frame_latency_ms,
                        "reasoning_latency_ms": self.telemetry.last_reasoning_latency_ms,
                        "frames_processed": self.telemetry.frames_processed,
                        "frames_dropped": self.telemetry.frames_dropped,
                        "processor_source": (
                            "sdk" if self.telemetry.sdk_active
                            else "mediapipe_fallback" if self._fallback_processor
                            else "none"
                        ),
                    },
                })
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    # ── Demo worker (simulated metrics) ────────────────────────────────

    async def _demo_worker(self) -> None:
        """Streams simulated metrics at ~5 FPS for demo without camera."""
        logger.info(f"[{self.session_id}] Demo worker started")
        fb_engine = ReasoningEngine()

        while self._active:
            try:
                if self._fallback_processor:
                    metrics = self._fallback_processor._simulated()
                else:
                    metrics = SpeakingMetrics(source="simulated")

                self._latest_metrics = metrics
                self.telemetry.frames_processed += 1

                await self._send({"type": "metrics", "data": metrics.to_dict()})

                fb = fb_engine._threshold_fb.evaluate(metrics)
                if fb:
                    await self._send({"type": "feedback", "data": fb.to_dict()})

                await asyncio.sleep(0.2)  # ~5 FPS
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.session_id}] Demo error: {e}")

        logger.info(f"[{self.session_id}] Demo worker stopped")

    # ── WebSocket send helper ──────────────────────────────────────────

    async def _send(self, data: Dict[str, Any]) -> None:
        """Send JSON to client. Never crashes on closed socket."""
        try:
            await self.ws.send_text(json.dumps(data))
        except Exception:
            pass  # client disconnected; outer handler will clean up


# ---------------------------------------------------------------------------
# Session Manager (thin registry)
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Maps session_id → CoachSession.
    No global mutable agent state — each session is fully isolated.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, CoachSession] = {}

    async def create_session(self, ws: Any) -> CoachSession:
        session_id = uuid.uuid4().hex[:12]
        session = CoachSession(session_id=session_id, ws=ws)
        self._sessions[session_id] = session
        logger.info(
            f"SessionManager: created {session_id} (total: {len(self._sessions)})"
        )
        return session

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()
            logger.info(
                f"SessionManager: removed {session_id} (total: {len(self._sessions)})"
            )

    async def close_all(self) -> None:
        for sid in list(self._sessions.keys()):
            await self.close_session(sid)

    def get_session(self, session_id: str) -> Optional[CoachSession]:
        return self._sessions.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def all_sessions(self) -> Dict[str, CoachSession]:
        return dict(self._sessions)
