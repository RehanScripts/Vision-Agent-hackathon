"""
SpeakAI — AI Service

================================================================================
THE REAL AI PARTICIPANT — JOINS A STREAM VIDEO CALL
================================================================================

This is the core of the system. `AIService` is the class that:

  1. Creates a Vision Agent with:
       • `gemini.Realtime(fps=2)` — multimodal video+audio LLM
       • `deepgram.STT()` — real-time speech-to-text
       • `elevenlabs.TTS()` — synthesized audio responses published back
       • `SpeakingCoachProcessor` — our custom VideoProcessorPublisher
  2. Wraps the Agent in an `AgentLauncher` for lifecycle management.
  3. Starts a session → Agent joins a Stream Video call as a participant.
  4. The Agent automatically:
       • Subscribes to the human's audio + video tracks via Stream's edge.
       • Forwards video frames through VideoForwarder → our processor.
       • Processes audio via Deepgram STT.
       • Sends video+audio to Gemini Realtime for multimodal understanding.
       • Generates spoken coaching responses via ElevenLabs TTS.
       • Publishes TTS audio BACK into the call as the AI participant.
  5. Our processor fires MetricsProducedEvent with structured SpeakingMetrics.
  6. The server layer subscribes to those events and streams them to the frontend.

No WebSocket frame hacks.  No manual base64 streaming.
The Agent IS a call participant using Stream's edge media routing.
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from config import sdk_cfg, processing_cfg
from models import SpeakingMetrics, CoachingFeedback, SessionTelemetry
from speaking_coach_processor import SpeakingCoachProcessor, MetricsProducedEvent
from reasoning_engine import ReasoningEngine

logger = logging.getLogger("speakai.service")


# ═══════════════════════════════════════════════════════════════════════════
# AI Service — one per coaching session
# ═══════════════════════════════════════════════════════════════════════════

class AIService:
    """
    Manages a single AI coaching session inside a Stream Video call.

    Lifecycle:
        service = AIService(session_id, on_metrics=..., on_feedback=..., on_status=...)
        await service.start(call_id="my-call")
        # ... Agent is now live in the call, processing + responding ...
        await service.stop()
    """

    def __init__(
        self,
        session_id: str,
        on_metrics: Optional[Callable[[SpeakingMetrics], Any]] = None,
        on_feedback: Optional[Callable[[CoachingFeedback], Any]] = None,
        on_status: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.telemetry = SessionTelemetry(session_id=session_id)

        # Callbacks for streaming data to the server/frontend layer
        self._on_metrics = on_metrics
        self._on_feedback = on_feedback
        self._on_status = on_status

        # SDK objects (set during start)
        self._processor: Optional[SpeakingCoachProcessor] = None
        self._agent: Any = None
        self._launcher: Any = None
        self._agent_session: Any = None

        # Background tasks
        self._reasoning_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None
        self._metrics_listener_task: Optional[asyncio.Task] = None

        # State
        self._reasoning = ReasoningEngine()
        self._active = False
        self._started_at: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def latest_metrics(self) -> SpeakingMetrics:
        if self._processor:
            return self._processor.latest
        return SpeakingMetrics()

    # ── Start: create Agent, join call ──────────────────────────────────

    async def start(self, call_id: str, call_type: str = "default") -> Dict[str, Any]:
        """
        Boot the AI agent and join the specified Stream Video call.
        Returns session metadata dict.
        """
        self._started_at = time.time()
        self.telemetry.call_id = call_id

        try:
            from vision_agents.core import Agent, User
            from vision_agents.core.agents import AgentLauncher
            from vision_agents.plugins import gemini, getstream, deepgram, elevenlabs

            # Create our processor
            self._processor = SpeakingCoachProcessor(
                fps=processing_cfg.processor_fps
            )

            # Agent factory
            async def create_agent(**kwargs: Any) -> Agent:
                agent = Agent(
                    edge=getstream.Edge(),
                    agent_user=User(
                        name="SpeakAI Coach",
                        id=f"speakai-{self.session_id[:8]}",
                    ),
                    instructions=(
                        "You are an AI public speaking coach observing a live video feed.\n"
                        "Analyse the speaker's:\n"
                        "- Eye contact and gaze direction\n"
                        "- Head stability and movement\n"
                        "- Posture and body language\n"
                        "- Facial expressiveness\n"
                        "- Speaking pace and filler words\n\n"
                        "Provide short, actionable coaching tips (≤20 words).\n"
                        "Be encouraging but direct. Speak naturally as a coach."
                    ),
                    llm=gemini.Realtime(fps=sdk_cfg.llm_fps),
                    stt=deepgram.STT(eager_turn_detection=True),
                    tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
                    processors=[self._processor],
                )
                return agent

            # Join call handler
            async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs: Any) -> None:
                """Join the Stream Video call as an AI participant."""
                await agent.create_user()
                call = await agent.create_call(call_type, call_id)

                logger.info(f"[{self.session_id}] 🤖 Agent joining call {call_id}...")

                async with agent.join(call):
                    logger.info(f"[{self.session_id}] 🤖 Agent joined call {call_id}")
                    self._agent = agent

                    # Agent is now live — it will:
                    # • Subscribe to participant audio/video tracks automatically
                    # • Forward video frames to our SpeakingCoachProcessor
                    # • Process audio via Deepgram STT
                    # • Send video+audio to Gemini Realtime
                    # • Generate TTS audio responses via ElevenLabs
                    # • Publish TTS audio back into the call

                    await agent.finish()  # Block until call ends

            # Launch
            self._launcher = AgentLauncher(
                create_agent=create_agent,
                join_call=join_call,
                agent_idle_timeout=300.0,
            )
            await self._launcher.start()

            # Start session (this triggers create_agent → join_call)
            self._agent_session = await self._launcher.start_session(
                call_id=call_id,
                call_type=call_type,
            )
            self._agent = self._agent_session.agent

            self._active = True
            self.telemetry.sdk_active = True
            self.telemetry.multimodal_active = True
            self.telemetry.agent_joined = True

            # Start background workers
            self._reasoning_task = asyncio.create_task(
                self._reasoning_worker(), name=f"reason-{self.session_id}"
            )
            self._status_task = asyncio.create_task(
                self._status_worker(), name=f"status-{self.session_id}"
            )
            self._metrics_listener_task = asyncio.create_task(
                self._metrics_listener(), name=f"metrics-{self.session_id}"
            )

            logger.info(
                f"[{self.session_id}] ✅ AI Service started — "
                f"Agent joined call '{call_id}' as real participant"
            )

            return {
                "session_id": self.session_id,
                "call_id": call_id,
                "agent_id": f"speakai-{self.session_id[:8]}",
                "sdk_active": True,
                "multimodal_active": True,
                "mode": "live",
            }

        except ImportError as e:
            logger.warning(f"[{self.session_id}] SDK import failed: {e}")
            self.telemetry.sdk_active = False
            raise
        except Exception as e:
            logger.error(f"[{self.session_id}] AI Service start failed: {e}", exc_info=True)
            self.telemetry.sdk_active = False
            raise

    # ── Stop: tear down agent + workers ─────────────────────────────────

    async def stop(self) -> Dict[str, Any]:
        """Stop the AI service, cancel workers, tear down the agent."""
        self._active = False
        duration = time.time() - (self._started_at or time.time())

        # Cancel background tasks
        for task in [self._reasoning_task, self._status_task, self._metrics_listener_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Close the agent session via launcher
        try:
            if self._launcher is not None and self._agent_session is not None:
                await self._launcher.close_session(
                    session_id=self._agent_session.agent.id, wait=True
                )
        except Exception as e:
            logger.debug(f"[{self.session_id}] Agent session close: {e}")

        # Stop the launcher
        try:
            if self._launcher is not None:
                await self._launcher.stop()
        except Exception as e:
            logger.debug(f"[{self.session_id}] Launcher stop: {e}")

        self._agent = None
        self._launcher = None
        self._agent_session = None

        summary = {
            "duration_seconds": round(duration, 1),
            **self.telemetry.to_dict(),
        }
        logger.info(f"[{self.session_id}] AI Service stopped — {summary}")
        return summary

    # ── Metrics listener (consumes MetricsProducedEvent) ────────────────

    async def _metrics_listener(self) -> None:
        """
        Poll the processor's latest metrics and forward to the callback.
        In a production setup we'd subscribe to MetricsProducedEvent on the
        EventManager, but for simplicity we poll at processor FPS.
        """
        fps = processing_cfg.processor_fps
        interval = 1.0 / fps if fps > 0 else 0.2

        while self._active:
            try:
                await asyncio.sleep(interval)
                if not self._active or self._processor is None:
                    break

                metrics = self._processor.latest
                if metrics.timestamp > 0 and self._on_metrics:
                    self.telemetry.frames_processed = self._processor.frames_processed
                    self.telemetry.last_frame_latency_ms = self._processor.last_latency_ms
                    self.telemetry.sdk_inferences += 1

                    cb = self._on_metrics(metrics)
                    if asyncio.iscoroutine(cb):
                        await cb

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.session_id}] Metrics listener error: {e}")

    # ── Reasoning worker (decoupled cadence) ────────────────────────────

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
                    self.latest_metrics,
                    agent=self._agent,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                self.telemetry.last_reasoning_latency_ms = round(elapsed, 1)

                if fb and self._on_feedback:
                    self.telemetry.reasoning_calls += 1
                    cb = self._on_feedback(fb)
                    if asyncio.iscoroutine(cb):
                        await cb

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.session_id}] Reasoning error: {e}")

        logger.info(f"[{self.session_id}] Reasoning worker stopped")

    # ── Status broadcaster ──────────────────────────────────────────────

    async def _status_worker(self) -> None:
        """Periodically pushes system_status to the server layer."""
        while self._active:
            try:
                await asyncio.sleep(processing_cfg.status_broadcast_interval)
                if not self._active:
                    break

                status = {
                    "session_id": self.session_id,
                    "call_id": self.telemetry.call_id,
                    "sdk_active": self.telemetry.sdk_active,
                    "multimodal_active": self.telemetry.multimodal_active,
                    "agent_joined": self.telemetry.agent_joined,
                    "frames_processed": self.telemetry.frames_processed,
                    "inference_latency_ms": self.telemetry.last_frame_latency_ms,
                    "reasoning_latency_ms": self.telemetry.last_reasoning_latency_ms,
                    "source": "sdk",
                }

                if self._on_status:
                    cb = self._on_status(status)
                    if asyncio.iscoroutine(cb):
                        await cb

            except asyncio.CancelledError:
                break
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Service Registry (per-session, zero global mutable agent state)
# ═══════════════════════════════════════════════════════════════════════════

class ServiceRegistry:
    """Maps session_id → AIService. Thread-safe via asyncio."""

    def __init__(self) -> None:
        self._services: Dict[str, AIService] = {}

    def create(
        self,
        session_id: str,
        on_metrics: Optional[Callable] = None,
        on_feedback: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
    ) -> AIService:
        service = AIService(
            session_id=session_id,
            on_metrics=on_metrics,
            on_feedback=on_feedback,
            on_status=on_status,
        )
        self._services[session_id] = service
        logger.info(f"ServiceRegistry: created {session_id} (total: {len(self._services)})")
        return service

    async def stop_service(self, session_id: str) -> Optional[Dict[str, Any]]:
        service = self._services.pop(session_id, None)
        if service:
            summary = await service.stop()
            logger.info(f"ServiceRegistry: removed {session_id} (total: {len(self._services)})")
            return summary
        return None

    async def stop_all(self) -> None:
        for sid in list(self._services.keys()):
            await self.stop_service(sid)

    def get(self, session_id: str) -> Optional[AIService]:
        return self._services.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self._services)

    @property
    def all_services(self) -> Dict[str, AIService]:
        return dict(self._services)
