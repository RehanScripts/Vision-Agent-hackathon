"""
SpeakAI — AI Service

================================================================================
THE REAL AI PARTICIPANT — JOINS A STREAM VIDEO CALL & COMMUNICATES
================================================================================

This is the core of the system. `AIService` is the class that:

  1. Creates a Vision Agent with:
       • `gemini.Realtime(fps=2)` — multimodal video+audio LLM
       • `elevenlabs.TTS()` — synthesized audio responses published back
       • `SpeakingCoachProcessor` — custom VideoProcessorPublisher
  2. Wraps the Agent in an `AgentLauncher` for lifecycle management.
  3. Starts a session → Agent joins a Stream Video call as a participant.
  4. The Agent automatically:
       • Subscribes to the human's audio + video tracks via Stream's edge.
       • Forwards video frames through VideoForwarder → our processor.
       • Sends video+audio to Gemini Realtime for multimodal understanding.
       • Generates spoken coaching responses via ElevenLabs TTS.
       • Publishes TTS audio BACK into the call as the AI participant.
  5. AudioProcessor extracts speech metrics + transcript from audio tracks.
  6. ConversationManager orchestrates bidirectional dialogue.
  7. Structured metrics, feedback, transcript, and chat are streamed to frontend.

No WebSocket frame hacks.  No manual base64 streaming.
The Agent IS a call participant using Stream's edge media routing.
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from ..core.config import sdk_cfg, processing_cfg, conversation_cfg
from ..core.models import (
    SpeakingMetrics,
    CoachingFeedback,
    SessionTelemetry,
    ChatMessage,
    TranscriptEntry,
    ConversationState,
)
from ..processing.processor import SpeakingCoachProcessor
from ..processing.reasoning import ReasoningEngine
from ..processing.audio_processor import AudioProcessor
from ..processing.conversation import ConversationManager

logger = logging.getLogger("speakai.service")


# ═══════════════════════════════════════════════════════════════════════════
# AI Service — one per coaching session
# ═══════════════════════════════════════════════════════════════════════════

class AIService:
    """
    Manages a single AI coaching + communication session inside a Stream Video call.

    Lifecycle:
        service = AIService(session_id, on_metrics=..., on_feedback=..., on_status=..., ...)
        await service.start(call_id="my-call")
        # ... Agent is now live in the call, processing + communicating ...
        await service.send_chat("Hello!")  # text input from frontend
        await service.stop()
    """

    def __init__(
        self,
        session_id: str,
        on_metrics: Optional[Callable[[SpeakingMetrics], Any]] = None,
        on_feedback: Optional[Callable[[CoachingFeedback], Any]] = None,
        on_status: Optional[Callable[[Dict[str, Any]], Any]] = None,
        on_chat: Optional[Callable[[ChatMessage], Any]] = None,
        on_transcript: Optional[Callable[[TranscriptEntry], Any]] = None,
        on_conversation_state: Optional[Callable[[ConversationState], Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.telemetry = SessionTelemetry(session_id=session_id)

        # Callbacks for streaming data to the server/frontend layer
        self._on_metrics = on_metrics
        self._on_feedback = on_feedback
        self._on_status = on_status
        self._on_chat = on_chat
        self._on_transcript = on_transcript
        self._on_conversation_state = on_conversation_state

        # SDK objects (set during start)
        self._processor: Optional[SpeakingCoachProcessor] = None
        self._audio_processor: Optional[AudioProcessor] = None
        self._conversation: Optional[ConversationManager] = None
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
        self._last_metrics_timestamp: float = 0.0
        self._join_ready = asyncio.Event()
        self._join_error: Optional[str] = None
        self._pending_chat: deque[str] = deque(maxlen=50)
        self._starting = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def latest_metrics(self) -> SpeakingMetrics:
        if self._processor:
            return self._processor.latest
        return SpeakingMetrics()

    @property
    def conversation_messages(self) -> List[ChatMessage]:
        """Get the full conversation history."""
        if self._conversation:
            return self._conversation.messages
        return []

    @property
    def transcript(self) -> List[TranscriptEntry]:
        """Get the full audio transcript."""
        if self._audio_processor:
            return self._audio_processor.transcript
        return []

    # ── Public API: send text chat from frontend ────────────────────────

    async def send_chat(self, text: str) -> None:
        """
        Handle a text message from the user (sent via WebSocket).
        Forwards to the ConversationManager which generates a response.
        """
        text = text.strip()
        if not text:
            return

        if self._conversation:
            await self._conversation.handle_user_text(text)
            return

        self._pending_chat.append(text)

    # ── Start: create Agent, join call ──────────────────────────────────

    async def start(self, call_id: str, call_type: str = "default") -> Dict[str, Any]:
        """
        Boot the AI agent and join the specified Stream Video call.
        Sets up audio processing and conversation management.
        Returns session metadata dict.
        """
        self._started_at = time.time()
        self._last_metrics_timestamp = 0.0
        self._join_ready = asyncio.Event()
        self._join_error = None
        self._starting = True
        self.telemetry.call_id = call_id

        try:
            from vision_agents.core import Agent, User
            from vision_agents.core.agents import AgentLauncher
            from vision_agents.plugins import gemini, getstream, elevenlabs

            # ── Create processors ───────────────────────────────────────
            self._processor = SpeakingCoachProcessor(
                fps=processing_cfg.processor_fps
            )

            # Audio processor — listens to call audio, extracts transcript + metrics
            self._audio_processor = AudioProcessor(
                on_transcript=self._handle_transcript,
                on_speech_state=self._handle_speech_state,
            )

            # Conversation manager — orchestrates bidirectional dialogue
            self._conversation = ConversationManager(
                session_id=self.session_id,
                on_chat_message=self._handle_chat_message,
                on_state_change=self._handle_conversation_state,
                on_agent_speech=self._handle_agent_speech,
            )

            # Flush any chat messages received while the service was starting
            while self._pending_chat:
                queued_text = self._pending_chat.popleft()
                await self._conversation.handle_user_text(queued_text)

            # ── Agent factory ───────────────────────────────────────────
            async def create_agent(**kwargs: Any) -> Agent:
                agent = Agent(
                    edge=getstream.Edge(),
                    agent_user=User(
                        name="SpeakAI Coach",
                        id=f"speakai-{self.session_id[:8]}",
                    ),
                    instructions=conversation_cfg.system_prompt,
                    llm=gemini.Realtime(fps=sdk_cfg.llm_fps),
                    tts=elevenlabs.TTS(model_id="eleven_turbo_v2_5"),
                    processors=[self._processor],
                )
                return agent

            # ── Join call handler ───────────────────────────────────────
            async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs: Any) -> None:
                """Join the Stream Video call as an AI participant."""
                try:
                    await agent.create_user()
                    call = await agent.create_call(call_type, call_id)

                    logger.info(f"[{self.session_id}] 🤖 Agent joining call {call_id}...")

                    async with agent.join(call):
                        logger.info(f"[{self.session_id}] 🤖 Agent joined call {call_id}")
                        self._agent = agent

                        # Attach agent to sub-processors
                        self._audio_processor.attach_agent(agent)
                        self._conversation.attach_agent(agent)

                        # Subscribe to audio tracks for processing
                        if hasattr(agent, "on"):
                            try:
                                agent.on("track_subscribed", self._on_track_subscribed)
                                agent.on("participant_joined", self._on_participant_joined)
                                agent.on("participant_left", self._on_participant_left)
                            except Exception as e:
                                logger.debug(f"Agent event subscription: {e}")

                        # Start conversation manager
                        await self._conversation.start()

                        self._join_ready.set()

                        # Agent is now live — it will:
                        # • Subscribe to participant audio/video tracks automatically
                        # • Forward video frames to our SpeakingCoachProcessor
                        # • Forward audio to Gemini Realtime for multimodal understanding
                        # • Our AudioProcessor extracts speech metrics + transcript
                        # • ConversationManager handles turn-taking + responses
                        # • Generate TTS audio responses via ElevenLabs
                        # • Publish TTS audio back into the call

                        await agent.finish()  # Block until call ends
                except Exception as exc:
                    self._join_error = str(exc)
                    self._join_ready.set()
                    raise

            # ── Launch ──────────────────────────────────────────────────
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

            try:
                await asyncio.wait_for(self._join_ready.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                raise RuntimeError("Agent join timed out")

            if self._join_error:
                raise RuntimeError(self._join_error)

            self._active = True
            self._starting = False
            self.telemetry.sdk_active = True
            self.telemetry.multimodal_active = True
            self.telemetry.agent_joined = True
            self.telemetry.audio_active = True

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
                f"Agent joined call '{call_id}' as real participant with communication"
            )

            return {
                "session_id": self.session_id,
                "call_id": call_id,
                "agent_id": f"speakai-{self.session_id[:8]}",
                "sdk_active": True,
                "multimodal_active": True,
                "audio_active": True,
                "conversation_active": True,
                "mode": "live",
            }

        except ImportError as e:
            logger.warning(f"[{self.session_id}] SDK import failed: {e}")
            self.telemetry.sdk_active = False
            self._starting = False
            raise
        except Exception as e:
            logger.error(f"[{self.session_id}] AI Service start failed: {e}", exc_info=True)
            self.telemetry.sdk_active = False
            self._starting = False
            raise

    # ── Track subscription (audio + video) ──────────────────────────────

    async def _on_track_subscribed(self, event: Any) -> None:
        """Called when a media track from a participant is subscribed."""
        try:
            track = getattr(event, "track", None)
            participant_id = getattr(event, "participant_id", None)
            track_kind = getattr(track, "kind", "unknown")

            if track_kind == "audio" and self._audio_processor:
                logger.info(
                    f"[{self.session_id}] Audio track subscribed: {participant_id}"
                )
                await self._audio_processor.process_audio(track, participant_id)

            elif track_kind == "video":
                logger.info(
                    f"[{self.session_id}] Video track subscribed: {participant_id}"
                )
                # Video is already handled by SpeakingCoachProcessor via SDK

        except Exception as e:
            logger.debug(f"[{self.session_id}] Track subscription handler: {e}")

    async def _on_participant_joined(self, event: Any = None) -> None:
        """Log when a participant joins the call."""
        pid = getattr(event, "participant_id", "unknown") if event else "unknown"
        logger.info(f"[{self.session_id}] Participant joined: {pid}")

    async def _on_participant_left(self, event: Any = None) -> None:
        """Handle participant leaving the call."""
        pid = getattr(event, "participant_id", "unknown") if event else "unknown"
        logger.info(f"[{self.session_id}] Participant left: {pid}")

    # ── Communication event handlers ────────────────────────────────────

    async def _handle_transcript(self, entry: TranscriptEntry) -> None:
        """Forward transcript entry to frontend + conversation manager."""
        self.telemetry.transcript_entries += 1

        # Forward to conversation manager for dialogue handling
        if self._conversation:
            await self._conversation.handle_transcript(entry)

        # Forward to frontend
        if self._on_transcript:
            cb = self._on_transcript(entry)
            if asyncio.iscoroutine(cb):
                await cb

    async def _handle_chat_message(self, msg: ChatMessage) -> None:
        """Forward chat message to frontend."""
        self.telemetry.chat_messages += 1

        if self._on_chat:
            cb = self._on_chat(msg)
            if asyncio.iscoroutine(cb):
                await cb

    async def _handle_conversation_state(self, state: ConversationState) -> None:
        """Forward conversation state to frontend."""
        self.telemetry.conversation_turns = state.turn_count

        if self._on_conversation_state:
            cb = self._on_conversation_state(state)
            if asyncio.iscoroutine(cb):
                await cb

    async def _handle_speech_state(self, state: Dict[str, Any]) -> None:
        """Handle speech state changes from audio processor."""
        # Update conversation manager with current user speaking state
        if self._conversation:
            self._conversation._state.is_user_speaking = state.get("is_user_speaking", False)

    async def _handle_agent_speech(self, text: str) -> None:
        """
        Called when the ConversationManager wants the agent to speak.
        The Agent's TTS module handles synthesis + publishing into the call.
        """
        if self._agent and hasattr(self._agent, "tts"):
            try:
                # The Vision Agent's TTS plugin synthesizes and publishes
                # audio directly into the Stream Video call
                await self._agent.tts.say(text)
                logger.info(f"[{self.session_id}] Agent spoke: {text[:60]}...")
            except Exception as e:
                logger.warning(f"[{self.session_id}] TTS failed: {e}")
        elif self._agent and hasattr(self._agent, "say"):
            try:
                await self._agent.say(text)
            except Exception as e:
                logger.warning(f"[{self.session_id}] Agent say failed: {e}")

    # ── Stop: tear down agent + workers ─────────────────────────────────

    async def stop(self) -> Dict[str, Any]:
        """Stop the AI service, cancel workers, tear down the agent."""
        self._active = False
        self._starting = False
        duration = time.time() - (self._started_at or time.time())

        # Stop conversation manager
        conversation_summary = {}
        if self._conversation:
            conversation_summary = await self._conversation.stop()

        # Stop audio processor
        if self._audio_processor:
            await self._audio_processor.close()

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
        self._pending_chat.clear()

        summary = {
            "duration_seconds": round(duration, 1),
            **self.telemetry.to_dict(),
            "conversation": conversation_summary,
        }
        logger.info(f"[{self.session_id}] AI Service stopped — {summary}")
        return summary

    # ── Metrics listener (consumes MetricsProducedEvent) ────────────────

    async def _metrics_listener(self) -> None:
        """
        Poll the processor's latest metrics and forward to the callback.
        Augments video metrics with audio-derived data.
        """
        fps = processing_cfg.processor_fps
        interval = 1.0 / fps if fps > 0 else 0.2

        while self._active:
            try:
                await asyncio.sleep(interval)
                if not self._active or self._processor is None:
                    break

                metrics = self._processor.latest

                # Augment with audio metrics (WPM, filler words)
                if self._audio_processor:
                    metrics = self._audio_processor.update_metrics(metrics)

                # Feed metrics to conversation manager for visual context
                if self._conversation:
                    self._conversation.update_metrics(metrics)

                if (
                    metrics.timestamp > 0
                    and metrics.timestamp != self._last_metrics_timestamp
                    and self._on_metrics
                ):
                    self._last_metrics_timestamp = metrics.timestamp
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
        last_reasoning_metrics_ts = 0.0

        while self._active:
            try:
                await asyncio.sleep(processing_cfg.reasoning_cooldown)
                if not self._active:
                    break

                metrics = self.latest_metrics
                if metrics.timestamp <= 0:
                    continue

                if metrics.timestamp == last_reasoning_metrics_ts:
                    continue

                last_reasoning_metrics_ts = metrics.timestamp

                t0 = time.perf_counter()
                fb = await self._reasoning.evaluate(
                    metrics,
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

                conversation_state = (
                    self._conversation.state.to_dict() if self._conversation else {}
                )

                status = {
                    "session_id": self.session_id,
                    "call_id": self.telemetry.call_id,
                    "sdk_active": self.telemetry.sdk_active,
                    "multimodal_active": self.telemetry.multimodal_active,
                    "agent_joined": self.telemetry.agent_joined,
                    "audio_active": self.telemetry.audio_active,
                    "frames_processed": self.telemetry.frames_processed,
                    "inference_latency_ms": self.telemetry.last_frame_latency_ms,
                    "reasoning_latency_ms": self.telemetry.last_reasoning_latency_ms,
                    "transcript_entries": self.telemetry.transcript_entries,
                    "chat_messages": self.telemetry.chat_messages,
                    "conversation_turns": self.telemetry.conversation_turns,
                    "conversation": conversation_state,
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
