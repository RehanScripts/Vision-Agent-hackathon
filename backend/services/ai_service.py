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

from ..core.config import sdk_cfg, processing_cfg, conversation_cfg, performance_cfg
from ..core.models import (
    SpeakingMetrics,
    CoachingFeedback,
    SessionTelemetry,
    ChatMessage,
    TranscriptEntry,
    ConversationState,
)
from ..core.state_machine import SessionStateMachine, SessionState, SessionMode
from ..core.health import SessionPolicy
from ..core.latency import LatencyTracer
from ..processing.processor import SpeakingCoachProcessor
from ..processing.reasoning import ReasoningEngine
from ..processing.audio_processor import AudioProcessor
from ..processing.conversation import ConversationManager

logger = logging.getLogger("speakai.service")

# SDK event types (imported lazily at runtime)
_sdk_event_types_loaded = False
_RealtimeUserSpeechTranscriptionEvent = None
_RealtimeAgentSpeechTranscriptionEvent = None
_TrackAddedEvent = None
_TrackRemovedEvent = None


def _load_sdk_event_types() -> None:
    """Lazy-load SDK event types to avoid import errors."""
    global _sdk_event_types_loaded
    global _RealtimeUserSpeechTranscriptionEvent
    global _RealtimeAgentSpeechTranscriptionEvent
    global _TrackAddedEvent, _TrackRemovedEvent

    if _sdk_event_types_loaded:
        return

    try:
        from vision_agents.core.llm.events import (
            RealtimeUserSpeechTranscriptionEvent,
            RealtimeAgentSpeechTranscriptionEvent,
        )
        _RealtimeUserSpeechTranscriptionEvent = RealtimeUserSpeechTranscriptionEvent
        _RealtimeAgentSpeechTranscriptionEvent = RealtimeAgentSpeechTranscriptionEvent
    except ImportError:
        pass

    try:
        from vision_agents.core.edge.events import TrackAddedEvent, TrackRemovedEvent
        _TrackAddedEvent = TrackAddedEvent
        _TrackRemovedEvent = TrackRemovedEvent
    except ImportError:
        pass

    _sdk_event_types_loaded = True


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

        # State machine + policy layer + latency tracer
        self._state_machine = SessionStateMachine(
            on_transition=self._on_state_transition,
        )
        self._policy = SessionPolicy()
        self._latency = LatencyTracer(session_id)

        # Legacy state (kept for backward compat during migration)
        self._reasoning = ReasoningEngine()
        self._active = False
        self._started_at: Optional[float] = None
        self._last_metrics_timestamp: float = 0.0
        self._join_ready = asyncio.Event()
        self._join_error: Optional[str] = None
        self._pending_chat: deque[str] = deque(maxlen=50)
        self._starting = False
        self._latest_emitted_metrics: Optional[SpeakingMetrics] = None

        # Pre-warm state
        self._llm_warmed = False
        self._tts_warmed = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def latest_metrics(self) -> SpeakingMetrics:
        if self._processor and self._processor.frames_processed > 0:
            return self._processor.latest
        # In audio-only mode, return the last emitted audio-only metrics
        if self._latest_emitted_metrics and self._latest_emitted_metrics.timestamp > 0:
            return self._latest_emitted_metrics
        if self._processor:
            return self._processor.latest
        return SpeakingMetrics()

    # ── State machine callback ──────────────────────────────────────────

    def _on_state_transition(
        self, prev: SessionState, new: SessionState, reason: str
    ) -> None:
        """Called on every state transition — updates telemetry and emits to frontend."""
        self.telemetry.session_state = new.value
        self.telemetry.session_mode = self._state_machine.mode.value

        # Propagate mode to conversation manager for context
        if self._conversation:
            self._conversation.set_session_mode(self._state_machine.mode.value)

        # Emit state change to frontend
        if self._on_status:
            import asyncio as _aio
            try:
                loop = _aio.get_running_loop()
                loop.create_task(self._emit_state_change(prev, new, reason))
            except RuntimeError:
                pass  # No event loop — skip

    async def _emit_state_change(
        self, prev: SessionState, new: SessionState, reason: str
    ) -> None:
        """Push state transition to frontend via status callback."""
        status = {
            "type": "state_transition",
            "session_state": new.value,
            "session_mode": self._state_machine.mode.value,
            "previous_state": prev.value,
            "reason": reason,
            "health": self._policy.check_health().to_dict(),
            "latency": self._latency.summary(),
        }
        if self._on_status:
            cb = self._on_status(status)
            if asyncio.iscoroutine(cb):
                await cb

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

        # ── LATENCY: mark join start ────────────────────────────────────
        self._latency.mark_join_started()

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
                    llm=gemini.Realtime(
                        fps=sdk_cfg.llm_fps,
                        api_key=sdk_cfg.gemini_api_key or None,
                        config={"temperature": performance_cfg.llm_temperature},
                    ),
                    tts=elevenlabs.TTS(
                        model_id=performance_cfg.tts_model,
                        api_key=sdk_cfg.elevenlabs_api_key or None,
                    ),
                    processors=[self._processor],
                )
                return agent

            # ── Join call handler ───────────────────────────────────────
            async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs: Any) -> None:
                """Join the Stream Video call as an AI participant."""
                try:
                    await agent.create_user()
                    call = await agent.create_call(call_type, call_id)

                    logger.info(f"[{self.session_id}] Agent joining call {call_id}...")

                    async with agent.join(call, participant_wait_timeout=0):
                        logger.info(f"[{self.session_id}] Agent joined call {call_id}")
                        self._agent = agent

                        # ── LATENCY: mark agent joined ─────────────────
                        self._latency.mark_agent_joined()

                        # ── STATE: INIT → READY ────────────────────────
                        self._policy.report_agent_joined(True)
                        self._policy.report_model_state(True)
                        try:
                            self._state_machine.transition(
                                SessionState.READY, reason="agent_joined_call"
                            )
                        except ValueError as e:
                            logger.warning(f"[{self.session_id}] State transition: {e}")

                        # Attach agent to sub-processors
                        self._processor.attach_agent(agent)
                        self._audio_processor.attach_agent(agent)
                        self._conversation.attach_agent(agent)

                        # Report audio as active (SDK handles audio transport)
                        self._policy.report_audio_state(True)

                        # Subscribe to SDK events via proper event bus
                        _load_sdk_event_types()
                        try:
                            agent.subscribe(self._on_sdk_event)
                            logger.info(f"[{self.session_id}] Subscribed to agent event bus")
                        except Exception as e:
                            logger.warning(f"[{self.session_id}] Agent event subscription failed: {e}")

                        # ── TRACK INTROSPECTION: discover existing tracks ──
                        # If tracks were added before our event subscription, we
                        # won't get TrackAddedEvent for them. Introspect the agent
                        # to find any already-connected tracks.
                        await self._introspect_tracks(agent)

                        # Start conversation manager
                        await self._conversation.start()

                        # ── PRE-WARM: LLM + TTS connections ──────
                        # Eliminates cold-start latency on first response
                        if performance_cfg.pre_warm_llm:
                            asyncio.create_task(
                                self._pre_warm_llm(agent),
                                name=f"prewarm-llm-{self.session_id}",
                            )
                        if performance_cfg.pre_warm_tts:
                            asyncio.create_task(
                                self._pre_warm_tts(agent),
                                name=f"prewarm-tts-{self.session_id}",
                            )

                        self._join_ready.set()

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
                self._state_machine.transition(SessionState.FAILED, reason="join_timeout")
                raise RuntimeError("Agent join timed out")

            if self._join_error:
                self._state_machine.transition(SessionState.FAILED, reason=self._join_error)
                raise RuntimeError(self._join_error)

            self._active = True
            self._starting = False
            self.telemetry.sdk_active = True
            self.telemetry.agent_joined = True
            self.telemetry.audio_active = True

            # ── POLICY: determine initial mode (NOT multimodal until video arrives) ──
            mode = self._policy.determine_mode()
            self._state_machine.set_mode(mode)
            self.telemetry.session_mode = mode.value
            self.telemetry.multimodal_active = (mode == SessionMode.MULTIMODAL)

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
            # Health monitor — drives state machine transitions
            self._health_task = asyncio.create_task(
                self._health_monitor(), name=f"health-{self.session_id}"
            )

            logger.info(
                f"[{self.session_id}] AI Service started — "
                f"state={self._state_machine.state.value}, "
                f"mode={self._state_machine.mode.value}, "
                f"call='{call_id}'"
            )

            return {
                "session_id": self.session_id,
                "call_id": call_id,
                "agent_id": f"speakai-{self.session_id[:8]}",
                "session_state": self._state_machine.state.value,
                "session_mode": self._state_machine.mode.value,
                "sdk_active": True,
                "multimodal_active": self.telemetry.multimodal_active,
                "audio_active": True,
                "conversation_active": True,
            }

        except ImportError as e:
            logger.warning(f"[{self.session_id}] SDK import failed: {e}")
            self.telemetry.sdk_active = False
            self._starting = False
            try:
                self._state_machine.transition(SessionState.FAILED, reason=f"import_error: {e}")
            except ValueError:
                pass
            raise
        except Exception as e:
            logger.error(f"[{self.session_id}] AI Service start failed: {e}", exc_info=True)
            self.telemetry.sdk_active = False
            self._starting = False
            try:
                self._state_machine.transition(SessionState.FAILED, reason=str(e)[:100])
            except ValueError:
                pass
            raise

    # ── SDK Event Handler (unified event bus) ──────────────────────────

    async def _on_sdk_event(self, event: Any) -> None:
        """
        Unified handler for ALL events from the Agent's event bus.
        Handles: transcription, track lifecycle, conversation items.
        """
        try:
            event_type = getattr(event, "type", type(event).__name__)

            # ── User speech transcription → filler words, WPM, transcript ──
            if _RealtimeUserSpeechTranscriptionEvent and isinstance(
                event, _RealtimeUserSpeechTranscriptionEvent
            ):
                text = getattr(event, "text", "").strip()
                if text:
                    logger.info(f"[{self.session_id}] User speech: {text[:80]}")

                    # ── LATENCY: mark first transcript ──
                    self._latency.mark_first_transcript()

                    # Update audio processor with real transcript
                    if self._audio_processor:
                        await self._audio_processor.handle_user_transcript(text)

                    # Forward as transcript entry
                    entry = TranscriptEntry(
                        speaker="user",
                        text=text,
                        timestamp=time.time(),
                        confidence=1.0,
                        is_final=True,
                    )
                    if self._on_transcript:
                        cb = self._on_transcript(entry)
                        if asyncio.iscoroutine(cb):
                            await cb

                    # Forward to conversation manager
                    if self._conversation:
                        await self._conversation.handle_transcript(entry)

            # ── Agent speech transcription → track agent responses ──
            elif _RealtimeAgentSpeechTranscriptionEvent and isinstance(
                event, _RealtimeAgentSpeechTranscriptionEvent
            ):
                text = getattr(event, "text", "").strip()
                if text:
                    logger.info(f"[{self.session_id}] Agent speech: {text[:80]}")

                    # Forward as chat message
                    msg = ChatMessage(
                        id=f"agent-{int(time.time()*1000)}",
                        role="assistant",
                        content=text,
                        timestamp=time.time(),
                        source="voice",
                    )
                    if self._on_chat:
                        cb = self._on_chat(msg)
                        if asyncio.iscoroutine(cb):
                            await cb

                    # Update conversation state
                    if self._conversation:
                        self._conversation._state.last_agent_response = text[:200]

            # ── Track lifecycle events ──
            elif _TrackAddedEvent and isinstance(event, _TrackAddedEvent):
                track_type = getattr(event, "track_type", "unknown")
                track_id = getattr(event, "track_id", "unknown")
                track = getattr(event, "track", None)
                participant_id = getattr(event, "participant_id", None)
                logger.info(
                    f"[{self.session_id}] Track added: type={track_type}, "
                    f"id={track_id}, participant={participant_id}, "
                    f"track_obj={type(track).__name__ if track else 'None'}"
                )
                # Report to policy layer
                if str(track_type).lower() in ("video", "screenshare"):
                    self._policy.report_frame()  # First track arrival = first frame signal

                    # ── CRITICAL: Explicitly trigger processor.process_video() ──
                    # The SDK's automatic processor pipeline via agents.processors=[]
                    # may not always call process_video(). This ensures the processor
                    # gets the track directly when we detect it via events.
                    if track is not None and self._processor is not None:
                        try:
                            logger.info(
                                f"[{self.session_id}] 🎥 Explicitly forwarding video "
                                f"track to processor"
                            )
                            await self._processor.process_video(
                                track=track,
                                participant_id=str(participant_id) if participant_id else None,
                                shared_forwarder=None,  # Let processor create its own
                            )
                        except Exception as e:
                            logger.error(
                                f"[{self.session_id}] Failed to forward track to processor: {e}",
                                exc_info=True,
                            )

                if str(track_type).lower() == "audio":
                    self._policy.report_audio_state(True)
                    # Forward audio track to audio processor
                    if track is not None and self._audio_processor is not None:
                        try:
                            await self._audio_processor.process_audio(
                                track=track,
                                participant_id=str(participant_id) if participant_id else None,
                            )
                        except Exception as e:
                            logger.debug(
                                f"[{self.session_id}] Audio track forward: {e}"
                            )

            elif _TrackRemovedEvent and isinstance(event, _TrackRemovedEvent):
                track_id = getattr(event, "track_id", "unknown")
                track_type = getattr(event, "track_type", "unknown")
                logger.info(
                    f"[{self.session_id}] Track removed: type={track_type}, id={track_id}"
                )
                # If video track removed, stop processor
                if str(track_type).lower() in ("video", "screenshare"):
                    if self._processor is not None:
                        try:
                            await self._processor.stop_processing()
                        except Exception:
                            pass

            else:
                # Log ALL other events for debugging
                if not hasattr(self, "_event_type_counts"):
                    self._event_type_counts: Dict[str, int] = {}
                count = self._event_type_counts.get(event_type, 0) + 1
                self._event_type_counts[event_type] = count
                if count <= 5:
                    logger.info(
                        f"[{self.session_id}] SDK event: {event_type} "
                        f"(#{count}) attrs={list(vars(event).keys()) if hasattr(event, '__dict__') else '?'}"
                    )

        except Exception as e:
            logger.debug(f"[{self.session_id}] SDK event handler error: {e}")

    async def _introspect_tracks(self, agent: Any) -> None:
        """
        Discover any video/audio tracks the agent already has access to.
        This covers the race condition where tracks are added before our
        event subscription starts.
        """
        try:
            # Check various SDK attribute patterns for track access
            tracks_found = 0

            # Pattern 1: agent.call.participants / remote tracks
            call = getattr(agent, 'call', None) or getattr(agent, '_call', None)
            if call:
                participants = (
                    getattr(call, 'participants', None) or
                    getattr(call, 'members', None) or
                    []
                )
                for p in participants if hasattr(participants, '__iter__') else []:
                    pid = getattr(p, 'id', getattr(p, 'user_id', 'unknown'))
                    # Skip the agent's own tracks
                    agent_id = getattr(agent, 'id', '')
                    if pid == agent_id:
                        continue

                    for track_attr in ('video_track', 'video', 'published_tracks'):
                        track = getattr(p, track_attr, None)
                        if track is not None:
                            logger.info(
                                f"[{self.session_id}] Introspected video track "
                                f"from participant {pid}"
                            )
                            if self._processor:
                                await self._processor.process_video(
                                    track=track,
                                    participant_id=str(pid),
                                )
                            tracks_found += 1
                            break

            # Pattern 2: agent._video_forwarder (some SDK versions expose this)
            forwarder = getattr(agent, '_video_forwarder', None)
            if forwarder and self._processor and tracks_found == 0:
                logger.info(
                    f"[{self.session_id}] Found agent's video forwarder — "
                    f"attaching processor"
                )
                # The processor's process_video was already called by the SDK
                # via processors=[] — but let's verify it's working
                if self._processor.frames_processed == 0:
                    logger.info(
                        f"[{self.session_id}] Processor has 0 frames — "
                        f"watchdog will handle fallback"
                    )

            # Pattern 3: agent._edge or edge transport object
            edge = getattr(agent, 'edge', getattr(agent, '_edge', None))
            if edge:
                remote_tracks = getattr(edge, 'remote_tracks', None)
                if remote_tracks and hasattr(remote_tracks, '__iter__'):
                    for rt in remote_tracks:
                        kind = getattr(rt, 'kind', '')
                        if 'video' in str(kind).lower() and self._processor:
                            logger.info(
                                f"[{self.session_id}] Introspected remote video track "
                                f"from edge transport"
                            )
                            await self._processor.process_video(
                                track=rt,
                                participant_id=None,
                            )
                            tracks_found += 1

            logger.info(
                f"[{self.session_id}] Track introspection complete: "
                f"found {tracks_found} tracks"
            )

        except Exception as e:
            logger.debug(f"[{self.session_id}] Track introspection: {e}")

    # ── Pre-warm: eliminate cold-start latency ──────────────────────────

    async def _pre_warm_llm(self, agent: Any) -> None:
        """
        Pre-warm the LLM connection by sending a trivial prompt.
        Eliminates ~500-1500ms cold-start on the first real user interaction.
        """
        try:
            if not agent or not hasattr(agent, "llm"):
                return

            t0 = time.perf_counter()
            response = await asyncio.wait_for(
                agent.llm.simple_response("Respond with OK"),
                timeout=3.0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._llm_warmed = True
            logger.info(
                f"[{self.session_id}] ⚡ LLM pre-warmed in {elapsed_ms:.0f}ms"
            )
        except asyncio.TimeoutError:
            logger.debug(f"[{self.session_id}] LLM pre-warm timed out")
        except Exception as e:
            logger.debug(f"[{self.session_id}] LLM pre-warm skipped: {e}")

    async def _pre_warm_tts(self, agent: Any) -> None:
        """
        Pre-warm the speech pipeline.

        In Realtime mode TTS=None so agent.say() is a no-op.
        We warm up via simple_response which goes through the
        Realtime LLM's native audio generation path.
        """
        try:
            if not agent:
                return

            t0 = time.perf_counter()
            # In Realtime mode, simple_response warms the audio output path
            await asyncio.wait_for(
                agent.simple_response("Respond with a single word: ready"),
                timeout=5.0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._tts_warmed = True
            logger.info(
                f"[{self.session_id}] ⚡ Speech pipeline pre-warmed in {elapsed_ms:.0f}ms"
            )
        except asyncio.TimeoutError:
            logger.debug(f"[{self.session_id}] Speech pre-warm timed out")
        except Exception as e:
            logger.debug(f"[{self.session_id}] Speech pre-warm skipped: {e}")

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

        In Realtime mode (Gemini handles STT+TTS), agent.say() is a no-op
        because TTS is None.  Use agent.simple_response() which sends the
        text to the Realtime LLM, and the model generates audio natively.
        """
        if not self._agent:
            logger.warning(f"[{self.session_id}] Cannot speak — agent not ready")
            return

        try:
            # Prefer simple_response (works in Realtime mode where TTS=None)
            await self._agent.simple_response(text)

            # ── LATENCY: mark first response ──
            self._latency.mark_first_response()

            logger.info(f"[{self.session_id}] Agent spoke: {text[:80]}...")
        except Exception as e:
            logger.error(f"[{self.session_id}] Agent speech failed: {e}", exc_info=True)

    # ── Stop: tear down agent + workers ─────────────────────────────────

    async def stop(self) -> Dict[str, Any]:
        """Stop the AI service, cancel workers, tear down the agent."""
        self._active = False
        self._starting = False
        duration = time.time() - (self._started_at or time.time())

        # ── STATE: reset to INIT ──
        try:
            self._state_machine.reset()
        except Exception:
            pass

        # Stop conversation manager
        conversation_summary = {}
        if self._conversation:
            conversation_summary = await self._conversation.stop()

        # Stop audio processor
        if self._audio_processor:
            await self._audio_processor.close()

        # Stop video processor (cleans up direct reader + watchdog)
        if self._processor:
            try:
                await self._processor.close()
            except Exception as e:
                logger.debug(f"[{self.session_id}] Processor close: {e}")

        # Cancel background tasks (including health monitor)
        for task in [self._reasoning_task, self._status_task,
                     self._metrics_listener_task,
                     getattr(self, '_health_task', None)]:
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
            "latency": self._latency.summary(),
        }
        logger.info(f"[{self.session_id}] AI Service stopped — {summary}")
        return summary

    # ── Metrics listener (consumes MetricsProducedEvent) ────────────────

    async def _metrics_listener(self) -> None:
        """
        Poll the processor's latest metrics and forward to the callback.

        PRODUCTION INTEGRITY:
        - Only real MediaPipe metrics are forwarded in multimodal mode.
        - In audio-only mode, only audio-derived metrics (WPM, fillers) are sent
          with visual fields explicitly zeroed and source="audio_only".
        - NO silent fallback generation. If no data exists, nothing is sent.
        - The policy layer gates all emission decisions.
        - Reports frame arrivals to the health system for mode transitions.
        """
        fps = processing_cfg.processor_fps
        interval = 1.0 / fps if fps > 0 else 0.2
        _last_frame_count = 0

        while self._active:
            try:
                await asyncio.sleep(interval)
                if not self._active or self._processor is None:
                    break

                has_video_frames = self._processor.frames_processed > 0
                current_frame_count = self._processor.frames_processed

                # Report new frames to health system (drives mode transitions)
                if current_frame_count > _last_frame_count:
                    self._policy.report_frame(self._processor.latest.timestamp)
                    self._latency.mark_first_frame()

                    # On first frame, log the pipeline source
                    if _last_frame_count == 0:
                        source = (
                            "direct_reader"
                            if self._processor._direct_reader_active
                            else "forwarder"
                        )
                        logger.info(
                            f"[{self.session_id}] 🎬 First video frame via {source} — "
                            f"activating multimodal pipeline"
                        )

                    _last_frame_count = current_frame_count

                if has_video_frames:
                    # Real MediaPipe metrics from processed video frames
                    metrics = self._processor.latest
                else:
                    # ── NO SILENT FALLBACK ──
                    # In audio-only mode, emit audio-derived metrics only.
                    # Visual fields stay at zero — never fabricated.
                    mode = self._state_machine.mode
                    if mode == SessionMode.AUDIO_ONLY and self._audio_processor:
                        metrics = SpeakingMetrics(
                            eye_contact=0.0,
                            head_stability=0.0,
                            posture_score=0.0,
                            facial_engagement=0.0,
                            attention_intensity=0.0,
                            words_per_minute=self._audio_processor.current_wpm,
                            filler_words=self._audio_processor.total_fillers,
                            timestamp=time.time(),
                            source="audio_only",
                        )
                    else:
                        # No data at all — skip emission
                        continue

                # Augment with audio metrics (WPM, filler words) — real data
                if self._audio_processor:
                    metrics = self._audio_processor.update_metrics(metrics)

                # Feed metrics to conversation manager for visual context
                if self._conversation:
                    self._conversation.update_metrics(metrics)

                # ── POLICY: gate emission ──
                if not self._policy.should_emit_metrics(metrics):
                    continue

                if metrics.timestamp > 0 and self._on_metrics:
                    # Skip duplicate timestamps (only for real video frames)
                    if has_video_frames and metrics.timestamp == self._last_metrics_timestamp:
                        continue

                    self._last_metrics_timestamp = metrics.timestamp
                    self._latest_emitted_metrics = metrics
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
        Freshness-gated: skips reasoning when metrics are stale.
        """
        logger.info(f"[{self.session_id}] Reasoning worker started")
        last_reasoning_metrics_ts = 0.0

        # Short warmup so initial metrics settle
        await asyncio.sleep(1.0)

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

                # ── FRESHNESS GATE: skip stale/synthetic metrics ──
                if not self._policy.validate_freshness(metrics):
                    logger.debug(
                        f"[{self.session_id}] Reasoning skipped: "
                        f"stale metrics (age={time.time() - metrics.timestamp:.1f}s, "
                        f"source={metrics.source})"
                    )
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

                audio_stats = {}
                if self._audio_processor:
                    audio_stats = {
                        "wpm": self._audio_processor.current_wpm,
                        "total_words": self._audio_processor.total_words,
                        "total_fillers": self._audio_processor.total_fillers,
                        "is_speaking": self._audio_processor.is_user_speaking,
                    }

                # ── POLICY: diagnostics from the policy layer ──
                policy_diag = self._policy.diagnostics()

                # ── PROCESSOR: pipeline health diagnostics ──
                processor_diag = {}
                if self._processor:
                    processor_diag = {
                        "frames_processed": self._processor.frames_processed,
                        "last_latency_ms": self._processor.last_latency_ms,
                        "direct_reader_active": self._processor._direct_reader_active,
                        "forwarder_active": self._processor._video_forwarder is not None,
                        "has_raw_track": self._processor._raw_track is not None,
                    }

                status = {
                    "session_id": self.session_id,
                    "call_id": self.telemetry.call_id,
                    # ── Explicit state + mode (requirement #2, #5) ──
                    "session_state": self._state_machine.state.value,
                    "session_mode": self._state_machine.mode.value,
                    # ── Legacy fields for backward compatibility ──
                    "sdk_active": self.telemetry.sdk_active,
                    "multimodal_active": self.telemetry.multimodal_active,
                    "agent_joined": self.telemetry.agent_joined,
                    "audio_active": self.telemetry.audio_active,
                    "frames_processed": self.telemetry.frames_processed,
                    "metrics_source": self._state_machine.mode.value,
                    "inference_latency_ms": self.telemetry.last_frame_latency_ms,
                    "reasoning_latency_ms": self.telemetry.last_reasoning_latency_ms,
                    "transcript_entries": self.telemetry.transcript_entries,
                    "chat_messages": self.telemetry.chat_messages,
                    "conversation_turns": self.telemetry.conversation_turns,
                    "conversation": conversation_state,
                    "audio": audio_stats,
                    # ── Health + latency + pipeline diagnostics ──
                    "health": policy_diag.get("health", {}),
                    "latency": self._latency.summary(),
                    "pipeline": processor_diag,
                    "performance": {
                        "llm_warmed": self._llm_warmed,
                        "tts_warmed": self._tts_warmed,
                        "reasoning_cooldown_s": processing_cfg.reasoning_cooldown,
                        "response_delay_s": conversation_cfg.response_delay,
                        "llm_timeout_s": getattr(conversation_cfg, 'llm_timeout', 5.0),
                    },
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

    # ── Health monitor (drives state machine) ───────────────────────────

    async def _health_monitor(self) -> None:
        """
        Periodically evaluates system health and drives state transitions.

        This is the ONLY place state transitions happen during normal operation.
        The policy layer determines what state to be in; this worker enforces it.
        """
        logger.info(f"[{self.session_id}] Health monitor started")

        while self._active:
            try:
                await asyncio.sleep(2.0)
                if not self._active:
                    break

                # Ask policy what state we should be in
                recommended = self._policy.determine_state()
                current = self._state_machine.state

                # Don't attempt transitions from FAILED or INIT
                if current in (SessionState.FAILED, SessionState.INIT):
                    continue

                # Attempt transition if state changed
                if recommended != current:
                    try:
                        health = self._policy.check_health()
                        reason = (
                            f"health_check: video={health.video}, "
                            f"audio={health.audio}, model={health.model}"
                        )
                        self._state_machine.transition(recommended, reason=reason)
                    except ValueError as e:
                        logger.debug(f"[{self.session_id}] Health transition blocked: {e}")

                # Always update mode
                mode = self._policy.determine_mode()
                self._state_machine.set_mode(mode)
                self.telemetry.session_state = self._state_machine.state.value
                self.telemetry.session_mode = mode.value
                self.telemetry.multimodal_active = (mode == SessionMode.MULTIMODAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.session_id}] Health monitor error: {e}")

        logger.info(f"[{self.session_id}] Health monitor stopped")
