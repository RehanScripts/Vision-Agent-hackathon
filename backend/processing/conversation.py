"""
SpeakAI — Conversation Manager

================================================================================
BIDIRECTIONAL COMMUNICATION ORCHESTRATOR
================================================================================

Manages the full conversation lifecycle between the user and AI agent:

  1. Maintains conversation history with bounded context window.
  2. Processes incoming user messages (voice transcripts or text chat).
  3. Generates AI responses using the Agent's LLM (Gemini Realtime).
  4. Manages turn-taking: detects when user finishes speaking,
     waits a grace period, then lets the agent respond.
  5. Supports proactive mode: agent comments on visual observations
     when user is silent for too long.
  6. Combines visual context (metrics) + audio context (transcript)
     for multimodal responses.

The conversation manager does NOT handle media transport — that's the
Agent's job via Stream Video. This module handles the dialogue logic.
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..core.models import (
    ChatMessage,
    TranscriptEntry,
    ConversationState,
    SpeakingMetrics,
    CoachingFeedback,
)
from ..core.config import conversation_cfg
from ..core.health import SessionPolicy, FRESHNESS_MAX_AGE_S

logger = logging.getLogger("speakai.conversation")


class ConversationManager:
    """
    Orchestrates the dialogue between user and AI agent.

    Responsibilities:
      - Maintain conversation history
      - Generate context-aware responses
      - Handle turn-taking
      - Support text and voice input
      - Provide visual+audio multimodal context to the LLM
    """

    def __init__(
        self,
        session_id: str,
        on_chat_message: Optional[Callable[[ChatMessage], Any]] = None,
        on_state_change: Optional[Callable[[ConversationState], Any]] = None,
        on_agent_speech: Optional[Callable[[str], Any]] = None,
    ):
        self.session_id = session_id

        # Callbacks
        self._on_chat_message = on_chat_message
        self._on_state_change = on_state_change
        self._on_agent_speech = on_agent_speech

        # Conversation history (bounded)
        self._messages: Deque[ChatMessage] = deque(
            maxlen=conversation_cfg.max_context_messages
        )

        # State
        self._state = ConversationState()
        self._agent: Any = None
        self._active = False

        # Turn-taking
        self._last_user_speech_end: float = 0.0
        self._last_agent_response: float = 0.0
        self._pending_response = False
        self._pending_response_task: Optional[asyncio.Task] = None

        # Voice transcript batching (merge fragmented final chunks)
        self._pending_voice_fragments: List[str] = []
        self._pending_voice_flush_task: Optional[asyncio.Task] = None

        # Proactive mode
        self._last_proactive_check: float = 0.0
        self._proactive_task: Optional[asyncio.Task] = None
        self._last_proactive_text: str = ""
        self._last_proactive_at: float = 0.0
        self._last_assistant_text: str = ""
        self._last_assistant_at: float = 0.0

        # Visual context (updated by processor)
        self._latest_metrics: Optional[SpeakingMetrics] = None

        # Session mode (set by service layer via set_session_mode)
        self._session_mode: str = "unavailable"

        # Add system message
        system_msg = ChatMessage(
            id=uuid.uuid4().hex[:8],
            role="system",
            content=conversation_cfg.system_prompt,
            source="system",
        )
        self._messages.append(system_msg)

        logger.info(f"[{session_id}] ConversationManager initialized")

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def messages(self) -> List[ChatMessage]:
        return list(self._messages)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def attach_agent(self, agent: Any) -> None:
        """Attach the Vision Agent for LLM access."""
        self._agent = agent
        logger.info(f"[{self.session_id}] ConversationManager attached to agent")

    def update_metrics(self, metrics: SpeakingMetrics) -> None:
        """Update the latest visual metrics for context."""
        self._latest_metrics = metrics

    def set_session_mode(self, mode: str) -> None:
        """Set the current session mode (from policy layer)."""
        if mode != self._session_mode:
            logger.info(f"[{self.session_id}] Conversation mode: {self._session_mode} → {mode}")
            self._session_mode = mode

    async def start(self) -> None:
        """Start the conversation manager."""
        self._active = True
        self._last_agent_response = time.time()  # prevent immediate proactive

        # Start proactive monitoring if enabled
        if conversation_cfg.proactive_mode:
            self._proactive_task = asyncio.create_task(
                self._proactive_worker(), name=f"proactive-{self.session_id}"
            )

        # Send welcome message after a short delay so frontend is ready
        asyncio.create_task(
            self._send_welcome(), name=f"welcome-{self.session_id}"
        )

        logger.info(f"[{self.session_id}] ConversationManager started")

    async def _send_welcome(self) -> None:
        """Send an initial greeting when the agent joins."""
        logger.info(f"[{self.session_id}] 🎤 Welcome task started, waiting 3s...")
        await asyncio.sleep(3.0)  # Let everything initialize
        if not self._active:
            logger.warning(f"[{self.session_id}] Welcome aborted — session no longer active")
            return

        welcome = (
            "Hi! I'm your AI speaking coach. Start speaking and I'll give you "
            "real-time feedback on pace, filler words, and more. Ask me anything!"
        )
        msg = ChatMessage(
            id=uuid.uuid4().hex[:8],
            role="assistant",
            content=welcome,
            source="system",
        )
        self._messages.append(msg)
        self._state.last_agent_response = welcome
        self._last_agent_response = time.time()

        if self._on_chat_message:
            cb = self._on_chat_message(msg)
            if asyncio.iscoroutine(cb):
                await cb

        # Also speak the welcome via TTS
        if self._on_agent_speech:
            cb = self._on_agent_speech(welcome)
            if asyncio.iscoroutine(cb):
                await cb

        logger.info(f"[{self.session_id}] Welcome message sent")

    async def stop(self) -> Dict[str, Any]:
        """Stop and return conversation summary."""
        self._active = False

        if self._pending_voice_flush_task and not self._pending_voice_flush_task.done():
            self._pending_voice_flush_task.cancel()
            try:
                await self._pending_voice_flush_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._pending_response_task and not self._pending_response_task.done():
            self._pending_response_task.cancel()
            try:
                await self._pending_response_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._proactive_task and not self._proactive_task.done():
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except (asyncio.CancelledError, Exception):
                pass

        summary = {
            "total_messages": len(self._messages),
            "conversation_turns": self._state.turn_count,
            "transcript_entries": self._state.transcript_length,
        }
        logger.info(f"[{self.session_id}] ConversationManager stopped: {summary}")
        return summary

    # ── User input handlers ─────────────────────────────────────────────

    async def handle_user_text(self, text: str) -> None:
        """
        Handle a text message from the user (typed in chat).
        Generates an AI response.
        """
        if not text.strip():
            return

        msg = ChatMessage(
            id=uuid.uuid4().hex[:8],
            role="user",
            content=text.strip(),
            source="text",
        )
        self._messages.append(msg)
        self._state.last_user_speech = text.strip()
        self._state.turn_count += 1

        # Notify frontend
        if self._on_chat_message:
            cb = self._on_chat_message(msg)
            if asyncio.iscoroutine(cb):
                await cb

        # Generate response
        await self._generate_response(trigger="text_input")

    async def handle_transcript(self, entry: TranscriptEntry) -> None:
        """
        Handle a transcript entry from the audio processor.
        Only generates a response on final transcripts after a pause.
        """
        self._state.transcript_length += 1

        if entry.speaker in ("user", "participant"):
            self._state.is_user_speaking = not entry.is_final

            if entry.is_final and entry.text.strip():
                self._pending_voice_fragments.append(entry.text.strip())

                if self._pending_voice_flush_task and not self._pending_voice_flush_task.done():
                    self._pending_voice_flush_task.cancel()

                self._pending_voice_flush_task = asyncio.create_task(
                    self._flush_voice_utterance(),
                    name=f"flush-voice-{self.session_id}",
                )

        elif entry.speaker == "agent":
            # Agent's own speech transcript
            self._state.is_agent_speaking = not entry.is_final

    async def _delayed_response(self) -> None:
        """Wait for response delay then generate response (turn-taking)."""
        await asyncio.sleep(conversation_cfg.response_delay)

        # Only respond if no new speech started during delay
        if (
            self._pending_response
            and not self._state.is_user_speaking
            and time.time() - self._last_user_speech_end >= conversation_cfg.response_delay
        ):
            self._pending_response = False
            await self._generate_response(trigger="voice_input")

    async def _flush_voice_utterance(self) -> None:
        """Debounce fragmented voice transcript chunks into a single utterance."""
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            return

        if not self._pending_voice_fragments:
            return

        merged_text = self._normalize_voice_text(" ".join(self._pending_voice_fragments))
        self._pending_voice_fragments.clear()

        if not merged_text:
            return

        msg = ChatMessage(
            id=uuid.uuid4().hex[:8],
            role="user",
            content=merged_text,
            source="voice",
        )
        self._messages.append(msg)
        self._state.last_user_speech = merged_text
        self._state.turn_count += 1

        if self._on_chat_message:
            cb = self._on_chat_message(msg)
            if asyncio.iscoroutine(cb):
                await cb

        self._last_user_speech_end = time.time()
        self._pending_response = True

        if self._pending_response_task and not self._pending_response_task.done():
            self._pending_response_task.cancel()

        self._pending_response_task = asyncio.create_task(
            self._delayed_response(),
            name=f"delay-response-{self.session_id}",
        )

    @staticmethod
    def _normalize_voice_text(text: str) -> str:
        """Normalize spacing and punctuation for merged transcript text."""
        cleaned = " ".join(text.split())
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        return cleaned.strip()

    # ── Response generation ─────────────────────────────────────────────

    async def _generate_response(self, trigger: str = "auto") -> None:
        """
        Generate an AI response using the Agent's LLM.
        Combines conversation context + visual metrics.

        FRESHNESS GATE: Visual-dependent responses are blocked unless
        metrics are fresh and from a real visual source.
        """
        now = time.time()

        # Rate limit: don't respond more than once per response_delay
        if now - self._last_agent_response < conversation_cfg.response_delay:
            return

        self._state.is_agent_speaking = True
        await self._broadcast_state()

        try:
            response_text = await self._call_llm(trigger)
            if not response_text or not response_text.strip():
                response_text = self._fallback_response(trigger)

            clean_response = response_text.strip() if response_text else ""

            if clean_response and self._is_redundant_response(clean_response, trigger):
                logger.info(f"[{self.session_id}] Skipping redundant {trigger} response")
                return

            if clean_response:
                msg = ChatMessage(
                    id=uuid.uuid4().hex[:8],
                    role="assistant",
                    content=clean_response,
                    source="llm",
                )
                self._messages.append(msg)
                self._state.last_agent_response = clean_response
                self._state.chat_messages = len(
                    [m for m in self._messages if m.role != "system"]
                )
                self._last_agent_response = time.time()
                self._last_assistant_text = clean_response
                self._last_assistant_at = self._last_agent_response

                if trigger == "proactive":
                    self._last_proactive_text = clean_response
                    self._last_proactive_at = self._last_agent_response

                # Notify frontend with chat message
                if self._on_chat_message:
                    cb = self._on_chat_message(msg)
                    if asyncio.iscoroutine(cb):
                        await cb

                # Tell agent to speak (TTS)
                if self._on_agent_speech:
                    cb = self._on_agent_speech(clean_response)
                    if asyncio.iscoroutine(cb):
                        await cb

        except Exception as e:
            logger.warning(f"[{self.session_id}] Response generation failed: {e}")
        finally:
            self._state.is_agent_speaking = False
            await self._broadcast_state()

    async def _call_llm(self, trigger: str) -> Optional[str]:
        """
        Build context-aware response text for the agent to speak via TTS.

        Priority:
          1. Visual guardrails (hard-coded for honesty about capabilities)
          2. Agent LLM (agent.llm.simple_response) for rich, dynamic responses
          3. Deterministic fallback if LLM unavailable or times out

        FRESHNESS + MODE GATE:
        - Visual claims require fresh metrics from a real visual source
          AND session_mode == "multimodal".
        - In audio_only mode, only audio-derived coaching is generated.
        """
        latest_user_text = (self._state.last_user_speech or "").strip()
        has_live_visual = self._has_live_visual_context()

        # In audio-only mode, never claim visual capability
        if self._session_mode == "audio_only":
            has_live_visual = False

        # Hard guardrails for visual honesty and unsupported tasks
        if self._is_visual_question(latest_user_text):
            guarded = self._visual_capability_response(latest_user_text, has_live_visual)
            if guarded:
                return guarded

        # Try the agent's LLM for a dynamic response
        if self._agent and hasattr(self._agent, "llm"):
            try:
                prompt = self._build_llm_prompt(trigger, latest_user_text, has_live_visual)
                response = await asyncio.wait_for(
                    self._agent.llm.simple_response(prompt),
                    timeout=conversation_cfg.llm_timeout if hasattr(conversation_cfg, 'llm_timeout') else 5.0,
                )
                text = getattr(response, "text", str(response)).strip()
                if text and len(text) > 5:
                    return text
            except asyncio.TimeoutError:
                logger.warning(f"[{self.session_id}] LLM response timed out — using fallback")
            except Exception as e:
                logger.debug(f"[{self.session_id}] LLM call failed: {e} — using fallback")

        return self._fallback_response(trigger)

    def _build_llm_prompt(self, trigger: str, user_text: str, has_live_visual: bool) -> str:
        """Build a context-aware prompt for the agent's LLM."""
        parts: list[str] = []

        # Context: metrics
        if self._latest_metrics and self._latest_metrics.timestamp > 0:
            m = self._latest_metrics
            if has_live_visual:
                parts.append(
                    f"Speaker metrics: eye contact {m.eye_contact:.0f}%, "
                    f"posture {m.posture_score:.0f}%, "
                    f"head stability {m.head_stability:.0f}%, "
                    f"engagement {m.facial_engagement:.0f}%, "
                    f"WPM {m.words_per_minute:.0f}, "
                    f"filler words {m.filler_words}."
                )
            else:
                parts.append(
                    f"Audio metrics: WPM {m.words_per_minute:.0f}, "
                    f"filler words {m.filler_words}. "
                    "No video feed available."
                )

        if trigger == "proactive":
            parts.append(
                "The speaker has been quiet. Give a brief, encouraging coaching tip "
                "based on their current metrics. Keep it under 2 sentences."
            )
        elif trigger == "text_input" and user_text:
            parts.append(f"The user asked: \"{user_text}\"")
            parts.append("Respond helpfully as a speaking coach. Keep it concise (1-3 sentences).")
        elif trigger == "voice_input" and user_text:
            parts.append(f"The user said: \"{user_text}\"")
            parts.append("Respond as a speaking coach. Be brief and supportive (1-2 sentences).")
        else:
            parts.append("Give a brief coaching observation. Keep it under 2 sentences.")

        return "\n".join(parts)

    def _has_live_visual_context(self) -> bool:
        """
        True when we have fresh, real visual metrics from video frames.
        Uses the same freshness window as the policy layer.
        """
        if not self._latest_metrics:
            return False

        m = self._latest_metrics
        if m.timestamp <= 0:
            return False

        # Treat stale metrics as no live visual signal.
        if (time.time() - m.timestamp) > FRESHNESS_MAX_AGE_S:
            return False

        # Simulated/estimated/audio-only values should not be used to claim real visibility.
        if (m.source or "").lower() in ("simulated", "live_estimation", "audio_only", "no_mediapipe"):
            return False

        # Session mode must be multimodal for visual claims
        if self._session_mode != "multimodal":
            return False

        return True

    @staticmethod
    def _is_visual_question(text: str) -> bool:
        t = text.lower()
        visual_keywords = (
            "see me", "visible", "can you see", "look", "expression", "face", "camera",
            "finger", "fingers", "fingure", "gesture", "hand"
        )
        return any(keyword in t for keyword in visual_keywords)

    @staticmethod
    def _is_finger_count_question(text: str) -> bool:
        t = text.lower()
        return (
            "finger" in t
            or "fingers" in t
            or "fingure" in t
            or "how many" in t and "hand" in t
        )

    def _visual_capability_response(self, user_text: str, has_live_visual: bool) -> Optional[str]:
        """Deterministic, honest responses for visibility/capability questions."""
        if not has_live_visual:
            return (
                "I don't have a reliable live video feed yet. "
                "Make sure your camera is on and publishing in the Stream call. "
                "I can still coach you on speech pace, clarity, and filler words!"
            )

        if self._is_finger_count_question(user_text):
            return (
                "I can assess eye contact, posture, and overall facial engagement, "
                "but I can’t reliably count fingers yet."
            )

        if "expression" in user_text.lower() and self._latest_metrics:
            engagement = self._latest_metrics.facial_engagement
            if engagement >= 70:
                return "Your expressions look reasonably engaged right now."
            return "Your expressions look a bit flat right now—try adding more facial emphasis."

        return None

    def _fallback_response(self, trigger: str) -> Optional[str]:
        """Generate a simple fallback response when LLM is unavailable."""
        if trigger == "proactive":
            # Audio-only coaching when no visual metrics available
            if not self._latest_metrics or not self._has_visual_metrics():
                tips = [
                    "Remember to maintain steady eye contact with your camera.",
                    "Keep your shoulders back and sit up straight for better presence.",
                    "Try to vary your vocal tone — it keeps listeners engaged.",
                    "Pause briefly between key points for emphasis.",
                    "You're doing great! Keep going with confidence.",
                ]
                import random
                return random.choice(tips)

            m = self._latest_metrics
            if m.eye_contact < 50:
                return "Try looking directly at the camera to improve eye contact."
            if m.posture_score < 70:
                return "Your posture could use some adjustment. Sit up straight!"
            if m.head_stability < 60:
                return "Try to keep your head steady — it helps with presence."
            return "You're doing well! Keep up the good work."

        if trigger == "text_input":
            user_text = (self._state.last_user_speech or "").lower()
            if "eye" in user_text or "contact" in user_text:
                return "Focus on looking at the camera lens for 2–3 seconds at a time. It builds trust with your audience."
            if "posture" in user_text or "sit" in user_text or "stand" in user_text:
                return "Keep shoulders relaxed, chest open, and sit upright for stronger presence."
            if "pace" in user_text or "speed" in user_text or "wpm" in user_text or "fast" in user_text or "slow" in user_text:
                return "Aim for a steady 120–150 WPM. Pause briefly after key points for emphasis."
            if "filler" in user_text or "um" in user_text or "uh" in user_text or "like" in user_text:
                return "Replace filler words with brief pauses. It sounds more confident and polished."
            if "nervous" in user_text or "anxiety" in user_text or "confident" in user_text:
                return "Take a deep breath before speaking. Open posture and steady eye contact project confidence."
            if "hello" in user_text or "hi" in user_text or "hey" in user_text:
                return "Hi there! I'm your speaking coach. Try practicing a short pitch and I'll give feedback."
            if "help" in user_text or "what can" in user_text or "how" in user_text:
                return "I coach on eye contact, posture, pace, and filler words. Start speaking or ask me a specific question!"
            if "tip" in user_text or "advice" in user_text or "suggest" in user_text:
                return "Here's a quick tip: vary your vocal tone and pause between key ideas. It keeps listeners engaged."
            return "Great question! I can help with eye contact, posture, pace, filler words, and confidence. What would you like to work on?"

        if trigger == "voice_input":
            return "I heard you! Try to speak clearly and at a steady pace."

        return None

    def _has_actionable_metrics(self) -> bool:
        """True when we have any metrics data (visual or audio) to act on."""
        if not self._latest_metrics:
            return False

        m = self._latest_metrics
        if m.timestamp <= 0:
            return False

        # Active session with any data means we can coach
        return True

    def _has_visual_metrics(self) -> bool:
        """True only when we have real non-zero visual metrics from MediaPipe."""
        if not self._latest_metrics:
            return False
        m = self._latest_metrics
        if m.timestamp <= 0:
            return False
        if (m.source or "").lower() in ("simulated", "live_estimation", "audio_only", "no_mediapipe"):
            return False
        signal = m.eye_contact + m.posture_score + m.head_stability + m.facial_engagement
        return signal > 5

    def _is_redundant_response(self, text: str, trigger: str) -> bool:
        """Suppress repeated assistant responses, especially proactive duplicates."""
        now = time.time()
        normalized = " ".join(text.lower().split())

        # Generic duplicate guard for all triggers
        if normalized and normalized == self._last_assistant_text.lower() and (now - self._last_assistant_at) < 12.0:
            return True

        # Stronger guard for proactive repetition
        if trigger == "proactive":
            if normalized and normalized == self._last_proactive_text.lower() and (now - self._last_proactive_at) < 45.0:
                return True

        return False

    # ── Proactive agent ─────────────────────────────────────────────────

    async def _proactive_worker(self) -> None:
        """
        Periodically checks if the agent should proactively comment.
        Works in both visual and audio-only modes.
        """
        while self._active:
            try:
                await asyncio.sleep(conversation_cfg.silence_prompt_timeout)
                if not self._active:
                    break

                now = time.time()
                silence_duration = now - max(
                    self._last_user_speech_end, self._last_agent_response
                )

                # Speak proactively if there's been enough silence
                if (
                    silence_duration >= conversation_cfg.silence_prompt_timeout
                    and not self._state.is_user_speaking
                    and not self._state.is_agent_speaking
                ):
                    self._last_proactive_check = now
                    await self._generate_response(trigger="proactive")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.session_id}] Proactive worker error: {e}")

    # ── State broadcasting ──────────────────────────────────────────────

    async def _broadcast_state(self) -> None:
        """Push conversation state to frontend."""
        if self._on_state_change:
            cb = self._on_state_change(self._state)
            if asyncio.iscoroutine(cb):
                await cb

    # ── Utility ─────────────────────────────────────────────────────────

    def get_context_summary(self) -> str:
        """Get a summary of the conversation for external use."""
        user_msgs = [m for m in self._messages if m.role == "user"]
        agent_msgs = [m for m in self._messages if m.role == "assistant"]
        return (
            f"Conversation: {len(user_msgs)} user messages, "
            f"{len(agent_msgs)} agent responses, "
            f"{self._state.turn_count} turns"
        )
