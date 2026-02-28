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

        # Proactive mode
        self._last_proactive_check: float = 0.0
        self._proactive_task: Optional[asyncio.Task] = None
        self._last_proactive_text: str = ""
        self._last_proactive_at: float = 0.0
        self._last_assistant_text: str = ""
        self._last_assistant_at: float = 0.0

        # Visual context (updated by processor)
        self._latest_metrics: Optional[SpeakingMetrics] = None

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

    async def start(self) -> None:
        """Start the conversation manager."""
        self._active = True

        # Start proactive monitoring if enabled
        if conversation_cfg.proactive_mode:
            self._proactive_task = asyncio.create_task(
                self._proactive_worker(), name=f"proactive-{self.session_id}"
            )

        logger.info(f"[{self.session_id}] ConversationManager started")

    async def stop(self) -> Dict[str, Any]:
        """Stop and return conversation summary."""
        self._active = False

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
                msg = ChatMessage(
                    id=uuid.uuid4().hex[:8],
                    role="user",
                    content=entry.text.strip(),
                    source="voice",
                )
                self._messages.append(msg)
                self._state.last_user_speech = entry.text.strip()
                self._state.turn_count += 1

                if self._on_chat_message:
                    cb = self._on_chat_message(msg)
                    if asyncio.iscoroutine(cb):
                        await cb

                # Mark speech end for turn-taking
                self._last_user_speech_end = time.time()
                self._pending_response = True

                # Wait for response delay, then respond
                asyncio.create_task(self._delayed_response())

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

    # ── Response generation ─────────────────────────────────────────────

    async def _generate_response(self, trigger: str = "auto") -> None:
        """
        Generate an AI response using the Agent's LLM.
        Combines conversation context + visual metrics.
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
        Build the prompt with full context and call the LLM.
        Uses Agent.llm if available, otherwise returns None.
        """
        if self._agent is None or not hasattr(self._agent, "llm"):
            return self._fallback_response(trigger)

        # Build context
        context_parts = []

        # Add visual context if available
        if self._latest_metrics:
            m = self._latest_metrics
            context_parts.append(
                f"[Visual observation] Eye contact: {m.eye_contact:.0f}%, "
                f"head stability: {m.head_stability:.0f}%, "
                f"posture: {m.posture_score:.0f}%, "
                f"engagement: {m.facial_engagement:.0f}%, "
                f"WPM: {m.words_per_minute}"
            )

        # Build conversation prompt
        recent_messages = list(self._messages)[-6:]  # Last 6 messages for context
        conversation_text = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in recent_messages
            if msg.role != "system"
        )

        prompt = (
            f"{conversation_cfg.system_prompt}\n\n"
            f"{''.join(context_parts)}\n\n"
            f"Recent conversation:\n{conversation_text}\n\n"
            "Respond naturally and directly to the latest user message. "
            "If it's a question, answer it first. Keep it concise (≤35 words)."
        )

        try:
            timeout = 4.5 if trigger in ("text_input", "voice_input") else 2.5
            response = await asyncio.wait_for(
                self._agent.llm.simple_response(prompt),
                timeout=timeout,
            )
            text = getattr(response, "text", str(response))
            if text and len(text.strip()) > 2:
                return text

            logger.warning(f"[{self.session_id}] LLM returned empty response, using fallback")
            return self._fallback_response(trigger)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.session_id}] LLM response timed out")
            return self._fallback_response(trigger)
        except Exception as e:
            logger.warning(f"[{self.session_id}] LLM call failed: {e}")
            return self._fallback_response(trigger)

    def _fallback_response(self, trigger: str) -> Optional[str]:
        """Generate a simple fallback response when LLM is unavailable."""
        if trigger == "proactive" and self._latest_metrics:
            if not self._has_actionable_metrics():
                return None

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
            if "eye" in user_text:
                return "Focus on looking at the camera lens for 2–3 seconds at a time."
            if "posture" in user_text:
                return "Keep shoulders relaxed, chest open, and sit upright for stronger presence."
            if "pace" in user_text or "speed" in user_text or "wpm" in user_text:
                return "Aim for a steady 120–150 WPM and pause briefly after key points."
            return "I can help with eye contact, posture, pace, or confidence. Ask one specific question."

        return None

    def _has_actionable_metrics(self) -> bool:
        """True only when we have meaningful (non-default) visual metrics."""
        if not self._latest_metrics:
            return False

        m = self._latest_metrics
        if m.timestamp <= 0:
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
        Periodically checks if the agent should proactively comment
        based on visual observations during silence.
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

                # Only speak proactively if there's been silence
                if (
                    silence_duration >= conversation_cfg.silence_prompt_timeout
                    and self._has_actionable_metrics()
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
