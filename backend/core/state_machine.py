"""
SpeakAI — Session State Machine

Enforces the lifecycle: INIT → READY → ACTIVE → DEGRADED → FAILED.
All state transitions go through this module so illegitimate states
are impossible and every transition is logged.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger("speakai.state")


class SessionState(str, Enum):
    """Strict session lifecycle states."""
    INIT = "init"            # Service created, not started
    READY = "ready"          # Agent joined call, waiting for tracks
    ACTIVE = "active"        # All channels healthy (video + audio + model)
    DEGRADED = "degraded"    # Partial capability (e.g. audio-only)
    FAILED = "failed"        # Unrecoverable error


class SessionMode(str, Enum):
    """Explicit operating mode — never inferred silently."""
    MULTIMODAL = "multimodal"    # Video + audio + model
    AUDIO_ONLY = "audio_only"    # Audio only, no video frames
    UNAVAILABLE = "unavailable"  # No capabilities active


# Legal state transitions
_TRANSITIONS: Dict[SessionState, Set[SessionState]] = {
    SessionState.INIT:     {SessionState.READY, SessionState.FAILED},
    SessionState.READY:    {SessionState.ACTIVE, SessionState.DEGRADED, SessionState.FAILED, SessionState.INIT},
    SessionState.ACTIVE:   {SessionState.DEGRADED, SessionState.FAILED, SessionState.INIT},
    SessionState.DEGRADED: {SessionState.ACTIVE, SessionState.FAILED, SessionState.INIT},
    SessionState.FAILED:   {SessionState.INIT},
}


class SessionStateMachine:
    """
    Enforces legal state transitions and notifies listeners.

    Usage:
        sm = SessionStateMachine(on_transition=my_callback)
        sm.transition(SessionState.READY)        # OK
        sm.transition(SessionState.ACTIVE)       # OK
        sm.transition(SessionState.INIT)         # illegal from ACTIVE → raises
    """

    def __init__(
        self,
        on_transition: Optional[Callable[[SessionState, SessionState, str], None]] = None,
    ) -> None:
        self._state = SessionState.INIT
        self._mode = SessionMode.UNAVAILABLE
        self._on_transition = on_transition
        self._history: List[Dict] = []
        self._entered_at = time.time()

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def mode(self) -> SessionMode:
        return self._mode

    @property
    def history(self) -> List[Dict]:
        return list(self._history)

    def transition(self, target: SessionState, reason: str = "") -> None:
        """
        Attempt a state transition. Raises ValueError on illegal transitions.
        """
        if target == self._state:
            return  # Idempotent — no-op for same state

        allowed = _TRANSITIONS.get(self._state, set())
        if target not in allowed:
            raise ValueError(
                f"Illegal state transition: {self._state.value} → {target.value}. "
                f"Allowed from {self._state.value}: {[s.value for s in allowed]}. "
                f"Reason: {reason}"
            )

        prev = self._state
        now = time.time()
        self._history.append({
            "from": prev.value,
            "to": target.value,
            "reason": reason,
            "timestamp": now,
            "duration_in_prev_ms": round((now - self._entered_at) * 1000, 1),
        })
        self._state = target
        self._entered_at = now

        logger.info(
            f"STATE: {prev.value} → {target.value}"
            + (f" ({reason})" if reason else "")
        )

        if self._on_transition:
            try:
                self._on_transition(prev, target, reason)
            except Exception as e:
                logger.error(f"State transition callback error: {e}")

    def set_mode(self, mode: SessionMode) -> None:
        """Update operating mode. Logged when it changes."""
        if mode == self._mode:
            return
        prev = self._mode
        self._mode = mode
        logger.info(f"MODE: {prev.value} → {mode.value}")

    def reset(self) -> None:
        """Reset to INIT state (for stop/cleanup)."""
        if self._state != SessionState.INIT:
            self.transition(SessionState.INIT, reason="reset")
        self._mode = SessionMode.UNAVAILABLE
