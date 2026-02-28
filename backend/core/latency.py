"""
SpeakAI — Structured Latency Tracer

Records wall-clock timestamps for critical milestones:
  join_started → agent_joined → first_frame → first_transcript → first_response

Computes and logs latency deltas.  This is the single source of truth
for latency diagnostics — no ad-hoc timing elsewhere.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("speakai.latency")


@dataclass
class LatencyTrace:
    """Immutable record of session latency milestones (wall-clock seconds)."""

    session_id: str = ""

    join_started: float = 0.0
    agent_joined: float = 0.0
    first_frame: float = 0.0
    first_transcript: float = 0.0
    first_response: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "session_id": self.session_id,
        }
        # Only include milestones that have been recorded
        for name in ("join_started", "agent_joined", "first_frame",
                      "first_transcript", "first_response"):
            ts = getattr(self, name)
            if ts > 0:
                d[name] = ts
        # Computed deltas
        d["deltas"] = self.deltas()
        return d

    def deltas(self) -> Dict[str, Optional[float]]:
        """Compute latency deltas between milestones (milliseconds)."""
        def _delta(a: float, b: float) -> Optional[float]:
            if a > 0 and b > 0:
                return round((b - a) * 1000, 1)
            return None

        return {
            "join_to_agent_ms": _delta(self.join_started, self.agent_joined),
            "agent_to_first_frame_ms": _delta(self.agent_joined, self.first_frame),
            "agent_to_first_transcript_ms": _delta(self.agent_joined, self.first_transcript),
            "agent_to_first_response_ms": _delta(self.agent_joined, self.first_response),
            "join_to_first_response_ms": _delta(self.join_started, self.first_response),
        }


class LatencyTracer:
    """
    Mutable tracer that records milestones and logs them.

    Usage:
        tracer = LatencyTracer("session-abc")
        tracer.mark_join_started()
        tracer.mark_agent_joined()
        tracer.mark_first_frame()
        tracer.mark_first_transcript()
        tracer.mark_first_response()
    """

    def __init__(self, session_id: str) -> None:
        self._trace = LatencyTrace(session_id=session_id)

    @property
    def trace(self) -> LatencyTrace:
        return self._trace

    def mark_join_started(self) -> None:
        self._trace.join_started = time.time()
        logger.info(f"[{self._trace.session_id}] LATENCY join_started")

    def mark_agent_joined(self) -> None:
        if self._trace.agent_joined > 0:
            return  # Already marked
        self._trace.agent_joined = time.time()
        delta = self._trace.deltas()
        logger.info(
            f"[{self._trace.session_id}] LATENCY agent_joined "
            f"(join→agent: {delta['join_to_agent_ms']}ms)"
        )

    def mark_first_frame(self) -> None:
        if self._trace.first_frame > 0:
            return
        self._trace.first_frame = time.time()
        delta = self._trace.deltas()
        logger.info(
            f"[{self._trace.session_id}] LATENCY first_frame "
            f"(agent→frame: {delta['agent_to_first_frame_ms']}ms)"
        )

    def mark_first_transcript(self) -> None:
        if self._trace.first_transcript > 0:
            return
        self._trace.first_transcript = time.time()
        delta = self._trace.deltas()
        logger.info(
            f"[{self._trace.session_id}] LATENCY first_transcript "
            f"(agent→transcript: {delta['agent_to_first_transcript_ms']}ms)"
        )

    def mark_first_response(self) -> None:
        if self._trace.first_response > 0:
            return
        self._trace.first_response = time.time()
        delta = self._trace.deltas()
        logger.info(
            f"[{self._trace.session_id}] LATENCY first_response "
            f"(agent→response: {delta['agent_to_first_response_ms']}ms, "
            f"join→response: {delta['join_to_first_response_ms']}ms)"
        )

    def summary(self) -> Dict[str, Any]:
        return self._trace.to_dict()
