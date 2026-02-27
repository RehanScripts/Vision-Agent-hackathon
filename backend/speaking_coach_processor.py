"""
SpeakAI — Speaking Coach Processor (Compatibility Shim)

This module re-exports the refactored classes from session_manager.py
for backwards compatibility. All processing logic now lives in
session_manager.py with per-session isolation.

For the primary implementation see: session_manager.py
"""

from session_manager import (
    SpeakingMetrics,
    FrameAnalyzer as SpeakingCoachAnalyzer,
    FeedbackEngine,
    SessionManager,
    VisionAgentSession,
    SessionLog,
    ReasoningEngine,
)

__all__ = [
    "SpeakingMetrics",
    "SpeakingCoachAnalyzer",
    "FeedbackEngine",
    "SessionManager",
    "VisionAgentSession",
    "SessionLog",
    "ReasoningEngine",
]
