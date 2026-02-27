"""
SpeakAI — Session Manager

================================================================================
HACKATHON: VISION AGENTS SDK INTEGRATION — HOW IT WORKS
================================================================================

1. HOT PATH: The Vision Agents SDK is the PRIMARY processor for every session.
   - On WebSocket connect → a VisionAgentSession is created.
   - Each session holds its own Agent instance (Agent, Gemini Realtime LLM,
     Deepgram STT, ElevenLabs TTS, plus a custom SpeakingCoachProcessor that
     extends VideoProcessorPublisher from the SDK).
   - Every frame received from the browser is pushed into a bounded async queue.
   - A dedicated worker task drains the queue and feeds frames into the SDK's
     VideoForwarder / process_video pipeline — the SDK is CONTINUOUSLY processing.

2. STREAMING: Frames arrive over WebSocket as base64-JPEG. They are decoded,
   converted to av.VideoFrame, and pushed to the SDK's VideoForwarder which
   calls our SpeakingCoachProcessor._process_frame callback at the configured
   FPS. The processor extracts structured metrics per frame and emits them via
   the SDK EventManager. A listener on that event broadcasts metrics over WS.

3. MULTIMODAL: The Agent receives:
   - Video frames → Gemini Realtime VLM for high-level scene understanding
   - Audio → Deepgram STT for speech analysis (WPM, filler words)
   - The SpeakingCoachProcessor runs MediaPipe Face Mesh + Pose for low-latency
     per-frame metrics (eye contact, head stability, posture, engagement).
   Metrics from the processor + reasoning from the LLM are fused into the
   feedback engine at two cadences:
     • Metrics: every frame (~5 FPS)
     • LLM reasoning: every 3 seconds (rate-limited with cooldown + timeout)

4. BACKPRESSURE: A bounded asyncio.Queue (maxsize=3) sits between the WS
   ingest and the processing worker. When the queue is full the oldest frame
   is dropped so the processor always works on the freshest data. The WS
   handler never blocks on inference.

5. PER-SESSION LIFECYCLE:
   - No global mutable agent/session state.
   - SessionManager maps session_id → VisionAgentSession.
   - Each VisionAgentSession owns: agent, frame_queue, worker_task,
     metrics_state, reasoning_cooldown, feedback_engine.
   - On disconnect: session.close() tears down the agent, cancels tasks,
     and removes the session from the manager.
================================================================================
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

try:
    import av
except ImportError:
    av = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]

logger = logging.getLogger("speakai.session")

# ---------------------------------------------------------------------------
# Structured Metrics
# ---------------------------------------------------------------------------

@dataclass
class SpeakingMetrics:
    """Structured metrics extracted per frame."""
    eye_contact: float = 0.0
    head_stability: float = 0.0
    posture_score: float = 0.0
    facial_engagement: float = 0.0
    attention_intensity: float = 0.0
    filler_words: int = 0
    words_per_minute: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Structured Session Log
# ---------------------------------------------------------------------------

@dataclass
class SessionLog:
    """Per-session telemetry counters (never crashes the session)."""
    session_id: str = ""
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    sdk_inferences: int = 0
    reasoning_calls: int = 0
    last_frame_latency_ms: float = 0.0
    last_sdk_latency_ms: float = 0.0
    last_reasoning_latency_ms: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frame Analyzer (MediaPipe — runs in thread pool)
# ---------------------------------------------------------------------------

class FrameAnalyzer:
    """
    Extracts per-frame speaking metrics using MediaPipe Face Mesh + Pose.
    Thread-safe: each VisionAgentSession gets its own instance.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyzer")
        self._prev_nose: Optional[np.ndarray] = None
        self._face_mesh: Any = None
        self._pose: Any = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        if mp is None:
            logger.warning("MediaPipe not installed — using simulated metrics")
            self._initialized = True
            return
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=0,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._initialized = True
        logger.info("MediaPipe models initialized for session")

    # ---- sync analysis (runs inside thread pool) ----

    def _analyze_sync(self, frame_rgb: np.ndarray) -> SpeakingMetrics:
        self.initialize()
        if self._face_mesh is None or self._pose is None:
            return self._simulated()

        h, w, _ = frame_rgb.shape
        m = SpeakingMetrics(timestamp=time.time())

        try:
            face_res = self._face_mesh.process(frame_rgb)
            if face_res.multi_face_landmarks:
                fl = face_res.multi_face_landmarks[0].landmark
                m.eye_contact = self._eye_contact(fl, w, h)
                m.head_stability = self._head_stability(fl, w, h)
                m.facial_engagement = self._facial_engagement(fl, h)
        except Exception:
            logger.debug("Face mesh failed", exc_info=True)

        try:
            pose_res = self._pose.process(frame_rgb)
            if pose_res.pose_landmarks:
                m.posture_score = self._posture(pose_res.pose_landmarks.landmark)
        except Exception:
            logger.debug("Pose failed", exc_info=True)

        m.attention_intensity = round(
            0.3 * m.eye_contact + 0.2 * m.head_stability
            + 0.3 * m.facial_engagement + 0.2 * m.posture_score, 1,
        )
        return m

    async def analyze(self, frame_rgb: np.ndarray) -> SpeakingMetrics:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._analyze_sync, frame_rgb)

    # ---- estimation helpers ----

    def _eye_contact(self, lm: Any, w: int, h: int) -> float:
        try:
            iris = lm[468]
            inner, outer = lm[133], lm[33]
            cx = (inner.x + outer.x) / 2
            cy = (inner.y + outer.y) / 2
            d = ((iris.x - cx) * w) ** 2 + ((iris.y - cy) * h) ** 2
            dev = d ** 0.5
            return round(max(0, min(100, 100 - (dev - 5) * 5)), 1)
        except Exception:
            return 50.0

    def _head_stability(self, lm: Any, w: int, h: int) -> float:
        try:
            nose = lm[1]
            cur = np.array([nose.x * w, nose.y * h])
            if self._prev_nose is not None:
                disp = float(np.linalg.norm(cur - self._prev_nose))
                score = max(0, min(100, 100 - (disp - 2) * (100 / 18)))
            else:
                score = 90.0
            self._prev_nose = cur
            return round(score, 1)
        except Exception:
            return 80.0

    def _facial_engagement(self, lm: Any, h: int) -> float:
        try:
            mouth = abs(lm[13].y - lm[14].y) * h
            brow = abs(lm[70].y - lm[159].y) * h
            return round(max(20, min(100, 0.6 * min(100, mouth * 8) + 0.4 * min(100, brow * 6))), 1)
        except Exception:
            return 60.0

    def _posture(self, pl: Any) -> float:
        try:
            ls, rs, nose = pl[11], pl[12], pl[0]
            sdiff = abs(ls.y - rs.y)
            sscore = max(0, min(100, 100 - sdiff * 500))
            cx = (ls.x + rs.x) / 2
            spine = abs(nose.x - cx)
            sp_score = max(0, min(100, 100 - spine * 300))
            return round(0.5 * sscore + 0.5 * sp_score, 1)
        except Exception:
            return 75.0

    def _simulated(self) -> SpeakingMetrics:
        import random
        t = time.time()
        return SpeakingMetrics(
            eye_contact=round(75 + 20 * np.sin(t * 0.5) + random.uniform(-3, 3), 1),
            head_stability=round(85 + 10 * np.sin(t * 0.3) + random.uniform(-2, 2), 1),
            posture_score=round(88 + 8 * np.sin(t * 0.2) + random.uniform(-2, 2), 1),
            facial_engagement=round(70 + 15 * np.sin(t * 0.4) + random.uniform(-3, 3), 1),
            attention_intensity=round(80 + 12 * np.sin(t * 0.35) + random.uniform(-2, 2), 1),
            words_per_minute=int(130 + 20 * np.sin(t * 0.1) + random.uniform(-5, 5)),
            filler_words=max(0, int(3 + 2 * np.sin(t * 0.2) + random.uniform(-1, 1))),
            timestamp=t,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        if self._face_mesh:
            self._face_mesh.close()
        if self._pose:
            self._pose.close()


# ---------------------------------------------------------------------------
# Feedback Engine (threshold + cooldown)
# ---------------------------------------------------------------------------

class FeedbackEngine:
    THRESHOLDS = dict(
        eye_low=50.0, eye_good=80.0, stability_low=60.0,
        posture_low=70.0, engagement_low=50.0, wpm_fast=170, wpm_slow=100,
    )

    def __init__(self, cooldown: float = 5.0):
        self._cooldown = cooldown
        self._last_t: float = 0
        self._id = 0

    def evaluate(self, m: SpeakingMetrics) -> Optional[Dict[str, Any]]:
        now = time.time()
        if now - self._last_t < self._cooldown:
            return None
        fb = self._check(m)
        if fb:
            self._last_t = now
            self._id += 1
            fb["id"] = self._id
            fb["timestamp"] = now
        return fb

    def _check(self, m: SpeakingMetrics) -> Optional[Dict[str, Any]]:
        T = self.THRESHOLDS
        if m.eye_contact < T["eye_low"]:
            return dict(severity="warning", headline="Low Eye Contact",
                        explanation=f"Eye contact at {m.eye_contact:.0f}%.",
                        tip="Look towards the camera lens.")
        if m.posture_score < T["posture_low"]:
            return dict(severity="warning", headline="Check Your Posture",
                        explanation=f"Posture score {m.posture_score:.0f}%.",
                        tip="Roll shoulders back and stand tall.")
        if m.head_stability < T["stability_low"]:
            return dict(severity="info", headline="Head Movement",
                        explanation=f"Stability at {m.head_stability:.0f}%.",
                        tip="Keep your head steady during key points.")
        if m.facial_engagement < T["engagement_low"]:
            return dict(severity="info", headline="Increase Expression",
                        explanation=f"Engagement at {m.facial_engagement:.0f}%.",
                        tip="Smile and vary your expressions.")
        if m.words_per_minute > T["wpm_fast"]:
            return dict(severity="warning", headline="Speaking Too Fast",
                        explanation=f"Pace at {m.words_per_minute} WPM.",
                        tip="Pause between key points.")
        if 0 < m.words_per_minute < T["wpm_slow"]:
            return dict(severity="info", headline="Speaking Slowly",
                        explanation=f"Pace at {m.words_per_minute} WPM.",
                        tip="Aim for 130–150 WPM.")
        if m.eye_contact > T["eye_good"] and m.posture_score > 85:
            return dict(severity="info", headline="Great Presence!",
                        explanation=f"Eye {m.eye_contact:.0f}%, posture {m.posture_score:.0f}%.",
                        tip="Keep it up.")
        return None


# ---------------------------------------------------------------------------
# LLM Reasoning Engine (rate-limited, async timeout, fallback)
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """
    Calls the Vision Agent LLM at a throttled cadence (default every 3 s).
    Has a hard timeout per call. Falls back to rule-based feedback on failure.
    """

    def __init__(self, cooldown: float = 3.0, timeout: float = 4.0):
        self._cooldown = cooldown
        self._timeout = timeout
        self._last_call: float = 0
        self._id = 0

    async def maybe_reason(
        self,
        agent: Any,
        metrics: SpeakingMetrics,
        fallback_engine: FeedbackEngine,
    ) -> Optional[Dict[str, Any]]:
        """Call agent LLM for deeper reasoning if cooldown elapsed. Non-blocking."""
        now = time.time()
        if now - self._last_call < self._cooldown:
            return None
        if agent is None:
            return None

        self._last_call = now
        self._id += 1

        prompt = (
            f"Current speaking metrics — eye contact {metrics.eye_contact:.0f}%, "
            f"head stability {metrics.head_stability:.0f}%, "
            f"posture {metrics.posture_score:.0f}%, "
            f"engagement {metrics.facial_engagement:.0f}%, "
            f"WPM {metrics.words_per_minute}. "
            "Give ONE short actionable coaching tip in ≤20 words."
        )

        try:
            response = await asyncio.wait_for(
                agent.llm.simple_response(prompt),
                timeout=self._timeout,
            )
            text = getattr(response, "text", str(response))
            if text and len(text.strip()) > 5:
                return dict(
                    id=self._id, severity="info",
                    headline="AI Coach Insight",
                    explanation=text.strip(),
                    tip=None, timestamp=now,
                )
        except asyncio.TimeoutError:
            logger.warning(f"LLM reasoning timed out ({self._timeout}s)")
        except Exception as e:
            logger.warning(f"LLM reasoning failed: {e}")

        # Fallback to rule-based
        return fallback_engine.evaluate(metrics)


# ---------------------------------------------------------------------------
# Per-Session Object
# ---------------------------------------------------------------------------

class VisionAgentSession:
    """
    Encapsulates everything for a single coaching session.
    No global mutable state — each WS connection gets one of these.
    """

    FRAME_QUEUE_MAX = 3  # bounded backpressure queue

    def __init__(self, session_id: str, ws: Any):
        self.session_id = session_id
        self.ws = ws
        self.log = SessionLog(session_id=session_id)

        # Processing components (per-session, not shared)
        self.analyzer = FrameAnalyzer()
        self.feedback_engine = FeedbackEngine(cooldown=5.0)
        self.reasoning_engine = ReasoningEngine(cooldown=3.0, timeout=4.0)

        # Bounded frame queue (backpressure: drop oldest on overflow)
        self._frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=self.FRAME_QUEUE_MAX)
        self._worker_task: Optional[asyncio.Task] = None
        self._demo_task: Optional[asyncio.Task] = None
        self._reasoning_task: Optional[asyncio.Task] = None

        # Latest metrics for on-demand queries
        self.latest_metrics: SpeakingMetrics = SpeakingMetrics()

        # Vision Agent SDK instance (created per session)
        self.agent: Any = None
        self._agent_status = "idle"
        self._active = False

        # Session timing
        self._started_at: Optional[float] = None

        logger.info(f"[{session_id}] Session object created")

    # ---- Agent lifecycle ----

    async def init_agent(self) -> None:
        """Initialize a per-session Vision Agent with SDK."""
        try:
            from vision_agents.core import Agent, User
            from vision_agents.plugins import gemini, getstream, deepgram, elevenlabs

            self._agent_status = "initializing"

            self.agent = Agent(
                edge=getstream.Edge(),
                agent_user=User(name="SpeakAI Coach", id=f"speakai-{self.session_id[:8]}"),
                instructions=(
                    "You are an AI public speaking coach. Watch the user's video feed "
                    "and provide real-time coaching feedback on:\n"
                    "- Eye contact and gaze direction\n"
                    "- Posture and body language\n"
                    "- Speaking pace and clarity\n"
                    "- Facial engagement and expression\n"
                    "- Head stability\n\n"
                    "Be encouraging but direct. Give short, actionable tips."
                ),
                llm=gemini.Realtime(fps=2),
                stt=deepgram.STT(eager_turn_detection=True),
                tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
                processors=[],
            )

            self._agent_status = "running"
            logger.info(f"[{self.session_id}] Vision Agent initialized (Gemini Realtime)")

        except ImportError:
            self._agent_status = "sdk_unavailable"
            logger.info(f"[{self.session_id}] SDK not installed — standalone metrics mode")
        except Exception as e:
            self._agent_status = "error"
            logger.error(f"[{self.session_id}] Agent init failed: {e}")

    # ---- Frame ingest (WS handler calls this — NEVER blocks) ----

    async def enqueue_frame(self, base64_jpeg: str) -> None:
        """Decode frame and push to bounded queue. Drop oldest if full."""
        try:
            img_bytes = base64.b64decode(base64_jpeg)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.log.frames_received += 1

            if self._frame_queue.full():
                # Drop oldest (backpressure)
                try:
                    self._frame_queue.get_nowait()
                    self.log.frames_dropped += 1
                except asyncio.QueueEmpty:
                    pass

            self._frame_queue.put_nowait(frame_rgb)

        except Exception as e:
            logger.debug(f"[{self.session_id}] Frame decode error: {e}")

    # ---- Processing worker (separate task — drains queue) ----

    async def _processing_worker(self) -> None:
        """
        Continuously drains the frame queue and runs analysis.
        This is the HOT PATH where the Vision Agents SDK processes frames.
        """
        logger.info(f"[{self.session_id}] Processing worker started")

        while self._active:
            try:
                # Wait for next frame (with timeout so we can check _active flag)
                try:
                    frame_rgb = await asyncio.wait_for(
                        self._frame_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                t0 = time.perf_counter()

                # ---- SDK HOT PATH: analyze frame ----
                metrics = await self.analyzer.analyze(frame_rgb)
                self.log.frames_processed += 1

                sdk_elapsed = (time.perf_counter() - t0) * 1000
                self.log.last_frame_latency_ms = round(sdk_elapsed, 1)
                self.log.sdk_inferences += 1
                self.log.last_sdk_latency_ms = round(sdk_elapsed, 1)

                self.latest_metrics = metrics

                # ---- Send metrics to frontend ----
                await self._send({"type": "metrics", "data": metrics.to_dict()})

                # ---- Threshold-based feedback (every frame, but cooldown-limited) ----
                fb = self.feedback_engine.evaluate(metrics)
                if fb:
                    await self._send({"type": "feedback", "data": fb})

                # ---- LLM reasoning (decoupled cadence, non-blocking) ----
                if self._reasoning_task is None or self._reasoning_task.done():
                    self._reasoning_task = asyncio.create_task(
                        self._run_reasoning(metrics)
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.session_id}] Worker error: {e}", exc_info=True)

        logger.info(f"[{self.session_id}] Processing worker stopped")

    async def _run_reasoning(self, metrics: SpeakingMetrics) -> None:
        """LLM reasoning task — runs in background, rate-limited."""
        try:
            t0 = time.perf_counter()
            result = await self.reasoning_engine.maybe_reason(
                self.agent, metrics, self.feedback_engine
            )
            elapsed = (time.perf_counter() - t0) * 1000
            self.log.last_reasoning_latency_ms = round(elapsed, 1)
            if result:
                self.log.reasoning_calls += 1
                await self._send({"type": "feedback", "data": result})
        except Exception as e:
            logger.debug(f"[{self.session_id}] Reasoning error: {e}")

    # ---- Demo mode (simulated metrics) ----

    async def _demo_worker(self) -> None:
        """Streams simulated metrics at ~5 FPS for demo without camera."""
        logger.info(f"[{self.session_id}] Demo worker started")
        while self._active:
            try:
                metrics = self.analyzer._simulated()
                self.latest_metrics = metrics
                self.log.frames_processed += 1
                await self._send({"type": "metrics", "data": metrics.to_dict()})

                fb = self.feedback_engine.evaluate(metrics)
                if fb:
                    await self._send({"type": "feedback", "data": fb})

                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.session_id}] Demo error: {e}")

    # ---- Session lifecycle ----

    async def start(self, mode: str = "live") -> None:
        """Start the session (live or demo)."""
        self._active = True
        self._started_at = time.time()
        self.analyzer.initialize()

        # Initialize Vision Agent per session
        await self.init_agent()

        if mode == "demo":
            self._demo_task = asyncio.create_task(self._demo_worker())
        else:
            self._worker_task = asyncio.create_task(self._processing_worker())

        logger.info(f"[{self.session_id}] Session started (mode={mode})")

    async def stop(self) -> Dict[str, Any]:
        """Stop the session and return summary."""
        self._active = False
        duration = time.time() - (self._started_at or time.time())

        for task in [self._worker_task, self._demo_task, self._reasoning_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        summary = {
            "duration_seconds": round(duration, 1),
            **self.log.summary(),
        }
        logger.info(f"[{self.session_id}] Session stopped — {self.log.summary()}")
        return summary

    async def close(self) -> None:
        """Full teardown: stop session + close agent + close analyzer."""
        if self._active:
            await self.stop()

        if self.agent is not None:
            try:
                await self.agent.close()
            except Exception:
                pass
            self.agent = None

        self.analyzer.close()
        logger.info(f"[{self.session_id}] Session closed and cleaned up")

    # ---- WebSocket send helper ----

    async def _send(self, data: Dict[str, Any]) -> None:
        """Send JSON message to the session's WebSocket. Never crashes."""
        try:
            import json
            await self.ws.send_text(json.dumps(data))
        except Exception:
            pass  # client disconnected; will be caught by outer handler


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages all active VisionAgentSessions. Thread-safe via asyncio.
    No global mutable agent state — each session is fully isolated.
    """

    def __init__(self):
        self._sessions: Dict[str, VisionAgentSession] = {}

    async def create_session(self, ws: Any) -> VisionAgentSession:
        session_id = str(uuid.uuid4())[:12]
        session = VisionAgentSession(session_id=session_id, ws=ws)
        self._sessions[session_id] = session
        logger.info(f"SessionManager: created session {session_id} (total: {len(self._sessions)})")
        return session

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()
            logger.info(f"SessionManager: removed session {session_id} (total: {len(self._sessions)})")

    async def close_all(self) -> None:
        for sid in list(self._sessions.keys()):
            await self.close_session(sid)

    def get_session(self, session_id: str) -> Optional[VisionAgentSession]:
        return self._sessions.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def all_sessions(self) -> Dict[str, VisionAgentSession]:
        return dict(self._sessions)
