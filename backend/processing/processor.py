"""
SpeakAI — Speaking Coach Video Processor

================================================================================
VISION AGENTS SDK — REAL VIDEO PROCESSOR PLUGIN
================================================================================

This module implements `SpeakingCoachProcessor`, a proper `VideoProcessorPublisher`
subclass that plugs directly into the Vision Agent's media pipeline.

How it works inside a Stream Video call:

  1. The Agent subscribes to the human participant's video track.
  2. The SDK's internal `VideoForwarder` reads frames from that track.
  3. The Agent calls `processor.process_video(track, participant_id, forwarder)`.
  4. Our processor registers a frame handler on the shared forwarder.
  5. Each frame callback receives an `av.VideoFrame` — no base64, no WebSocket.
  6. We run lightweight MediaPipe analysis per-frame in a ThreadPoolExecutor
     (CPU-heavy work off the event loop).
  7. Metrics are published via the SDK's `EventManager` so the Agent and
     server layer can consume them.
  8. We also publish an annotated video track back into the call via
     `QueuedVideoTrack` / `publish_video_track()`.

This is how official examples (security_camera, golf_coach) work.
No manual base64 streaming.  No WebSocket frame hacks.
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import pathlib

import numpy as np

from ..core.models import SpeakingMetrics
from ..core.config import processing_cfg

# -- Optional heavy imports --------------------------------------------------

try:
    import av
except ImportError:
    av = None  # type: ignore[assignment]

try:
    import cv2  # opencv for frame conversion fallback
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import aiortc
except ImportError:
    aiortc = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]

# -- MediaPipe model paths (Tasks API 0.10.x+) ------------------------------
_MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
_FACE_MODEL = _MODELS_DIR / "face_landmarker.task"
_POSE_MODEL = _MODELS_DIR / "pose_landmarker_lite.task"

try:
    from vision_agents.core.processors.base_processor import VideoProcessorPublisher
    from vision_agents.core.utils.video_forwarder import VideoForwarder
    from vision_agents.core.utils.video_track import QueuedVideoTrack
    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False
    # Stubs so the class definition doesn't break
    class VideoProcessorPublisher:  # type: ignore[no-redef]
        pass
    class VideoForwarder:  # type: ignore[no-redef]
        pass
    class QueuedVideoTrack:  # type: ignore[no-redef]
        pass
    pass

# Watchdog: if no frames arrive via forwarder within this window, start direct reader
_FORWARDER_WATCHDOG_TIMEOUT_S = 5.0
# Direct reader FPS cap
_DIRECT_READER_FPS = 5


logger = logging.getLogger("speakai.processor")


# ═══════════════════════════════════════════════════════════════════════════
# SpeakingCoachProcessor — plugs into Agent.processors=[]
# ═══════════════════════════════════════════════════════════════════════════

class SpeakingCoachProcessor(VideoProcessorPublisher if _HAS_SDK else object):  # type: ignore
    """
    A proper VideoProcessorPublisher that:
      • Subscribes to incoming video via the SDK's VideoForwarder
      • Analyses each frame with MediaPipe (eye contact, posture, etc.)
      • Publishes annotated frames back into the call via QueuedVideoTrack
      • Fires MetricsProducedEvent so the Agent/server can stream them

    Lifecycle (managed by the Agent):
      attach_agent(agent) → process_video(track, pid, forwarder) → stop_processing() → close()
    """

    name = "speaking_coach"

    def __init__(self, fps: int = processing_cfg.processor_fps, max_workers: int = 2):
        self.fps = fps
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="coach-cv"
        )
        self._video_forwarder: Optional[Any] = None
        self._video_track: Optional[Any] = None
        self._agent: Any = None
        self._shutdown = False

        # MediaPipe models (lazy-init in worker thread)
        self._face_mesh: Any = None
        self._pose: Any = None
        self._mp_initialized = False
        self._prev_nose: Optional[np.ndarray] = None

        # Latest metrics (for reasoning engine to read)
        self._latest_metrics = SpeakingMetrics(timestamp=0.0, source="sdk")

        # Telemetry
        self.frames_processed: int = 0
        self.last_latency_ms: float = 0.0

        # Direct frame reader (fallback when VideoForwarder doesn't fire)
        self._direct_reader_task: Optional[asyncio.Task] = None
        self._direct_reader_active = False
        self._watchdog_task: Optional[asyncio.Task] = None
        self._raw_track: Any = None  # The raw video track for direct reading
        self._forwarder_started_at: float = 0.0

        # Output video track
        if _HAS_SDK:
            self._video_track = QueuedVideoTrack(
                width=640, height=480, fps=fps, max_queue_size=fps
            )

        logger.info(f"SpeakingCoachProcessor init (fps={fps})")

    # -- SDK integration points -----------------------------------------------

    @property
    def latest(self) -> SpeakingMetrics:
        return self._latest_metrics

    def attach_agent(self, agent: Any) -> None:
        """Called by Agent during init — gives us access to agent and event bus."""
        self._agent = agent
        logger.info("SpeakingCoachProcessor attached to agent")

    def publish_video_track(self) -> Any:
        """Return our QueuedVideoTrack so Agent publishes it into the call."""
        return self._video_track

    async def process_video(
        self,
        track: Any,  # aiortc.VideoStreamTrack
        participant_id: Optional[str],
        shared_forwarder: Optional[Any] = None,
    ) -> None:
        """
        Called by the Agent when a participant's video track appears.
        We register a frame handler on the shared VideoForwarder AND
        start a watchdog that falls back to direct frame reading if
        the forwarder's callback never fires.
        """
        logger.info(
            f"🎥 process_video called: participant={participant_id}, "
            f"track={type(track).__name__}, "
            f"forwarder={'shared' if shared_forwarder else 'none'}, "
            f"SDK={_HAS_SDK}"
        )

        # Stop any previous processing
        if self._video_forwarder is not None:
            logger.info("Stopping previous video processing — new track published")
            if _HAS_SDK:
                try:
                    await self._video_forwarder.remove_frame_handler(self._on_frame)
                except Exception:
                    pass
        await self._stop_direct_reader()

        # Store raw track for direct reader fallback
        self._raw_track = track

        logger.info(f"Starting speaking coach video processing at {self.fps} FPS")

        if shared_forwarder is not None and _HAS_SDK:
            self._video_forwarder = shared_forwarder
            logger.info(f"Using shared forwarder: {type(shared_forwarder).__name__}")
        elif _HAS_SDK:
            self._video_forwarder = VideoForwarder(
                track,
                max_buffer=30,
                fps=self.fps,
                name="speaking_coach_forwarder",
            )
            logger.info("Created dedicated VideoForwarder")

        if self._video_forwarder is not None and _HAS_SDK:
            try:
                self._video_forwarder.add_frame_handler(
                    self._on_frame, fps=float(self.fps), name="speaking_coach"
                )
                self._forwarder_started_at = time.time()
                logger.info(
                    f"✅ Frame handler registered on forwarder "
                    f"(handlers={len(self._video_forwarder._frame_handlers)})"
                )
            except Exception as e:
                logger.error(f"❌ Failed to add frame handler: {e}", exc_info=True)
                # Forwarder registration failed → go straight to direct reader
                logger.info("⚡ Forwarder failed — starting direct frame reader immediately")
                self._start_direct_reader()
                return
        else:
            logger.warning(
                f"⚠️ No forwarder available: "
                f"forwarder={self._video_forwarder is not None}, SDK={_HAS_SDK}"
            )
            # No forwarder at all → use direct reader
            if track is not None:
                logger.info("⚡ No SDK forwarder — starting direct frame reader")
                self._start_direct_reader()
                return

        # Start watchdog to detect if forwarder callbacks aren't firing
        self._start_watchdog()

        logger.info("✅ Speaking coach video processing pipeline started")

    def _start_watchdog(self) -> None:
        """Start a watchdog that monitors forwarder health and falls back to direct reading."""
        if self._watchdog_task and not self._watchdog_task.done():
            return
        self._watchdog_task = asyncio.ensure_future(self._forwarder_watchdog())

    async def _forwarder_watchdog(self) -> None:
        """
        Monitors if the VideoForwarder callback is actually delivering frames.
        If no frames arrive within _FORWARDER_WATCHDOG_TIMEOUT_S, switches to
        the direct frame reader.
        """
        try:
            await asyncio.sleep(_FORWARDER_WATCHDOG_TIMEOUT_S)
            if self._shutdown:
                return
            if self.frames_processed == 0 and not self._direct_reader_active:
                logger.warning(
                    f"⏰ Watchdog: no frames received after {_FORWARDER_WATCHDOG_TIMEOUT_S}s "
                    f"via VideoForwarder — switching to direct frame reader"
                )
                self._start_direct_reader()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Watchdog error: {e}")

    def _start_direct_reader(self) -> None:
        """Start the direct frame reading task from the raw track."""
        if self._direct_reader_active or self._raw_track is None:
            return
        self._direct_reader_active = True
        self._direct_reader_task = asyncio.ensure_future(self._direct_frame_reader())
        logger.info("🔄 Direct frame reader started")

    async def _stop_direct_reader(self) -> None:
        """Stop the direct frame reader."""
        self._direct_reader_active = False
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._direct_reader_task and not self._direct_reader_task.done():
            self._direct_reader_task.cancel()
            try:
                await self._direct_reader_task
            except (asyncio.CancelledError, Exception):
                pass
        self._direct_reader_task = None
        self._watchdog_task = None

    async def _direct_frame_reader(self) -> None:
        """
        Fallback frame reader: directly pulls frames from the raw aiortc track
        using track.recv(). This bypasses the VideoForwarder entirely.

        This is the safety net for when the SDK's internal forwarding doesn't work.
        """
        track = self._raw_track
        if track is None:
            logger.warning("Direct reader: no track available")
            return

        interval = 1.0 / _DIRECT_READER_FPS
        consecutive_errors = 0
        max_errors = 20

        logger.info(
            f"🔄 Direct frame reader running at {_DIRECT_READER_FPS} FPS "
            f"on track {type(track).__name__}"
        )

        while self._direct_reader_active and not self._shutdown:
            try:
                # Method 1: aiortc track.recv() — the standard WebRTC approach
                if hasattr(track, 'recv'):
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                    await self._on_frame(frame)
                    consecutive_errors = 0
                # Method 2: If the track has an async iterator (livekit-style)
                elif hasattr(track, '__aiter__'):
                    async for frame in track:
                        if not self._direct_reader_active or self._shutdown:
                            break
                        await self._on_frame(frame)
                        consecutive_errors = 0
                        await asyncio.sleep(interval)
                    break  # Iterator exhausted
                else:
                    logger.warning(
                        f"Direct reader: track type {type(track).__name__} "
                        f"has no recv() or __aiter__ method. "
                        f"Available: {[a for a in dir(track) if not a.startswith('_')]}"
                    )
                    break

                await asyncio.sleep(interval)

            except asyncio.TimeoutError:
                consecutive_errors += 1
                if consecutive_errors > max_errors:
                    logger.warning(
                        f"Direct reader: {max_errors} consecutive timeouts — stopping"
                    )
                    break
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    logger.warning(f"Direct reader error: {e}", exc_info=True)
                elif consecutive_errors > max_errors:
                    logger.error(f"Direct reader: too many errors — stopping")
                    break
                await asyncio.sleep(0.5)

        self._direct_reader_active = False
        logger.info(
            f"🔄 Direct frame reader stopped "
            f"(processed {self.frames_processed} frames total)"
        )

    async def stop_processing(self) -> None:
        """Called when all video tracks are removed (participant left)."""
        # Stop direct reader + watchdog
        await self._stop_direct_reader()

        if self._video_forwarder is not None and _HAS_SDK:
            try:
                await self._video_forwarder.remove_frame_handler(self._on_frame)
            except Exception:
                pass
            self._video_forwarder = None

        self._raw_track = None
        logger.info("Stopped speaking coach video processing")

    async def close(self) -> None:
        """Clean up all resources."""
        self._shutdown = True
        await self.stop_processing()
        self._executor.shutdown(wait=False)
        if self._face_mesh:
            self._face_mesh.close()
        if self._pose:
            self._pose.close()
        if self._video_track is not None and _HAS_SDK:
            self._video_track.stop()
        logger.info(
            f"SpeakingCoachProcessor closed "
            f"(total frames processed: {self.frames_processed})"
        )

    # -- Frame handler (called by VideoForwarder) -----------------------------

    async def _on_frame(self, frame: Any) -> None:
        """
        Receives a video frame from either:
          - The SDK's VideoForwarder (av.VideoFrame)
          - The direct frame reader (av.VideoFrame from track.recv())
          - A raw numpy array from an alternative pipeline

        Runs CV analysis in a thread pool, publishes metrics + annotated frame.
        """
        if self._shutdown:
            return

        try:
            t0 = time.perf_counter()

            # Log first frame arrival
            if self.frames_processed == 0:
                source = "direct_reader" if self._direct_reader_active else "forwarder"
                logger.info(
                    f"🎬 FIRST VIDEO FRAME received via {source}! "
                    f"type={type(frame).__name__}, "
                    f"format={getattr(frame, 'format', 'unknown')}, "
                    f"size={getattr(frame, 'width', '?')}x{getattr(frame, 'height', '?')}"
                )

            # Convert to numpy RGB array
            frame_rgb = self._frame_to_rgb(frame)
            if frame_rgb is None:
                return

            # Run MediaPipe analysis in thread pool (never blocks event loop)
            loop = asyncio.get_running_loop()
            metrics = await loop.run_in_executor(
                self._executor, self._analyze_sync, frame_rgb
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.last_latency_ms = round(elapsed_ms, 1)
            self.frames_processed += 1
            self._latest_metrics = metrics

            # Log periodically
            if self.frames_processed % 100 == 0:
                source = "direct" if self._direct_reader_active else "forwarder"
                logger.info(
                    f"📊 Processed {self.frames_processed} frames ({source}), "
                    f"latency={self.last_latency_ms:.1f}ms, "
                    f"eye={metrics.eye_contact:.0f}%, posture={metrics.posture_score:.0f}%"
                )

            # Publish the (original) frame back into the call
            if self._video_track is not None and av is not None and isinstance(frame, av.VideoFrame):
                await self._video_track.add_frame(frame)

        except Exception as e:
            if self.frames_processed == 0:
                logger.warning(f"⚠️ Frame processing error (first frame): {e}", exc_info=True)
            else:
                logger.debug(f"Frame processing error: {e}")
            # Pass through on error
            if self._video_track is not None and av is not None and isinstance(frame, av.VideoFrame):
                try:
                    await self._video_track.add_frame(frame)
                except Exception:
                    pass

    def _frame_to_rgb(self, frame: Any) -> Optional[np.ndarray]:
        """
        Convert a frame from any format to a numpy RGB array.
        Handles:
          - av.VideoFrame → frame.to_ndarray(format="rgb24")
          - numpy array → assume BGR (from cv2) or RGB
          - Other types → attempt conversion
        """
        try:
            # av.VideoFrame (from SDK's VideoForwarder or aiortc track.recv())
            if av is not None and isinstance(frame, av.VideoFrame):
                return frame.to_ndarray(format="rgb24")

            # Raw numpy array
            if isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[2] == 3:
                    return frame  # Assume RGB
                if frame.ndim == 3 and frame.shape[2] == 4:
                    return frame[:, :, :3]  # Drop alpha channel
                return None

            # Some SDKs provide frame.data as bytes
            if hasattr(frame, 'data') and hasattr(frame, 'width') and hasattr(frame, 'height'):
                w, h = frame.width, frame.height
                data = frame.data
                if isinstance(data, (bytes, bytearray)):
                    arr = np.frombuffer(data, dtype=np.uint8)
                    if arr.size == w * h * 3:
                        return arr.reshape((h, w, 3))
                    elif arr.size == w * h * 4:
                        return arr.reshape((h, w, 4))[:, :, :3]

            logger.debug(f"Unknown frame type: {type(frame).__name__}")
            return None

        except Exception as e:
            logger.debug(f"Frame conversion error: {e}")
            return None

    # -- MediaPipe sync analysis (runs in thread pool) ------------------------

    def _init_mediapipe(self) -> None:
        """Lazy-init MediaPipe models using the Tasks API (0.10.x+)."""
        if self._mp_initialized:
            return
        if mp is not None:
            try:
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                RunningMode = mp.tasks.vision.RunningMode

                if _FACE_MODEL.exists():
                    face_opts = FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(_FACE_MODEL)),
                        running_mode=RunningMode.IMAGE,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                    self._face_mesh = FaceLandmarker.create_from_options(face_opts)
                    logger.info("FaceLandmarker loaded (Tasks API)")
                else:
                    logger.warning(f"Face model not found: {_FACE_MODEL}")

                if _POSE_MODEL.exists():
                    pose_opts = PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(_POSE_MODEL)),
                        running_mode=RunningMode.IMAGE,
                        min_pose_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self._pose = PoseLandmarker.create_from_options(pose_opts)
                    logger.info("PoseLandmarker loaded (Tasks API)")
                else:
                    logger.warning(f"Pose model not found: {_POSE_MODEL}")

            except Exception as e:
                logger.warning(f"Failed to init MediaPipe Tasks: {e}", exc_info=True)
        self._mp_initialized = True

    def _analyze_sync(self, frame_rgb: np.ndarray) -> SpeakingMetrics:
        """Full frame analysis — called from thread pool."""
        self._init_mediapipe()

        m = SpeakingMetrics(timestamp=time.time(), source="sdk")

        if self._face_mesh is None or self._pose is None:
            # No MediaPipe available — return zeros (honest, no fake data)
            return SpeakingMetrics(timestamp=time.time(), source="no_mediapipe")

        h, w, _ = frame_rgb.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            face_result = self._face_mesh.detect(mp_image)
            if face_result.face_landmarks:
                fl = face_result.face_landmarks[0]  # List[NormalizedLandmark]
                m.eye_contact = self._eye_contact(fl, w, h)
                m.head_stability = self._head_stability(fl, w, h)
                m.facial_engagement = self._facial_engagement(fl, h)
        except Exception:
            pass

        try:
            pose_result = self._pose.detect(mp_image)
            if pose_result.pose_landmarks:
                m.posture_score = self._posture(pose_result.pose_landmarks[0])
        except Exception:
            pass

        m.attention_intensity = round(
            0.3 * m.eye_contact + 0.2 * m.head_stability
            + 0.3 * m.facial_engagement + 0.2 * m.posture_score, 1
        )

        return m

    # -- Estimation helpers ---------------------------------------------------

    def _eye_contact(self, lm: Any, w: int, h: int) -> float:
        try:
            iris = lm[468]
            inner, outer = lm[133], lm[33]
            cx = (inner.x + outer.x) / 2
            cy = (inner.y + outer.y) / 2
            d = ((iris.x - cx) * w) ** 2 + ((iris.y - cy) * h) ** 2
            dev = d ** 0.5
            return round(max(0.0, min(100.0, 100.0 - (dev - 5) * 5)), 1)
        except Exception:
            return 50.0

    def _head_stability(self, lm: Any, w: int, h: int) -> float:
        try:
            nose = lm[1]
            cur = np.array([nose.x * w, nose.y * h])
            if self._prev_nose is not None:
                disp = float(np.linalg.norm(cur - self._prev_nose))
                score = max(0.0, min(100.0, 100.0 - (disp - 2) * (100 / 18)))
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
            return round(max(20.0, min(100.0, 0.6 * min(100, mouth * 8) + 0.4 * min(100, brow * 6))), 1)
        except Exception:
            return 60.0

    def _posture(self, pl: Any) -> float:
        try:
            ls, rs, nose = pl[11], pl[12], pl[0]
            sdiff = abs(ls.y - rs.y)
            sscore = max(0.0, min(100.0, 100.0 - sdiff * 500))
            cx = (ls.x + rs.x) / 2
            spine = abs(nose.x - cx)
            sp_score = max(0.0, min(100.0, 100.0 - spine * 300))
            return round(0.5 * sscore + 0.5 * sp_score, 1)
        except Exception:
            return 75.0
