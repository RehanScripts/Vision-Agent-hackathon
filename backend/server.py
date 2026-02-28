"""
SpeakAI — FastAPI Server

================================================================================
Architecture:
  • Per-WebSocket session backed by AIService (real Stream call participant)
    or DemoService (simulated fallback)
  • AIService creates a Vision Agent that joins a Stream Video call:
      - Subscribes to participant audio + video tracks
      - Processes frames via SpeakingCoachProcessor (VideoProcessorPublisher)
      - Runs Gemini Realtime for multimodal understanding
      - Publishes TTS audio responses back into the call
  • No base64 frame hacks — the Agent uses Stream's edge media routing
  • Structured metrics + feedback streamed to frontend over WebSocket
  • System status debug messages every 5 s
================================================================================

Endpoints:
  WS  /ws/metrics           — real-time session stream
  GET /health               — server health
  GET /sessions             — list active sessions with telemetry
  GET /session/{session_id} — single session detail

Client → Server messages:
  { type: "start_session", call_id: "..." }  → join a Stream call
  { type: "start_demo" }                     → simulated metrics
  { type: "stop_session" }                   → stop current service
  { type: "send_message", text: "..." }      → send chat to AI agent
  { type: "ping" }                           → keepalive

Server → Client messages:
  { type: "metrics", data: {...} }           → per-frame metrics
  { type: "feedback", data: {...} }          → coaching feedback
  { type: "chat", data: {...} }              → conversation message
  { type: "transcript", data: {...} }        → speech transcript entry
  { type: "conversation_state", data: {...} }→ conversation state
  { type: "system_status", payload: {...} }  → debug telemetry
  { type: "session_started", data: {...} }   → ack
  { type: "session_stopped", data: {...} }   → ack + summary
  { type: "demo_started", data: {...} }      → ack
  { type: "pong" }                           → keepalive ack
  { type: "error", message: "..." }          → error
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import server_cfg, sdk_cfg
from .services.ai_service import AIService
from .services.registry import ServiceRegistry
from .services.demo_service import DemoService

import jwt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("speakai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Service Registry
# ---------------------------------------------------------------------------

registry = ServiceRegistry()

# Track demo services separately (they don't go through AIService)
_demo_services: Dict[str, DemoService] = {}

# ---------------------------------------------------------------------------
# FastAPI Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 SpeakAI Backend starting...")
    logger.info(f"   Stream keys configured: {sdk_cfg.has_all_keys}")
    yield
    logger.info("🛑 Shutting down — closing all services...")
    await registry.stop_all()
    for sid in list(_demo_services.keys()):
        await _demo_services.pop(sid).stop()
    logger.info("🛑 SpeakAI Backend stopped")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SpeakAI — Real-Time AI Speaking Coach",
    version="4.1.0",
    description=(
        "AI agent that joins Stream Video calls as a real participant, "
        "analyses the speaker's video+audio via Vision Agents SDK, and "
        "publishes spoken coaching responses back into the call."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(server_cfg.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "4.1.0",
        "sdk_keys_configured": sdk_cfg.has_all_keys,
        "active_sessions": registry.active_count,
        "active_demos": len(_demo_services),
    }


@app.get("/token")
async def token(user_id: str):
    if not sdk_cfg.stream_api_key or not sdk_cfg.stream_api_secret:
        return {"error": "Stream API keys not configured"}

    now = int(time.time())
    payload = {
        "user_id": user_id,
        "iat": now,
        "exp": now + 3600,
    }
    stream_token = jwt.encode(payload, sdk_cfg.stream_api_secret, algorithm="HS256")

    return {
        "api_key": sdk_cfg.stream_api_key,
        "token": stream_token,
        "user_id": user_id,
    }


@app.get("/sessions")
async def list_sessions():
    result: Dict[str, Any] = {}
    for sid, svc in registry.all_services.items():
        result[sid] = {
            "active": svc.is_active,
            "telemetry": svc.telemetry.to_dict(),
        }
    for sid, demo in _demo_services.items():
        result[sid] = {
            "active": demo.is_active,
            "mode": "demo",
            "telemetry": demo.telemetry.to_dict(),
        }
    return result


@app.get("/session/{session_id}")
async def session_detail(session_id: str):
    svc = registry.get(session_id)
    if svc:
        return {
            "session_id": session_id,
            "active": svc.is_active,
            "telemetry": svc.telemetry.to_dict(),
        }
    demo = _demo_services.get(session_id)
    if demo:
        return {
            "session_id": session_id,
            "active": demo.is_active,
            "mode": "demo",
            "telemetry": demo.telemetry.to_dict(),
        }
    return {"error": "session not found"}


# ---------------------------------------------------------------------------
# Video Upload Analysis
# ---------------------------------------------------------------------------

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accept a video upload, run MediaPipe frame-by-frame analysis,
    and return aggregated speaking metrics + timeline.
    """
    import tempfile, os

    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        return JSONResponse(status_code=400, content={"error": f"Expected video, got {content_type}"})

    try:
        suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)[:200]}"})

    try:
        result = await asyncio.get_running_loop().run_in_executor(None, _process_video_file, tmp_path)
        return result
    except Exception as e:
        logger.error(f"Video analysis error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {str(e)[:200]}"})
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _process_video_file(video_path: str) -> Dict[str, Any]:
    """Process video frames with PyAV (Vision Agents SDK) + MediaPipe."""
    import av as _av

    from .processing.processor import SpeakingCoachProcessor

    try:
        container = _av.open(video_path)
    except Exception as e:
        return {"error": f"Could not open video: {str(e)[:200]}"}

    video_stream = container.streams.video[0]
    total_frames = video_stream.frames or 0
    fps = float(video_stream.average_rate or video_stream.guessed_rate or 30)
    duration_s = float(video_stream.duration * video_stream.time_base) if video_stream.duration else (total_frames / fps if fps > 0 else 0)
    width = video_stream.codec_context.width
    height = video_stream.codec_context.height
    sample_interval = max(1, int(fps / 3))  # ~3 FPS analysis

    processor = SpeakingCoachProcessor(fps=3, max_workers=1)
    all_metrics: list = []
    frame_idx = 0

    for frame in container.decode(video=0):
        if frame_idx % sample_interval == 0:
            frame_rgb = frame.to_ndarray(format="rgb24")
            all_metrics.append(processor._analyze_sync(frame_rgb))
        frame_idx += 1

    container.close()
    if processor._face_mesh:
        processor._face_mesh.close()
    if processor._pose:
        processor._pose.close()
    processor._executor.shutdown(wait=False)

    if total_frames == 0:
        total_frames = frame_idx

    if not all_metrics:
        return {"error": "No frames could be processed from the video"}

    def avg(f: str) -> float:
        return round(sum(getattr(m, f) for m in all_metrics) / len(all_metrics), 1)

    timeline = []
    for i, m in enumerate(all_metrics):
        timeline.append({
            "time_s": round(i * (sample_interval / fps), 1),
            "eye_contact": round(m.eye_contact, 1),
            "head_stability": round(m.head_stability, 1),
            "posture_score": round(m.posture_score, 1),
            "facial_engagement": round(m.facial_engagement, 1),
            "attention_intensity": round(m.attention_intensity, 1),
        })

    overall = round((avg("eye_contact") + avg("head_stability") + avg("posture_score") + avg("facial_engagement")) / 4, 1)

    return {
        "status": "ok",
        "video_info": {
            "duration_s": round(duration_s, 1),
            "total_frames": total_frames,
            "fps": round(fps, 1),
            "resolution": f"{width}x{height}",
            "frames_analyzed": len(all_metrics),
        },
        "metrics": {
            "eye_contact": avg("eye_contact"),
            "head_stability": avg("head_stability"),
            "posture_score": avg("posture_score"),
            "facial_engagement": avg("facial_engagement"),
            "attention_intensity": avg("attention_intensity"),
            "overall_score": overall,
        },
        "timeline": timeline,
    }


# ---------------------------------------------------------------------------
# WebSocket: Per-Session Metrics Stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket):
    """
    WebSocket endpoint — one AIService or DemoService per connection.
    The AI agent joins a Stream Video call as a real participant.
    Metrics and feedback are streamed back to the frontend.
    """
    await ws.accept()

    session_id = uuid.uuid4().hex[:12]
    current_service: Any = None  # AIService or DemoService
    start_task: Any = None

    # Helper to send JSON safely
    async def send(data: Dict[str, Any]) -> None:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            pass

    # Callbacks for AIService / DemoService
    async def on_metrics(m: Any) -> None:
        await send({"type": "metrics", "data": m.to_dict()})

    async def on_feedback(fb: Any) -> None:
        await send({"type": "feedback", "data": fb.to_dict()})

    async def on_status(status: Dict[str, Any]) -> None:
        await send({"type": "system_status", "payload": status})

    # Communication callbacks (chat, transcript, conversation state)
    async def on_chat(msg: Any) -> None:
        await send({"type": "chat", "data": msg.to_dict()})

    async def on_transcript(entry: Any) -> None:
        await send({"type": "transcript", "data": entry.to_dict()})

    async def on_conversation_state(state: Any) -> None:
        await send({"type": "conversation_state", "data": state.to_dict()})

    try:
        while True:
            raw = await ws.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type", "")

            # ── Start live session (AI agent joins a Stream call) ──
            if msg_type == "start_session":
                if current_service is not None:
                    await send({"type": "error", "message": "Session already active"})
                    continue

                call_id = message.get("call_id", f"speakai-{session_id}")

                if not sdk_cfg.has_all_keys:
                    await send({
                        "type": "error",
                        "message": (
                            "Live agent unavailable. Set STREAM_API_KEY, STREAM_API_SECRET, "
                            "GEMINI_API_KEY, and ELEVENLABS_API_KEY."
                        ),
                    })
                    continue

                try:
                    service = registry.create(
                        session_id=session_id,
                        on_metrics=on_metrics,
                        on_feedback=on_feedback,
                        on_status=on_status,
                        on_chat=on_chat,
                        on_transcript=on_transcript,
                        on_conversation_state=on_conversation_state,
                    )
                    current_service = service
                    await send({
                        "type": "session_starting",
                        "data": {
                            "session_id": session_id,
                            "call_id": call_id,
                            "session_state": "init",
                            "session_mode": "unavailable",
                        },
                    })

                    async def _start_live_session() -> None:
                        try:
                            info = await service.start(call_id=call_id)
                            await send({"type": "session_started", "data": info})
                        except Exception as e:
                            logger.error(f"[{session_id}] Failed to start AI service: {e}")
                            await registry.stop_service(session_id)
                            await send({
                                "type": "error",
                                "message": f"Failed to start AI agent: {str(e)[:100]}",
                            })

                    start_task = asyncio.create_task(_start_live_session())
                except Exception as e:
                    logger.error(f"[{session_id}] Failed to start AI service: {e}")
                    await registry.stop_service(session_id)
                    await send({
                        "type": "error",
                        "message": f"Failed to start AI agent: {str(e)[:100]}",
                    })

            # ── Start demo session ──
            elif msg_type == "start_demo":
                if current_service is not None:
                    await send({"type": "error", "message": "Session already active"})
                    continue

                demo = DemoService(
                    session_id=session_id,
                    on_metrics=on_metrics,
                    on_feedback=on_feedback,
                    on_status=on_status,
                )
                info = await demo.start()
                _demo_services[session_id] = demo
                current_service = demo
                await send({"type": "demo_started", "data": info})

            # ── Stop session ──
            elif msg_type == "stop_session":
                if current_service is None:
                    continue

                if start_task and not start_task.done():
                    start_task.cancel()
                    try:
                        await start_task
                    except BaseException:
                        pass
                    start_task = None

                if isinstance(current_service, DemoService):
                    summary = await current_service.stop()
                    _demo_services.pop(session_id, None)
                else:
                    summary = await registry.stop_service(session_id) or {}

                current_service = None
                await send({"type": "session_stopped", "data": summary})

            # ── Send chat message to AI agent ──
            elif msg_type == "send_message":
                text = message.get("text", "").strip()
                if not text:
                    continue
                if current_service is None:
                    await send({"type": "error", "message": "No active session"})
                    continue
                if isinstance(current_service, DemoService):
                    # In demo mode, echo back a simulated response
                    await send({"type": "chat", "data": {
                        "id": uuid.uuid4().hex[:8],
                        "role": "user",
                        "content": text,
                        "timestamp": time.time(),
                        "source": "text",
                    }})
                    await send({"type": "chat", "data": {
                        "id": uuid.uuid4().hex[:8],
                        "role": "assistant",
                        "content": "I'm in demo mode. Start a live session to chat with the AI coach!",
                        "timestamp": time.time(),
                        "source": "demo",
                    }})
                else:
                    try:
                        await current_service.send_chat(text)
                    except Exception as e:
                        logger.debug(f"[{session_id}] Chat send error: {e}")
                        await send({"type": "error", "message": "Failed to send message"})

            # ── Keepalive ──
            elif msg_type == "ping":
                await send({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}", exc_info=True)
    finally:
        # Clean up on disconnect
        if start_task and not start_task.done():
            start_task.cancel()
            try:
                await start_task
            except BaseException:
                pass

        if current_service is not None:
            if isinstance(current_service, DemoService):
                await current_service.stop()
                _demo_services.pop(session_id, None)
            else:
                await registry.stop_service(session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.server:app",
        host=server_cfg.host,
        port=server_cfg.port,
        reload=True,
        log_level="info",
    )
