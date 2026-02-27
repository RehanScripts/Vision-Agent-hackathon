"""
SpeakAI — FastAPI Server

================================================================================
Architecture:
  • Per-WebSocket-session lifecycle (no global mutable agent state)
  • Bounded frame queue with backpressure (drop-oldest, maxsize=3)
  • Dedicated frame-processing worker per session
  • Reasoning worker decoupled at ~3 s cadence
  • System-status debug messages every 5 s
  • Structured logging with session_id + latency telemetry
  • All decode / inference / reasoning wrapped in try/except
================================================================================

Endpoints:
  WS  /ws/metrics        — real-time bi-directional session stream
  GET /health            — server + session overview
  GET /sessions          — list active sessions with telemetry
  GET /session/{id}      — single session detail
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import server_cfg
from session import SessionManager, CoachSession

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("speakai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Session Manager (registry — not global agent state)
# ---------------------------------------------------------------------------

session_mgr = SessionManager()

# ---------------------------------------------------------------------------
# FastAPI Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 SpeakAI Backend starting...")
    yield
    logger.info("🛑 Shutting down — closing all sessions...")
    await session_mgr.close_all()
    logger.info("🛑 SpeakAI Backend stopped")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SpeakAI — Public Speaking Coach API",
    version="3.0.0",
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
        "active_sessions": session_mgr.active_count,
        "version": "3.0.0",
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions with telemetry."""
    result = {}
    for sid, sess in session_mgr.all_sessions.items():
        result[sid] = {
            "mode": sess._mode,
            "active": sess._active,
            "sdk_active": sess.telemetry.sdk_active,
            "multimodal_active": sess.telemetry.multimodal_active,
            "telemetry": sess.telemetry.to_dict(),
            "latest_metrics": sess._latest_metrics.to_dict(),
        }
    return result


@app.get("/session/{session_id}")
async def session_detail(session_id: str):
    sess = session_mgr.get_session(session_id)
    if sess is None:
        return {"error": "session not found"}
    return {
        "session_id": session_id,
        "mode": sess._mode,
        "active": sess._active,
        "sdk_active": sess.telemetry.sdk_active,
        "multimodal_active": sess.telemetry.multimodal_active,
        "telemetry": sess.telemetry.to_dict(),
        "latest_metrics": sess._latest_metrics.to_dict(),
    }


# ---------------------------------------------------------------------------
# WebSocket: Per-Session Metrics Stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket):
    """
    WebSocket endpoint — one CoachSession per connection.

    Client messages:
      { type: "start_session" }         → start live analysis
      { type: "start_demo" }            → start simulated metrics
      { type: "stop_session" }          → stop session
      { type: "frame", data: "base64" } → send webcam frame
      { type: "audio", data: "base64" } → send audio chunk (multimodal)
      { type: "ping" }                  → keepalive

    Server messages:
      { type: "metrics", data: {...} }           → per-frame metrics
      { type: "feedback", data: {...} }          → coaching feedback
      { type: "system_status", payload: {...} }  → debug telemetry
      { type: "session_started", data: {...} }   → ack
      { type: "session_stopped", data: {...} }   → ack + summary
      { type: "demo_started", data: {...} }      → ack
      { type: "pong" }                           → keepalive ack
    """
    await ws.accept()

    # Create per-connection session
    session: CoachSession = await session_mgr.create_session(ws)
    session_id = session.session_id

    try:
        while True:
            raw = await ws.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type", "")

            # --- Frame: push to bounded queue (NEVER blocks) ---
            if msg_type == "frame":
                frame_data = message.get("data", "")
                if frame_data and session._active:
                    await session.enqueue_frame(frame_data)

            # --- Audio chunk: route to SDK if multimodal ---
            elif msg_type == "audio":
                # Future: route to SDK audio pipeline
                pass

            # --- Start live session ---
            elif msg_type == "start_session":
                info = await session.start(mode="live")
                await ws.send_text(json.dumps({
                    "type": "session_started",
                    "data": info,
                }))

            # --- Start demo session ---
            elif msg_type == "start_demo":
                info = await session.start(mode="demo")
                await ws.send_text(json.dumps({
                    "type": "demo_started",
                    "data": info,
                }))

            # --- Stop session ---
            elif msg_type == "stop_session":
                summary = await session.stop()
                await ws.send_text(json.dumps({
                    "type": "session_stopped",
                    "data": summary,
                }))

            # --- Keepalive ---
            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}", exc_info=True)
    finally:
        # Clean up session on disconnect
        await session_mgr.close_session(session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=server_cfg.host,
        port=server_cfg.port,
        reload=True,
        log_level="info",
    )
