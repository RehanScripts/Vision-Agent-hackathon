"""
SpeakAI — Vision Agent Backend Server (Refactored)

================================================================================
Architecture:
  - Per-WebSocket-session Vision Agent lifecycle (no global mutable state)
  - Bounded frame queue with backpressure (drop-oldest, maxsize=3)
  - Dedicated processing worker task per session
  - Metrics at ~5 FPS, LLM reasoning decoupled at ~3 s cadence
  - Structured logging with session_id and latency telemetry
  - All frame decode / SDK inference / LLM calls wrapped in try/except
================================================================================

Endpoints:
  WS /ws/metrics  — real-time bi-directional session stream
  GET /health      — server + session overview
  GET /sessions    — list active sessions with telemetry
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from session_manager import SessionManager, VisionAgentSession

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("speakai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Global Session Manager (not mutable agent state — just a registry)
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
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
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
        "uptime": "running",
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions with telemetry."""
    result = {}
    for sid, sess in session_mgr.all_sessions.items():
        result[sid] = {
            "agent_status": sess._agent_status,
            "active": sess._active,
            "log": sess.log.summary(),
            "latest_metrics": sess.latest_metrics.to_dict(),
        }
    return result


# ---------------------------------------------------------------------------
# WebSocket: Per-Session Metrics Stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket):
    """
    WebSocket endpoint — one VisionAgentSession per connection.

    Client messages:
      { type: "start_session" }         → start live analysis
      { type: "start_demo" }            → start simulated metrics
      { type: "stop_session" }          → stop session
      { type: "frame", data: "base64" } → send webcam frame
      { type: "ping" }                  → keepalive

    Server messages:
      { type: "metrics", data: {...} }          → metrics update (~5 FPS)
      { type: "feedback", data: {...} }         → coaching feedback (rate-limited)
      { type: "session_started", data: {...} }  → session started ack
      { type: "session_stopped", data: {...} }  → session stopped ack + summary
      { type: "demo_started", data: {...} }     → demo started ack
      { type: "pong" }                          → keepalive response
    """
    await ws.accept()

    # Create per-connection session
    session: VisionAgentSession = await session_mgr.create_session(ws)
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

            # --- Start live session ---
            elif msg_type == "start_session":
                await session.start(mode="live")
                await ws.send_text(json.dumps({
                    "type": "session_started",
                    "data": {"session_id": session_id, "mode": "live"},
                }))

            # --- Start demo session ---
            elif msg_type == "start_demo":
                await session.start(mode="demo")
                await ws.send_text(json.dumps({
                    "type": "demo_started",
                    "data": {"session_id": session_id, "mode": "simulated"},
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
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
