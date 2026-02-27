"""
SpeakAI — Vision Agent Backend Server

FastAPI server that:
1. Runs a Vision Agent using GetStream's edge network + Gemini VLM + YOLO Pose
2. Processes webcam frames for public speaking metrics via MediaPipe
3. Streams metrics to the Next.js frontend over WebSocket
4. Generates coaching feedback in real-time

Architecture:
  Browser ↔ GetStream WebRTC ↔ Vision Agent (Gemini + YOLO)
  Browser ↔ WebSocket ↔ This Server (metrics + feedback)
  Browser ↔ getUserMedia → WebSocket frames → MediaPipe analysis
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from speaking_coach_processor import (
    SpeakingCoachAnalyzer,
    FeedbackEngine,
    MetricsBroadcaster,
    SpeakingMetrics,
    broadcaster,
)

load_dotenv()

logger = logging.getLogger("speakai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

analyzer = SpeakingCoachAnalyzer(fps=5)
feedback_engine = FeedbackEngine(cooldown_seconds=6.0)

# Track session state
session_state = {
    "active": False,
    "started_at": None,
    "frame_count": 0,
    "agent_status": "idle",  # idle | initializing | running | error
}

# Vision Agent reference (optional — when SDK is installed)
vision_agent = None
vision_agent_task = None


# ---------------------------------------------------------------------------
# Vision Agent Setup (optional — requires SDK + API keys)
# ---------------------------------------------------------------------------

async def try_start_vision_agent():
    """Attempt to initialize the Vision Agents SDK agent."""
    global vision_agent, session_state

    try:
        from vision_agents.core import Agent, User
        from vision_agents.plugins import gemini, getstream, deepgram, elevenlabs

        session_state["agent_status"] = "initializing"

        agent = Agent(
            edge=getstream.Edge(),
            agent_user=User(
                name="SpeakAI Coach",
                id="speakai-coach",
            ),
            instructions=(
                "You are an AI public speaking coach. Watch the user's video feed "
                "and provide real-time coaching feedback on their:\n"
                "- Eye contact and gaze direction\n"
                "- Posture and body language\n"
                "- Speaking pace and clarity\n"
                "- Facial engagement and expression\n"
                "- Head stability\n\n"
                "Be encouraging but direct. Give short, actionable tips. "
                "Focus on one improvement at a time."
            ),
            llm=gemini.Realtime(fps=2),
            stt=deepgram.STT(eager_turn_detection=True),
            tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        )

        vision_agent = agent
        session_state["agent_status"] = "running"
        logger.info("✅ Vision Agent initialized with Gemini Realtime + Deepgram + ElevenLabs")
        return agent

    except ImportError:
        session_state["agent_status"] = "unavailable"
        logger.info("ℹ️  Vision Agents SDK not installed — running in standalone metrics mode")
        return None
    except Exception as e:
        session_state["agent_status"] = "error"
        logger.error(f"❌ Vision Agent initialization failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Simulated Metrics Stream (fallback when no camera frames arrive)
# ---------------------------------------------------------------------------

async def simulated_metrics_loop():
    """Generates simulated metrics for demo/testing purposes."""
    logger.info("🎬 Starting simulated metrics stream")
    while session_state["active"]:
        metrics = analyzer._simulated_metrics()
        await broadcaster.broadcast(metrics.to_dict())

        # Check for feedback
        feedback = feedback_engine.evaluate(metrics)
        if feedback:
            await broadcaster.broadcast_feedback(feedback)

        await asyncio.sleep(0.2)  # 5 FPS


# ---------------------------------------------------------------------------
# FastAPI Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("🚀 SpeakAI Backend starting...")
    analyzer.initialize()

    # Try starting Vision Agent (non-blocking)
    asyncio.create_task(try_start_vision_agent())

    yield

    # Cleanup
    analyzer.close()
    if vision_agent:
        try:
            await vision_agent.close()
        except Exception:
            pass
    logger.info("🛑 SpeakAI Backend stopped")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SpeakAI — Public Speaking Coach API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "agent_status": session_state["agent_status"],
        "session_active": session_state["active"],
        "frame_count": session_state["frame_count"],
        "connected_clients": len(broadcaster._clients),
    }


@app.post("/session/start")
async def start_session():
    """Start a coaching session."""
    global session_state
    session_state["active"] = True
    session_state["started_at"] = time.time()
    session_state["frame_count"] = 0
    logger.info("▶️  Session started")
    return {"status": "started", "started_at": session_state["started_at"]}


@app.post("/session/stop")
async def stop_session():
    """Stop the active coaching session."""
    global session_state
    duration = 0
    if session_state["started_at"]:
        duration = time.time() - session_state["started_at"]
    session_state["active"] = False
    session_state["started_at"] = None
    logger.info(f"⏹️  Session stopped (duration: {duration:.1f}s)")
    return {
        "status": "stopped",
        "duration_seconds": round(duration, 1),
        "total_frames": session_state["frame_count"],
    }


@app.get("/session/status")
async def session_status():
    """Get current session status."""
    duration = 0
    if session_state["started_at"]:
        duration = time.time() - session_state["started_at"]
    return {
        "active": session_state["active"],
        "duration_seconds": round(duration, 1),
        "frame_count": session_state["frame_count"],
        "agent_status": session_state["agent_status"],
        "latest_metrics": broadcaster.latest_metrics,
    }


# ---------------------------------------------------------------------------
# WebSocket: Metrics Stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket):
    """
    WebSocket endpoint for real-time metrics streaming.

    Client can:
    - Receive metrics updates (type: "metrics")
    - Receive feedback messages (type: "feedback")
    - Send video frames as base64 JPEG (type: "frame")
    - Send control commands (type: "start_session", "stop_session", "start_demo")
    """
    await ws.accept()
    broadcaster.register(ws)

    try:
        while True:
            data = await ws.receive_text()
            message = json.loads(data)

            msg_type = message.get("type", "")

            if msg_type == "frame":
                # Process incoming video frame
                await _handle_frame(message, ws)

            elif msg_type == "start_session":
                session_state["active"] = True
                session_state["started_at"] = time.time()
                session_state["frame_count"] = 0
                await ws.send_text(json.dumps({
                    "type": "session_started",
                    "data": {"started_at": session_state["started_at"]},
                }))

            elif msg_type == "stop_session":
                session_state["active"] = False
                await ws.send_text(json.dumps({
                    "type": "session_stopped",
                    "data": {"frame_count": session_state["frame_count"]},
                }))

            elif msg_type == "start_demo":
                # Start simulated metrics for demo mode
                session_state["active"] = True
                session_state["started_at"] = time.time()
                asyncio.create_task(simulated_metrics_loop())
                await ws.send_text(json.dumps({
                    "type": "demo_started",
                    "data": {"mode": "simulated"},
                }))

            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        broadcaster.unregister(ws)


async def _handle_frame(message: dict, ws: WebSocket):
    """Decode base64 JPEG frame, analyze, broadcast metrics."""
    try:
        frame_data = message.get("data", "")
        if not frame_data:
            return

        # Decode base64 → numpy array
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Analyze frame
        session_state["frame_count"] += 1
        metrics = await analyzer.analyze_frame(frame_rgb)

        # Broadcast metrics to all clients
        await broadcaster.broadcast(metrics.to_dict())

        # Generate feedback if thresholds breached
        feedback = feedback_engine.evaluate(metrics)
        if feedback:
            await broadcaster.broadcast_feedback(feedback)

    except Exception as e:
        logger.error(f"❌ Frame processing error: {e}")


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
