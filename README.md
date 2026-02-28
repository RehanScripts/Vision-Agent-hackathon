# SpeakAI — Real-Time AI Public Speaking Coach

A premium AI-powered public speaking coach. The AI agent joins a Stream Video call as a real participant, analyses the speaker's video + audio via Vision Agents SDK (Gemini Realtime + ElevenLabs TTS), and provides real-time spoken coaching feedback.

## Tech Stack

### Frontend
- **Next.js 16** (App Router + Turbopack)
- **TypeScript**
- **Tailwind CSS v4**
- **Framer Motion** — animations & transitions
- **@stream-io/video-react-sdk** — live video calls
- **Lucide React** — icon system
- **Recharts** — analytics charts
- **clsx + tailwind-merge** — utility class composition

### Backend
- **Python 3.13** + **FastAPI** / **Uvicorn**
- **Vision Agents SDK** (`getstream`, `gemini`, `elevenlabs`, `ultralytics`)
- **Gemini Realtime** — multimodal video+audio LLM
- **ElevenLabs TTS** — spoken coaching responses
- **MediaPipe** — per-frame face/pose analysis
- **WebSocket** — real-time metrics streaming

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with hero, features, and CTA |
| `/dashboard` | Live session view with camera feed, metrics, and AI feedback |
| `/analytics` | Performance analytics with charts and AI summary |
| `/history` | Session history list |
| `/practice` | Practice mode scenario selection |
| `/settings` | App preferences and configuration |

## Project Structure

```
├── .env.example             # Environment variable template
├── next.config.ts           # Next.js configuration
├── package.json             # Frontend dependencies
│
├── backend/                 # Python FastAPI backend
│   ├── __init__.py
│   ├── server.py            # FastAPI app — REST + WebSocket endpoints
│   ├── requirements.txt     # Python dependencies
│   ├── core/                # Configuration & data models
│   │   ├── config.py        # Centralised settings from env vars
│   │   └── models.py        # SpeakingMetrics, CoachingFeedback, SessionTelemetry
│   ├── services/            # Session management
│   │   ├── ai_service.py    # AI agent — joins Stream calls as participant
│   │   ├── demo_service.py  # Simulated metrics fallback
│   │   └── registry.py      # Session registry (maps session_id → service)
│   └── processing/          # Video analysis & reasoning
│       ├── processor.py     # SpeakingCoachProcessor (MediaPipe + VideoProcessorPublisher)
│       └── reasoning.py     # ReasoningEngine + ThresholdFeedback
│
└── src/                     # Next.js frontend
    ├── app/
    │   ├── layout.tsx              # Root layout
    │   ├── page.tsx                # Landing page
    │   ├── globals.css             # Global styles & theme
    │   └── (dashboard)/
    │       ├── layout.tsx          # Dashboard shell (sidebar + topbar)
    │       ├── dashboard/page.tsx  # Live session
    │       ├── analytics/page.tsx  # Analytics
    │       ├── history/page.tsx    # Session history
    │       ├── practice/page.tsx   # Practice mode
    │       └── settings/page.tsx   # Settings
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.tsx         # Collapsible sidebar navigation
    │   │   └── Topbar.tsx          # Top navigation bar
    │   ├── ui/
    │   │   ├── GlassCard.tsx       # Reusable glass card component
    │   │   ├── AnimatedNumber.tsx  # Animated count-up numbers
    │   │   ├── Badge.tsx           # Status badges with variants
    │   │   └── MetricCard.tsx      # Metric display cards
    │   ├── dashboard/
    │   │   ├── CameraFeed.tsx      # Local webcam preview with overlays
    │   │   └── StreamCallPanel.tsx # Stream Video call UI (agent joins here)
    │   ├── analytics/
    │   │   └── PerformanceChart.tsx # Recharts performance line chart
    │   └── background/
    │       └── GradientBlobs.tsx   # Animated background blobs
    ├── hooks/
    │   ├── useMetricsStream.ts     # WebSocket hook for real-time metrics
    │   └── useWebcam.ts            # getUserMedia webcam management
    └── lib/
        ├── sessionStore.ts         # localStorage session persistence & analytics
        └── utils.ts                # cn() utility function
```

## Getting Started

### 1. Environment variables

Copy `.env.example` → `.env` and fill in your keys:

```bash
cp .env.example .env
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
cd ..
uvicorn backend.server:app --reload --host 0.0.0.0 --port 8080
```

### 3. Frontend

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the landing page, then navigate to `/dashboard`.

## Design System

- **Background:** `#0B0F19`
- **Glass cards:** `bg-white/5 + backdrop-blur-xl`
- **Accent Blue:** `#4F8CFF`
- **Accent Violet:** `#8B5CF6`
- **Success:** `#22C55E`
- **Warning:** `#F59E0B`
- **Critical:** `#EF4444`
- **Dark theme only**
