# SpeakAI — Real-Time AI Public Speaking Coach

A premium, futuristic AI SaaS dashboard UI for a real-time public speaking coach. Built with Next.js, Tailwind CSS, Framer Motion, and Recharts.

> ⚠️ This is a **UI-only** project. No backend logic is implemented yet.

## Tech Stack

- **Next.js 16** (App Router)
- **TypeScript**
- **Tailwind CSS v4**
- **Framer Motion** — animations & transitions
- **Lucide React** — icon system
- **Recharts** — analytics charts
- **clsx + tailwind-merge** — utility class composition

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
src/
├── app/
│   ├── layout.tsx              # Root layout
│   ├── page.tsx                # Landing page
│   ├── globals.css             # Global styles
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
│   │   ├── CameraFeed.tsx      # Camera feed placeholder with effects
│   │   └── FeedbackPanel.tsx   # AI feedback floating panel
│   ├── analytics/
│   │   └── PerformanceChart.tsx # Recharts performance line chart
│   └── background/
│       └── GradientBlobs.tsx   # Animated background blobs
└── lib/
    └── utils.ts                # cn() utility function
```

## Getting Started

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
