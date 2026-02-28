"use client";

import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Upload,
  Video,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  X,
  Play,
  Clock,
  Film,
  BarChart3,
  Eye,
  PersonStanding,
  Activity,
  Smile,
  Brain,
} from "lucide-react";
import GlassCard from "@/components/ui/GlassCard";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface TimelinePoint {
  time_s: number;
  eye_contact: number;
  head_stability: number;
  posture_score: number;
  facial_engagement: number;
  attention_intensity: number;
}

interface AnalysisResult {
  status: string;
  video_info: {
    duration_s: number;
    total_frames: number;
    fps: number;
    resolution: string;
    frames_analyzed: number;
  };
  metrics: {
    eye_contact: number;
    head_stability: number;
    posture_score: number;
    facial_engagement: number;
    attention_intensity: number;
    overall_score: number;
  };
  timeline: TimelinePoint[];
  error?: string;
}

type UploadStatus = "idle" | "selected" | "uploading" | "analyzing" | "done" | "error";

/* ------------------------------------------------------------------ */
/*  Mini sparkline                                                     */
/* ------------------------------------------------------------------ */

function Sparkline({ data, color, height = 32 }: { data: number[]; color: string; height?: number }) {
  if (data.length < 2) return null;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const w = 200;
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${height - ((v - min) / range) * (height - 4) - 2}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${height}`} className="w-full" style={{ height }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ filter: `drop-shadow(0 0 4px ${color}40)` }}
      />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Score ring                                                         */
/* ------------------------------------------------------------------ */

function ScoreRing({ value, color, label, size = 80 }: { value: number; color: string; label: string; size?: number }) {
  const r = (size - 10) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="5" />
          <motion.circle
            cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="5"
            strokeLinecap="round" strokeDasharray={c}
            initial={{ strokeDashoffset: c }}
            animate={{ strokeDashoffset: c * (1 - value / 100) }}
            transition={{ duration: 1.2, ease: [0.25, 0.1, 0.25, 1] as const }}
            style={{ filter: `drop-shadow(0 0 6px ${color}40)` }}
          />
        </svg>
        <span className="absolute text-sm font-bold text-white/80">{Math.round(value)}%</span>
      </div>
      <span className="text-[10px] text-white/40 uppercase tracking-wider">{label}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Feedback generator                                                 */
/* ------------------------------------------------------------------ */

function generateFeedback(m: AnalysisResult["metrics"]): { severity: "good" | "warn" | "bad"; text: string }[] {
  const fb: { severity: "good" | "warn" | "bad"; text: string }[] = [];

  if (m.eye_contact >= 70) fb.push({ severity: "good", text: "Great eye contact — you look confident and engaged." });
  else if (m.eye_contact >= 45) fb.push({ severity: "warn", text: "Eye contact is inconsistent — try looking at the camera more often." });
  else fb.push({ severity: "bad", text: "Low eye contact detected — practice maintaining a steady gaze toward your audience." });

  if (m.posture_score >= 75) fb.push({ severity: "good", text: "Posture is upright and professional." });
  else if (m.posture_score >= 50) fb.push({ severity: "warn", text: "Posture could improve — sit or stand straighter." });
  else fb.push({ severity: "bad", text: "Slouching detected — align your shoulders and sit upright." });

  if (m.head_stability >= 70) fb.push({ severity: "good", text: "Head is steady — no distracting movements." });
  else fb.push({ severity: "warn", text: "Excessive head movement detected — try to keep your head still while speaking." });

  if (m.facial_engagement >= 65) fb.push({ severity: "good", text: "Expressive facial engagement — good energy!" });
  else fb.push({ severity: "warn", text: "Facial expression is flat — add more expression to keep your audience engaged." });

  if (m.overall_score >= 70) fb.push({ severity: "good", text: `Overall score: ${Math.round(m.overall_score)}% — solid performance!` });
  else if (m.overall_score >= 50) fb.push({ severity: "warn", text: `Overall score: ${Math.round(m.overall_score)}% — room for improvement.` });
  else fb.push({ severity: "bad", text: `Overall score: ${Math.round(m.overall_score)}% — practice regularly to improve.` });

  return fb;
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function VideoUploadPanel({ className }: { className?: string }) {
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  /* ---- File selection ---- */
  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("video/")) { setError("Please select a video file."); return; }
    if (f.size > 200 * 1024 * 1024) { setError("Max file size is 200 MB."); return; }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setStatus("selected");
    setError(null);
    setResult(null);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }, [handleFile]);
  const onDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setDragOver(true); }, []);
  const onDragLeave = useCallback(() => setDragOver(false), []);

  /* ---- Analyze ---- */
  const analyze = useCallback(async () => {
    if (!file) return;
    setStatus("uploading");
    setError(null);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const resp = await new Promise<AnalysisResult>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener("progress", (e) => {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 50);
            setProgress(pct);
            if (pct >= 50) setStatus("analyzing");
          }
        });
        xhr.addEventListener("load", () => {
          try { xhr.status >= 200 && xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(JSON.parse(xhr.responseText).error || `Error ${xhr.status}`)); }
          catch { reject(new Error(`Server error ${xhr.status}`)); }
        });
        xhr.addEventListener("error", () => reject(new Error("Network error")));
        xhr.open("POST", "http://localhost:8080/analyze-video");
        xhr.send(formData);

        // Simulate analysis progress after upload completes
        let ap = 50;
        const iv = setInterval(() => { if (ap < 95) { ap += Math.random() * 8; setProgress(Math.min(95, Math.round(ap))); } else clearInterval(iv); }, 500);
        xhr.addEventListener("load", () => clearInterval(iv));
        xhr.addEventListener("error", () => clearInterval(iv));
      });

      if (resp.error) throw new Error(resp.error);
      setProgress(100);
      setResult(resp);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setStatus("error");
    }
  }, [file]);

  /* ---- Reset ---- */
  const reset = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null); setStatus("idle"); setProgress(0); setResult(null); setError(null);
  }, [preview]);

  const metricColor = (v: number) => (v >= 70 ? "#22C55E" : v >= 50 ? "#F59E0B" : "#EF4444");

  /* ================================================================== */

  return (
    <GlassCard className={cn("p-0 overflow-hidden", className)} hover={false}>
      {/* Header */}
      <div className="flex items-center gap-2 px-5 py-3 border-b border-white/[0.06]">
        <Video className="w-4 h-4 text-[#8B5CF6]" />
        <h3 className="text-sm font-semibold text-white/80">Video Analysis</h3>
        <span className="ml-auto text-[10px] text-white/30 uppercase tracking-wider">
          {status === "done" ? "Complete" : status === "analyzing" ? "Processing…" : status === "uploading" ? "Uploading…" : "Upload a video"}
        </span>
      </div>

      <AnimatePresence mode="wait">
        {/* ── IDLE / ERROR: Drop zone ── */}
        {(status === "idle" || status === "error") && (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
            onClick={() => inputRef.current?.click()}
            className={cn(
              "flex flex-col items-center justify-center gap-4 p-10 cursor-pointer transition-colors min-h-[260px]",
              dragOver ? "bg-[#4F8CFF]/10" : "hover:bg-white/[0.02]",
            )}
          >
            <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }} />
            <motion.div
              animate={dragOver ? { scale: 1.1, y: -4 } : { scale: 1, y: 0 }}
              className="w-14 h-14 rounded-2xl bg-[#8B5CF6]/10 border border-[#8B5CF6]/20 flex items-center justify-center"
            >
              <Upload className="w-6 h-6 text-[#8B5CF6]/70" />
            </motion.div>
            <div className="text-center">
              <p className="text-sm font-medium text-white/70">Drop a video here or <span className="text-[#8B5CF6]">browse</span></p>
              <p className="text-xs text-white/30 mt-1">MP4, WebM, MOV — up to 200 MB</p>
            </div>
            {error && (
              <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#EF4444]/10 border border-[#EF4444]/20">
                <AlertTriangle className="w-3.5 h-3.5 text-[#EF4444]" />
                <span className="text-xs text-[#EF4444]/80">{error}</span>
              </div>
            )}
          </motion.div>
        )}

        {/* ── SELECTED: preview + Analyze button ── */}
        {status === "selected" && preview && (
          <motion.div key="preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="relative">
            <video src={preview} className="w-full aspect-video object-cover" controls playsInline />
            <div className="absolute bottom-0 inset-x-0 p-3 bg-gradient-to-t from-black/80 to-transparent flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Film className="w-3.5 h-3.5 text-white/50" />
                <span className="text-xs text-white/60 truncate max-w-[180px]">{file?.name}</span>
                <span className="text-[10px] text-white/30">{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : ""}</span>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={reset} className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"><X className="w-3.5 h-3.5 text-white/60" /></button>
                <button onClick={analyze} className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-lg bg-[#8B5CF6] text-white hover:bg-[#8B5CF6]/90 transition-colors">
                  <Play className="w-3.5 h-3.5" /> Analyze Speaking
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* ── UPLOADING / ANALYZING ── */}
        {(status === "uploading" || status === "analyzing") && (
          <motion.div key="progress" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center gap-5 p-10 min-h-[260px]"
          >
            <div className="relative">
              <Loader2 className="w-12 h-12 text-[#8B5CF6] animate-spin" />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[10px] font-bold text-white/60">{progress}%</span>
              </div>
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-white/70">{status === "uploading" ? "Uploading video…" : "Analyzing with MediaPipe…"}</p>
              <p className="text-xs text-white/30 mt-1">{status === "uploading" ? "Sending to server" : "Processing eye contact, posture & engagement"}</p>
            </div>
            <div className="w-full max-w-xs h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <motion.div className="h-full rounded-full bg-[#8B5CF6]" style={{ boxShadow: "0 0 12px rgba(139,92,246,0.3)" }}
                animate={{ width: `${progress}%` }} transition={{ duration: 0.3 }} />
            </div>
          </motion.div>
        )}

        {/* ── DONE: Full results ── */}
        {status === "done" && result && (
          <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-0">
            {/* Video info bar */}
            <div className="flex items-center gap-3 px-5 py-2.5 bg-white/[0.02] border-b border-white/[0.06] flex-wrap">
              <div className="flex items-center gap-1.5 text-[10px] text-white/40">
                <CheckCircle2 className="w-3 h-3 text-[#22C55E]" /> Analysis Complete
              </div>
              <div className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-white/[0.06] text-white/40">
                <Clock className="w-3 h-3" /> {result.video_info.duration_s}s
              </div>
              <div className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-white/[0.06] text-white/40">
                <Video className="w-3 h-3" /> {result.video_info.resolution}
              </div>
              <div className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-white/[0.06] text-white/40">
                <BarChart3 className="w-3 h-3" /> {result.video_info.frames_analyzed} frames
              </div>
              <button onClick={reset} className="ml-auto text-[10px] text-[#8B5CF6]/70 hover:text-[#8B5CF6] transition-colors flex items-center gap-1">
                <Upload className="w-3 h-3" /> New Video
              </button>
            </div>

            {/* Score rings */}
            <div className="px-5 py-5">
              <div className="flex items-center justify-around flex-wrap gap-4">
                <ScoreRing value={result.metrics.overall_score} color={metricColor(result.metrics.overall_score)} label="Overall" size={96} />
                <ScoreRing value={result.metrics.eye_contact} color="#22C55E" label="Eye Contact" />
                <ScoreRing value={result.metrics.posture_score} color="#8B5CF6" label="Posture" />
                <ScoreRing value={result.metrics.head_stability} color="#4F8CFF" label="Stability" />
                <ScoreRing value={result.metrics.facial_engagement} color="#F59E0B" label="Engagement" />
              </div>
            </div>

            {/* Timeline sparklines */}
            <div className="px-5 pb-4">
              <p className="text-[10px] text-white/30 uppercase tracking-wider mb-3">Timeline</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                {([
                  { label: "Eye Contact", key: "eye_contact" as const, color: "#22C55E", icon: <Eye className="w-3 h-3" /> },
                  { label: "Stability", key: "head_stability" as const, color: "#4F8CFF", icon: <Brain className="w-3 h-3" /> },
                  { label: "Posture", key: "posture_score" as const, color: "#8B5CF6", icon: <PersonStanding className="w-3 h-3" /> },
                  { label: "Engagement", key: "facial_engagement" as const, color: "#F59E0B", icon: <Smile className="w-3 h-3" /> },
                  { label: "Attention", key: "attention_intensity" as const, color: "#EC4899", icon: <Activity className="w-3 h-3" /> },
                ] as const).map(({ label, key, color, icon }) => (
                  <div key={key} className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-2.5">
                    <div className="flex items-center gap-1 mb-1.5">
                      <span style={{ color }} className="opacity-70">{icon}</span>
                      <span className="text-[10px] text-white/40">{label}</span>
                      <span className="ml-auto text-[10px] font-semibold" style={{ color }}>{Math.round(result.metrics[key])}%</span>
                    </div>
                    <Sparkline data={result.timeline.map((t) => t[key])} color={color} height={24} />
                  </div>
                ))}
              </div>
            </div>

            {/* AI Feedback */}
            <div className="px-5 pb-5">
              <p className="text-[10px] text-white/30 uppercase tracking-wider mb-3">Speaking Feedback</p>
              <div className="space-y-2">
                {generateFeedback(result.metrics).map((fb, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 * i, duration: 0.3 }}
                    className={cn(
                      "flex items-start gap-2 px-3 py-2 rounded-lg border text-xs",
                      fb.severity === "good" ? "bg-[#22C55E]/5 border-[#22C55E]/15 text-[#22C55E]/80" :
                      fb.severity === "warn" ? "bg-[#F59E0B]/5 border-[#F59E0B]/15 text-[#F59E0B]/80" :
                                               "bg-[#EF4444]/5 border-[#EF4444]/15 text-[#EF4444]/80"
                    )}
                  >
                    {fb.severity === "good" ? <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" /> :
                     fb.severity === "warn" ? <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" /> :
                                              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />}
                    {fb.text}
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </GlassCard>
  );
}
