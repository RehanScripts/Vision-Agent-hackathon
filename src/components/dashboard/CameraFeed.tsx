"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useCallback } from "react";
import Badge from "@/components/ui/Badge";
import { ScanLine, VideoOff, Camera, AlertTriangle, Wifi, WifiOff } from "lucide-react";
import { useWebcam } from "@/hooks/useWebcam";
import type { ConnectionStatus, SessionStatus } from "@/hooks/useMetricsStream";

interface CameraFeedProps {
  connectionStatus?: ConnectionStatus;
  sessionStatus?: SessionStatus;
  streaming?: boolean;
}

export default function CameraFeed({
  connectionStatus = "disconnected",
  sessionStatus = "idle",
  streaming: _streaming = false,
}: CameraFeedProps) {
  void _streaming;
  const {
    status: cameraStatus,
    error: cameraError,
    videoRef,
    startCamera,
    stopCamera,
  } = useWebcam();

  const handleToggleCamera = useCallback(async () => {
    if (cameraStatus === "active") {
      stopCamera();
    } else {
      await startCamera();
    }
  }, [cameraStatus, startCamera, stopCamera]);

  const isLive = cameraStatus === "active";
  const isConnected = connectionStatus === "connected";
  const isAnalyzing = isLive && isConnected && sessionStatus !== "idle";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="relative w-full aspect-video rounded-2xl overflow-hidden border border-white/[0.08] bg-black/40"
    >
      {/* Neon border glow */}
      <div className="absolute inset-0 rounded-2xl shadow-[inset_0_0_30px_rgba(79,140,255,0.08)]" />

      {/* Live video element */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-500 ${
          isLive ? "opacity-100" : "opacity-0"
        }`}
        style={{ transform: "scaleX(-1)" }}
      />

      {/* Scanning grid overlay (visible when analyzing) */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: `
                linear-gradient(rgba(79,140,255,0.5) 1px, transparent 1px),
                linear-gradient(90deg, rgba(79,140,255,0.5) 1px, transparent 1px)
              `,
              backgroundSize: "40px 40px",
            }}
          />
        )}
      </AnimatePresence>

      {/* Scanning line animation */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#4F8CFF]/40 to-transparent"
            style={{ animation: "scanline 4s ease-in-out infinite" }}
          />
        )}
      </AnimatePresence>

      {/* Camera placeholder — when NOT active */}
      <AnimatePresence>
        {!isLive && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex flex-col items-center justify-center"
          >
            {cameraStatus === "denied" || cameraStatus === "error" ? (
              <div className="flex flex-col items-center gap-3">
                <motion.div
                  animate={{ scale: [1, 1.05, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="w-20 h-20 rounded-full bg-[#EF4444]/10 border border-[#EF4444]/20 flex items-center justify-center"
                >
                  {cameraStatus === "denied" ? (
                    <VideoOff className="w-8 h-8 text-[#EF4444]/60" />
                  ) : (
                    <AlertTriangle className="w-8 h-8 text-[#F59E0B]/60" />
                  )}
                </motion.div>
                <p className="text-white/40 text-sm max-w-xs text-center">
                  {cameraError || "Camera unavailable"}
                </p>
                <button
                  onClick={handleToggleCamera}
                  className="px-4 py-2 text-xs font-medium text-white/70 bg-white/[0.06] border border-white/10 rounded-lg hover:bg-white/[0.1] transition-colors"
                >
                  Retry Camera
                </button>
              </div>
            ) : cameraStatus === "requesting" ? (
              <div className="flex flex-col items-center gap-3">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="w-16 h-16 rounded-full border-2 border-white/10 border-t-[#4F8CFF]/50"
                />
                <p className="text-white/40 text-sm">Requesting camera access...</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <motion.div
                  animate={{ opacity: [0.3, 0.6, 0.3] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  className="w-20 h-20 rounded-full bg-white/[0.04] border border-white/[0.08] flex items-center justify-center cursor-pointer hover:bg-white/[0.06] transition-colors"
                  onClick={handleToggleCamera}
                >
                  <Camera className="w-8 h-8 text-white/30" />
                </motion.div>
                <p className="text-white/20 text-sm">Click to start camera</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Top badges */}
      <div className="absolute top-4 left-4 flex items-center gap-2">
        {isAnalyzing ? (
          <Badge variant="critical" pulse>LIVE ANALYSIS</Badge>
        ) : isLive ? (
          <Badge variant="green" pulse>CAMERA ON</Badge>
        ) : (
          <Badge variant="neutral">OFFLINE</Badge>
        )}
      </div>

      <div className="absolute top-4 right-4 flex items-center gap-2">
        {isConnected ? (
          <Badge variant="blue"><Wifi className="w-3 h-3" />Connected</Badge>
        ) : connectionStatus === "reconnecting" ? (
          <Badge variant="warning"><WifiOff className="w-3 h-3" />Reconnecting</Badge>
        ) : (
          <Badge variant="neutral"><WifiOff className="w-3 h-3" />Offline</Badge>
        )}
        {isAnalyzing && (
          <Badge variant="blue"><ScanLine className="w-3 h-3" />AI Tracking</Badge>
        )}
      </div>

      {/* Camera toggle button (when live) */}
      {isLive && (
        <motion.button
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          onClick={handleToggleCamera}
          className="absolute bottom-14 right-4 p-2 rounded-lg bg-black/40 border border-white/10 text-white/50 hover:text-white/80 hover:bg-black/60 transition-all"
        >
          <VideoOff className="w-4 h-4" />
        </motion.button>
      )}

      {/* Bottom status bar */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/60 to-transparent">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <motion.div
              animate={{
                opacity: isLive ? [1, 0.4, 1] : 0.3,
                backgroundColor: isLive ? "#EF4444" : "#666666",
              }}
              transition={{ duration: 1.5, repeat: isLive ? Infinity : 0 }}
              className="w-2 h-2 rounded-full"
            />
            <span className="text-xs text-white/50">
              {isAnalyzing ? "Analyzing" : isLive ? "Camera Active" : "Standby"}
            </span>
          </div>
          <span className="text-xs text-white/30 font-mono">
            {isLive ? "1280×720 • 30fps" : "—"}
          </span>
        </div>
      </div>

      {/* Corner markers */}
      {["top-3 left-3 rotate-0", "top-3 right-3 rotate-90", "bottom-3 left-3 -rotate-90", "bottom-3 right-3 rotate-180"].map((pos) => (
        <svg key={pos} className={`absolute ${pos} w-6 h-6 text-[#4F8CFF]/30`}>
          <path d="M0 8V2a2 2 0 012-2h6" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      ))}

      <style jsx>{`
        @keyframes scanline {
          0%, 100% { top: 0%; }
          50% { top: 100%; }
        }
      `}</style>
    </motion.div>
  );
}
