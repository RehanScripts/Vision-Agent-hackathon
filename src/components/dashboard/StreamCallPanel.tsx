"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  StreamVideo,
  StreamCall,
  StreamTheme,
  StreamVideoClient,
  SpeakerLayout,
  CallControls,
} from "@stream-io/video-react-sdk";
import { cn } from "@/lib/utils";
import GlassCard from "@/components/ui/GlassCard";

interface StreamCallPanelProps {
  className?: string;
  onReadyChange?: (ready: boolean) => void;
  onMediaReadyChange?: (ready: boolean) => void;
  onCallIdChange?: (callId: string) => void;
}

type CallStatus = "idle" | "joining" | "joined" | "error";

export default function StreamCallPanel({
  className,
  onReadyChange,
  onMediaReadyChange,
  onCallIdChange,
}: StreamCallPanelProps) {
  const [status, setStatus] = useState<CallStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [callId, setCallId] = useState("speakai-live");
  const [client, setClient] = useState<StreamVideoClient | null>(null);
  const [call, setCall] = useState<ReturnType<StreamVideoClient["call"]> | null>(null);
  const [userId, setUserId] = useState("user-loading");
  const clientRef = useRef<StreamVideoClient | null>(null);
  const callRef = useRef<ReturnType<StreamVideoClient["call"]> | null>(null);

  useEffect(() => {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      setUserId(`user-${crypto.randomUUID().slice(0, 8)}`);
      return;
    }
    setUserId(`user-${Math.random().toString(36).slice(2, 10)}`);
  }, []);

  const apiKey = process.env.NEXT_PUBLIC_STREAM_API_KEY ?? "";

  const leaveCall = useCallback(async () => {
    try {
      await callRef.current?.leave();
    } catch {
      // ignore leave errors
    }
    try {
      await clientRef.current?.disconnectUser?.();
    } catch {
      // ignore disconnect errors
    }
    callRef.current = null;
    clientRef.current = null;
    setCall(null);
    setClient(null);
    setStatus("idle");
    setError(null);
    onReadyChange?.(false);
    onMediaReadyChange?.(false);
  }, [onMediaReadyChange, onReadyChange]);

  const joinCall = useCallback(async () => {
    if (status === "joining" || status === "joined") return;

    if (!apiKey) {
      setStatus("error");
      setError("Missing NEXT_PUBLIC_STREAM_API_KEY in .env");
      return;
    }

    setStatus("joining");
    setError(null);

    if (callRef.current || clientRef.current) {
      await leaveCall();
    }

    try {
      const response = await fetch(
        `http://localhost:8080/token?user_id=${encodeURIComponent(userId)}`
      );
      const data = (await response.json()) as {
        token?: string;
        api_key?: string;
        error?: string;
      };

      if (!response.ok || data.error) {
        throw new Error(data.error || "Failed to fetch Stream token");
      }

      const token = data.token;
      const resolvedKey = data.api_key || apiKey;

      if (!token || !resolvedKey) {
        throw new Error("Missing Stream token or API key");
      }

      const videoClient = new StreamVideoClient({
        apiKey: resolvedKey,
        user: { id: userId, name: userId },
        token,
      });

      const videoCall = videoClient.call("default", callId);
      await videoCall.join({ create: true });

      try {
        await videoCall.camera.enable();
        await videoCall.microphone.enable();
        onMediaReadyChange?.(true);
      } catch {
        onMediaReadyChange?.(false);
        setError("Camera/Mic publish failed. Allow permissions and rejoin call.");
      }

      clientRef.current = videoClient;
      callRef.current = videoCall;
      setClient(videoClient);
      setCall(videoCall);
      setStatus("joined");
      onReadyChange?.(true);
      onCallIdChange?.(callId);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to join call";
      setStatus("error");
      setError(message);
      onReadyChange?.(false);
      onMediaReadyChange?.(false);
    }
  }, [apiKey, callId, leaveCall, onCallIdChange, onMediaReadyChange, onReadyChange, status, userId]);

  useEffect(() => {
    return () => {
      void leaveCall();
    };
    // Run cleanup only on true unmount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <GlassCard className={cn("p-4", className)} hover={false} glow="blue">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-white/40">Stream Call</p>
          <h3 className="text-sm font-semibold text-white/80">Live Video Session</h3>
        </div>
        <span
          className={cn(
            "text-[10px] px-2 py-1 rounded-full border",
            status === "joined" && "text-[#22C55E] border-[#22C55E]/30 bg-[#22C55E]/10",
            status === "joining" && "text-[#F59E0B] border-[#F59E0B]/30 bg-[#F59E0B]/10",
            status === "idle" && "text-white/40 border-white/10 bg-white/5",
            status === "error" && "text-[#EF4444] border-[#EF4444]/30 bg-[#EF4444]/10"
          )}
        >
          {status}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <label className="text-[11px] text-white/40">Call ID</label>
          <input
            value={callId}
            onChange={(event) => setCallId(event.target.value)}
            className="mt-1 w-full rounded-lg bg-white/[0.04] border border-white/10 text-xs px-3 py-2 text-white/70 focus:outline-none focus:ring-2 focus:ring-[#4F8CFF]/40"
          />
        </div>
        <div>
          <label className="text-[11px] text-white/40">User ID</label>
          <div className="mt-1 w-full rounded-lg bg-white/[0.04] border border-white/10 text-xs px-3 py-2 text-white/60">
            <span suppressHydrationWarning>{userId}</span>
          </div>
        </div>
      </div>

      {error && (
        <p className="mt-3 text-xs text-[#EF4444]/80">{error}</p>
      )}

      <div className="mt-4 flex items-center gap-2">
        {status !== "joined" ? (
          <button
            onClick={joinCall}
            disabled={status === "joining" || userId === "user-loading"}
            className="px-3 py-2 rounded-lg text-xs font-medium bg-[#4F8CFF]/15 text-[#4F8CFF] border border-[#4F8CFF]/20 hover:bg-[#4F8CFF]/25 transition-colors"
          >
            {status === "joining" ? "Joining..." : "Join Call"}
          </button>
        ) : (
          <button
            onClick={leaveCall}
            className="px-3 py-2 rounded-lg text-xs font-medium bg-[#EF4444]/15 text-[#EF4444] border border-[#EF4444]/20 hover:bg-[#EF4444]/25 transition-colors"
          >
            Leave Call
          </button>
        )}
        <span className="text-[11px] text-white/40">
          Join the call before starting a live session.
        </span>
      </div>

      {client && call && (
        <div className="mt-4 rounded-xl border border-white/[0.08] bg-black/30 overflow-hidden">
          <StreamVideo client={client}>
            <StreamCall call={call}>
              <StreamTheme className="speakai-stream-theme">
                <div className="h-[220px]">
                  <SpeakerLayout participantsBarPosition="bottom" />
                </div>
                <div className="px-2 pb-2">
                  <CallControls />
                </div>
              </StreamTheme>
            </StreamCall>
          </StreamVideo>
        </div>
      )}
    </GlassCard>
  );
}
