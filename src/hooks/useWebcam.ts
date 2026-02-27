/**
 * SpeakAI — Webcam Hook
 *
 * Manages getUserMedia webcam access with:
 * - Permission state tracking
 * - Frame capture at configurable FPS for WebSocket streaming
 * - Graceful error handling with user-friendly messages
 * - Cleanup on unmount
 */

import { useState, useEffect, useRef, useCallback } from "react";

export type CameraStatus =
  | "idle"
  | "requesting"
  | "active"
  | "denied"
  | "error"
  | "not_found";

export interface UseWebcamOptions {
  /** Desired width */
  width?: number;
  /** Desired height */
  height?: number;
  /** Capture FPS for frame extraction */
  captureFps?: number;
  /** Auto-start on mount */
  autoStart?: boolean;
  /** JPEG quality (0-1) */
  jpegQuality?: number;
}

export function useWebcam(options: UseWebcamOptions = {}) {
  const {
    width = 1280,
    height = 720,
    captureFps = 5,
    autoStart = false,
    jpegQuality = 0.6,
  } = options;

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onFrameRef = useRef<((base64: string) => void) | null>(null);

  // -- Start Camera ----------------------------------------------------------

  const startCamera = useCallback(async () => {
    setStatus("requesting");
    setError(null);

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: width },
          height: { ideal: height },
          facingMode: "user",
        },
        audio: false,
      });

      setStream(mediaStream);
      setStatus("active");

      // Attach to video element
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play().catch(() => {});
      }

      return mediaStream;
    } catch (err: unknown) {
      const error = err as DOMException;
      if (error.name === "NotAllowedError") {
        setStatus("denied");
        setError("Camera permission denied. Please allow camera access.");
      } else if (error.name === "NotFoundError") {
        setStatus("not_found");
        setError("No camera found. Please connect a webcam.");
      } else {
        setStatus("error");
        setError(`Camera error: ${error.message}`);
      }
      return null;
    }
  }, [width, height]);

  // -- Stop Camera -----------------------------------------------------------

  const stopCamera = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setStatus("idle");
    setError(null);
  }, [stream]);

  // -- Frame Capture ---------------------------------------------------------

  const startCapture = useCallback(
    (onFrame: (base64: string) => void) => {
      onFrameRef.current = onFrame;

      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }

      // Create hidden canvas for frame extraction
      if (!canvasRef.current) {
        canvasRef.current = document.createElement("canvas");
      }

      const interval = 1000 / captureFps;

      captureIntervalRef.current = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL("image/jpeg", jpegQuality);
        // Strip the data:image/jpeg;base64, prefix
        const base64 = dataUrl.split(",")[1];
        if (base64 && onFrameRef.current) {
          onFrameRef.current(base64);
        }
      }, interval);
    },
    [captureFps, jpegQuality]
  );

  const stopCapture = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    onFrameRef.current = null;
  }, []);

  // -- Lifecycle -------------------------------------------------------------

  useEffect(() => {
    if (autoStart) {
      startCamera();
    }
    return () => {
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    // State
    status,
    error,
    stream,

    // Refs
    videoRef,

    // Actions
    startCamera,
    stopCamera,
    startCapture,
    stopCapture,
  };
}
