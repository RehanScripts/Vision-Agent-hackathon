/**
 * SpeakAI — Webcam Hook
 *
 * Manages getUserMedia webcam access with:
 * - Permission state tracking
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
  /** Auto-start on mount */
  autoStart?: boolean;
}

export function useWebcam(options: UseWebcamOptions = {}) {
  const { width = 1280, height = 720, autoStart = false } = options;

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);

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
  };
}
