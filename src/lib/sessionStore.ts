/**
 * SpeakAI — Session Store
 *
 * localStorage-backed persistence for completed sessions.
 * Keeps the last 100 sessions.  All data is typed and serializable.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SessionMetricsSnapshot {
  eye_contact: number;
  head_stability: number;
  posture_score: number;
  facial_engagement: number;
  attention_intensity: number;
  filler_words: number;
  words_per_minute: number;
}

export interface StoredSession {
  id: string;
  title: string;
  type: string; // "Interview" | "Pitch" | "MUN Debate" | "Presentation" | "Free Practice"
  date: string; // ISO date string
  durationSeconds: number;
  score: number; // 0-100 overall composite
  metrics: SessionMetricsSnapshot;
  /** Timeline snapshots taken every 30s during a session */
  timeline: Array<{ time: number; metrics: SessionMetricsSnapshot }>;
}

export interface UserSettings {
  cameraPreview: boolean;
  interruptions: boolean;
  darkMode: boolean;
  sessionType: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SESSIONS_KEY = "speakai_sessions";
const SETTINGS_KEY = "speakai_settings";
const MAX_SESSIONS = 100;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readJSON<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJSON<T>(key: string, data: T): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(data));
  } catch {
    // quota exceeded — silently ignore
  }
}

// ---------------------------------------------------------------------------
// Session CRUD
// ---------------------------------------------------------------------------

export function getSessions(): StoredSession[] {
  return readJSON<StoredSession[]>(SESSIONS_KEY, []);
}

export function addSession(session: StoredSession): void {
  const sessions = getSessions();
  sessions.unshift(session); // newest first
  writeJSON(SESSIONS_KEY, sessions.slice(0, MAX_SESSIONS));
}

export function clearSessions(): void {
  writeJSON(SESSIONS_KEY, []);
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export const DEFAULT_SETTINGS: UserSettings = {
  cameraPreview: true,
  interruptions: true,
  darkMode: true,
  sessionType: "Interview",
};

export function getSettings(): UserSettings {
  return readJSON<UserSettings>(SETTINGS_KEY, DEFAULT_SETTINGS);
}

export function saveSettings(settings: UserSettings): void {
  writeJSON(SETTINGS_KEY, settings);
}

// ---------------------------------------------------------------------------
// Computed analytics from stored sessions
// ---------------------------------------------------------------------------

export interface AnalyticsOverview {
  totalSessions: number;
  totalMinutes: number;
  avgScore: number;
  avgWPM: number;
  avgEyeContact: number;
  avgFillerWords: number;
  avgPosture: number;
  scoreChange: number; // compared to previous 5 sessions
  /** Per-metric trend: last 10 sessions */
  trend: Array<{
    index: number;
    wpm: number;
    eyeContact: number;
    posture: number;
    score: number;
  }>;
}

export function computeAnalytics(): AnalyticsOverview {
  const sessions = getSessions();
  const n = sessions.length;

  if (n === 0) {
    return {
      totalSessions: 0,
      totalMinutes: 0,
      avgScore: 0,
      avgWPM: 0,
      avgEyeContact: 0,
      avgFillerWords: 0,
      avgPosture: 0,
      scoreChange: 0,
      trend: [],
    };
  }

  const totalMinutes = Math.round(
    sessions.reduce((s, x) => s + x.durationSeconds, 0) / 60
  );
  const avg = (arr: number[]) =>
    arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0;

  const scores = sessions.map((s) => s.score);
  const recentScores = scores.slice(0, 5);
  const olderScores = scores.slice(5, 10);

  const scoreChange =
    olderScores.length > 0 ? avg(recentScores) - avg(olderScores) : 0;

  // Build trend from last 10 sessions (reversed so oldest first)
  const last10 = sessions.slice(0, 10).reverse();
  const trend = last10.map((s, i) => ({
    index: i,
    wpm: s.metrics.words_per_minute,
    eyeContact: Math.round(s.metrics.eye_contact),
    posture: Math.round(s.metrics.posture_score),
    score: s.score,
  }));

  return {
    totalSessions: n,
    totalMinutes,
    avgScore: avg(scores),
    avgWPM: avg(sessions.map((s) => s.metrics.words_per_minute)),
    avgEyeContact: avg(sessions.map((s) => Math.round(s.metrics.eye_contact))),
    avgFillerWords: avg(sessions.map((s) => s.metrics.filler_words)),
    avgPosture: avg(sessions.map((s) => Math.round(s.metrics.posture_score))),
    scoreChange,
    trend,
  };
}
