"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import GlassCard from "@/components/ui/GlassCard";
import {
  MessageSquare,
  Send,
  Mic,
  MicOff,
  Bot,
  User,
  Volume2,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  source?: string; // "text" | "voice" | "llm" | "demo"
}

export interface TranscriptEntryData {
  speaker: string;
  text: string;
  timestamp: number;
  confidence: number;
  is_final: boolean;
}

export interface ConversationStateData {
  is_user_speaking: boolean;
  is_agent_speaking: boolean;
  turn_count: number;
  last_user_speech: string;
  last_agent_response: string;
}

interface ChatPanelProps {
  className?: string;
  messages: ChatMessageData[];
  transcript: TranscriptEntryData[];
  conversationState: ConversationStateData | null;
  onSendMessage: (text: string) => void;
  sessionActive: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ChatPanel({
  className,
  messages,
  transcript,
  conversationState,
  onSendMessage,
  sessionActive,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [activeTab, setActiveTab] = useState<"chat" | "transcript">("chat");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    if (activeTab === "chat") {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, activeTab]);

  useEffect(() => {
    if (activeTab === "transcript") {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [transcript, activeTab]);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || !sessionActive) return;
    onSendMessage(text);
    setInput("");
    inputRef.current?.focus();
  }, [input, onSendMessage, sessionActive]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const isUserSpeaking = conversationState?.is_user_speaking ?? false;
  const isAgentSpeaking = conversationState?.is_agent_speaking ?? false;
  const turnCount = conversationState?.turn_count ?? 0;

  // Filter out system messages for display
  const displayMessages = messages.filter((m) => m.role !== "system");

  return (
    <GlassCard className={cn("p-0 flex flex-col", className)} hover={false} glow="violet">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-[#8B5CF6]" />
          <h3 className="text-sm font-semibold text-white/80">AI Communication</h3>
        </div>
        <div className="flex items-center gap-2">
          {/* Speaking indicators */}
          <AnimatePresence>
            {isUserSpeaking && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#4F8CFF]/10 border border-[#4F8CFF]/20"
              >
                <Mic className="w-3 h-3 text-[#4F8CFF]" />
                <span className="text-[10px] text-[#4F8CFF]">User speaking</span>
              </motion.div>
            )}
          </AnimatePresence>
          <AnimatePresence>
            {isAgentSpeaking && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#8B5CF6]/10 border border-[#8B5CF6]/20"
              >
                <Volume2 className="w-3 h-3 text-[#8B5CF6]" />
                <span className="text-[10px] text-[#8B5CF6]">AI speaking</span>
              </motion.div>
            )}
          </AnimatePresence>
          {turnCount > 0 && (
            <span className="text-[10px] text-white/30">
              {turnCount} turn{turnCount !== 1 ? "s" : ""}
            </span>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/[0.06]">
        <button
          onClick={() => setActiveTab("chat")}
          className={cn(
            "flex-1 px-4 py-2 text-xs font-medium transition-colors",
            activeTab === "chat"
              ? "text-[#8B5CF6] border-b-2 border-[#8B5CF6]"
              : "text-white/40 hover:text-white/60"
          )}
        >
          Chat ({displayMessages.length})
        </button>
        <button
          onClick={() => setActiveTab("transcript")}
          className={cn(
            "flex-1 px-4 py-2 text-xs font-medium transition-colors",
            activeTab === "transcript"
              ? "text-[#4F8CFF] border-b-2 border-[#4F8CFF]"
              : "text-white/40 hover:text-white/60"
          )}
        >
          Transcript ({transcript.length})
        </button>
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-y-auto min-h-[200px] max-h-[400px]">
        {activeTab === "chat" ? (
          <div className="p-3 space-y-3">
            {displayMessages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-[180px] text-white/20">
                <Bot className="w-8 h-8 mb-2 opacity-50" />
                <p className="text-xs">
                  {sessionActive
                    ? "AI agent is listening. Type a message or speak..."
                    : "Start a session to chat with the AI coach"}
                </p>
              </div>
            ) : (
              displayMessages.map((msg) => (
                <ChatBubble key={msg.id} message={msg} />
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        ) : (
          <div className="p-3 space-y-2">
            {transcript.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-[180px] text-white/20">
                <MicOff className="w-8 h-8 mb-2 opacity-50" />
                <p className="text-xs">
                  {sessionActive
                    ? "Waiting for speech..."
                    : "Start a session to see live transcript"}
                </p>
              </div>
            ) : (
              transcript.map((entry, i) => (
                <TranscriptBubble key={`${entry.timestamp}-${i}`} entry={entry} />
              ))
            )}
            <div ref={transcriptEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="p-3 border-t border-white/[0.06]">
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              sessionActive
                ? "Type a message to the AI coach..."
                : "Start a session first"
            }
            disabled={!sessionActive}
            className="flex-1 bg-white/[0.04] border border-white/10 rounded-lg px-3 py-2 text-xs text-white/80 placeholder:text-white/20 focus:outline-none focus:ring-2 focus:ring-[#8B5CF6]/40 disabled:opacity-40 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSend}
            disabled={!sessionActive || !input.trim()}
            className="p-2 rounded-lg bg-[#8B5CF6]/15 text-[#8B5CF6] border border-[#8B5CF6]/20 hover:bg-[#8B5CF6]/25 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </GlassCard>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ChatBubble({ message }: { message: ChatMessageData }) {
  const isUser = message.role === "user";
  const isVoice = message.source === "voice";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex gap-2",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center",
          isUser
            ? "bg-[#4F8CFF]/15 border border-[#4F8CFF]/20"
            : "bg-[#8B5CF6]/15 border border-[#8B5CF6]/20"
        )}
      >
        {isUser ? (
          <User className="w-3 h-3 text-[#4F8CFF]" />
        ) : (
          <Bot className="w-3 h-3 text-[#8B5CF6]" />
        )}
      </div>

      {/* Bubble */}
      <div
        className={cn(
          "max-w-[80%] rounded-xl px-3 py-2",
          isUser
            ? "bg-[#4F8CFF]/10 border border-[#4F8CFF]/15"
            : "bg-[#8B5CF6]/10 border border-[#8B5CF6]/15"
        )}
      >
        <p className="text-xs text-white/80 leading-relaxed">{message.content}</p>
        <div className="flex items-center gap-1.5 mt-1">
          {isVoice && <Mic className="w-2.5 h-2.5 text-white/20" />}
          <span className="text-[10px] text-white/20">
            {new Date(message.timestamp * 1000).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        </div>
      </div>
    </motion.div>
  );
}

function TranscriptBubble({ entry }: { entry: TranscriptEntryData }) {
  const isUser = entry.speaker === "user" || entry.speaker === "participant";

  return (
    <motion.div
      initial={{ opacity: 0, x: isUser ? 10 : -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(
        "flex items-start gap-2 text-xs",
        !entry.is_final && "opacity-50"
      )}
    >
      <span
        className={cn(
          "flex-shrink-0 text-[10px] font-medium uppercase tracking-wider",
          isUser ? "text-[#4F8CFF]/60" : "text-[#8B5CF6]/60"
        )}
      >
        {isUser ? "USER" : "AI"}
      </span>
      <span className="text-white/60 leading-relaxed">{entry.text}</span>
      {!entry.is_final && (
        <motion.span
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="text-[10px] text-[#F59E0B]"
        >
          ...
        </motion.span>
      )}
    </motion.div>
  );
}
