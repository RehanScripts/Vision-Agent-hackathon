"use client";

import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Camera, Mic, User } from "lucide-react";
import Badge from "@/components/ui/Badge";
import { useEffect, useState } from "react";

const pageTitles: Record<string, string> = {
  "/dashboard": "Live Session",
  "/analytics": "Analytics",
  "/history": "Session History",
  "/practice": "Practice Mode",
  "/settings": "Settings",
};

export default function Topbar() {
  const pathname = usePathname();
  const title = pageTitles[pathname] || "Dashboard";

  const [timer, setTimer] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setTimer((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, "0")}:${m
      .toString()
      .padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <motion.header
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="sticky top-0 z-40 h-16 flex items-center justify-between px-6 bg-[#0B0F19]/60 backdrop-blur-2xl border-b border-white/[0.06]"
    >
      {/* Left - Page Title */}
      <div>
        <motion.h1
          key={pathname}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className="text-lg font-semibold text-white"
        >
          {title}
        </motion.h1>
      </div>

      {/* Right - Status indicators */}
      <div className="flex items-center gap-4">
        {/* Camera Status */}
        <div className="flex items-center gap-2 text-sm text-white/50">
          <div className="relative">
            <Camera className="w-4 h-4 text-[#22C55E]" />
            <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-[#22C55E] rounded-full" />
          </div>
          <span className="hidden sm:inline text-white/40 text-xs">
            Camera
          </span>
        </div>

        {/* Mic Status */}
        <div className="flex items-center gap-2 text-sm text-white/50">
          <div className="relative">
            <Mic className="w-4 h-4 text-[#22C55E]" />
            <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-[#22C55E] rounded-full" />
          </div>
          <span className="hidden sm:inline text-white/40 text-xs">Mic</span>
        </div>

        {/* Divider */}
        <div className="h-6 w-px bg-white/[0.08]" />

        {/* Session Timer */}
        <Badge variant="blue" pulse>
          <span className="font-mono text-xs">{formatTime(timer)}</span>
        </Badge>

        {/* Divider */}
        <div className="h-6 w-px bg-white/[0.08]" />

        {/* Profile Avatar */}
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#4F8CFF]/40 to-[#8B5CF6]/40 flex items-center justify-center cursor-pointer hover:opacity-80 transition-opacity">
          <User className="w-4 h-4 text-white/80" />
        </div>
      </div>
    </motion.header>
  );
}
