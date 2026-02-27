"use client";

import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import GlassCard from "@/components/ui/GlassCard";
import Badge from "@/components/ui/Badge";
import {
  Mic,
  Video,
  Timer,
  Target,
  Presentation,
  MessageSquare,
  Swords,
  Play,
  Clock,
} from "lucide-react";

const practiceScenarios = [
  {
    id: 1,
    title: "Job Interview",
    description: "Practice answering common behavioral and technical interview questions.",
    icon: MessageSquare,
    color: "#4F8CFF",
    duration: "5-10 min",
    difficulty: "Medium",
  },
  {
    id: 2,
    title: "Startup Pitch",
    description: "Rehearse your elevator pitch with real-time pacing and clarity feedback.",
    icon: Presentation,
    color: "#8B5CF6",
    duration: "3-5 min",
    difficulty: "Hard",
  },
  {
    id: 3,
    title: "MUN Debate",
    description: "Practice opening statements and rebuttals for Model United Nations.",
    icon: Swords,
    color: "#F59E0B",
    duration: "5-15 min",
    difficulty: "Hard",
  },
  {
    id: 4,
    title: "Free Practice",
    description: "Open-ended session with full AI coaching and no time constraints.",
    icon: Mic,
    color: "#22C55E",
    duration: "Unlimited",
    difficulty: "Easy",
  },
];

const difficultyColors: Record<string, "blue" | "green" | "warning" | "critical"> = {
  Easy: "green",
  Medium: "blue",
  Hard: "warning",
};

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 15 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as const } },
};

export default function PracticePage() {
  const router = useRouter();

  const handleQuickStart = () => {
    router.push("/dashboard?type=Free+Practice&autostart=demo");
  };

  const handleScenario = (title: string) => {
    router.push(`/dashboard?type=${encodeURIComponent(title)}&autostart=demo`);
  };

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={fadeUp}>
        <h2 className="text-xl font-semibold text-white/90">Practice Mode</h2>
        <p className="text-sm text-white/40 mt-1">
          Choose a scenario and start practicing with AI coaching.
        </p>
      </motion.div>

      {/* Quick Start */}
      <motion.div variants={fadeUp}>
        <GlassCard className="p-6" hover={false} glow="blue">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-[#4F8CFF] to-[#8B5CF6] flex items-center justify-center shadow-lg shadow-[#4F8CFF]/20">
                <Play className="w-5 h-5 text-white ml-0.5" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-white/90">Quick Start</h3>
                <p className="text-xs text-white/40 mt-0.5">
                  Jump into a free-form practice session with full AI analysis.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs text-white/30">
                <Video className="w-3.5 h-3.5" />
                <span>Camera</span>
                <Mic className="w-3.5 h-3.5 ml-2" />
                <span>Mic</span>
              </div>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleQuickStart}
                className="px-5 py-2.5 text-sm font-semibold rounded-xl bg-gradient-to-r from-[#4F8CFF] to-[#8B5CF6] text-white shadow-lg shadow-[#4F8CFF]/20"
              >
                Start Now
              </motion.button>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* Scenarios Grid */}
      <motion.div variants={fadeUp}>
        <h3 className="text-sm font-semibold text-white/60 mb-4">Practice Scenarios</h3>
      </motion.div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {practiceScenarios.map((scenario) => (
          <motion.div key={scenario.id} variants={fadeUp}>
            <GlassCard className="p-5 cursor-pointer group h-full">
              <div className="flex items-start gap-4">
                <div
                  className="flex-shrink-0 w-11 h-11 rounded-xl flex items-center justify-center border border-white/[0.06]"
                  style={{ backgroundColor: `${scenario.color}12` }}
                >
                  <scenario.icon
                    className="w-5 h-5"
                    style={{ color: scenario.color }}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1.5">
                    <h4 className="text-sm font-semibold text-white/80">
                      {scenario.title}
                    </h4>
                    <Badge variant={difficultyColors[scenario.difficulty]}>
                      {scenario.difficulty}
                    </Badge>
                  </div>
                  <p className="text-xs text-white/40 leading-relaxed mb-3">
                    {scenario.description}
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 text-xs text-white/30">
                      <Clock className="w-3 h-3" />
                      <span>{scenario.duration}</span>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => handleScenario(scenario.title)}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-white/[0.05] border border-white/[0.08] text-white/60 hover:text-white hover:bg-white/[0.08] transition-all"
                    >
                      <Target className="w-3 h-3" />
                      Practice
                    </motion.button>
                  </div>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        ))}
      </div>

      {/* Tips */}
      <motion.div variants={fadeUp}>
        <GlassCard className="p-5" hover={false}>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-white/30 mb-3">
            💡 Practice Tips
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              "Warm up with 2-3 minutes of free practice before scenario sessions.",
              "Focus on one metric per session — don't try to fix everything at once.",
              "Review your analytics after each session to track improvement trends.",
            ].map((tip, i) => (
              <p key={i} className="text-xs text-white/40 leading-relaxed">
                {tip}
              </p>
            ))}
          </div>
        </GlassCard>
      </motion.div>

      {/* Permission Indicators */}
      <motion.div variants={fadeUp}>
        <div className="flex items-center gap-4 text-xs text-white/30">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-[#22C55E]" />
            <span>Camera access granted</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-[#22C55E]" />
            <span>Microphone access granted</span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
