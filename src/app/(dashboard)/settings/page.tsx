"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import GlassCard from "@/components/ui/GlassCard";
import Badge from "@/components/ui/Badge";
import {
  Camera,
  MessageSquare,
  Presentation,
  Moon,
  Save,
  ChevronDown,
  Check,
} from "lucide-react";

function Toggle({
  enabled,
  onToggle,
}: {
  enabled: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
        enabled ? "bg-[#4F8CFF]" : "bg-white/10"
      }`}
    >
      <motion.div
        className="absolute top-0.5 w-5 h-5 rounded-full bg-white shadow-sm"
        animate={{ left: enabled ? 22 : 2 }}
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
      />
    </button>
  );
}

function SelectDropdown({
  value,
  options,
  onChange,
}: {
  value: string;
  options: string[];
  onChange: (val: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full px-4 py-2.5 text-sm text-white/80 bg-white/[0.04] border border-white/[0.08] rounded-xl hover:bg-white/[0.06] transition-colors"
      >
        <span>{value}</span>
        <motion.div animate={{ rotate: open ? 180 : 0 }}>
          <ChevronDown className="w-4 h-4 text-white/40" />
        </motion.div>
      </button>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: -5 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-full left-0 right-0 mt-2 py-1 bg-[#141824] border border-white/[0.08] rounded-xl shadow-2xl z-10 overflow-hidden"
        >
          {options.map((option) => (
            <button
              key={option}
              onClick={() => {
                onChange(option);
                setOpen(false);
              }}
              className="flex items-center justify-between w-full px-4 py-2.5 text-sm text-white/60 hover:text-white hover:bg-white/[0.04] transition-colors"
            >
              <span>{option}</span>
              {value === option && (
                <Check className="w-4 h-4 text-[#4F8CFF]" />
              )}
            </button>
          ))}
        </motion.div>
      )}
    </div>
  );
}

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 15 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as const } },
};

export default function SettingsPage() {
  const [cameraPreview, setCameraPreview] = useState(true);
  const [interruptions, setInterruptions] = useState(true);
  const [darkMode, setDarkMode] = useState(true);
  const [sessionType, setSessionType] = useState("Interview");
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="max-w-2xl space-y-6"
    >
      <motion.div variants={fadeUp}>
        <h2 className="text-xl font-semibold text-white/90 mb-1">Settings</h2>
        <p className="text-sm text-white/40">
          Customize your SpeakAI experience.
        </p>
      </motion.div>

      {/* Session Settings */}
      <motion.div variants={fadeUp}>
        <GlassCard className="p-6 space-y-6" hover={false}>
          <div>
            <h3 className="text-sm font-semibold text-white/70 mb-4 flex items-center gap-2">
              <Presentation className="w-4 h-4 text-[#4F8CFF]" />
              Session Settings
            </h3>
            <div className="space-y-5">
              {/* Camera Preview */}
              <div className="flex items-center justify-between">
                <div className="flex items-start gap-3">
                  <Camera className="w-4 h-4 text-white/40 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-white/80">
                      Camera Preview
                    </p>
                    <p className="text-xs text-white/40 mt-0.5">
                      Show camera feed during live sessions
                    </p>
                  </div>
                </div>
                <Toggle
                  enabled={cameraPreview}
                  onToggle={() => setCameraPreview(!cameraPreview)}
                />
              </div>

              {/* Divider */}
              <div className="border-t border-white/[0.04]" />

              {/* Real-time Interruptions */}
              <div className="flex items-center justify-between">
                <div className="flex items-start gap-3">
                  <MessageSquare className="w-4 h-4 text-white/40 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-white/80">
                      Real-time Interruptions
                    </p>
                    <p className="text-xs text-white/40 mt-0.5">
                      AI coach provides live feedback during sessions
                    </p>
                  </div>
                </div>
                <Toggle
                  enabled={interruptions}
                  onToggle={() => setInterruptions(!interruptions)}
                />
              </div>

              {/* Divider */}
              <div className="border-t border-white/[0.04]" />

              {/* Session Type */}
              <div>
                <div className="flex items-start gap-3 mb-3">
                  <Presentation className="w-4 h-4 text-white/40 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-white/80">
                      Session Type
                    </p>
                    <p className="text-xs text-white/40 mt-0.5">
                      Choose the context for AI coaching
                    </p>
                  </div>
                </div>
                <div className="ml-7">
                  <SelectDropdown
                    value={sessionType}
                    options={[
                      "Interview",
                      "Pitch",
                      "MUN Debate",
                      "Presentation",
                      "General Practice",
                    ]}
                    onChange={setSessionType}
                  />
                </div>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* Appearance */}
      <motion.div variants={fadeUp}>
        <GlassCard className="p-6" hover={false}>
          <h3 className="text-sm font-semibold text-white/70 mb-4 flex items-center gap-2">
            <Moon className="w-4 h-4 text-[#8B5CF6]" />
            Appearance
          </h3>
          <div className="flex items-center justify-between">
            <div className="flex items-start gap-3">
              <Moon className="w-4 h-4 text-white/40 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-white/80">Dark Mode</p>
                <p className="text-xs text-white/40 mt-0.5">
                  Application theme preference
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge variant="blue">Active</Badge>
              <Toggle
                enabled={darkMode}
                onToggle={() => setDarkMode(!darkMode)}
              />
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* Save Button */}
      <motion.div variants={fadeUp} className="flex items-center gap-3">
        <motion.button
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          onClick={handleSave}
          className="flex items-center gap-2 px-6 py-3 text-sm font-semibold rounded-xl bg-gradient-to-r from-[#4F8CFF] to-[#8B5CF6] text-white shadow-lg shadow-[#4F8CFF]/20 hover:shadow-[#4F8CFF]/30 transition-shadow"
        >
          {saved ? (
            <>
              <Check className="w-4 h-4" />
              Saved
            </>
          ) : (
            <>
              <Save className="w-4 h-4" />
              Save Changes
            </>
          )}
        </motion.button>
        {saved && (
          <motion.span
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-xs text-[#22C55E]"
          >
            Settings saved successfully
          </motion.span>
        )}
      </motion.div>
    </motion.div>
  );
}
