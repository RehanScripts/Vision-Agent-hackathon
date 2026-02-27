"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Sparkles,
  ArrowRight,
  Mic,
  Eye,
  BarChart3,
  Zap,
  Shield,
  Globe,
} from "lucide-react";
import GradientBlobs from "@/components/background/GradientBlobs";

const features = [
  {
    icon: Mic,
    title: "Speech Analysis",
    description:
      "Real-time WPM tracking and filler word detection powered by AI.",
  },
  {
    icon: Eye,
    title: "Eye Contact Tracking",
    description:
      "Computer vision monitors your gaze and engagement levels live.",
  },
  {
    icon: BarChart3,
    title: "Deep Analytics",
    description:
      "Detailed performance breakdowns with session-over-session trends.",
  },
  {
    icon: Zap,
    title: "Instant Feedback",
    description:
      "AI coach intervenes in real-time with actionable suggestions.",
  },
  {
    icon: Shield,
    title: "Posture Detection",
    description:
      "Body language analysis ensures confident and open presentation.",
  },
  {
    icon: Globe,
    title: "Multi-Scenario",
    description:
      "Practice for interviews, pitches, MUN debates, and more.",
  },
];

const container = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.08 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as const },
  },
};

export default function LandingPage() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      <GradientBlobs />

      {/* Nav */}
      <motion.nav
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between h-16 px-6 md:px-12 bg-[#0B0F19]/60 backdrop-blur-2xl border-b border-white/[0.06]"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#4F8CFF] to-[#8B5CF6] flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="text-lg font-bold bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
            SpeakAI
          </span>
        </div>
        <div className="flex items-center gap-4">
          <Link
            href="/dashboard"
            className="px-4 py-2 text-sm font-medium text-white/60 hover:text-white transition-colors"
          >
            Dashboard
          </Link>
          <Link
            href="/dashboard"
            className="px-5 py-2 text-sm font-semibold rounded-xl bg-gradient-to-r from-[#4F8CFF] to-[#8B5CF6] text-white hover:opacity-90 transition-opacity shadow-lg shadow-[#4F8CFF]/20"
          >
            Get Started
          </Link>
        </div>
      </motion.nav>

      {/* Hero */}
      <section className="relative flex flex-col items-center justify-center min-h-screen px-6 pt-24 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="max-w-4xl mx-auto"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 mb-8 text-xs font-semibold rounded-full bg-white/[0.05] border border-white/[0.08] text-white/60"
          >
            <Zap className="w-3.5 h-3.5 text-[#F59E0B]" />
            AI-Powered Public Speaking Coach
          </motion.div>

          <h1 className="text-5xl md:text-7xl font-bold leading-[1.1] tracking-tight">
            <span className="bg-gradient-to-b from-white to-white/50 bg-clip-text text-transparent">
              Speak with
            </span>
            <br />
            <span className="bg-gradient-to-r from-[#4F8CFF] to-[#8B5CF6] bg-clip-text text-transparent">
              Confidence
            </span>
          </h1>

          <p className="mt-6 text-lg md:text-xl text-white/40 max-w-2xl mx-auto leading-relaxed">
            Real-time AI analysis of your speech patterns, eye contact, posture,
            and delivery — helping you become an exceptional communicator.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-10">
            <Link href="/dashboard">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="flex items-center gap-2 px-8 py-3.5 text-sm font-semibold rounded-2xl bg-gradient-to-r from-[#4F8CFF] to-[#8B5CF6] text-white shadow-2xl shadow-[#4F8CFF]/20 hover:shadow-[#4F8CFF]/30 transition-shadow"
              >
                Start Free Session
                <ArrowRight className="w-4 h-4" />
              </motion.button>
            </Link>
            <Link href="/analytics">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="flex items-center gap-2 px-8 py-3.5 text-sm font-medium rounded-2xl bg-white/[0.05] border border-white/[0.08] text-white/60 hover:text-white hover:bg-white/[0.08] transition-all"
              >
                View Analytics
              </motion.button>
            </Link>
          </div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-10"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="w-5 h-8 rounded-full border border-white/[0.15] flex items-start justify-center p-1.5"
          >
            <div className="w-1 h-1.5 bg-white/30 rounded-full" />
          </motion.div>
        </motion.div>
      </section>

      {/* Features */}
      <section className="relative px-6 md:px-12 pb-32 max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-white/90">
            Everything you need to{" "}
            <span className="bg-gradient-to-r from-[#4F8CFF] to-[#8B5CF6] bg-clip-text text-transparent">
              present better
            </span>
          </h2>
          <p className="mt-4 text-white/40 max-w-xl mx-auto">
            Comprehensive AI analysis across every dimension of public speaking.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          {features.map((feature) => (
            <motion.div
              key={feature.title}
              variants={item}
              className="group relative p-6 rounded-2xl border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#4F8CFF]/10 to-[#8B5CF6]/10 border border-white/[0.06] flex items-center justify-center mb-4">
                <feature.icon className="w-5 h-5 text-[#4F8CFF]" />
              </div>
              <h3 className="text-sm font-semibold text-white/80 mb-2">
                {feature.title}
              </h3>
              <p className="text-xs text-white/40 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* CTA */}
      <section className="relative px-6 md:px-12 pb-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto text-center p-12 rounded-3xl border border-white/[0.06] bg-white/[0.02] backdrop-blur-xl"
        >
          <h3 className="text-2xl md:text-3xl font-bold text-white/90 mb-4">
            Ready to transform your speaking?
          </h3>
          <p className="text-white/40 mb-8 text-sm">
            Join thousands of professionals using SpeakAI to deliver
            unforgettable presentations.
          </p>
          <Link href="/dashboard">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="px-8 py-3.5 text-sm font-semibold rounded-2xl bg-gradient-to-r from-[#4F8CFF] to-[#8B5CF6] text-white shadow-2xl shadow-[#4F8CFF]/20"
            >
              Start Your Free Trial
            </motion.button>
          </Link>
        </motion.div>
      </section>
    </div>
  );
}
