"use client";

import Sidebar from "@/components/layout/Sidebar";
import Topbar from "@/components/layout/Topbar";
import GradientBlobs from "@/components/background/GradientBlobs";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="relative min-h-screen">
      <GradientBlobs />
      <Sidebar />
      {/* Main content area — offset by sidebar width */}
      <div className="pl-[260px] max-lg:pl-[72px] transition-all duration-300">
        <Topbar />
        <main className="p-6">{children}</main>
      </div>
    </div>
  );
}
