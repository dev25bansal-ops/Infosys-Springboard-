import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as SonnerToaster } from "@/components/ui/sonner";

export const metadata: Metadata = {
  title: "Flash Crash Watchdog — Real-time ML Anomaly Detection",
  description: "Live ML-powered flash crash detection dashboard. Monitors BTC/USDT order book microstructure and fires alerts before price dislocation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      {/* system font stack (no Google Fonts dependency — works offline/behind a proxy) */}
      <body className="antialiased bg-background text-foreground font-sans">
        {children}
        <Toaster />
        <SonnerToaster position="top-right" richColors closeButton expand={false} duration={6000} />
      </body>
    </html>
  );
}
