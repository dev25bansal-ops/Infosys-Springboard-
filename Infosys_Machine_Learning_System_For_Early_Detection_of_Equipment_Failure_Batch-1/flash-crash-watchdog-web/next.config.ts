import type { NextConfig } from "next";

const securityHeaders = [
  // SEC-11: baseline security headers.
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  // HSTS only over TLS. The gateway usually terminates TLS; harmless if not.
  { key: "Strict-Transport-Security", value: "max-age=31536000; includeSubDomains" },
];

const nextConfig: NextConfig = {
  output: "standalone",
  // SEC-15: removed `ignoreBuildErrors: true` — the app now type-checks clean,
  // and a `next build` must enforce types (CI gates on `tsc --noEmit` too).
  reactStrictMode: false,
  async headers() {
    return [{ source: "/(.*)", headers: securityHeaders }];
  },
};

export default nextConfig;
