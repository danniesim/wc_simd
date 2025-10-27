import type { NextConfig } from "next";

// Proxy all requests under /backend/* to the Flask backend base.
// Target base is configured via NEXT_PUBLIC_TIMETRVLR_BACKEND_BASE (fallback: http://127.0.0.1:8001).
const BACKEND_BASE = (process.env.NEXT_PUBLIC_TIMETRVLR_BACKEND_BASE || "http://127.0.0.1:8001").replace(/\/$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${BACKEND_BASE}/:path*`,
      },
    ];
  },
};

export default nextConfig;
