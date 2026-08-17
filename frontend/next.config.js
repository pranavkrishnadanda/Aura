const path = require("node:path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  turbopack: {
    // The repo root also holds a lockfile, so Turbopack cannot infer which
    // directory is the project root and warns on every build. Say it explicitly.
    root: path.join(__dirname),
  },
  // The former `rewrites()` proxy to /api/* was dead config: lib/api.ts calls the
  // backend at an absolute NEXT_PUBLIC_API_URL and never went through it.
};

module.exports = nextConfig;
