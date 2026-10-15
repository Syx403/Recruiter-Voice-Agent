import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Configuration C UI (alt/). `npm run dev`: hot reload here; API and WebSockets are proxied to the FastAPI server.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: { "/api": "http://127.0.0.1:8100", "/ws": { target: "ws://127.0.0.1:8100", ws: true } },
  },
  // Two entries: the recruiter console (index.html) and the candidate interview page (interview.html).
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: { input: { console: "index.html", interview: "interview.html" } },
  },
});
