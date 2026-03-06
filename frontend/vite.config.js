import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    // FIX: set explicit dev server port so the URL is predictable
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // FIX: rewrite strips the /api prefix before forwarding to FastAPI
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },

  // FIX: configure build output directory explicitly so CI/CD scripts can
  // rely on it without reading vite internals.
  build: {
    outDir: "dist",
    sourcemap: false,   // set to true if you need source maps in production
  },
});
