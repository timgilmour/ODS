import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Dev-only: the FastAPI backend (app/main.py) serves /api and /health
      // on :3015. In production the same container serves ui/dist statically
      // at "/" alongside those same routes, so no proxy is needed there.
      '/api': 'http://localhost:3015',
      '/health': 'http://localhost:3015',
    },
  },
})
