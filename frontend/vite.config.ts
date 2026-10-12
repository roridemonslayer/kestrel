import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The API runs on :8077 (uvicorn api.main:app --port 8077); /api requests are forwarded to it.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8077', rewrite: (path) => path.replace(/^\/api/, '') },
    },
  },
})
