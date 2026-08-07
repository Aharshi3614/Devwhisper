import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/stream': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/reset': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/history': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/webhook': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/statistics': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/index': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/repos': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
