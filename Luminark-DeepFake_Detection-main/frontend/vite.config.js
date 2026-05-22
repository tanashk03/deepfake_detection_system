import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Use VITE_PROXY_TARGET for Docker internal buffering, fallback to VITE_API_URL or localhost
const backendUrl = process.env.VITE_PROXY_TARGET || process.env.VITE_API_URL || 'http://localhost:8000'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 3000,
        host: '0.0.0.0', // Allow external connections in Docker
        proxy: {
            '/api': {
                target: backendUrl,
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api/, ''),
            },
            // Also proxy /explain and /infer directly (without /api prefix)
            '/explain': {
                target: backendUrl,
                changeOrigin: true,
            },
            '/infer': {
                target: backendUrl,
                changeOrigin: true,
            },
            '/health': {
                target: backendUrl,
                changeOrigin: true,
            },
        },
    },
})
