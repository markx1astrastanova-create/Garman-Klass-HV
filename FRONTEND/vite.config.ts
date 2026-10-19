import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: './',
  build: { outDir: mode === 'streamlit' ? 'streamlit_dist' : 'dist' },
}))
