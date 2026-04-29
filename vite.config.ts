import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: process.env.VITE_BASE_PATH || "/",
  plugins: [react()],
  optimizeDeps: {
    exclude: ["@huggingface/transformers"]
  },
  worker: {
    format: "es"
  }
});
