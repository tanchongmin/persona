#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models/gemma-4-E2B-it-q4"

mkdir -p "$MODEL_DIR"

hf download google/gemma-4-E2B-it-qat-q4_0-gguf \
  gemma-4-E2B_q4_0-it.gguf \
  --local-dir "$MODEL_DIR"

hf download bartowski/google_gemma-4-E2B-it-GGUF \
  mmproj-google_gemma-4-E2B-it-f16.gguf \
  --local-dir "$MODEL_DIR"

echo "Downloaded quantized Gemma model to $MODEL_DIR/gemma-4-E2B_q4_0-it.gguf"
echo "Downloaded Gemma vision projector to $MODEL_DIR/mmproj-google_gemma-4-E2B-it-f16.gguf"
