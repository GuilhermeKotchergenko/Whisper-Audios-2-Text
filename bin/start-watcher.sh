#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input_dir="${1:-$project_dir/data/incoming}"

exec "$project_dir/.venv/bin/python" "$project_dir/src/transcribe_audio.py" \
  "$input_dir" \
  --language "${WHISPER_LANGUAGE:-pt}" \
  --model "${WHISPER_MODEL:-large-v3}" \
  --device "${WHISPER_DEVICE:-auto}" \
  --watch
