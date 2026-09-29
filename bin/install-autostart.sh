#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service_dir="$HOME/.config/systemd/user"
service_file="$service_dir/whisper-audios.service"

mkdir -p "$service_dir"
sed "s|__PROJECT_DIR__|$project_dir|g" \
  "$project_dir/systemd/whisper-audios.service.template" > "$service_file"
systemctl --user daemon-reload
systemctl --user enable --now whisper-audios.service
systemctl --user status whisper-audios.service --no-pager
