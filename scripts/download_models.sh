#!/usr/bin/env bash
# Download pretrained bridge-bot binary checkpoints from GitHub Releases into checkpoints/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-v1.0}"
ASSET="bridge-bot-models-${VERSION}.tar.gz"
URL="https://github.com/frigoow/bridge-bot/releases/download/${VERSION}/${ASSET}"

mkdir -p "$ROOT/checkpoints"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading ${URL}..."
if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar "$URL" -o "$TMP/$ASSET"
elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress "$URL" -O "$TMP/$ASSET"
else
    echo "Error: neither curl nor wget is installed." >&2
    exit 1
fi

tar -xzf "$TMP/$ASSET" -C "$ROOT"
echo "Extracted pretrained checkpoints into $ROOT/checkpoints/:"
ls -lh "$ROOT/checkpoints/"*.bin
