#!/usr/bin/env bash
# Downloads the Amazon Review Data (2018) 5-core files used by this project (~7.9 GB in total:
# Books ~7.1 GB, Movies and TV ~0.8 GB). Source page: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2
# Uses curl, which is preinstalled on macOS and most Linux systems. Re-running the script resumes
# a partial download and skips files that are already complete.
# If a URL fails, open the source page, copy the current link for the "Movies and TV" and
# "Books" 5-core files, and update BASE / FILES below.
set -euo pipefail
BASE="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall"
FILES=(Movies_and_TV_5.json.gz Books_5.json.gz)
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"

size_of() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }

for f in "${FILES[@]}"; do
  url="$BASE/$f"
  out="$DEST/$f"
  remote=$(curl -fsSIL "$url" | tr -d '\r' | awk 'tolower($1) == "content-length:" {n = $2} END {print n}')
  if [ -n "$remote" ] && [ "$(size_of "$out")" = "$remote" ]; then
    echo "$f is already complete ($remote bytes), skipping"
    continue
  fi
  echo "Downloading $f (${remote:-unknown} bytes)"
  curl -fL --retry 3 -C - -o "$out" "$url"
  if [ -n "$remote" ] && [ "$(size_of "$out")" != "$remote" ]; then
    echo "ERROR: $f has $(size_of "$out") bytes, expected $remote. Re-run this script to resume." >&2
    exit 1
  fi
done
echo "Downloaded to $DEST"
