#!/usr/bin/env bash
# Download the two native libraries the Go runtime links against, into third_party/:
#   - ONNX Runtime shared library      (loaded at run time via ONNXRUNTIME_SHARED_LIBRARY_PATH)
#   - HF tokenizers static library     (linked at build time via CGO_LDFLAGS)
# Usage: scripts/fetch_deps.sh            # detects OS/arch
#        ORT_VERSION=1.29.1 TOKENIZERS_VERSION=1.27.0 scripts/fetch_deps.sh
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.29.1}"          # must match the API version github.com/yalue/onnxruntime_go expects
TOKENIZERS_VERSION="${TOKENIZERS_VERSION:-1.27.0}"  # must match github.com/daulet/tokenizers in go.mod
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

os="$(uname -s)"; arch="$(uname -m)"
case "$os" in
  Darwin) ort_os=osx; tok_os=darwin; libext=dylib ;;
  Linux)  ort_os=linux; tok_os=linux; libext=so ;;
  *) echo "unsupported OS: $os" >&2; exit 1 ;;
esac
case "$arch" in
  arm64|aarch64) ort_arch=$([ "$ort_os" = osx ] && echo arm64 || echo aarch64); tok_arch=arm64 ;;
  x86_64|amd64)  ort_arch=$([ "$ort_os" = osx ] && echo x86_64 || echo x64);   tok_arch=x86_64 ;;
  *) echo "unsupported arch: $arch" >&2; exit 1 ;;
esac

ort_dir="$TP/onnxruntime-$ort_os-$ort_arch-$ORT_VERSION"
if [ ! -d "$ort_dir" ]; then
  url="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-$ort_os-$ort_arch-$ORT_VERSION.tgz"
  echo "fetching $url"
  curl -fsSL "$url" | tar xz -C "$TP"
fi

tok_dir="$TP/tokenizers"
if [ ! -f "$tok_dir/libtokenizers.a" ]; then
  mkdir -p "$tok_dir"
  url="https://github.com/daulet/tokenizers/releases/download/v$TOKENIZERS_VERSION/libtokenizers.$tok_os-$tok_arch.tar.gz"
  echo "fetching $url"
  curl -fsSL "$url" | tar xz -C "$tok_dir"
fi

ort_lib=""
for cand in "$ort_dir/lib/libonnxruntime.so.$ORT_VERSION" "$ort_dir/lib/libonnxruntime.$ORT_VERSION.dylib" \
            "$ort_dir/lib/libonnxruntime.so" "$ort_dir/lib/libonnxruntime.dylib"; do
  if [ -f "$cand" ]; then ort_lib="$cand"; break; fi
done
[ -n "$ort_lib" ] || { echo "libonnxruntime not found under $ort_dir/lib" >&2; exit 1; }

cat > "$TP/env.sh" <<EOF
# source this file:  . third_party/env.sh
export CGO_LDFLAGS="-L$tok_dir"
export ONNXRUNTIME_SHARED_LIBRARY_PATH="$ort_lib"
EOF
echo "ok. now:  . third_party/env.sh && go build ./cmd/layad"
