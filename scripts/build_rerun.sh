#!/usr/bin/env bash
# brain uses the rerun C++ SDK 0.20 API; conda-forge only ships newer versions.
# Same version and flags as $ROBOTEDGE_ROOT/scripts/install_rerun_cpp.sh, installed into vendor/ (no sudo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RERUN_VERSION="0.20.3"
PREFIX="$ROOT/vendor/rerun_sdk"
VERSION_FILE="$PREFIX/lib/cmake/rerun_sdk/rerun_sdkConfigVersion.cmake"
if [ -f "$VERSION_FILE" ] && grep -q "set(PACKAGE_VERSION \"${RERUN_VERSION}\")" "$VERSION_FILE"; then
  echo "rerun C++ SDK ${RERUN_VERSION} present in $PREFIX"
  exit 0
fi

CACHE="$ROOT/vendor/cache"
ARCHIVE="$CACHE/rerun_cpp_sdk-${RERUN_VERSION}-multiplatform.zip"
URL="https://github.com/rerun-io/rerun/releases/download/${RERUN_VERSION}/rerun_cpp_sdk-${RERUN_VERSION}-multiplatform.zip"
mkdir -p "$CACHE"
if ! unzip -tq "$ARCHIVE" >/dev/null 2>&1; then
  curl --fail --location --retry 5 --output "$ARCHIVE" "$URL"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unzip -q "$ARCHIVE" -d "$WORK"
# Arrow 18 comes prebuilt from conda-forge (pixi.toml) instead of being compiled here.
cmake -S "$WORK/rerun_cpp_sdk" -B "$WORK/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:?run inside pixi}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DRERUN_DOWNLOAD_AND_BUILD_ARROW=OFF \
  -DRERUN_ARROW_LINK_SHARED=ON
cmake --build "$WORK/build" --parallel
cmake --install "$WORK/build"
test -f "$VERSION_FILE"
echo "rerun C++ SDK ${RERUN_VERSION} installed in $PREFIX"
