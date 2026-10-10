#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTEDGE_ROOT="${ROBOTEDGE_ROOT:-/Users/johnnytan5/Robotedge-5v5-RCAP2026}"
SDK_ZIP="${SDK_ZIP:-$HOME/Downloads/sdk_release.zip}"
PACKAGES="${PACKAGES:-brain}"

mkdir -p "$ROOT/ros_ws/src"
ln -sfn "$ROBOTEDGE_ROOT/src" "$ROOT/ros_ws/src/robotedge"

if [ ! -d "$ROOT/vendor/booster_sdk/include/booster" ]; then
  tmp="$(mktemp -d)"
  unzip -q "$SDK_ZIP" -d "$tmp"
  mkdir -p "$ROOT/vendor/booster_sdk"
  cp -R "$tmp/sdk_release/include" "$ROOT/vendor/booster_sdk/include"
  rm -rf "$tmp"
fi

cd "$ROOT/ros_ws"
# shellcheck disable=SC2086
colcon build --symlink-install \
  --packages-up-to $PACKAGES \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  "-DCMAKE_CXX_FLAGS=-I$ROOT/vendor/booster_sdk/include"
