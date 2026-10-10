#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTEDGE_ROOT="${ROBOTEDGE_ROOT:-/Users/johnnytan5/Robotedge-5v5-RCAP2026}"
SDK_ZIP="${SDK_ZIP:-$HOME/Downloads/sdk_release.zip}"
PACKAGES="${PACKAGES:-brain}"
SRC="$ROOT/ros_ws/src"

# Message packages are used straight from the Robotedge working tree (read-only).
mkdir -p "$SRC"
[ -L "$SRC/robotedge" ] && rm "$SRC/robotedge"
mkdir -p "$SRC/robotedge"
for pkg in booster_msgs booster_ros2_interface robocup_ros2_interface; do
  ln -sfn "$ROBOTEDGE_ROOT/src/$pkg" "$SRC/robotedge/$pkg"
done

# brain is copied from the Robotedge working tree and patched for macOS clang (behaviour-identical;
# see patches/brain-clang.patch). Re-copied on every build so local brain edits are always what runs.
rsync -a --delete --exclude build --exclude install --exclude log --exclude __pycache__ \
  "$ROBOTEDGE_ROOT/src/brain/" "$SRC/brain/"
if ! patch -p1 -d "$SRC/brain" --forward --silent < "$ROOT/patches/brain-clang.patch"; then
  echo "patches/brain-clang.patch no longer applies to $ROBOTEDGE_ROOT/src/brain; refresh it." >&2
  exit 1
fi

if [ ! -d "$ROOT/vendor/booster_sdk/include/booster" ]; then
  tmp="$(mktemp -d)"
  unzip -q "$SDK_ZIP" -d "$tmp"
  mkdir -p "$ROOT/vendor/booster_sdk"
  cp -R "$tmp/sdk_release/include" "$ROOT/vendor/booster_sdk/include"
  rm -rf "$tmp"
fi

bash "$ROOT/scripts/build_rerun.sh"

cd "$ROOT/ros_ws"
# shellcheck disable=SC2086
colcon build --symlink-install \
  --packages-up-to $PACKAGES \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  "-DCMAKE_CXX_FLAGS=-I$ROOT/vendor/booster_sdk/include" \
  "-Drerun_sdk_DIR=$ROOT/vendor/rerun_sdk/lib/cmake/rerun_sdk"
