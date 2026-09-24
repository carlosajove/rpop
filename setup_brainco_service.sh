#!/usr/bin/env bash
# Build Unitree's brainco_hand_service (BrainCo Revo2 hand <-> DDS bridge) on this workstation.
# Normally this service runs on the robot's PC2 (it needs the USB-serial link to the hands); building it here
# gives you the binary + a known-good dependency set to copy over, and lets you run it against a hand on USB.
#
# Step 1 (needs sudo, run once):
#   sudo apt install -y libspdlog-dev libfmt-dev libyaml-cpp-dev libboost-program-options-dev
# Step 2: bash setup_brainco_service.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$HOME/.local"
for pkg in libspdlog-dev libfmt-dev libyaml-cpp-dev libboost-program-options-dev; do
  dpkg -s "$pkg" >/dev/null 2>&1 || { echo "missing apt package $pkg -> run: sudo apt install -y libspdlog-dev libfmt-dev libyaml-cpp-dev libboost-program-options-dev"; exit 1; }
done
# unitree_sdk2 (C++), installed into ~/.local (already done once; idempotent)
if [ ! -f "$PREFIX/lib/libunitree_sdk2.a" ]; then
  cmake -S "$ROOT/unitree_sdk2" -B "$ROOT/unitree_sdk2/build" -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF
  cmake --build "$ROOT/unitree_sdk2/build" -j"$(nproc)" && cmake --install "$ROOT/unitree_sdk2/build"
fi
# brainco_hand_service: its CMakeLists hardcodes /usr/local/include/ddscxx, so point it at ~/.local instead
cmake -S "$ROOT/brainco_hand_service" -B "$ROOT/brainco_hand_service/build" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_CXX_FLAGS="-I$PREFIX/include -I$PREFIX/include/ddscxx" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib -Wl,-rpath,$ROOT/brainco_hand_service/lib/x86_64"
cmake --build "$ROOT/brainco_hand_service/build" -j"$(nproc)"
echo "built: $ROOT/brainco_hand_service/build/brainco_hand_server"
echo "run (hands on USB):  ./brainco_hand_server -n <dds interface>   -> topics rt/brainco/{left,right}/{cmd,state}"
