#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE=${1:-Release}          # ./build.sh Debug
JOBS=${JOBS:-$(nproc)}            # override with JOBS=8 ./build.sh

cmake -S "$(dirname "$0")" -B build -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build build -j"$JOBS"
