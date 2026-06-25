// GARY — Phase 0 seed entry point.
//
// For now this is a toolchain smoke test. The signaling-game seed and the
// mutual-information meter (the "meaning-meter") land here next, per ADR-0001
// (the spine) and ADR-0002 (C/C++-first, portable core + optional CUDA backend).
#include <cstdio>

#include "gary/version.hpp"

int main() {
  std::printf("GARY %s — %s\n", gary::version(), gary::tagline());
  std::printf("core: portable C++ (CPU, Raspberry-capable)\n");
  std::printf("cuda backend: %s\n",
              gary::has_cuda_backend() ? "enabled (sm_120 / Blackwell)" : "disabled");
  std::printf("phase 0: signaling-game seed + mutual-information meter — next.\n");
  return 0;
}
