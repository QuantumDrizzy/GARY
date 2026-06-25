// GARY — Phase 1 + 1b: a population of independent signaling games.
// Honest benchmark: CPU (single thread) vs GPU FP64 (reference) vs GPU FP32 (Phase 1b).
// The only change from FP64 to FP32 is the weight precision -- same algorithm and memory
// layout. All three must agree on the statistics (mean MI, %% perfect); the GPU just scales.
#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "gary/population.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n", gary::version(), gary::tagline());
  std::printf("core: portable C++ | cuda: %s\n\n",
              gary::has_cuda_backend() ? "ON (sm_120 / Blackwell)" : "off");

  const int n_sym = 8, rounds = 10000;
  const double phi = 0.05, Hmax = std::log2(static_cast<double>(n_sym));

  std::printf("Phase 1/1b - population of independent signaling games\n");
  std::printf("N=M=%d, forgetting=%.2f, %d rounds. Perfect code = %.3f bits.\n", n_sym, phi, rounds,
              Hmax);
  std::printf("baseline: CPU single-thread; GPU kernel-only (cudaEvent). Phase 1b change: FP64 -> FP32.\n\n");

#if defined(GARY_WITH_CUDA)
  std::printf("   games     CPU(1thr)    GPU-FP64    GPU-FP32     FP32 vs CPU   FP32 vs FP64   MI(cpu/64/32)\n");
  for (int n : {1024, 4096, 16384}) {
    const gary::PopulationResult c = gary::run_population_cpu(n, n_sym, rounds, phi, 1);
    const gary::PopulationResult g64 = gary::run_population_cuda(n, n_sym, rounds, phi, 1);
    const gary::PopulationResult g32 = gary::run_population_cuda_f32(n, n_sym, rounds, phi, 1);
    std::printf("  %6d   %8.1fms  %8.1fms  %8.1fms     %7.1fx      %7.1fx      %.3f/%.3f/%.3f\n", n,
                c.ms, g64.ms, g32.ms, c.ms / g32.ms, g64.ms / g32.ms, c.mean_mi, g64.mean_mi,
                g32.mean_mi);
  }
  std::printf("\nVERDICT (Phase 1b): FP32 is NOT a win here.\n");
  std::printf(" - speed: ~1.0-1.2x vs FP64 => memory/branch-divergence bound, NOT FP64-compute\n");
  std::printf("   bound. The earlier 'FP64 is the bottleneck' guess was wrong.\n");
  std::printf(" - correctness: FP32 MI=3.000 != FP64/CPU ~2.933 => precision artifact (float\n");
  std::printf("   flush-to-zero in the forgetting decay fakes a perfect code). FP64 is the reference.\n");
  std::printf(" - real optimization (parked): lazy/scaled forgetting + coalesced layout, not precision.\n");
#else
  std::printf("   games     CPU(1thr)    MI    [CUDA off: build with scripts/build_cuda.bat]\n");
  for (int n : {1024, 4096, 16384}) {
    const gary::PopulationResult c = gary::run_population_cpu(n, n_sym, rounds, phi, 1);
    std::printf("  %6d   %8.1fms   %.3f (%.0f%% perfect)\n", n, c.ms, c.mean_mi,
                c.frac_perfect * 100.0);
  }
#endif
  return 0;
}
