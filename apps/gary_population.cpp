// GARY — Phase 1: a population of independent signaling games.
// Honest benchmark: CPU (single thread) vs GPU (kernel-only, cudaEvent timing).
// The population statistics (mean MI, %% perfect) must match; the GPU just scales.
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

  std::printf("Phase 1 - population of independent signaling games\n");
  std::printf("N=M=%d, forgetting=%.2f, %d rounds. Perfect code = %.3f bits.\n", n_sym, phi, rounds,
              Hmax);
  std::printf("baseline: CPU single-thread vs GPU kernel-only (cudaEvent), both double precision.\n\n");

  std::printf("   games        CPU(1 thread)            GPU(sm_120)             speedup\n");
  for (int n : {1024, 4096, 16384}) {
    const gary::PopulationResult c = gary::run_population_cpu(n, n_sym, rounds, phi, 1);
#if defined(GARY_WITH_CUDA)
    const gary::PopulationResult g = gary::run_population_cuda(n, n_sym, rounds, phi, 1);
    std::printf("  %6d   %8.1f ms (MI %.3f %3.0f%%P)   %8.1f ms (MI %.3f %3.0f%%P)   %6.1fx\n", n,
                c.ms, c.mean_mi, c.frac_perfect * 100.0, g.ms, g.mean_mi, g.frac_perfect * 100.0,
                c.ms / g.ms);
#else
    std::printf("  %6d   %8.1f ms (MI %.3f %3.0f%%P)   [CUDA off: -DGARY_BUILD_CUDA=ON]\n", n, c.ms,
                c.mean_mi, c.frac_perfect * 100.0);
#endif
  }
  std::printf("\n(%%P = %% of games reaching a near-perfect code, MI >= 99%% of %.3f bits)\n", Hmax);
  return 0;
}
