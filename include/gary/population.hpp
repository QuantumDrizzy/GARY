// GARY — Phase 1: a population of independent signaling games.
// CPU reference is portable (Pi-capable). The CUDA backend (one thread per game)
// is compiled only when GARY_WITH_CUDA is defined.
#pragma once

#include <cstdint>

namespace gary {

struct PopulationResult {
  int n_games;
  int rounds;
  double mean_mi;       // mean final I(state; signal) in bits across the population
  double frac_perfect;  // fraction of games with MI >= 0.99 * log2(N)
  double ms;            // wall-clock milliseconds
};

// Run `n_games` independent signaling games (N = M = n_sym) for `rounds` rounds each,
// Roth-Erev with `forgetting`. CPU reference (single thread).
PopulationResult run_population_cpu(int n_games, int n_sym, int rounds, double forgetting,
                                    std::uint64_t seed);

#if defined(GARY_WITH_CUDA)
// Same computation, batched on the GPU (one CUDA thread per game). Statistically
// equivalent to the CPU version (same algorithm/params, independent RNG streams).
// run_population_cuda is the FP64 reference. run_population_cuda_f32 is an FP32 A/B.
// [KNOWN_LIMIT] Phase 1b measured FP32 and found it is NOT a faithful optimization here:
//   (1) barely faster (~1.0-1.2x) -- the kernel is memory / branch-divergence bound (the
//       per-round forgetting decay over uncoalesced global memory), not FP64-compute bound;
//   (2) it CHANGES the result -- float flush-to-zero in the decay sharpens the strategy to a
//       fake-perfect MI (3.000 vs the correct ~2.933 at 10k rounds).
// Kept only as a documented negative experiment. Use the FP64 path; the real win is memory
// access (lazy/scaled forgetting + coalescing), not precision.
PopulationResult run_population_cuda(int n_games, int n_sym, int rounds, double forgetting,
                                     std::uint64_t seed);
PopulationResult run_population_cuda_f32(int n_games, int n_sym, int rounds, double forgetting,
                                         std::uint64_t seed);
#endif

}  // namespace gary
