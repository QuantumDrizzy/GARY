// GARY — Phase 6: dynamical systems. Deterministic chaos is structured, not random.
// The 4th lens of ADR-0001 ("chaos is not disorder"). Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>

namespace gary {

// Largest Lyapunov exponent of the logistic map x -> r*x*(1-x). Positive = chaos
// (sensitive dependence), negative = a stable periodic orbit. `transient` iterations are
// discarded before averaging log|f'(x)| over `iters` steps.
double lyapunov_exponent(double r, int iters, int transient);

struct ChaosMi {
  double chaos_self_mi;     // I(x_{n+1}; x_n) for the chaotic series, bits
  double random_self_mi;    // the same for an IID uniform series (the null floor)
  double chaos_marginal_h;  // H(x_n) of the chaotic series  (both look "random")
  double random_marginal_h; // H of the IID series
  double lyapunov;          // Lyapunov exponent at r (confirms chaos)
};

// Compare a chaotic logistic series (at parameter r) with IID uniform noise. Both have broad
// histograms (similar marginal entropy), but the consecutive mutual information I(x_{n+1};x_n)
// reveals the chaotic series is deterministic STRUCTURE while the random one is not (~0).
// `bins` discretises [0,1]. This reuses GARY's mutual-information meter: chaos is not disorder.
ChaosMi chaos_vs_random(double r, int n, int bins, std::uint64_t seed);

}  // namespace gary
