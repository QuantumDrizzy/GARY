// GARY — Phase 10: the Henon strange attractor (2D). Chaos that lives on a low-dimensional
// fractal. Closes the dynamical-systems lens. Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>

namespace gary {

struct HenonResult {
  double lyapunov;               // largest Lyapunov exponent (> 0 = chaotic)
  double correlation_dimension;  // D2 (Grassberger-Procaccia); ~1.22 for the strange attractor
};

// The Henon map x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n. With the classic a=1.4, b=0.3 it
// is chaotic (positive Lyapunov) yet its attractor is a fractal of non-integer correlation
// dimension ~1.22 -- a strange attractor: structure hiding inside the chaos. `n_points` points
// are sampled for the Grassberger-Procaccia dimension estimate.
HenonResult henon_attractor(double a, double b, int n_points, std::uint64_t seed);

}  // namespace gary
