// GARY — Phase 8: order from chaos. The period-doubling cascade of the logistic map and the
// universal Feigenbaum constant, measured. Deepens the dynamical-systems lens of Phase 6.
// Portable core, no CUDA (Pi-capable).
#pragma once

#include <vector>

namespace gary {

// Period of the logistic map's attractor at parameter r (x -> r*x*(1-x)), measured after a
// transient. Returns 1, 2, 4, ... for periodic orbits; 0 if chaotic (period > max_period or
// the attractor is a continuum). `tol` clusters revisited points.
int attractor_period(double r, int max_period, int transient, int samples, double tol);

struct BifurcationResult {
  std::vector<double> doubling_points;  // r where the period first doubles (1->2, 2->4, ...)
  std::vector<int> periods;             // the period reached at each doubling point
  double feigenbaum_delta;              // estimate of Feigenbaum's constant (true ~ 4.66920)
};

// Sweep r to find the period-doubling cascade and estimate the Feigenbaum constant
// delta = lim (r_n - r_{n-1}) / (r_{n+1} - r_n). The known value is 4.66920160...
BifurcationResult feigenbaum_cascade(double r_start, double r_end, double r_step);

}  // namespace gary
