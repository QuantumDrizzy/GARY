#include "gary/bifurcation.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gary {

int attractor_period(double r, int max_period, int transient, int samples, double tol) {
  double x = 0.5;
  for (int i = 0; i < transient; ++i) x = r * x * (1.0 - x);

  std::vector<double> pts;
  pts.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    x = r * x * (1.0 - x);
    pts.push_back(x);
  }

  // Count distinct attractor points (clusters separated by more than tol).
  std::sort(pts.begin(), pts.end());
  int clusters = 1;
  for (size_t i = 1; i < pts.size(); ++i)
    if (pts[i] - pts[i - 1] > tol) ++clusters;

  return (clusters > max_period) ? 0 : clusters;  // 0 = chaotic / period too high
}

BifurcationResult feigenbaum_cascade(double r_start, double r_end, double r_step) {
  BifurcationResult res;
  const int max_period = 64, transient = 20000, samples = 1024;
  const double tol = 1e-4;

  int prev_period = attractor_period(r_start, max_period, transient, samples, tol);
  if (prev_period <= 0) prev_period = 1;

  for (double r = r_start; r <= r_end; r += r_step) {
    const int p = attractor_period(r, max_period, transient, samples, tol);
    if (p == 2 * prev_period) {  // a clean period doubling
      res.doubling_points.push_back(r);
      res.periods.push_back(p);
      prev_period = p;
    }
  }

  // Feigenbaum delta = (r_n - r_{n-1}) / (r_{n+1} - r_n); use the last (most converged) triple.
  res.feigenbaum_delta = 0.0;
  const int k = static_cast<int>(res.doubling_points.size());
  if (k >= 3) {
    const std::vector<double>& d = res.doubling_points;
    const double denom = d[k - 1] - d[k - 2];
    if (std::fabs(denom) > 1e-12) res.feigenbaum_delta = (d[k - 2] - d[k - 3]) / denom;
  }

  return res;
}

}  // namespace gary
