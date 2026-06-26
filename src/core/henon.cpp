#include "gary/henon.hpp"

#include <cmath>
#include <vector>

namespace gary {

HenonResult henon_attractor(double a, double b, int n_points, std::uint64_t /*seed*/) {
  HenonResult res;

  // ── Largest Lyapunov exponent (Benettin, via the Jacobian J = [[-2a x, 1],[b, 0]]) ──
  {
    double x = 0.1, y = 0.1;
    for (int i = 0; i < 1000; ++i) { const double nx = 1.0 - a * x * x + y; y = b * x; x = nx; }

    double vx = 1.0, vy = 0.0, sum = 0.0;
    const int steps = 100000;
    for (int i = 0; i < steps; ++i) {
      const double nvx = -2.0 * a * x * vx + vy;  // tangent vector through the Jacobian
      const double nvy = b * vx;
      const double norm = std::sqrt(nvx * nvx + nvy * nvy);
      sum += std::log(norm);
      vx = nvx / norm;
      vy = nvy / norm;
      const double nx = 1.0 - a * x * x + y;
      y = b * x;
      x = nx;
    }
    res.lyapunov = sum / steps;
  }

  // ── Correlation dimension D2 (Grassberger-Procaccia) ────────────────────────
  {
    std::vector<double> px(n_points), py(n_points);
    double x = 0.1, y = 0.1;
    for (int i = 0; i < 1000; ++i) { const double nx = 1.0 - a * x * x + y; y = b * x; x = nx; }
    for (int i = 0; i < n_points; ++i) {
      const double nx = 1.0 - a * x * x + y;
      y = b * x;
      x = nx;
      px[i] = x;
      py[i] = y;
    }

    const int R = 14;
    const double rmin = 0.005, rmax = 1.0;
    std::vector<double> radii(R);
    std::vector<long> counts(R, 0);
    for (int k = 0; k < R; ++k) radii[k] = rmin * std::pow(rmax / rmin, static_cast<double>(k) / (R - 1));

    for (int i = 0; i < n_points; ++i)
      for (int j = i + 1; j < n_points; ++j) {
        const double dx = px[i] - px[j], dy = py[i] - py[j];
        const double d = std::sqrt(dx * dx + dy * dy);
        for (int k = 0; k < R; ++k)
          if (d < radii[k]) ++counts[k];
      }

    const double total = static_cast<double>(n_points) * (n_points - 1) / 2.0;

    // D2 = slope of log C(r) vs log r in the scaling region (C between 1e-3 and 0.5).
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int m = 0;
    for (int k = 0; k < R; ++k) {
      const double C = counts[k] / total;
      if (C > 1e-3 && C < 0.5) {
        const double lr = std::log(radii[k]), lc = std::log(C);
        sx += lr;
        sy += lc;
        sxx += lr * lr;
        sxy += lr * lc;
        ++m;
      }
    }
    res.correlation_dimension = (m >= 2) ? (m * sxy - sx * sy) / (m * sxx - sx * sx) : 0.0;
  }

  return res;
}

}  // namespace gary
