#include "gary/info_theory.hpp"

#include <cmath>

namespace gary {

static inline double xlog2x(double x) noexcept {
  return (x > 0.0) ? x * std::log2(x) : 0.0;
}

double entropy_bits(const std::vector<double>& p) noexcept {
  double h = 0.0;
  for (double pi : p) h -= xlog2x(pi);
  return h;
}

double mutual_information_bits(const std::vector<double>& joint, int n_x, int n_y) noexcept {
  if (n_x <= 0 || n_y <= 0) return 0.0;
  std::vector<double> px(static_cast<size_t>(n_x), 0.0);
  std::vector<double> py(static_cast<size_t>(n_y), 0.0);
  for (int i = 0; i < n_x; ++i)
    for (int j = 0; j < n_y; ++j) {
      const double pij = joint[static_cast<size_t>(i) * n_y + j];
      px[i] += pij;
      py[j] += pij;
    }
  double mi = 0.0;
  for (int i = 0; i < n_x; ++i)
    for (int j = 0; j < n_y; ++j) {
      const double pij = joint[static_cast<size_t>(i) * n_y + j];
      if (pij > 0.0 && px[i] > 0.0 && py[j] > 0.0)
        mi += pij * std::log2(pij / (px[i] * py[j]));
    }
  return mi;
}

}  // namespace gary
