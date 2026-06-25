#include "gary/chaos.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "gary/info_theory.hpp"

namespace gary {
namespace {

int bin_of(double x, int bins) {
  int b = static_cast<int>(x * bins);
  if (b < 0) b = 0;
  if (b >= bins) b = bins - 1;
  return b;
}

// I(x_{n+1}; x_n) of a series, estimated by binning into `bins` bins.
double consecutive_mi(const std::vector<double>& s, int bins) {
  const int n = static_cast<int>(s.size()) - 1;
  if (n <= 0) return 0.0;
  std::vector<double> joint(static_cast<size_t>(bins) * bins, 0.0);
  for (int i = 0; i < n; ++i)
    joint[static_cast<size_t>(bin_of(s[i], bins)) * bins + bin_of(s[i + 1], bins)] += 1.0;
  for (auto& v : joint) v /= n;
  return mutual_information_bits(joint, bins, bins);
}

double marginal_entropy(const std::vector<double>& s, int bins) {
  std::vector<double> p(static_cast<size_t>(bins), 0.0);
  for (double x : s) p[bin_of(x, bins)] += 1.0;
  for (auto& v : p) v /= static_cast<double>(s.size());
  return entropy_bits(p);
}

}  // namespace

double lyapunov_exponent(double r, int iters, int transient) {
  double x = 0.5;
  for (int i = 0; i < transient; ++i) x = r * x * (1.0 - x);
  double sum = 0.0;
  for (int i = 0; i < iters; ++i) {
    const double deriv = r * (1.0 - 2.0 * x);
    sum += std::log(std::fabs(deriv) + 1e-300);
    x = r * x * (1.0 - x);
  }
  return sum / iters;
}

ChaosMi chaos_vs_random(double r, int n, int bins, std::uint64_t seed) {
  std::vector<double> chaos;
  chaos.reserve(static_cast<size_t>(n));
  double x = 0.5;
  for (int i = 0; i < 1000; ++i) x = r * x * (1.0 - x);  // discard transient
  for (int i = 0; i < n; ++i) {
    x = r * x * (1.0 - x);
    chaos.push_back(x);
  }

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> rnd;
  rnd.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) rnd.push_back(u(rng));

  ChaosMi m;
  m.chaos_self_mi = consecutive_mi(chaos, bins);
  m.random_self_mi = consecutive_mi(rnd, bins);
  m.chaos_marginal_h = marginal_entropy(chaos, bins);
  m.random_marginal_h = marginal_entropy(rnd, bins);
  m.lyapunov = lyapunov_exponent(r, n, 1000);
  return m;
}

}  // namespace gary
