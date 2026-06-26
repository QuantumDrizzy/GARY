#include "gary/quantum_learning.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace gary {
namespace {

constexpr double kPi = 3.14159265358979323846;

int sample_weighted(const std::vector<double>& w, int off, int n, std::mt19937_64& rng) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += w[off + i];
  std::uniform_real_distribution<double> u(0.0, sum);
  const double r = u(rng);
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    acc += w[off + i];
    if (r <= acc) return i;
  }
  return n - 1;
}

int argmax_at(const std::vector<double>& w, int off, int n) {
  int best = 0;
  double bv = w[off];
  for (int i = 1; i < n; ++i)
    if (w[off + i] > bv) { bv = w[off + i]; best = i; }
  return best;
}

// CHSH win probability for chosen options under each model.
double win_prob(bool quantum, int n_angles, int kA, int kB, int x, int y) {
  if (quantum) {
    const double aA = kA * kPi / n_angles;
    const double bB = kB * kPi / n_angles;
    const double E = std::cos(aA - bB);            // Bell-state correlation
    return 0.5 * (1.0 + ((x & y) ? -E : E));       // win iff (a xor b) == (x and y)
  }
  return ((kA ^ kB) == (x & y)) ? 1.0 : 0.0;       // deterministic output bits
}

}  // namespace

LearningChshResult learn_chsh(bool quantum, int n_angles, int rounds, int checkpoints,
                              double learning_rate, double forgetting, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> bit(0, 1);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  const int K = quantum ? n_angles : 2;  // options per input: angles, or output bits {0,1}
  std::vector<double> wA(static_cast<size_t>(2) * K, 1.0);  // Alice: [input x][option]
  std::vector<double> wB(static_cast<size_t>(2) * K, 1.0);  // Bob
  const double keep = 1.0 - forgetting;

  LearningChshResult res;
  res.classical_bound = 0.75;
  const double c = std::cos(kPi / 8.0);
  res.tsirelson = c * c;

  int block = rounds / std::max(1, checkpoints);
  if (block < 1) block = 1;
  int wins_block = 0, n_block = 0;

  for (int t = 0; t < rounds; ++t) {
    const int x = bit(rng), y = bit(rng);
    const int kA = sample_weighted(wA, x * K, K, rng);
    const int kB = sample_weighted(wB, y * K, K, rng);
    const bool win = (u01(rng) < win_prob(quantum, n_angles, kA, kB, x, y));

    if (forgetting > 0.0) {
      for (double& v : wA) v *= keep;
      for (double& v : wB) v *= keep;
    }
    if (win) {
      wA[x * K + kA] += learning_rate;
      wB[y * K + kB] += learning_rate;
    }

    wins_block += win ? 1 : 0;
    ++n_block;
    if (n_block >= block) {
      res.win_rate.push_back(static_cast<double>(wins_block) / n_block);
      wins_block = 0;
      n_block = 0;
    }
  }
  if (n_block > 0) res.win_rate.push_back(static_cast<double>(wins_block) / n_block);

  // Value of the learned GREEDY strategy (each input -> its best option), exact over inputs.
  double sum = 0.0;
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y) {
      const int kA = argmax_at(wA, x * K, K);
      const int kB = argmax_at(wB, y * K, K);
      sum += win_prob(quantum, n_angles, kA, kB, x, y);
    }
  res.final_win_rate = sum / 4.0;
  return res;
}

}  // namespace gary
