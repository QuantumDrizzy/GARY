#include "gary/decipher.hpp"

#include <random>
#include <vector>

#include "gary/info_theory.hpp"
#include "gary/signaling.hpp"

namespace gary {
namespace {

int argmax_row(const std::vector<double>& m, int row, int ncol) {
  int best = 0;
  double bv = m[static_cast<size_t>(row) * ncol];
  for (int j = 1; j < ncol; ++j) {
    const double v = m[static_cast<size_t>(row) * ncol + j];
    if (v > bv) { bv = v; best = j; }
  }
  return best;
}

int sample_row(const std::vector<double>& p, int row, int ncol, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng), acc = 0.0;
  for (int j = 0; j < ncol; ++j) {
    acc += p[static_cast<size_t>(row) * ncol + j];
    if (r <= acc) return j;
  }
  return ncol - 1;
}

}  // namespace

DecipherResult lingua_cosmica(int n_sym, int train_rounds, int obs_rounds, double forgetting,
                              Anchor anchor, double crib_fraction, std::uint64_t seed) {
  const int N = n_sym, M = n_sym;

  // 1. A sender-receiver pair invents a code.
  SignalingGame game(N, M, seed, forgetting);
  game.run(train_rounds);
  const std::vector<double> S = game.sender_strategy();    // p(signal | state)
  const std::vector<double> R = game.receiver_strategy();  // p(action | signal)

  std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
  std::uniform_int_distribution<int> states(0, N - 1);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  // 2. The third agent collects signal -> state evidence under its access mode.
  std::vector<double> evidence(static_cast<size_t>(M) * N, 0.0);
  for (int i = 0; i < obs_rounds; ++i) {
    const int t = states(rng);
    const int s = sample_row(S, t, M, rng);
    if (anchor == Anchor::Behavior) {
      const int a = sample_row(R, s, N, rng);  // the receiver's action leaks the meaning
      evidence[static_cast<size_t>(s) * N + a] += 1.0;
    } else if (anchor == Anchor::Crib) {
      if (u01(rng) < crib_fraction)            // known plaintext on a fraction of rounds
        evidence[static_cast<size_t>(s) * N + t] += 1.0;
    }
    // CiphertextOnly: no grounding evidence is available.
  }

  // 3. Build the predictor signal -> state. With no evidence (ciphertext only) the labeling
  //    is arbitrary (identity): structure may be recovered, but not grounded meaning.
  std::vector<int> shat(M);
  for (int s = 0; s < M; ++s) {
    double rowsum = 0.0;
    for (int j = 0; j < N; ++j) rowsum += evidence[static_cast<size_t>(s) * N + j];
    shat[s] = (rowsum > 0.0) ? argmax_row(evidence, s, N) : (s % N);
  }

  // 4. Evaluate on a held-out test stream.
  const int test_rounds = obs_rounds;
  std::vector<double> conf(static_cast<size_t>(N) * N, 0.0);  // conf[true][predicted]
  int correct = 0;
  for (int i = 0; i < test_rounds; ++i) {
    const int t = states(rng);
    const int s = sample_row(S, t, M, rng);
    const int pred = shat[s];
    conf[static_cast<size_t>(t) * N + pred] += 1.0;
    if (pred == t) ++correct;
  }
  for (auto& c : conf) c /= test_rounds;

  DecipherResult res;
  res.accuracy = static_cast<double>(correct) / test_rounds;
  res.mi_bits = mutual_information_bits(conf, N, N);
  return res;
}

}  // namespace gary
