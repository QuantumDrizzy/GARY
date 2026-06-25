#include "gary/dark_forest.hpp"

#include "gary/info_theory.hpp"

namespace gary {

DarkForestGame::DarkForestGame(int n_states, int n_signals, double benefit, double cost,
                               double p_detect, std::uint64_t seed, double learning_rate,
                               double floor)
    : n_states_(n_states),
      n_signals_(n_signals),
      n_opts_(n_signals + 1),  // last option = SILENCE
      benefit_(benefit),
      cost_(cost),
      p_detect_(p_detect),
      lr_(learning_rate),
      floor_(floor),
      sender_w_(static_cast<size_t>(n_states) * (n_signals + 1), 1.0),
      receiver_w_(static_cast<size_t>(n_signals) * n_states, 1.0),
      rng_(seed) {}

static int sample_weighted(const double* w, int n, std::mt19937_64& rng) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += w[i];
  std::uniform_real_distribution<double> u(0.0, sum);
  const double r = u(rng);
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    acc += w[i];
    if (r <= acc) return i;
  }
  return n - 1;
}

int DarkForestGame::sample_sender(int state) {
  return sample_weighted(&sender_w_[static_cast<size_t>(state) * n_opts_], n_opts_, rng_);
}

int DarkForestGame::sample_receiver(int signal) {
  return sample_weighted(&receiver_w_[static_cast<size_t>(signal) * n_states_], n_states_, rng_);
}

void DarkForestGame::step() {
  std::uniform_int_distribution<int> states(0, n_states_ - 1);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  const int t = states(rng_);
  const int o = sample_sender(t);

  double payoff;
  if (o == n_signals_) {  // SILENCE — safe, no information
    payoff = 0.0;
  } else {
    const int s = o;
    const int a = sample_receiver(s);
    const bool understood = (a == t);
    const bool detected = (u01(rng_) < p_detect_);
    payoff = (understood ? benefit_ : 0.0) - (detected ? cost_ : 0.0);
    if (understood)  // receiver bears no cost; reinforce successful decoding
      receiver_w_[static_cast<size_t>(s) * n_states_ + a] += benefit_;
  }

  // Floored payoff reinforcement on the chosen sender option. Silence (payoff 0) is the
  // unchanging safe baseline; a signal grows only if its net payoff beats it.
  double& w = sender_w_[static_cast<size_t>(t) * n_opts_ + o];
  w += lr_ * payoff;
  if (w < floor_) w = floor_;
}

void DarkForestGame::run(int rounds) {
  for (int i = 0; i < rounds; ++i) step();
}

DarkForestResult DarkForestGame::measure() const {
  const int N = n_states_, M = n_signals_, O = n_opts_;
  const double pt = 1.0 / N;

  double silence = 0.0;
  std::vector<double> js(static_cast<size_t>(N) * M, 0.0);  // joint p(state, signal)
  double total_sig = 0.0;

  for (int t = 0; t < N; ++t) {
    const double* w = &sender_w_[static_cast<size_t>(t) * O];
    double sum = 0.0;
    for (int o = 0; o < O; ++o) sum += w[o];
    silence += pt * (w[M] / sum);  // option M = silence
    for (int s = 0; s < M; ++s) {
      const double p = pt * (w[s] / sum);
      js[static_cast<size_t>(t) * M + s] = p;
      total_sig += p;
    }
  }

  DarkForestResult r;
  r.silence_rate = silence;
  if (total_sig < 1e-12) {  // total silence: code undefined
    r.mi_bits = 0.0;
    r.success_rate = 0.0;
    return r;
  }

  // MI among signaling rounds: renormalise the signal-only joint.
  std::vector<double> cj(static_cast<size_t>(N) * M);
  for (size_t i = 0; i < cj.size(); ++i) cj[i] = js[i] / total_sig;
  r.mi_bits = mutual_information_bits(cj, N, M);

  // Success among signaling rounds.
  double succ = 0.0;
  for (int t = 0; t < N; ++t)
    for (int s = 0; s < M; ++s) {
      const double* rw = &receiver_w_[static_cast<size_t>(s) * N];
      double rsum = 0.0;
      for (int a = 0; a < N; ++a) rsum += rw[a];
      succ += js[static_cast<size_t>(t) * M + s] * (rw[t] / rsum);
    }
  r.success_rate = succ / total_sig;
  return r;
}

}  // namespace gary
