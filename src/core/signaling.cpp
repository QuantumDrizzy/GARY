#include "gary/signaling.hpp"

#include "gary/info_theory.hpp"

namespace gary {

SignalingGame::SignalingGame(int n_states, int n_signals, std::uint64_t seed, double forgetting)
    : n_states_(n_states),
      n_signals_(n_signals),
      n_acts_(n_states),
      sender_w_(static_cast<size_t>(n_states) * n_signals, 1.0),
      receiver_w_(static_cast<size_t>(n_signals) * n_states, 1.0),
      rng_(seed),
      forgetting_(forgetting) {}

// Sample an index in [0, n) proportional to the weights w[0..n).
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

int SignalingGame::sender_signal(int state) {
  return sample_weighted(&sender_w_[static_cast<size_t>(state) * n_signals_], n_signals_, rng_);
}

int SignalingGame::receiver_action(int signal) {
  return sample_weighted(&receiver_w_[static_cast<size_t>(signal) * n_acts_], n_acts_, rng_);
}

bool SignalingGame::step() {
  std::uniform_int_distribution<int> states(0, n_states_ - 1);
  const int t = states(rng_);
  const int s = sender_signal(t);
  const int a = receiver_action(s);
  const bool success = (a == t);

  if (forgetting_ > 0.0) {  // Roth-Erev with forgetting (Barrett & Zollman 2009)
    const double keep = 1.0 - forgetting_;
    for (double& w : sender_w_) w *= keep;
    for (double& w : receiver_w_) w *= keep;
  }
  if (success) {  // reinforcement
    sender_w_[static_cast<size_t>(t) * n_signals_ + s] += 1.0;
    receiver_w_[static_cast<size_t>(s) * n_acts_ + a] += 1.0;
  }
  return success;
}

double SignalingGame::run(int rounds) {
  int wins = 0;
  for (int i = 0; i < rounds; ++i) wins += step() ? 1 : 0;
  return rounds > 0 ? static_cast<double>(wins) / rounds : 0.0;
}

std::vector<double> SignalingGame::state_signal_joint() const {
  std::vector<double> joint(static_cast<size_t>(n_states_) * n_signals_, 0.0);
  const double pt = 1.0 / n_states_;  // uniform state prior
  for (int t = 0; t < n_states_; ++t) {
    const double* w = &sender_w_[static_cast<size_t>(t) * n_signals_];
    double sum = 0.0;
    for (int s = 0; s < n_signals_; ++s) sum += w[s];
    for (int s = 0; s < n_signals_; ++s)
      joint[static_cast<size_t>(t) * n_signals_ + s] = pt * (w[s] / sum);
  }
  return joint;
}

double SignalingGame::mutual_information_bits() const {
  return gary::mutual_information_bits(state_signal_joint(), n_states_, n_signals_);
}

std::vector<double> SignalingGame::sender_strategy() const {
  std::vector<double> p(static_cast<size_t>(n_states_) * n_signals_);
  for (int t = 0; t < n_states_; ++t) {
    const double* w = &sender_w_[static_cast<size_t>(t) * n_signals_];
    double sum = 0.0;
    for (int s = 0; s < n_signals_; ++s) sum += w[s];
    for (int s = 0; s < n_signals_; ++s)
      p[static_cast<size_t>(t) * n_signals_ + s] = w[s] / sum;
  }
  return p;
}

std::vector<double> SignalingGame::receiver_strategy() const {
  std::vector<double> p(static_cast<size_t>(n_signals_) * n_acts_);
  for (int s = 0; s < n_signals_; ++s) {
    const double* w = &receiver_w_[static_cast<size_t>(s) * n_acts_];
    double sum = 0.0;
    for (int a = 0; a < n_acts_; ++a) sum += w[a];
    for (int a = 0; a < n_acts_; ++a)
      p[static_cast<size_t>(s) * n_acts_ + a] = w[a] / sum;
  }
  return p;
}

}  // namespace gary
