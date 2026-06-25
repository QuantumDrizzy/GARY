// GARY — Phase 2: the dark forest. Signaling under existential risk.
// Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace gary {

struct DarkForestResult {
  double silence_rate;  // P(sender stays silent), uniform state prior
  double mi_bits;       // I(state; signal) among signaling rounds (code quality)
  double success_rate;  // P(partner identifies the state | a signal was sent)
};

// A signaling game in the dark-forest regime. Each round the sender either emits one
// of n_signals signals or stays SILENT (a safe, no-information option). Emitting can be
// detected (probability p_detect) and punished with a catastrophic cost C. The sender
// learns by floored payoff reinforcement (silence is the payoff-0 safe default); the
// receiver learns by reinforcement on successful decoding (it bears no cost).
//
// The question: as C rises, does silence become the equilibrium (communication collapses)?
// Note the chicken-and-egg the model exposes honestly: a code cannot form without
// signaling, but signaling is risky before a code exists — so risk can suppress the code
// from ever forming, at a far lower C than a warm equilibrium would tolerate.
class DarkForestGame {
 public:
  DarkForestGame(int n_states, int n_signals, double benefit, double cost, double p_detect,
                 std::uint64_t seed, double learning_rate = 0.1, double floor = 0.01);

  void step();
  void run(int rounds);

  // Analytic metrics from the current (frozen) strategies.
  DarkForestResult measure() const;

 private:
  int sample_sender(int state);
  int sample_receiver(int signal);

  int n_states_;
  int n_signals_;
  int n_opts_;  // n_signals_ + 1; option index n_signals_ == SILENCE
  double benefit_, cost_, p_detect_, lr_, floor_;
  std::vector<double> sender_w_;    // [n_states  * n_opts_]
  std::vector<double> receiver_w_;  // [n_signals * n_states]
  std::mt19937_64 rng_;
};

}  // namespace gary
