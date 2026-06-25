// GARY — Lewis/Skyrms signaling game with Roth-Erev reinforcement.
// Portable core, no CUDA (Pi-capable). Phase 0: meaning emerges from interaction
// with nothing injected, and the meaning-meter (mutual information) measures it.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace gary {

// A Lewis-Skyrms signaling game.
//   Nature draws a state t in [0, n_states)   (uniform prior).
//   Sender:   state  -> signal   (stochastic, learned).
//   Receiver: signal -> action   (stochastic, learned).
//   Payoff:   success iff action == state.   Actions = states.
// Learning is Roth-Erev reinforcement (Polya urn): used (state,signal) and
// (signal,action) pairs are reinforced on success. A "signaling system" emerges.
class SignalingGame {
 public:
  SignalingGame(int n_states, int n_signals, std::uint64_t seed);

  // Play one round; returns true on success (action == state).
  bool step();

  // Play `rounds` rounds; returns the success rate over them.
  double run(int rounds);

  // I(state; signal) in bits, analytic from the current sender strategy under a
  // uniform state prior. 0 = no code; log2(n_states) = a perfect code.
  double mutual_information_bits() const;

  // Joint p(state, signal) (row-major n_states * n_signals) from the strategy.
  std::vector<double> state_signal_joint() const;

  int n_states() const noexcept { return n_states_; }
  int n_signals() const noexcept { return n_signals_; }

 private:
  int sender_signal(int state);
  int receiver_action(int signal);

  int n_states_;
  int n_signals_;
  int n_acts_;
  std::vector<double> sender_w_;    // [n_states  * n_signals] reinforcement weights
  std::vector<double> receiver_w_;  // [n_signals * n_acts]
  std::mt19937_64 rng_;
};

}  // namespace gary
