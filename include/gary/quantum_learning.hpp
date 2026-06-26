// GARY — Phase 9: quantum coordination games WITH learning. Can learning agents discover that
// entanglement beats the classical bound? Extends the quantum lens (Phase 4) with the learning
// spine (Phase 0). Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>
#include <vector>

namespace gary {

struct LearningChshResult {
  std::vector<double> win_rate;  // win rate per training checkpoint (the learning curve)
  double final_win_rate;         // win rate of the learned GREEDY strategy
  double classical_bound;        // 0.75 (provable classical maximum)
  double tsirelson;              // cos^2(pi/8) ~ 0.8536 (quantum maximum)
};

// Two agents learn a CHSH coordination strategy by reinforcement (Roth-Erev + forgetting).
// quantum=true: they share a Bell state and learn measurement ANGLES (entanglement is a
// resource to exploit). quantum=false: they learn classical output BITS (bounded by 0.75).
// Does learning, with nothing hand-designed, discover that entanglement beats the classical
// bound? n_angles discretises [0, pi) for the quantum agents.
LearningChshResult learn_chsh(bool quantum, int n_angles, int rounds, int checkpoints,
                              double learning_rate, double forgetting, std::uint64_t seed);

}  // namespace gary
