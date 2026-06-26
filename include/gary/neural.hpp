// GARY — Phase 11: neural emergent communication. The tabular -> neural jump of the spine.
// Tiny from-scratch MLP agents trained by REINFORCE. Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>
#include <vector>

namespace gary {

struct NeuralCommResult {
  std::vector<double> mi_bits;   // I(state; signal) at each training checkpoint
  std::vector<double> success;   // success rate at each checkpoint
  double final_mi;
  double final_success;
  double max_mi;                 // log2(n_states)
};

// Neural emergent communication: a tiny MLP sender (one-hot state -> hidden(tanh) -> signal
// logits) and a tiny MLP receiver (one-hot signal -> hidden -> action logits) are trained
// TOGETHER by REINFORCE (policy gradient with a running baseline) on the referential signaling
// game (reward 1 iff the receiver's action equals the state). Does communication emerge with
// NEURAL agents, not tables? Measures I(state; signal) (from the sender's policy) and success.
NeuralCommResult neural_emergent_comm(int n_states, int n_signals, int hidden, int rounds,
                                      int checkpoints, double learning_rate, std::uint64_t seed);

}  // namespace gary
