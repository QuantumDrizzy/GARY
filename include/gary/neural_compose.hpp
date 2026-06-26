// GARY — Phase 12: neural compositionality. Do neural agents that learn to communicate about
// composite meanings develop a COMPOSITIONAL code, or just a successful holistic one?
// Joins Phase 7 (compositionality / topographic similarity) with Phase 11 (neural agents).
// Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>
#include <vector>

namespace gary {

struct NeuralComposeResult {
  std::vector<double> success;   // full-meaning reconstruction rate per checkpoint
  std::vector<double> topo_sim;  // topographic similarity (compositionality) per checkpoint
  double final_success;
  double final_topo_sim;
};

// A multi-head MLP sender maps a composite meaning (n_features features, each of n_values
// values) to a message of n_features symbols (alphabet n_symbols); a multi-head MLP receiver
// maps the message back to the n_features feature values. Both are trained by REINFORCE on the
// per-feature reconstruction reward. Measures full-meaning success and the topographic
// similarity of the emergent code (0 = holistic, 1 = perfectly compositional).
// `holistic_input`: if false (default) the sender sees a feature-structured input (one one-hot
// block per feature) — a compositional inductive bias. If true the sender sees the meaning as a
// single atomic one-hot (no feature structure). Comparing the two disentangles how much of the
// emergent compositionality comes from the architecture vs the communication pressure.
NeuralComposeResult neural_compositionality(int n_features, int n_values, int n_symbols,
                                            int hidden, int rounds, int checkpoints,
                                            double learning_rate, std::uint64_t seed,
                                            bool holistic_input = false);

struct NeuralIteratedResult {
  std::vector<double> topo_sim;  // topographic similarity per generation
  double initial_topo_sim;       // generation 0 (a random language)
  double final_topo_sim;
};

// Neural iterated learning (Kirby / Ren et al.) with BOTH pressures. Each generation: (1) a
// LEARNING phase — the new sender imitates the previous generation's language on a BOTTLENECK
// subset of meanings (compressibility/learnability pressure); then (2) an INTERACTION phase —
// the sender plays the referential game with a fresh receiver, trained by REINFORCE
// (expressivity pressure). The sender uses a HOLISTIC input (no architectural bias, cf. Phase
// 13). A bottleneck ALONE collapses to a degenerate constant language (topo-sim 0); only the
// COMBINATION of compressibility + expressivity yields compositionality. Does it emerge?
NeuralIteratedResult neural_iterated_learning(int n_features, int n_values, int n_symbols,
                                              int hidden, int bottleneck, int generations,
                                              int epochs, int interact_rounds,
                                              double learning_rate, std::uint64_t seed);

}  // namespace gary
