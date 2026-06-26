// GARY — Phase 7: compositionality via iterated learning. Does grammar emerge?
// Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>
#include <vector>

namespace gary {

// What inductive bias the learner brings to each generation.
enum class LearnerBias {
  Compositional,  // infers per-position (feature -> symbol) maps; generalises to unseen meanings
  Holistic        // memorises seen (meaning -> signal), random for unseen (no structure)
};

struct IteratedResult {
  std::vector<double> topo_sim;     // topographic similarity per generation (0..gen)
  double final_topo_sim;            // last generation's topographic similarity
  double final_expressivity;        // fraction of meanings with a distinct signal (1 = fully expressive)
};

// Iterated learning (Kirby-style cultural transmission). Meanings have `n_features` features,
// each taking `n_values` values (M = n_values^n_features meanings). A signal is `n_features`
// symbols over an alphabet of size `n_symbols`. Each generation learns the language from a
// BOTTLENECK sample of the previous generation's signals, then must produce signals for ALL
// meanings. Topographic similarity — the correlation between meaning-distance and
// signal-distance over all meaning pairs — measures emergent compositional structure
// (0 = holistic/random, 1 = perfectly compositional). Starts from a random language.
IteratedResult iterated_learning(int n_features, int n_values, int n_symbols, int bottleneck,
                                 int generations, LearnerBias bias, std::uint64_t seed);

}  // namespace gary
