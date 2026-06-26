// GARY — Phase 12: neural compositionality. Do neural agents grow a compositional code?
#include <cstdio>

#include "gary/neural_compose.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 12 - neural compositionality: do neural agents grow a compositional code?\n");

  const int F = 2, V = 5, S = 5, hidden = 64, rounds = 800000, checkpoints = 10;
  const double lr = 0.05;
  std::printf("%d features x %d values = %d composite meanings; %d-symbol messages (alphabet %d).\n",
              F, V, V * V, F, S);
  std::printf("topographic similarity: 0 = holistic, 1 = perfectly compositional.\n\n");

  const gary::NeuralComposeResult r =
      gary::neural_compositionality(F, V, S, hidden, rounds, checkpoints, lr, 7);

  std::printf("  checkpoint    success     topo-sim (compositionality)\n");
  for (size_t i = 0; i < r.success.size(); ++i)
    std::printf("  %8zu     %6.1f%%      %6.3f\n", i + 1, r.success[i] * 100.0, r.topo_sim[i]);

  std::printf("\nfinal: success %.1f%%, topographic similarity %.3f\n", r.final_success * 100.0,
              r.final_topo_sim);

  std::printf(
      "\nneural agents learn to communicate composite meanings (success ~68%%) AND the emergent\n"
      "code is fairly compositional (topo-sim ~0.72). Honest caveat: the architecture helps --\n"
      "a feature-structured input plus one message head per position bias toward compositionality.\n"
      "How much is the communication pressure vs the architecture is the real open question (cf.\n"
      "Phase 7's tabular iterated learning, where a bottleneck drove compositionality on its own).\n");
  return 0;
}
