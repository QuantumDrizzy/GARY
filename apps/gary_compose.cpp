// GARY — Phase 7: compositionality via iterated learning. Does grammar emerge?
// A transmission chain with a bottleneck. A compositional learner turns a random language into
// a compositional one over generations; a holistic learner does not. Topographic similarity
// (meaning-distance vs signal-distance correlation) is the measure.
#include <cstdio>

#include "gary/iterated.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 7 - compositionality via iterated learning (Kirby-style)\n");

  const int F = 2, V = 6, S = 6, B = 14, G = 30;  // 36 meanings, bottleneck 14, 30 generations
  std::printf("%d features x %d values = %d meanings; alphabet %d; bottleneck %d; %d generations.\n",
              F, V, V * V, S, B, G);
  std::printf("topographic similarity: 0 = holistic/random, 1 = perfectly compositional.\n\n");

  const gary::IteratedResult comp =
      gary::iterated_learning(F, V, S, B, G, gary::LearnerBias::Compositional, 7);
  const gary::IteratedResult holi =
      gary::iterated_learning(F, V, S, B, G, gary::LearnerBias::Holistic, 7);

  std::printf("  generation   topo-sim (compositional learner)   topo-sim (holistic, control)\n");
  for (int g = 0; g <= G; g += 5) {
    std::printf("  %8d              %6.3f                           %6.3f\n", g, comp.topo_sim[g],
                holi.topo_sim[g]);
  }

  std::printf("\nfinal: compositional learner -> topo-sim %.3f (expressivity %.2f); holistic -> %.3f\n",
              comp.final_topo_sim, comp.final_expressivity, holi.final_topo_sim);
  std::printf(
      "\ngrammar emerges from cultural transmission: under a compositional bias + a bottleneck,\n"
      "a random language regularises into a compositional one (topo-sim climbs toward 1). The\n"
      "holistic control stays near 0 -- structure is not automatic, it is selected for.\n");
  return 0;
}
