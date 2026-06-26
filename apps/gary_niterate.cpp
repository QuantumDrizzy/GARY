// GARY — Phase 14: neural iterated learning. Does a transmission bottleneck recover
// compositionality where communication pressure alone (Phase 13's holistic run) did not?
#include <cstdio>

#include "gary/neural_compose.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 14 - neural iterated learning: does compositionality emerge from transmission?\n");
  std::printf("HOLISTIC input (no architectural bias). TWO pressures per generation: a bottleneck\n");
  std::printf("(learnability) + an interaction game (expressivity). A bottleneck ALONE collapses to a\n");
  std::printf("degenerate constant language; only both pressures together can yield compositionality.\n\n");

  const int F = 2, V = 5, S = 5, hidden = 32, bottleneck = 12, gens = 30, epochs = 100, interact = 20000;
  const double lr = 0.05;
  const gary::NeuralIteratedResult r =
      gary::neural_iterated_learning(F, V, S, hidden, bottleneck, gens, epochs, interact, lr, 7);

  std::printf("  generation    topo-sim (compositionality)\n");
  for (size_t i = 0; i < r.topo_sim.size(); i += 4)
    std::printf("  %8zu       %6.3f\n", i, r.topo_sim[i]);
  if ((r.topo_sim.size() - 1) % 4 != 0)
    std::printf("  %8zu       %6.3f\n", r.topo_sim.size() - 1, r.topo_sim.back());

  const double gain = r.final_topo_sim - r.initial_topo_sim;
  std::printf("\ninitial (gen 0, random): %.3f  ->  final: %.3f   (gain %.3f)\n", r.initial_topo_sim,
              r.final_topo_sim, gain);

  if (gain > 0.2)
    std::printf(
        "\ncompositionality RECOVERS via the bottleneck, even with a holistic input -- the Kirby\n"
        "result in neural form. Communication pressure alone (Phase 13: 0.376) did not; cultural\n"
        "transmission through a bottleneck does. The arc closes: structure is selected by the\n"
        "bottleneck, not handed over by the architecture.\n");
  else
    std::printf(
        "\nhonest, nuanced result: a bottleneck ALONE collapses to 0 (a constant language); adding\n"
        "the expressivity pressure prevents that collapse and nudges compositionality up (%.3f ->\n"
        "%.3f) -- but only WEAKLY, nowhere near the architecture-biased run (Phase 12: 0.715) or\n"
        "the tabular bottleneck (Phase 7: ~1.0). In this naive neural setup, strong compositionality\n"
        "needs the architectural prior or far stronger transmission -- it is not handed over for\n"
        "free by cultural transmission alone.\n",
        r.initial_topo_sim, r.final_topo_sim);
  return 0;
}
