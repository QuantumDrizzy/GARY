// GARY — Phase 13: disentangling neural compositionality. Architecture vs communication
// pressure. Same task and training; only the sender's input representation changes.
#include <cstdio>

#include "gary/neural_compose.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 13 - disentangling compositionality: architecture vs communication pressure\n");
  std::printf("same task (2 features x 5 values, 2-symbol messages), same training -- only the\n");
  std::printf("SENDER's input representation changes. Does the compositionality survive a holistic input?\n\n");

  const int F = 2, V = 5, S = 5, hidden = 64, rounds = 800000, cp = 10;
  const double lr = 0.05;

  const gary::NeuralComposeResult structured =
      gary::neural_compositionality(F, V, S, hidden, rounds, cp, lr, 7, /*holistic_input=*/false);
  const gary::NeuralComposeResult holistic =
      gary::neural_compositionality(F, V, S, hidden, rounds, cp, lr, 7, /*holistic_input=*/true);

  std::printf("  sender input             success     topo-sim (compositionality)\n");
  std::printf("  structured (per-feature)  %6.1f%%      %6.3f\n", structured.final_success * 100.0,
              structured.final_topo_sim);
  std::printf("  holistic   (atomic)       %6.1f%%      %6.3f\n", holistic.final_success * 100.0,
              holistic.final_topo_sim);

  const double drop = structured.final_topo_sim - holistic.final_topo_sim;
  std::printf("\nverdict (topo-sim drop = %.3f):\n", drop);
  if (drop > 0.15)
    std::printf(
        "  compositionality DROPS with a holistic input -- the structured input (the\n"
        "  ARCHITECTURE) was driving it, not the communication pressure alone. My Phase 12\n"
        "  caveat was right: don't read 0.72 as 'pressure composes'.\n");
  else
    std::printf(
        "  compositionality largely SURVIVES a holistic input -- the communication pressure\n"
        "  composes even without the structured input. The architecture was not the whole story.\n");
  return 0;
}
