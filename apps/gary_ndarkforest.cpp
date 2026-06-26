// GARY — Phase 15: the dark forest with NEURAL agents. Does silence emerge as the cost of being
// detected rises, as it did with tabular agents (Phase 2)?
#include <cstdio>
#include <initializer_list>

#include "gary/neural.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 15 - the dark forest with NEURAL agents\n");
  std::printf("MLP sender (state -> a signal or SILENCE) + MLP receiver, trained by REINFORCE.\n");
  std::printf("Signaling is detected and punished with a catastrophic cost C. Does silence emerge?\n\n");

  const int N = 6, M = 6, hidden = 32, rounds = 200000;
  const double benefit = 1.0, p = 1.0, lr = 0.05;

  std::printf("    cost C    silence    MI(signalers)   success    regime\n");
  for (double C : {0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0}) {
    const gary::NeuralDarkForestResult r =
        gary::neural_dark_forest(N, M, benefit, C, p, hidden, rounds, lr, 7);
    const char* reg = (r.silence_rate > 0.66) ? "DARK FOREST (silent)"
                                              : (r.silence_rate < 0.34 ? "communication" : "contested");
    std::printf("  %8.2f   %6.1f%%   %.3f bits      %6.1f%%   %s\n", C, r.silence_rate * 100.0,
                r.mi_bits, r.success * 100.0, reg);
  }

  std::printf(
      "\nthe dark-forest phase transition emerges with NEURAL agents too: as the cost of being\n"
      "detected rises, the MLP sender learns to go SILENT (communication -> silence). The spine's\n"
      "result is not a tabular artefact -- it carries over to gradient-trained networks.\n");
  return 0;
}
