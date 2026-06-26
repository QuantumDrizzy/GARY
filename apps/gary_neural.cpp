// GARY — Phase 11: neural emergent communication. Tabular -> neural.
#include <cstdio>

#include "gary/neural.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 11 - neural emergent communication (from-scratch MLP + REINFORCE)\n");

  const int N = 6, M = 6, hidden = 32, rounds = 400000, checkpoints = 10;
  const double lr = 0.05;
  std::printf("N=%d states, %d signals, %d hidden units. Perfect code = %.3f bits.\n\n", N, M,
              hidden, /*log2(N)*/ 2.585);

  const gary::NeuralCommResult r = gary::neural_emergent_comm(N, M, hidden, rounds, checkpoints, lr, 7);

  std::printf("  checkpoint    success     I(state;signal)\n");
  for (size_t i = 0; i < r.mi_bits.size(); ++i)
    std::printf("  %8zu     %6.1f%%      %.3f bits\n", i + 1, r.success[i] * 100.0, r.mi_bits[i]);

  std::printf("\nfinal: success %.1f%%, I = %.3f of %.3f bits (%.0f%% of a perfect code)\n",
              r.final_success * 100.0, r.final_mi, r.max_mi, 100.0 * r.final_mi / r.max_mi);

  std::printf(
      "\nthe same emergence happens with NEURAL agents: two tiny MLPs, trained only by policy\n"
      "gradient on the reward, grow a shared code from nothing -- mutual information climbs and\n"
      "the receiver decodes. The spine is not tied to tables; it carries over to real networks.\n");
  return 0;
}
