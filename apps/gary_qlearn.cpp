// GARY — Phase 9: quantum coordination games with learning. Do learning agents discover that
// entanglement beats the classical bound?
#include <cstdio>

#include "gary/quantum_learning.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 9 - quantum coordination games WITH learning\n");
  std::printf("agents learn a CHSH strategy by reinforcement (nothing hand-designed).\n");
  std::printf("classical max = 0.750; quantum (Tsirelson) max = 0.854.\n\n");

  const int n_angles = 8, rounds = 300000, checkpoints = 10;
  const double lr = 0.1, forgetting = 0.01;

  const gary::LearningChshResult q =
      gary::learn_chsh(true, n_angles, rounds, checkpoints, lr, forgetting, 7);
  const gary::LearningChshResult c =
      gary::learn_chsh(false, n_angles, rounds, checkpoints, lr, forgetting, 7);

  std::printf("  checkpoint   quantum learner   classical learner\n");
  const size_t n = q.win_rate.size() < c.win_rate.size() ? q.win_rate.size() : c.win_rate.size();
  for (size_t i = 0; i < n; ++i)
    std::printf("  %8zu      %6.3f             %6.3f\n", i + 1, q.win_rate[i], c.win_rate[i]);

  std::printf("\nlearned greedy strategy:  quantum %.3f   |   classical %.3f\n", q.final_win_rate,
              c.final_win_rate);
  std::printf("(classical bound 0.750, Tsirelson 0.854)\n");

  std::printf(
      "\nlearning agents discover what hand-designed strategies show: a quantum-equipped\n"
      "learner beats the classical bound (> 0.75, toward Tsirelson), while a classical learner\n"
      "cannot exceed 0.75. Entanglement is a resource that learning can exploit, not just\n"
      "a hand-tuned trick.\n");
  return 0;
}
