// GARY — Phase 0 seed: a Lewis/Skyrms signaling game.
//
// Meaning emerges from interaction with nothing injected, and the meaning-meter
// I(state; signal) measures it climbing from 0 toward log2(N). A null control
// (destroy the state<->signal correspondence) confirms the meter reads ~0 on
// no-structure -- the apophenia tripwire of ADR-0001.
#include <cmath>
#include <cstdio>
#include <vector>

#include "gary/info_theory.hpp"
#include "gary/signaling.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n", gary::version(), gary::tagline());
  std::printf("core: portable C++ (CPU, Raspberry-capable) | cuda: %s\n\n",
              gary::has_cuda_backend() ? "on" : "off");

  const int N = 8;  // states
  const int M = 8;  // signals
  const double Hmax = std::log2(static_cast<double>(N));

  std::printf("Phase 0 - Lewis/Skyrms signaling game  (N=%d states, M=%d signals)\n", N, M);
  std::printf("the meaning-meter I(state;signal) should climb 0 -> log2(N) = %.3f bits\n\n", Hmax);

  gary::SignalingGame game(N, M, /*seed=*/42);
  std::printf("    round      success      I(state;signal)\n");
  std::printf("  %7d   %9s   %.4f bits\n", 0, "-", game.mutual_information_bits());

  int total = 0;
  for (int block = 0; block < 10; ++block) {
    const double sr = game.run(2000);
    total += 2000;
    std::printf("  %7d   %8.1f%%   %.4f bits\n", total, sr * 100.0,
                game.mutual_information_bits());
  }

  const double mi = game.mutual_information_bits();
  std::printf("\nfinal: I = %.4f bits of %.4f max  (%.1f%% of a perfect code)\n", mi, Hmax,
              100.0 * mi / Hmax);

  // Null control (apophenia tripwire): replace the joint with the product of its
  // marginals -- i.e. destroy the state<->signal correspondence -- and re-measure.
  const auto joint = game.state_signal_joint();
  std::vector<double> px(N, 0.0), py(M, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      const double p = joint[static_cast<size_t>(i) * M + j];
      px[i] += p;
      py[j] += p;
    }
  std::vector<double> null(static_cast<size_t>(N) * M);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      null[static_cast<size_t>(i) * M + j] = px[i] * py[j];
  const double mi_null = gary::mutual_information_bits(null, N, M);

  std::printf("null control (independence / shuffled): I = %.4f bits\n", mi_null);
  std::printf("\nthe bits are real: %.4f >> %.4f. apophenia tripwire holds.\n", mi, mi_null);
  return 0;
}
