// GARY — Phase 2: the dark forest. Sweep the catastrophic cost C and watch whether
// silence becomes the equilibrium. Firewall: C=0 must recover communication; large C
// must force silence. Everything in between is the measured phase transition.
#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "gary/dark_forest.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());

  const int N = 8, M = 8, rounds = 20000, seeds = 16;
  const double B = 1.0, p_detect = 1.0, lr = 0.1;

  std::printf("Phase 2 - the dark forest: signaling under existential risk\n");
  std::printf("benefit B=%.1f, detection p=%.1f, N=M=%d, %d rounds, mean over %d seeds.\n", B,
              p_detect, N, rounds, seeds);
  std::printf("does silence become the equilibrium as the cost C of being detected rises?\n\n");

  std::printf("    cost C    silence    MI(signalers)   success    regime\n");
  for (double C : {0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0}) {
    double sil = 0.0, mi = 0.0, suc = 0.0;
    for (int sd = 0; sd < seeds; ++sd) {
      gary::DarkForestGame g(N, M, B, C, p_detect, static_cast<std::uint64_t>(100 + sd), lr);
      g.run(rounds);
      const gary::DarkForestResult r = g.measure();
      sil += r.silence_rate;
      mi += r.mi_bits;
      suc += r.success_rate;
    }
    sil /= seeds;
    mi /= seeds;
    suc /= seeds;
    const char* regime =
        (sil > 0.66) ? "DARK FOREST (silent)" : (sil < 0.34 ? "communication" : "contested");
    std::printf("  %8.2f   %6.1f%%   %.3f bits      %6.1f%%   %s\n", C, sil * 100.0, mi,
                suc * 100.0, regime);
  }

  std::printf(
      "\nfirewall: C=0 must recover communication (low silence, MI>0); large C must force "
      "silence.\n");
  return 0;
}
