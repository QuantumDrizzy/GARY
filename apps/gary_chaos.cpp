// GARY — Phase 6: dynamical systems. The 4th lens: chaos is not disorder.
#include <cstdio>
#include <initializer_list>

#include "gary/chaos.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 6 - dynamical systems: chaos is not disorder\n");
  std::printf("logistic map x -> r*x*(1-x). Lyapunov exponent: >0 chaos, <0 periodic.\n\n");

  std::printf("     r        lyapunov    regime\n");
  for (double r : {2.50, 3.20, 3.50, 3.5699, 3.60, 3.83, 3.99}) {
    const double l = gary::lyapunov_exponent(r, 200000, 2000);
    const char* reg = (l > 0.001) ? "CHAOS" : (l < -0.001 ? "periodic" : "edge");
    std::printf("  %.4f     %+8.4f    %s\n", r, l, reg);
  }
  std::printf("\n(r=3.83 is a period-3 WINDOW inside the chaos: order hiding within disorder)\n");

  std::printf("\n--- chaos vs randomness: both look noisy, only one has structure ---\n");
  const gary::ChaosMi m = gary::chaos_vs_random(3.99, 200000, 20, 7);
  std::printf("                      marginal H     consecutive I(x_n+1; x_n)\n");
  std::printf("  chaotic (r=3.99)     %.3f bits        %.4f bits\n", m.chaos_marginal_h,
              m.chaos_self_mi);
  std::printf("  IID random           %.3f bits        %.4f bits\n", m.random_marginal_h,
              m.random_self_mi);

  std::printf(
      "\nsame apparent randomness (marginal H), opposite structure: the chaotic series is\n"
      "deterministic (high consecutive MI), the random one is not (~0). Chaos is not disorder --\n"
      "the inverse of apophenia: real structure hiding inside apparent noise.\n");
  return 0;
}
