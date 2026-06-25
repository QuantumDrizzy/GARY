// GARY — Phase 4: quantum substrate. The CHSH game: entanglement vs the classical bound.
#include <cstdio>

#include "gary/quantum.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 4 - quantum substrate: does entanglement beat the classical bound?\n");
  std::printf("CHSH game: win iff (a XOR b) == (x AND y). Classical max = 0.750 (proven).\n\n");

  const gary::ChshResult r = gary::chsh_game(2000000, 12345);
  std::printf("  classical best strategy : %.4f   (proven bound %.3f)\n", r.classical_win,
              r.classical_bound);
  std::printf("  quantum (entangled)     : %.4f   (Tsirelson %.4f)\n", r.quantum_win, r.tsirelson);

  std::printf(
      "\nfirewall: no classical strategy can exceed 0.75; the quantum sim reaches ~0.854.\n"
      "entanglement is a real, measurable coordination resource beyond any classical code.\n");
  return 0;
}
