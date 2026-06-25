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

  // ── Quantum mutual information: the quantum meaning-meter ──────────────────
  std::printf("\n--- quantum mutual information (the meaning-meter, quantum) ---\n");
  double bell[16] = {0};
  bell[0] = 0.5; bell[3] = 0.5; bell[12] = 0.5; bell[15] = 0.5;     // |Phi+><Phi+|
  double classical[16] = {0};
  classical[0] = 0.5; classical[15] = 0.5;                          // 1/2|00><00|+1/2|11><11|
  double product[16] = {0};
  product[0] = 0.25; product[5] = 0.25; product[10] = 0.25; product[15] = 0.25;  // I/4

  struct { const char* name; double* rho; } cases[] = {
      {"Bell (entangled)", bell}, {"classical-correlated", classical}, {"product (independent)", product}};
  std::printf("  state                  S(AB)   S(A)   S(B)    QMI\n");
  for (auto& cs : cases) {
    const gary::QmiResult q = gary::quantum_mutual_information(cs.rho);
    std::printf("  %-21s  %5.2f  %5.2f  %5.2f   %5.2f bits\n", cs.name, q.s_joint, q.s_a, q.s_b,
                q.qmi);
  }
  std::printf("\nthe entangled Bell state carries QMI = 2 bits -- DOUBLE the classical max (1 bit);\n");
  std::printf("the product state = 0 (the null). Entanglement holds correlation no classical code can.\n");
  return 0;
}
