// GARY — Phase 4: quantum substrate. Does entanglement beat the classical coordination
// bound? The CHSH game on a 2-qubit state vector. Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>

namespace gary {

struct ChshResult {
  double classical_win;    // best classical strategy's measured win rate
  double quantum_win;      // entangled quantum strategy's measured win rate
  double classical_bound;  // proven classical maximum (0.75)
  double tsirelson;        // quantum maximum cos^2(pi/8) ~ 0.8536
};

// Simulate the CHSH game: players get inputs x,y in {0,1}, output a,b in {0,1}, win iff
// (a XOR b) == (x AND y). A classical strategy can never exceed 0.75. An entangled pair,
// simulated here on a 2-qubit state vector with Born-rule measurement, approaches the
// Tsirelson bound cos^2(pi/8). Entanglement = a measurable coordination resource that no
// classical shared code can match.
ChshResult chsh_game(int shots, std::uint64_t seed);

struct QmiResult {
  double s_joint;  // S(rho_AB) von Neumann entropy, bits
  double s_a;      // S(rho_A)
  double s_b;      // S(rho_B)
  double qmi;      // I(A:B) = S(A) + S(B) - S(AB)
};

// Quantum mutual information I(A:B), in bits, of a 2-qubit (4x4, row-major, real-symmetric)
// density matrix. A maximally entangled Bell state gives QMI = 2 bits -- double the classical
// maximum (1 bit); a classically correlated state gives 1 bit; a product state gives 0.
// The quantum meaning-meter: entanglement carries correlation no classical state can hold.
QmiResult quantum_mutual_information(const double rho[16]);

}  // namespace gary
