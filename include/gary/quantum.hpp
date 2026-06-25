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

}  // namespace gary
