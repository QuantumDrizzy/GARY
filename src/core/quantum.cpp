#include "gary/quantum.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <random>

namespace gary {
namespace {

using cd = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;

// Apply Ry(theta) to `qubit` (0 = LSB) of a 2-qubit state psi[4], index = 2*b1 + b0.
void apply_ry(std::array<cd, 4>& psi, int qubit, double theta) {
  const double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
  for (int i = 0; i < 4; ++i) {
    if (((i >> qubit) & 1) == 0) {
      const int j = i | (1 << qubit);
      const cd a = psi[i], b = psi[j];
      psi[i] = c * a - s * b;
      psi[j] = s * a + c * b;
    }
  }
}

}  // namespace

ChshResult chsh_game(int shots, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::uniform_int_distribution<int> bit(0, 1);

  // Optimal CHSH measurement angles (as Ry rotation angles): correlation = cos(aA - aB).
  const double aA[2] = {0.0, kPi / 2.0};
  const double aB[2] = {kPi / 4.0, -kPi / 4.0};

  long qwins = 0, cwins = 0;
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

  for (int i = 0; i < shots; ++i) {
    const int x = bit(rng), y = bit(rng);

    // Quantum: shared Bell state |Phi+> = (|00> + |11>)/sqrt(2), local measurement rotations.
    std::array<cd, 4> psi = {cd(inv_sqrt2, 0), cd(0, 0), cd(0, 0), cd(inv_sqrt2, 0)};
    apply_ry(psi, 0, aA[x]);  // qubit 0 = Alice
    apply_ry(psi, 1, aB[y]);  // qubit 1 = Bob

    double r = u01(rng), cum = 0.0;
    int outcome = 3;
    for (int k = 0; k < 4; ++k) {
      cum += std::norm(psi[k]);
      if (r <= cum) { outcome = k; break; }
    }
    const int a = outcome & 1, b = (outcome >> 1) & 1;
    if (((a ^ b) == (x & y))) ++qwins;

    // Classical best strategy: always output 0,0 -> a XOR b = 0 -> win iff x AND y == 0.
    if (0 == (x & y)) ++cwins;
  }

  ChshResult res;
  res.classical_win = static_cast<double>(cwins) / shots;
  res.quantum_win = static_cast<double>(qwins) / shots;
  res.classical_bound = 0.75;
  const double c = std::cos(kPi / 8.0);
  res.tsirelson = c * c;
  return res;
}

}  // namespace gary
