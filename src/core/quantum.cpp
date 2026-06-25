#include "gary/quantum.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

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

// ── Quantum information: von Neumann entropy + quantum mutual information ──────
namespace {

// Eigenvalues of an n x n real-symmetric matrix (row-major) via cyclic Jacobi rotations.
std::vector<double> jacobi_eigenvalues(std::vector<double> a, int n) {
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
    if (off < 1e-28) break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        const double apq = a[p * n + q];
        if (std::fabs(apq) < 1e-300) continue;
        const double phi = 0.5 * std::atan2(2.0 * apq, a[q * n + q] - a[p * n + p]);
        const double c = std::cos(phi), s = std::sin(phi);
        for (int i = 0; i < n; ++i) {  // columns p,q
          const double aip = a[i * n + p], aiq = a[i * n + q];
          a[i * n + p] = c * aip - s * aiq;
          a[i * n + q] = s * aip + c * aiq;
        }
        for (int i = 0; i < n; ++i) {  // rows p,q
          const double api = a[p * n + i], aqi = a[q * n + i];
          a[p * n + i] = c * api - s * aqi;
          a[q * n + i] = s * api + c * aqi;
        }
      }
  }
  std::vector<double> eig(n);
  for (int i = 0; i < n; ++i) eig[i] = a[i * n + i];
  return eig;
}

double von_neumann_entropy_bits(const std::vector<double>& eig) {
  double h = 0.0;
  for (double l : eig)
    if (l > 1e-12) h -= l * std::log2(l);
  return h;
}

}  // namespace

QmiResult quantum_mutual_information(const double rho[16]) {
  // Reduced density matrices for 2 qubits, index = 2*A + B.
  double rhoA[4] = {0, 0, 0, 0}, rhoB[4] = {0, 0, 0, 0};
  for (int a = 0; a < 2; ++a)
    for (int ap = 0; ap < 2; ++ap) {
      double s = 0.0;
      for (int b = 0; b < 2; ++b) s += rho[(2 * a + b) * 4 + (2 * ap + b)];
      rhoA[a * 2 + ap] = s;
    }
  for (int b = 0; b < 2; ++b)
    for (int bp = 0; bp < 2; ++bp) {
      double s = 0.0;
      for (int a = 0; a < 2; ++a) s += rho[(2 * a + b) * 4 + (2 * a + bp)];
      rhoB[b * 2 + bp] = s;
    }

  QmiResult r;
  r.s_joint = von_neumann_entropy_bits(jacobi_eigenvalues(std::vector<double>(rho, rho + 16), 4));
  r.s_a = von_neumann_entropy_bits(jacobi_eigenvalues(std::vector<double>(rhoA, rhoA + 4), 2));
  r.s_b = von_neumann_entropy_bits(jacobi_eigenvalues(std::vector<double>(rhoB, rhoB + 4), 2));
  r.qmi = r.s_a + r.s_b - r.s_joint;
  return r;
}

}  // namespace gary
