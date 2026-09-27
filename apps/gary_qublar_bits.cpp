// GARY meter on a QuBLAR-shaped 6-bit posterior.
//
// QuBLAR emits the bits. Blaze compresses them only on a compressed verdict.
// GARY asks whether the dependence survives a shuffle. GARY is not the motor.
#include <cmath>
#include <cstdio>
#include <vector>

#include "gary/info_theory.hpp"
#include "gary/version.hpp"

int main() {
  // Fixture used across the ecosystem: mass 0.5 on 000000 and 0.5 on 001100.
  // Bits indexed 0..5 (MSB-left). Correlated pair: bit 2 and bit 3.
  constexpr int n_bits = 6;
  std::vector<double> hist(1u << n_bits, 0.0);
  hist[0b000000] = 0.5;
  hist[0b001100] = 0.5;

  // Build I(bit2; bit3) from the histogram (not a hand-typed joint).
  std::vector<double> joint(4, 0.0);
  for (int s = 0; s < (1 << n_bits); ++s) {
    const double w = hist[static_cast<size_t>(s)];
    if (w == 0.0) continue;
    const int b2 = (s >> 3) & 1;  // bit 2 of 000000 / 001100 encoding
    const int b3 = (s >> 2) & 1;  // bit 3
    joint[static_cast<size_t>(b2) * 2 + static_cast<size_t>(b3)] += w;
  }
  const double mi_structured = gary::mutual_information_bits(joint, 2, 2);

  // Shuffled null of the same histogram: product of marginals (phase-0 style).
  std::vector<double> px(2, 0.0), py(2, 0.0);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      const double pij = joint[static_cast<size_t>(i) * 2 + j];
      px[static_cast<size_t>(i)] += pij;
      py[static_cast<size_t>(j)] += pij;
    }
  std::vector<double> nul(4);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      nul[static_cast<size_t>(i) * 2 + j] = px[static_cast<size_t>(i)] * py[static_cast<size_t>(j)];
  const double mi_null = gary::mutual_information_bits(nul, 2, 2);

  // Phase-0 independent-joint tolerance is 1e-9 (test_core.cpp).
  constexpr double null_tol = 1e-9;
  const bool null_ok = std::fabs(mi_null) <= null_tol;
  const bool structured_above = mi_structured > null_tol + 0.5;  // clearly above null

  std::printf("GARY %s — QuBLAR-shaped 6-bit meter\n", gary::version());
  std::printf("fixture: p[000000]=0.5, p[001100]=0.5\n");
  std::printf("structured I(bit2;bit3) = %.17g bits\n", mi_structured);
  std::printf("null      I(product)    = %.17g bits\n", mi_null);
  std::printf("null_tol=%g  null_ok=%s  structured_above_null=%s\n", null_tol,
              null_ok ? "yes" : "no", structured_above ? "yes" : "no");

  if (!null_ok || !structured_above) {
    std::printf("FAIL: GARY meter gate\n");
    return 1;
  }
  std::printf("PASS: structured MI clearly above shuffle null\n");
  return 0;
}
