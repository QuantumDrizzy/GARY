// GARY core tests -- dependency-free (portable, Pi-friendly). Returns non-zero on failure.
#include <cmath>
#include <cstdio>
#include <vector>

#include "gary/info_theory.hpp"
#include "gary/signaling.hpp"

static int g_fail = 0;

#define CHECK(cond, msg)                          \
  do {                                            \
    if (!(cond)) {                                \
      std::printf("FAIL: %s\n", msg);             \
      ++g_fail;                                   \
    } else {                                      \
      std::printf("ok:   %s\n", msg);             \
    }                                             \
  } while (0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((a) - (b)) <= (tol), msg)

int main() {
  using namespace gary;

  // Entropy of a uniform distribution over 4 outcomes is 2 bits.
  CHECK_NEAR(entropy_bits({0.25, 0.25, 0.25, 0.25}), 2.0, 1e-9, "H(uniform-4) = 2 bits");

  // Mutual information of an independent joint is 0.
  {
    std::vector<double> j = {0.25, 0.25, 0.25, 0.25};
    CHECK_NEAR(mutual_information_bits(j, 2, 2), 0.0, 1e-9, "MI(independent) = 0");
  }

  // Mutual information of a perfect 2x2 diagonal code is 1 bit.
  {
    std::vector<double> j = {0.5, 0.0, 0.0, 0.5};
    CHECK_NEAR(mutual_information_bits(j, 2, 2), 1.0, 1e-9, "MI(perfect 2x2) = 1 bit");
  }

  // An untrained signaling game carries ~0 bits (signal independent of state).
  {
    SignalingGame g(8, 8, 1);
    CHECK(g.mutual_information_bits() < 1e-6, "untrained MI ~ 0 (apophenia floor)");
  }

  // A trained game: success and shared meaning both rise well above chance.
  {
    SignalingGame g(4, 4, 7);
    const double sr = g.run(20000);
    const double mi = g.mutual_information_bits();
    std::printf("   [trained 4x4] success = %.1f%%, MI = %.3f / %.3f bits\n", sr * 100.0, mi,
                std::log2(4.0));
    CHECK(mi > 0.5 * std::log2(4.0), "trained MI > half of a perfect code");
    CHECK(sr > 0.5, "trained success > 50%");
  }

  if (g_fail)
    std::printf("\n%d test(s) FAILED\n", g_fail);
  else
    std::printf("\nall tests passed\n");
  return g_fail ? 1 : 0;
}
