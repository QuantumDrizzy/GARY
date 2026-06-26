// GARY — Phase 10: the Henon strange attractor. Chaos living on a low-dimensional fractal.
#include <cstdio>

#include "gary/henon.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 10 - the Henon strange attractor (2D)\n");
  std::printf("map: x' = 1 - a*x^2 + y,  y' = b*x   with a=1.4, b=0.3\n\n");

  const gary::HenonResult h = gary::henon_attractor(1.4, 0.3, 3000, 1);

  std::printf("  largest Lyapunov exponent : %+.3f   (> 0 => chaotic)\n", h.lyapunov);
  std::printf("  correlation dimension D2  : %.3f    (literature ~1.22)\n", h.correlation_dimension);

  std::printf(
      "\na strange attractor: the dynamics are chaotic (positive Lyapunov), yet they live on a\n"
      "FRACTAL of non-integer dimension ~1.22 -- not a point, not a line, not the plane. The\n"
      "chaos is not space-filling noise; it is structure of a precise, fractional dimension.\n"
      "Chaos is not disorder -- it has a measurable shape.\n");
  return 0;
}
