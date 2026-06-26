// GARY — Phase 8: order from chaos. The period-doubling cascade of the logistic map and the
// universal Feigenbaum constant, measured from the cascade itself.
#include <cstdio>
#include <initializer_list>
#include <string>

#include "gary/bifurcation.hpp"
#include "gary/chaos.hpp"
#include "gary/version.hpp"

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 8 - order from chaos: the period-doubling cascade + Feigenbaum's constant\n\n");

  std::printf("period of the logistic attractor across r (with the chaos meter):\n");
  for (double r : {2.90, 3.20, 3.50, 3.55, 3.566, 3.60, 3.83, 3.99}) {
    const int p = gary::attractor_period(r, 64, 20000, 1024, 1e-4);
    const double lam = gary::lyapunov_exponent(r, 100000, 5000);
    const std::string period = (p > 0) ? std::to_string(p) : std::string("chaos");
    std::printf("  r=%.3f   period %-7s  lyapunov %+.3f\n", r, period.c_str(), lam);
  }
  std::printf("(r=3.83 -> period 3: order re-emerging from inside the chaos)\n\n");

  const gary::BifurcationResult bc = gary::feigenbaum_cascade(2.90, 3.5695, 0.00005);
  std::printf("period-doubling cascade found (%zu doublings):\n", bc.doubling_points.size());
  for (size_t i = 0; i < bc.doubling_points.size(); ++i)
    std::printf("  period -> %2d   at r = %.5f\n", bc.periods[i], bc.doubling_points[i]);

  std::printf("\nFeigenbaum constant from the cascade: delta = %.4f   (true value 4.66920)\n",
              bc.feigenbaum_delta);

  std::printf(
      "\norder is not the opposite of chaos: the cascade interleaves them, and one universal\n"
      "constant governs the route in. Structure all the way down -- chaos is not disorder.\n");
  return 0;
}
