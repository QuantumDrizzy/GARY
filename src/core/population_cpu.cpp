#include "gary/population.hpp"

#include <chrono>
#include <cmath>

#include "gary/signaling.hpp"

namespace gary {

PopulationResult run_population_cpu(int n_games, int n_sym, int rounds, double forgetting,
                                    std::uint64_t seed) {
  const double Hmax = std::log2(static_cast<double>(n_sym));
  double sum_mi = 0.0;
  int perfect = 0;

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int g = 0; g < n_games; ++g) {
    SignalingGame game(n_sym, n_sym, seed + static_cast<std::uint64_t>(g), forgetting);
    game.run(rounds);
    const double mi = game.mutual_information_bits();
    sum_mi += mi;
    if (mi >= 0.99 * Hmax) ++perfect;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  PopulationResult res;
  res.n_games = n_games;
  res.rounds = rounds;
  res.mean_mi = sum_mi / n_games;
  res.frac_perfect = static_cast<double>(perfect) / n_games;
  res.ms = ms;
  return res;
}

}  // namespace gary
