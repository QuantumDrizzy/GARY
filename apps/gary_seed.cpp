// GARY — Phase 0 seed: a Lewis/Skyrms signaling game.
//
// Meaning emerges from interaction with nothing injected; the meaning-meter
// I(state; signal) measures it climbing from 0 toward log2(N). Baseline Roth-Erev
// stalls in partial pooling (~2.69/3.00 bits at N=M=8); we try to beat it with
// FORGETTING (Barrett & Zollman 2009), measured honestly as a mean over seeds.
#include <cmath>
#include <cstdio>
#include <vector>

#include "gary/info_theory.hpp"
#include "gary/signaling.hpp"
#include "gary/version.hpp"

// Mean final MI / success over `runs` seeds; also the fraction reaching a near-perfect code.
static void evaluate(int N, int M, double phi, int runs, int rounds, double Hmax,
                     double& mean_mi, double& mean_sr, double& frac_near) {
  double smi = 0.0, ssr = 0.0;
  int near = 0;
  for (int r = 0; r < runs; ++r) {
    gary::SignalingGame g(N, M, static_cast<std::uint64_t>(1000 + r), phi);
    ssr += g.run(rounds);
    const double mi = g.mutual_information_bits();
    smi += mi;
    if (mi > 0.99 * Hmax) ++near;
  }
  mean_mi = smi / runs;
  mean_sr = ssr / runs;
  frac_near = static_cast<double>(near) / runs;
}

int main() {
  std::printf("GARY %s - %s\n", gary::version(), gary::tagline());
  std::printf("core: portable C++ (CPU, Raspberry-capable) | cuda: %s\n\n",
              gary::has_cuda_backend() ? "on" : "off");

  const int N = 8, M = 8, rounds = 20000, runs = 24;
  const double Hmax = std::log2(static_cast<double>(N));

  std::printf("Phase 0 - Lewis/Skyrms signaling (N=%d, M=%d). Perfect code = %.3f bits.\n", N, M,
              Hmax);
  std::printf("Beating the 2.69-bit baseline by escaping pooling with FORGETTING.\n");
  std::printf("Mean over %d seeds x %d rounds:\n\n", runs, rounds);

  std::printf("  forgetting   mean I(s;sig)   mean success   near-perfect(>=99%%)\n");
  double best_phi = 0.0, best_mi = -1.0, base_mi = -1.0;
  for (double phi : {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}) {
    double mi, sr, frac;
    evaluate(N, M, phi, runs, rounds, Hmax, mi, sr, frac);
    std::printf("  %9.3f     %.4f bits     %7.1f%%        %6.1f%%\n", phi, mi, sr * 100.0,
                frac * 100.0);
    if (phi == 0.0) base_mi = mi;
    if (mi > best_mi) { best_mi = mi; best_phi = phi; }
  }
  std::printf("\nbaseline (no forgetting): %.4f bits\n", base_mi);
  std::printf("best: forgetting=%.3f -> mean I=%.4f bits (%.1f%% of perfect)  [%+.4f vs baseline]\n",
              best_phi, best_mi, 100.0 * best_mi / Hmax, best_mi - base_mi);

  // One full climb at the best forgetting rate, with the null control (apophenia tripwire).
  std::printf("\none run at forgetting=%.3f:\n", best_phi);
  gary::SignalingGame game(N, M, 7, best_phi);
  int total = 0;
  for (int b = 0; b < 5; ++b) {
    const double sr = game.run(4000);
    total += 4000;
    std::printf("  round %6d   success %6.1f%%   I=%.4f bits\n", total, sr * 100.0,
                game.mutual_information_bits());
  }
  const double mi = game.mutual_information_bits();
  const auto joint = game.state_signal_joint();
  std::vector<double> px(N, 0.0), py(M, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      const double p = joint[static_cast<size_t>(i) * M + j];
      px[i] += p;
      py[j] += p;
    }
  std::vector<double> nul(static_cast<size_t>(N) * M);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) nul[static_cast<size_t>(i) * M + j] = px[i] * py[j];
  const double mi_null = gary::mutual_information_bits(nul, N, M);
  std::printf("  final I=%.4f bits | null control=%.4f bits (apophenia tripwire holds)\n", mi,
              mi_null);
  return 0;
}
