// GARY core tests -- dependency-free (portable, Pi-friendly). Returns non-zero on failure.
#include <cmath>
#include <cstdio>
#include <vector>

#include "gary/bifurcation.hpp"
#include "gary/chaos.hpp"
#include "gary/dark_forest.hpp"
#include "gary/decipher.hpp"
#include "gary/henon.hpp"
#include "gary/info_theory.hpp"
#include "gary/iterated.hpp"
#include "gary/neural.hpp"
#include "gary/neural_compose.hpp"
#include "gary/quantum.hpp"
#include "gary/quantum_learning.hpp"
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

  // Dark forest: no cost recovers communication; a huge cost forces silence.
  {
    DarkForestGame g0(8, 8, 1.0, 0.0, 1.0, 3, 0.1);
    g0.run(20000);
    const DarkForestResult r0 = g0.measure();
    std::printf("   [dark forest C=0] silence = %.1f%%, MI = %.3f bits\n", r0.silence_rate * 100.0,
                r0.mi_bits);
    CHECK(r0.silence_rate < 0.34, "dark forest C=0 -> communication (low silence)");

    DarkForestGame gh(8, 8, 1.0, 5.0, 1.0, 3, 0.1);
    gh.run(20000);
    const DarkForestResult rh = gh.measure();
    std::printf("   [dark forest C=5] silence = %.1f%%\n", rh.silence_rate * 100.0);
    CHECK(rh.silence_rate > 0.66, "dark forest C=5 -> silence dominates");
  }

  // Lingua cosmica: a behavioural anchor grounds decipherment (accuracy high); ciphertext
  // alone does not ground meaning (accuracy ~ chance) though structure (MI) is recovered.
  {
    const DecipherResult beh = lingua_cosmica(8, 20000, 20000, 0.05, Anchor::Behavior, 0.0, 5);
    std::printf("   [lingua: behavior]   accuracy = %.1f%%, MI = %.3f bits\n",
                beh.accuracy * 100.0, beh.mi_bits);
    CHECK(beh.accuracy > 0.8, "lingua cosmica: behavior anchor deciphers (accuracy > 80%)");

    const DecipherResult cph =
        lingua_cosmica(8, 20000, 20000, 0.05, Anchor::CiphertextOnly, 0.0, 5);
    std::printf("   [lingua: ciphertext] accuracy = %.1f%%, MI = %.3f bits\n",
                cph.accuracy * 100.0, cph.mi_bits);
    CHECK(cph.accuracy < 0.40, "lingua cosmica: ciphertext-only fails to ground meaning");
  }

  // Quantum: entanglement beats the classical CHSH bound; the classical strategy cannot.
  {
    const ChshResult q = chsh_game(500000, 99);
    std::printf("   [CHSH] classical = %.4f, quantum = %.4f (Tsirelson %.4f)\n", q.classical_win,
                q.quantum_win, q.tsirelson);
    CHECK(q.classical_win <= 0.76, "CHSH classical strategy stays at/below 0.75 bound");
    CHECK(q.quantum_win > 0.83, "CHSH quantum (entangled) beats the classical bound");
  }

  // Quantum mutual information: Bell = 2 bits (double the classical max), product = 0 (null).
  {
    double bell[16] = {0};
    bell[0] = 0.5; bell[3] = 0.5; bell[12] = 0.5; bell[15] = 0.5;
    double prod[16] = {0};
    prod[0] = 0.25; prod[5] = 0.25; prod[10] = 0.25; prod[15] = 0.25;
    const QmiResult qb = quantum_mutual_information(bell);
    const QmiResult qp = quantum_mutual_information(prod);
    std::printf("   [QMI] Bell = %.3f bits, product = %.3f bits\n", qb.qmi, qp.qmi);
    CHECK(std::fabs(qb.qmi - 2.0) < 1e-3, "QMI(Bell) = 2 bits (double the classical max)");
    CHECK(std::fabs(qp.qmi - 0.0) < 1e-3, "QMI(product) = 0 (the null)");
  }

  // Dynamical systems: the logistic map is chaotic at r=3.99, periodic at r=3.20; and a
  // chaotic series has high consecutive MI (structure) while IID noise has ~0 (the null).
  {
    CHECK(lyapunov_exponent(3.99, 100000, 2000) > 0.0, "logistic r=3.99 is chaotic (lyapunov>0)");
    CHECK(lyapunov_exponent(3.20, 100000, 2000) < 0.0, "logistic r=3.20 is periodic (lyapunov<0)");
    const ChaosMi m = chaos_vs_random(3.99, 100000, 20, 3);
    std::printf("   [chaos] consecutive MI: chaos=%.3f, random=%.3f bits\n", m.chaos_self_mi,
                m.random_self_mi);
    CHECK(m.chaos_self_mi > 1.0, "chaotic series has high consecutive MI (deterministic structure)");
    CHECK(m.random_self_mi < 0.1, "IID series has ~0 consecutive MI (the null)");
  }

  // Compositionality: a compositional learner makes grammar emerge (high topo-sim) under a
  // bottleneck; a holistic learner does not (near chance). Structure is selected, not automatic.
  {
    const IteratedResult comp = iterated_learning(2, 6, 6, 14, 30, LearnerBias::Compositional, 11);
    const IteratedResult holi = iterated_learning(2, 6, 6, 14, 30, LearnerBias::Holistic, 11);
    std::printf("   [compositionality] compositional = %.3f, holistic = %.3f topo-sim\n",
                comp.final_topo_sim, holi.final_topo_sim);
    CHECK(comp.final_topo_sim > 0.7, "compositional learner: grammar emerges (topo-sim > 0.7)");
    CHECK(holi.final_topo_sim < 0.4, "holistic control: no compositional structure emerges");
  }

  // Order from chaos: the logistic map's period doubles (2, 4, ...) and the Feigenbaum constant
  // emerges from the cascade (true value ~4.669).
  {
    CHECK(attractor_period(3.20, 64, 20000, 1024, 1e-4) == 2, "logistic r=3.20 has period 2");
    CHECK(attractor_period(3.50, 64, 20000, 1024, 1e-4) == 4, "logistic r=3.50 has period 4");
    const BifurcationResult bc = feigenbaum_cascade(2.90, 3.5695, 0.00005);
    std::printf("   [feigenbaum] %zu doublings, delta = %.3f (true 4.669)\n",
                bc.doubling_points.size(), bc.feigenbaum_delta);
    CHECK(bc.feigenbaum_delta > 4.3 && bc.feigenbaum_delta < 5.0,
          "Feigenbaum constant estimated near 4.669 from the cascade");
  }

  // Quantum learning: a quantum-equipped learner beats the classical CHSH bound; a classical
  // learner cannot; neither exceeds the Tsirelson bound (the quantum sim stays physical).
  {
    const LearningChshResult q = learn_chsh(true, 8, 300000, 10, 0.1, 0.01, 7);
    const LearningChshResult c = learn_chsh(false, 8, 300000, 10, 0.1, 0.01, 7);
    std::printf("   [q-learning] quantum learned = %.3f, classical = %.3f\n", q.final_win_rate,
                c.final_win_rate);
    CHECK(q.final_win_rate > 0.78, "quantum learner beats the classical 0.75 bound");
    CHECK(q.final_win_rate <= q.tsirelson + 1e-6, "quantum learner stays at/below Tsirelson");
    CHECK(c.final_win_rate <= 0.76, "classical learner cannot exceed 0.75");
  }

  // Henon strange attractor: chaotic (positive Lyapunov) yet a fractal of non-integer
  // correlation dimension ~1.22 -- structure inside the chaos.
  {
    const HenonResult h = henon_attractor(1.4, 0.3, 3000, 1);
    std::printf("   [henon] lyapunov = %.3f, correlation dimension = %.3f\n", h.lyapunov,
                h.correlation_dimension);
    CHECK(h.lyapunov > 0.0, "Henon map is chaotic (positive Lyapunov)");
    CHECK(h.correlation_dimension > 1.0 && h.correlation_dimension < 1.5,
          "Henon attractor has fractal dimension ~1.22");
  }

  // Neural emergent communication: two from-scratch MLPs trained by REINFORCE grow a shared
  // code (MI and success rise well above chance) -- the spine carries from tables to networks.
  {
    const NeuralCommResult r = neural_emergent_comm(6, 6, 32, 400000, 10, 0.05, 7);
    std::printf("   [neural] success = %.1f%%, MI = %.3f / %.3f bits\n", r.final_success * 100.0,
                r.final_mi, r.max_mi);
    CHECK(r.final_mi > 1.0, "neural agents grow shared information (MI > 1 bit)");
    CHECK(r.final_success > 0.4, "neural receiver decodes well above chance");
  }

  // Neural compositionality: multi-head MLP agents learn to communicate composite meanings
  // (success well above chance); topographic similarity measures how compositional the code is.
  {
    const NeuralComposeResult r = neural_compositionality(2, 5, 5, 64, 400000, 8, 0.05, 7);
    std::printf("   [neural-compose] success = %.1f%%, topo-sim = %.3f\n", r.final_success * 100.0,
                r.final_topo_sim);
    CHECK(r.final_success > 0.4, "neural agents communicate composite meanings (success > 40%)");
    CHECK(r.final_topo_sim > -1.01 && r.final_topo_sim < 1.01, "topographic similarity is in range");
  }

  // Disentangling compositionality: under identical training, a feature-structured input yields
  // a more compositional code than a holistic input -- the architecture (not pressure alone)
  // was driving Phase 12's compositionality.
  {
    const NeuralComposeResult st = neural_compositionality(2, 5, 5, 64, 400000, 8, 0.05, 7, false);
    const NeuralComposeResult ho = neural_compositionality(2, 5, 5, 64, 400000, 8, 0.05, 7, true);
    std::printf("   [disentangle] structured topo-sim = %.3f, holistic = %.3f\n", st.final_topo_sim,
                ho.final_topo_sim);
    CHECK(st.final_topo_sim > ho.final_topo_sim,
          "structured input is more compositional than holistic (architecture helps)");
  }

  // Neural iterated learning: with a holistic input (no architectural bias), a transmission
  // bottleneck recovers compositionality over generations (topo-sim rises from the random start).
  {
    const NeuralIteratedResult r = neural_iterated_learning(2, 5, 5, 32, 12, 30, 100, 20000, 0.05, 7);
    std::printf("   [neural-iterated] initial topo-sim = %.3f, final = %.3f\n", r.initial_topo_sim,
                r.final_topo_sim);
    CHECK(r.final_topo_sim > 0.1,
          "two pressures (bottleneck + expressivity) avoid the degenerate collapse to topo-sim 0");
  }

  if (g_fail)
    std::printf("\n%d test(s) FAILED\n", g_fail);
  else
    std::printf("\nall tests passed\n");
  return g_fail ? 1 : 0;
}
