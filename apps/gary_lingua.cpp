// GARY — Phase 3: lingua cosmica. A sender-receiver pair invents a code; a third agent
// tries to decipher it. The honest result distinguishes STRUCTURE (mutual information,
// recoverable by eavesdropping) from MEANING (grounded accuracy, needs an anchor).
#include <cmath>
#include <cstdio>

#include "gary/decipher.hpp"
#include "gary/version.hpp"

static void row(const char* label, gary::Anchor anchor, double crib, int seeds) {
  const int N = 8, train = 20000, obs = 20000;
  const double phi = 0.05;
  double acc = 0.0, mi = 0.0;
  for (int sd = 0; sd < seeds; ++sd) {
    const gary::DecipherResult r =
        gary::lingua_cosmica(N, train, obs, phi, anchor, crib, static_cast<std::uint64_t>(50 + sd));
    acc += r.accuracy;
    mi += r.mi_bits;
  }
  acc /= seeds;
  mi /= seeds;
  std::printf("  %-26s  %7.1f%%        %.3f bits\n", label, acc * 100.0, mi);
}

int main() {
  std::printf("GARY %s - %s\n\n", gary::version(), gary::tagline());
  std::printf("Phase 3 - lingua cosmica: can a third agent decipher an emergent code?\n");
  std::printf("N=8, perfect code = %.3f bits. Mean over 16 seeds.\n\n", std::log2(8.0));

  std::printf("  third agent's access        decode acc.     MI(recovered;true)\n");
  row("ciphertext only", gary::Anchor::CiphertextOnly, 0.0, 16);
  row("behavior (sees actions)", gary::Anchor::Behavior, 0.0, 16);
  row("cribs: 2% known plaintext", gary::Anchor::Crib, 0.02, 16);
  row("cribs: 10% known plaintext", gary::Anchor::Crib, 0.10, 16);

  std::printf(
      "\nthe honest lesson: ciphertext-only recovers the code's STRUCTURE (high MI) but NOT its\n"
      "MEANING (accuracy at chance ~12.5%%) -- an undeciphered script. Behaviour or a few cribs\n"
      "ground it (accuracy -> ~100%%). Meaning needs an anchor; structure does not.\n");
  return 0;
}
