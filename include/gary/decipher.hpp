// GARY — Phase 3: lingua cosmica. Can a third agent decipher an emergent code?
// Portable core, no CUDA (Pi-capable).
#pragma once

#include <cstdint>

namespace gary {

// What the eavesdropping third agent gets to observe.
enum class Anchor {
  CiphertextOnly,  // signals alone — no grounding to states
  Behavior,        // also sees the receiver's actions (meaning leaks through behavior)
  Crib             // sees the true state for a fraction of rounds (known plaintext)
};

struct DecipherResult {
  double accuracy;  // fraction of test signals decoded to the TRUE state (grounded meaning)
  double mi_bits;   // I(decoded; true state) — STRUCTURE recovered, even without grounding
};

// Train a sender-receiver pair to a code, then a third agent deciphers it under `anchor`.
// crib_fraction applies only to Anchor::Crib.
//
// The honest distinction this exposes: with CiphertextOnly the agent recovers the code's
// STRUCTURE (mi_bits high) but NOT its MEANING (accuracy at chance) — the signature of an
// undeciphered script. A behavioural anchor or a few cribs ground it (accuracy -> high).
DecipherResult lingua_cosmica(int n_sym, int train_rounds, int obs_rounds, double forgetting,
                              Anchor anchor, double crib_fraction, std::uint64_t seed);

}  // namespace gary
