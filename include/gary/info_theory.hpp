// GARY — information theory: the meaning-meter. Portable core, no CUDA (Pi-capable).
#pragma once

#include <vector>

namespace gary {

// Shannon entropy H(X) in bits of a probability vector (need not be normalised
// exactly; zero-probability entries contribute 0).
double entropy_bits(const std::vector<double>& p) noexcept;

// Mutual information I(X;Y) in bits from a joint distribution given as a row-major
// [n_x * n_y] matrix that sums to 1. Returns 0 for degenerate sizes.
//
// This is the meaning-meter: 0 bits = the signal carries nothing about the state;
// log2(n_x) bits = the signal pins the state down perfectly (a perfect code).
double mutual_information_bits(const std::vector<double>& joint, int n_x, int n_y) noexcept;

}  // namespace gary
