#include "gary/iterated.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace gary {
namespace {

int int_pow(int base, int exp) {
  int r = 1;
  for (int i = 0; i < exp; ++i) r *= base;
  return r;
}

std::vector<int> decode_meaning(int m, int n_features, int n_values) {
  std::vector<int> f(n_features);
  for (int p = 0; p < n_features; ++p) {
    f[p] = m % n_values;
    m /= n_values;
  }
  return f;
}

int hamming(const std::vector<int>& a, const std::vector<int>& b) {
  int d = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) ++d;
  return d;
}

// Pearson correlation between meaning-distance and signal-distance over all meaning pairs.
double topographic_similarity(const std::vector<std::vector<int>>& lang, int n_features,
                              int n_values) {
  const int M = static_cast<int>(lang.size());
  std::vector<double> dm, ds;
  for (int i = 0; i < M; ++i) {
    const std::vector<int> fi = decode_meaning(i, n_features, n_values);
    for (int j = i + 1; j < M; ++j) {
      const std::vector<int> fj = decode_meaning(j, n_features, n_values);
      dm.push_back(hamming(fi, fj));
      ds.push_back(hamming(lang[i], lang[j]));
    }
  }
  const int n = static_cast<int>(dm.size());
  if (n == 0) return 0.0;
  double mx = 0.0, my = 0.0;
  for (int k = 0; k < n; ++k) { mx += dm[k]; my += ds[k]; }
  mx /= n;
  my /= n;
  double sxy = 0.0, sxx = 0.0, syy = 0.0;
  for (int k = 0; k < n; ++k) {
    const double ax = dm[k] - mx, ay = ds[k] - my;
    sxy += ax * ay;
    sxx += ax * ax;
    syy += ay * ay;
  }
  if (sxx < 1e-12 || syy < 1e-12) return 0.0;
  return sxy / std::sqrt(sxx * syy);
}

}  // namespace

IteratedResult iterated_learning(int n_features, int n_values, int n_symbols, int bottleneck,
                                 int generations, LearnerBias bias, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const int M = int_pow(n_values, n_features);
  std::uniform_int_distribution<int> sym(0, n_symbols - 1);
  std::uniform_int_distribution<int> any_meaning(0, M - 1);

  // Generation 0: a random (holistic) language.
  std::vector<std::vector<int>> lang(M, std::vector<int>(n_features));
  for (int m = 0; m < M; ++m)
    for (int p = 0; p < n_features; ++p) lang[m][p] = sym(rng);

  IteratedResult res;
  res.topo_sim.push_back(topographic_similarity(lang, n_features, n_values));

  for (int g = 0; g < generations; ++g) {
    // Bottleneck: which meanings the next generation gets to observe (sampled with replacement).
    std::vector<char> seen(M, 0);
    for (int b = 0; b < bottleneck; ++b) seen[any_meaning(rng)] = 1;

    std::vector<std::vector<int>> next(M, std::vector<int>(n_features));

    if (bias == LearnerBias::Compositional) {
      // Per position p (aligned to feature p): the most common symbol for each feature value.
      std::vector<std::vector<std::vector<int>>> cnt(
          n_features, std::vector<std::vector<int>>(n_values, std::vector<int>(n_symbols, 0)));
      for (int m = 0; m < M; ++m) {
        if (!seen[m]) continue;
        const std::vector<int> f = decode_meaning(m, n_features, n_values);
        for (int p = 0; p < n_features; ++p) cnt[p][f[p]][lang[m][p]]++;
      }
      std::vector<std::vector<int>> smap(n_features, std::vector<int>(n_values));
      for (int p = 0; p < n_features; ++p)
        for (int v = 0; v < n_values; ++v) {
          int best = -1, best_count = 0;
          for (int s = 0; s < n_symbols; ++s)
            if (cnt[p][v][s] > best_count) { best_count = cnt[p][v][s]; best = s; }
          smap[p][v] = (best >= 0) ? best : (v % n_symbols);  // unseen value -> a default
        }
      for (int m = 0; m < M; ++m) {
        const std::vector<int> f = decode_meaning(m, n_features, n_values);
        for (int p = 0; p < n_features; ++p) next[m][p] = smap[p][f[p]];
      }
    } else {  // Holistic: memorise seen, randomise unseen — no generalising structure.
      for (int m = 0; m < M; ++m) {
        if (seen[m]) {
          next[m] = lang[m];
        } else {
          for (int p = 0; p < n_features; ++p) next[m][p] = sym(rng);
        }
      }
    }

    lang.swap(next);
    res.topo_sim.push_back(topographic_similarity(lang, n_features, n_values));
  }

  res.final_topo_sim = res.topo_sim.back();

  // Expressivity: fraction of meanings that map to a distinct signal.
  std::vector<std::vector<int>> sigs = lang;
  std::sort(sigs.begin(), sigs.end());
  const int distinct = static_cast<int>(std::unique(sigs.begin(), sigs.end()) - sigs.begin());
  res.final_expressivity = static_cast<double>(distinct) / M;

  return res;
}

}  // namespace gary
