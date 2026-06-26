#include "gary/neural_compose.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace gary {
namespace {

// A 1-hidden-layer MLP with a general (dense) input and MULTIPLE softmax heads.
// in -> hid (tanh) -> n_heads * head_size logits (each head softmaxed independently).
struct MultiHeadMLP {
  int in = 0, hid = 0, n_heads = 0, head = 0, out = 0;
  std::vector<double> W1, b1, W2, b2;
  std::vector<double> h, logits, probs;  // forward buffers

  void init(int in_, int hid_, int n_heads_, int head_size_, std::mt19937_64& rng) {
    in = in_; hid = hid_; n_heads = n_heads_; head = head_size_; out = n_heads_ * head_size_;
    W1.assign(static_cast<size_t>(hid) * in, 0.0);
    b1.assign(hid, 0.0);
    W2.assign(static_cast<size_t>(out) * hid, 0.0);
    b2.assign(out, 0.0);
    h.assign(hid, 0.0); logits.assign(out, 0.0); probs.assign(out, 0.0);
    std::normal_distribution<double> n1(0.0, 1.0 / std::sqrt(static_cast<double>(in)));
    std::normal_distribution<double> n2(0.0, 1.0 / std::sqrt(static_cast<double>(hid)));
    for (double& w : W1) w = n1(rng);
    for (double& w : W2) w = n2(rng);
  }

  void forward(const std::vector<double>& x) {
    for (int k = 0; k < hid; ++k) {
      double s = b1[k];
      for (int i = 0; i < in; ++i) s += W1[static_cast<size_t>(k) * in + i] * x[i];
      h[k] = std::tanh(s);
    }
    for (int o = 0; o < out; ++o) {
      double s = b2[o];
      for (int k = 0; k < hid; ++k) s += W2[static_cast<size_t>(o) * hid + k] * h[k];
      logits[o] = s;
    }
    for (int hd = 0; hd < n_heads; ++hd) {  // softmax each head block
      const int base = hd * head;
      double maxl = -1e300;
      for (int j = 0; j < head; ++j) maxl = std::max(maxl, logits[base + j]);
      double sum = 0.0;
      for (int j = 0; j < head; ++j) { probs[base + j] = std::exp(logits[base + j] - maxl); sum += probs[base + j]; }
      for (int j = 0; j < head; ++j) probs[base + j] /= sum;
    }
  }

  void sample(std::mt19937_64& rng, std::vector<int>& out_choice) const {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    out_choice.resize(n_heads);
    for (int hd = 0; hd < n_heads; ++hd) {
      const int base = hd * head;
      double r = u(rng), acc = 0.0;
      int c = head - 1;
      for (int j = 0; j < head; ++j) { acc += probs[base + j]; if (r <= acc) { c = j; break; } }
      out_choice[hd] = c;
    }
  }

  int argmax_head(int hd) const {
    const int base = hd * head;
    int best = 0;
    double bv = probs[base];
    for (int j = 1; j < head; ++j) if (probs[base + j] > bv) { bv = probs[base + j]; best = j; }
    return best;
  }

  // REINFORCE ascent: chosen[hd] per head, advantage A.
  void reinforce(const std::vector<double>& x, const std::vector<int>& chosen, double A, double lr) {
    std::vector<double> g_logits(out, 0.0);
    for (int hd = 0; hd < n_heads; ++hd) {
      const int base = hd * head;
      for (int j = 0; j < head; ++j)
        g_logits[base + j] = A * (((j == chosen[hd]) ? 1.0 : 0.0) - probs[base + j]);
    }
    std::vector<double> g_h(hid, 0.0);
    for (int k = 0; k < hid; ++k) {
      double s = 0.0;
      for (int o = 0; o < out; ++o) s += W2[static_cast<size_t>(o) * hid + k] * g_logits[o];
      g_h[k] = s * (1.0 - h[k] * h[k]);
    }
    for (int o = 0; o < out; ++o) {
      for (int k = 0; k < hid; ++k) W2[static_cast<size_t>(o) * hid + k] += lr * g_logits[o] * h[k];
      b2[o] += lr * g_logits[o];
    }
    for (int k = 0; k < hid; ++k) {
      const double gk = lr * g_h[k];
      for (int i = 0; i < in; ++i) W1[static_cast<size_t>(k) * in + i] += gk * x[i];
      b1[k] += gk;
    }
  }
};

std::vector<int> decode(int m, int n_features, int base) {
  std::vector<int> f(n_features);
  for (int p = 0; p < n_features; ++p) { f[p] = m % base; m /= base; }
  return f;
}

int hamming(const std::vector<int>& a, const std::vector<int>& b) {
  int d = 0;
  for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++d;
  return d;
}

// One-hot concatenation: n_features blocks of size `base`, set the value index in each block.
void encode_onehot(const std::vector<int>& feats, int base, std::vector<double>& out) {
  std::fill(out.begin(), out.end(), 0.0);
  for (size_t p = 0; p < feats.size(); ++p) out[p * base + feats[p]] = 1.0;
}

// Topographic similarity of the greedy code (meaning -> message), over all meaning pairs.
double topo_sim(const std::vector<std::vector<int>>& msg, int n_features, int n_values) {
  const int M = static_cast<int>(msg.size());
  std::vector<double> dm, ds;
  for (int i = 0; i < M; ++i) {
    const std::vector<int> fi = decode(i, n_features, n_values);
    for (int j = i + 1; j < M; ++j) {
      const std::vector<int> fj = decode(j, n_features, n_values);
      dm.push_back(hamming(fi, fj));
      ds.push_back(hamming(msg[i], msg[j]));
    }
  }
  const int n = static_cast<int>(dm.size());
  if (n == 0) return 0.0;
  double mx = 0, my = 0;
  for (int k = 0; k < n; ++k) { mx += dm[k]; my += ds[k]; }
  mx /= n; my /= n;
  double sxy = 0, sxx = 0, syy = 0;
  for (int k = 0; k < n; ++k) { const double ax = dm[k] - mx, ay = ds[k] - my; sxy += ax * ay; sxx += ax * ax; syy += ay * ay; }
  if (sxx < 1e-12 || syy < 1e-12) return 0.0;
  return sxy / std::sqrt(sxx * syy);
}

}  // namespace

NeuralComposeResult neural_compositionality(int n_features, int n_values, int n_symbols,
                                            int hidden, int rounds, int checkpoints,
                                            double learning_rate, std::uint64_t seed,
                                            bool holistic_input) {
  std::mt19937_64 rng(seed);
  const int M = static_cast<int>(std::pow(n_values, n_features) + 0.5);
  std::uniform_int_distribution<int> meaning(0, M - 1);

  const int sender_in = holistic_input ? M : n_features * n_values;
  MultiHeadMLP sender, receiver;
  sender.init(sender_in, hidden, n_features, n_symbols, rng);                // meaning -> message
  receiver.init(n_features * n_symbols, hidden, n_features, n_values, rng);  // message -> features

  std::vector<double> sin(sender_in), rin(n_features * n_symbols);
  std::vector<int> msg, pred;

  // The sender's view of a meaning: structured (per-feature one-hots) or holistic (atomic one-hot).
  auto set_sender_input = [&](int meaning_idx, const std::vector<int>& f) {
    std::fill(sin.begin(), sin.end(), 0.0);
    if (holistic_input) {
      sin[meaning_idx] = 1.0;
    } else {
      for (int p = 0; p < n_features; ++p) sin[static_cast<size_t>(p) * n_values + f[p]] = 1.0;
    }
  };

  NeuralComposeResult res;
  int block = rounds / std::max(1, checkpoints);
  if (block < 1) block = 1;
  double baseline = 0.0;
  int full_block = 0, n_block = 0;

  for (int t = 0; t < rounds; ++t) {
    const int m = meaning(rng);
    const std::vector<int> feats = decode(m, n_features, n_values);

    set_sender_input(m, feats);
    sender.forward(sin);
    sender.sample(rng, msg);

    encode_onehot(msg, n_symbols, rin);
    receiver.forward(rin);
    receiver.sample(rng, pred);

    int correct = 0;
    for (int p = 0; p < n_features; ++p) correct += (pred[p] == feats[p]) ? 1 : 0;
    const double reward = static_cast<double>(correct) / n_features;  // dense per-feature reward
    const bool full = (correct == n_features);
    const double adv = reward - baseline;

    sender.reinforce(sin, msg, adv, learning_rate);
    receiver.reinforce(rin, pred, adv, learning_rate);
    baseline += 0.001 * (reward - baseline);

    full_block += full ? 1 : 0;
    ++n_block;
    if (n_block >= block) {
      // Greedy code: meaning -> argmax message; measure topographic similarity.
      std::vector<std::vector<int>> code(M, std::vector<int>(n_features));
      for (int mm = 0; mm < M; ++mm) {
        const std::vector<int> f = decode(mm, n_features, n_values);
        set_sender_input(mm, f);
        sender.forward(sin);
        for (int p = 0; p < n_features; ++p) code[mm][p] = sender.argmax_head(p);
      }
      res.topo_sim.push_back(topo_sim(code, n_features, n_values));
      res.success.push_back(static_cast<double>(full_block) / n_block);
      full_block = 0; n_block = 0;
    }
  }

  res.final_success = res.success.empty() ? 0.0 : res.success.back();
  res.final_topo_sim = res.topo_sim.empty() ? 0.0 : res.topo_sim.back();
  return res;
}

NeuralIteratedResult neural_iterated_learning(int n_features, int n_values, int n_symbols,
                                              int hidden, int bottleneck, int generations,
                                              int epochs, int interact_rounds,
                                              double learning_rate, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const int M = static_cast<int>(std::pow(n_values, n_features) + 0.5);
  std::uniform_int_distribution<int> meaning(0, M - 1);

  std::vector<double> in(M, 0.0);  // holistic input: one-hot over meanings (no feature structure)
  std::vector<double> rin(static_cast<size_t>(n_features) * n_symbols);
  auto onehot = [&](int m) { std::fill(in.begin(), in.end(), 0.0); in[m] = 1.0; };

  auto greedy_language = [&](MultiHeadMLP& mlp) {
    std::vector<std::vector<int>> lang(M, std::vector<int>(n_features));
    for (int m = 0; m < M; ++m) {
      onehot(m);
      mlp.forward(in);
      for (int p = 0; p < n_features; ++p) lang[m][p] = mlp.argmax_head(p);
    }
    return lang;
  };

  MultiHeadMLP teacher;
  teacher.init(M, hidden, n_features, n_symbols, rng);

  NeuralIteratedResult res;
  res.initial_topo_sim = topo_sim(greedy_language(teacher), n_features, n_values);
  res.topo_sim.push_back(res.initial_topo_sim);

  std::vector<int> msg, pred;
  for (int g = 0; g < generations; ++g) {
    const std::vector<std::vector<int>> teach_lang = greedy_language(teacher);

    std::vector<char> seen(M, 0);
    for (int b = 0; b < bottleneck; ++b) seen[meaning(rng)] = 1;
    std::vector<int> bottleneck_set;
    for (int m = 0; m < M; ++m) if (seen[m]) bottleneck_set.push_back(m);

    MultiHeadMLP student;
    student.init(M, hidden, n_features, n_symbols, rng);

    // (1) LEARNING phase: imitate the teacher's messages on the bottleneck (compressibility).
    for (int e = 0; e < epochs; ++e)
      for (int m : bottleneck_set) {
        onehot(m);
        student.forward(in);
        student.reinforce(in, teach_lang[m], 1.0, learning_rate);
      }

    // (2) INTERACTION phase: the student communicates with a fresh receiver, by REINFORCE
    //     (expressivity pressure — a degenerate language fails the game and is pushed away).
    MultiHeadMLP receiver;
    receiver.init(n_features * n_symbols, hidden, n_features, n_values, rng);
    double baseline = 0.0;
    for (int t = 0; t < interact_rounds; ++t) {
      const int m = meaning(rng);
      const std::vector<int> feats = decode(m, n_features, n_values);
      onehot(m);
      student.forward(in);
      student.sample(rng, msg);
      encode_onehot(msg, n_symbols, rin);
      receiver.forward(rin);
      receiver.sample(rng, pred);
      int correct = 0;
      for (int p = 0; p < n_features; ++p) correct += (pred[p] == feats[p]) ? 1 : 0;
      const double reward = static_cast<double>(correct) / n_features;
      const double adv = reward - baseline;
      student.reinforce(in, msg, adv, learning_rate);
      receiver.reinforce(rin, pred, adv, learning_rate);
      baseline += 0.001 * (reward - baseline);
    }

    teacher = student;  // next generation
    res.topo_sim.push_back(topo_sim(greedy_language(teacher), n_features, n_values));
  }

  res.final_topo_sim = res.topo_sim.back();
  return res;
}

}  // namespace gary
