#include "gary/neural.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "gary/info_theory.hpp"

namespace gary {
namespace {

// A tiny 1-hidden-layer policy MLP with a ONE-HOT input (so the input layer is a lookup of one
// column of W1). in -> hidden (tanh) -> out (softmax). Trained by REINFORCE.
struct PolicyMLP {
  int in = 0, hid = 0, out = 0;
  std::vector<double> W1, b1, W2, b2;     // W1: hid*in, b1: hid, W2: out*hid, b2: out
  std::vector<double> h, logits, probs;   // forward buffers

  void init(int in_, int hid_, int out_, std::mt19937_64& rng) {
    in = in_; hid = hid_; out = out_;
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

  // Forward with one-hot input at index `xi`.
  void forward(int xi) {
    for (int k = 0; k < hid; ++k)
      h[k] = std::tanh(W1[static_cast<size_t>(k) * in + xi] + b1[k]);
    double maxl = -1e300;
    for (int o = 0; o < out; ++o) {
      double s = b2[o];
      for (int k = 0; k < hid; ++k) s += W2[static_cast<size_t>(o) * hid + k] * h[k];
      logits[o] = s;
      if (s > maxl) maxl = s;
    }
    double sum = 0.0;
    for (int o = 0; o < out; ++o) { probs[o] = std::exp(logits[o] - maxl); sum += probs[o]; }
    for (int o = 0; o < out; ++o) probs[o] /= sum;
  }

  int sample(std::mt19937_64& rng) const {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng), acc = 0.0;
    for (int o = 0; o < out; ++o) { acc += probs[o]; if (r <= acc) return o; }
    return out - 1;
  }

  // REINFORCE ascent for a one-hot input `xi`, chosen output `a`, advantage `A`.
  void reinforce(int xi, int a, double A, double lr) {
    std::vector<double> g_logits(out);
    for (int o = 0; o < out; ++o) g_logits[o] = A * (((o == a) ? 1.0 : 0.0) - probs[o]);
    // backprop to hidden (use current W2 before updating)
    std::vector<double> g_h(hid, 0.0);
    for (int k = 0; k < hid; ++k) {
      double s = 0.0;
      for (int o = 0; o < out; ++o) s += W2[static_cast<size_t>(o) * hid + k] * g_logits[o];
      g_h[k] = s * (1.0 - h[k] * h[k]);  // tanh'
    }
    // ascend
    for (int o = 0; o < out; ++o) {
      for (int k = 0; k < hid; ++k) W2[static_cast<size_t>(o) * hid + k] += lr * g_logits[o] * h[k];
      b2[o] += lr * g_logits[o];
    }
    for (int k = 0; k < hid; ++k) {
      W1[static_cast<size_t>(k) * in + xi] += lr * g_h[k];  // x is 1 at xi, 0 elsewhere
      b1[k] += lr * g_h[k];
    }
  }
};

}  // namespace

NeuralCommResult neural_emergent_comm(int n_states, int n_signals, int hidden, int rounds,
                                      int checkpoints, double learning_rate, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> states(0, n_states - 1);

  PolicyMLP sender, receiver;
  sender.init(n_states, hidden, n_signals, rng);
  receiver.init(n_signals, hidden, n_states, rng);

  NeuralCommResult res;
  res.max_mi = std::log2(static_cast<double>(n_states));
  int block = rounds / std::max(1, checkpoints);
  if (block < 1) block = 1;

  double baseline = 0.0;
  int wins_block = 0, n_block = 0;

  for (int t_i = 0; t_i < rounds; ++t_i) {
    const int t = states(rng);
    sender.forward(t);
    const int s = sender.sample(rng);
    receiver.forward(s);
    const int a = receiver.sample(rng);

    const double reward = (a == t) ? 1.0 : 0.0;
    const double adv = reward - baseline;
    sender.reinforce(t, s, adv, learning_rate);
    receiver.reinforce(s, a, adv, learning_rate);
    baseline += 0.001 * (reward - baseline);  // running-mean baseline

    wins_block += (a == t) ? 1 : 0;
    ++n_block;
    if (n_block >= block) {
      // MI from the sender's policy: joint p(t,s) = (1/N) * pi_sender(s | t).
      std::vector<double> joint(static_cast<size_t>(n_states) * n_signals, 0.0);
      const double pt = 1.0 / n_states;
      for (int st = 0; st < n_states; ++st) {
        sender.forward(st);
        for (int sg = 0; sg < n_signals; ++sg)
          joint[static_cast<size_t>(st) * n_signals + sg] = pt * sender.probs[sg];
      }
      res.mi_bits.push_back(mutual_information_bits(joint, n_states, n_signals));
      res.success.push_back(static_cast<double>(wins_block) / n_block);
      wins_block = 0;
      n_block = 0;
    }
  }

  res.final_mi = res.mi_bits.empty() ? 0.0 : res.mi_bits.back();
  res.final_success = res.success.empty() ? 0.0 : res.success.back();
  return res;
}

}  // namespace gary
