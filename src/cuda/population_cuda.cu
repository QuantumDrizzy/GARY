// GARY — Phase 1 CUDA backend: one signaling game per thread, batched on Blackwell.
// Mirrors the CPU Roth-Erev + forgetting logic so the population statistics match.
#include "gary/population.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace gary {
namespace {

__global__ void population_kernel(double* sender_w, double* receiver_w, double* mi_out,
                                  int n_games, int n_sym, int rounds, double forgetting,
                                  unsigned long long seed) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_games) return;

  const int n2 = n_sym * n_sym;
  double* sw = sender_w + static_cast<size_t>(g) * n2;
  double* rw = receiver_w + static_cast<size_t>(g) * n2;
  for (int i = 0; i < n2; ++i) { sw[i] = 1.0; rw[i] = 1.0; }

  curandStatePhilox4_32_10_t st;
  curand_init(seed, static_cast<unsigned long long>(g), 0ULL, &st);
  const double keep = 1.0 - forgetting;

  for (int r = 0; r < rounds; ++r) {
    const int t = static_cast<int>(curand(&st) % static_cast<unsigned>(n_sym));

    double* srow = sw + static_cast<size_t>(t) * n_sym;
    double sum = 0.0;
    for (int s = 0; s < n_sym; ++s) sum += srow[s];
    double x = curand_uniform_double(&st) * sum, acc = 0.0;
    int sig = n_sym - 1;
    for (int s = 0; s < n_sym; ++s) { acc += srow[s]; if (x <= acc) { sig = s; break; } }

    double* rrow = rw + static_cast<size_t>(sig) * n_sym;
    double sum2 = 0.0;
    for (int a = 0; a < n_sym; ++a) sum2 += rrow[a];
    double y = curand_uniform_double(&st) * sum2, acc2 = 0.0;
    int act = n_sym - 1;
    for (int a = 0; a < n_sym; ++a) { acc2 += rrow[a]; if (y <= acc2) { act = a; break; } }

    const bool success = (act == t);
    if (forgetting > 0.0) {
      for (int i = 0; i < n2; ++i) { sw[i] *= keep; rw[i] *= keep; }
    }
    if (success) { srow[sig] += 1.0; rrow[act] += 1.0; }
  }

  // I(state; signal) from the sender strategy, uniform state prior.
  const double pt = 1.0 / n_sym;
  double ps[32];
  for (int s = 0; s < n_sym; ++s) ps[s] = 0.0;
  for (int t = 0; t < n_sym; ++t) {
    double* srow = sw + static_cast<size_t>(t) * n_sym;
    double rs = 0.0;
    for (int s = 0; s < n_sym; ++s) rs += srow[s];
    for (int s = 0; s < n_sym; ++s) ps[s] += pt * (srow[s] / rs);
  }
  double mi = 0.0;
  for (int t = 0; t < n_sym; ++t) {
    double* srow = sw + static_cast<size_t>(t) * n_sym;
    double rs = 0.0;
    for (int s = 0; s < n_sym; ++s) rs += srow[s];
    for (int s = 0; s < n_sym; ++s) {
      const double pts = pt * (srow[s] / rs);
      if (pts > 0.0 && ps[s] > 0.0) mi += pts * log2(pts / (pt * ps[s]));
    }
  }
  mi_out[g] = mi;
}

}  // namespace

PopulationResult run_population_cuda(int n_games, int n_sym, int rounds, double forgetting,
                                     std::uint64_t seed) {
  const int n2 = n_sym * n_sym;
  double *d_sw = nullptr, *d_rw = nullptr, *d_mi = nullptr;
  cudaMalloc(&d_sw, static_cast<size_t>(n_games) * n2 * sizeof(double));
  cudaMalloc(&d_rw, static_cast<size_t>(n_games) * n2 * sizeof(double));
  cudaMalloc(&d_mi, static_cast<size_t>(n_games) * sizeof(double));

  const int threads = 128;
  const int blocks = (n_games + threads - 1) / threads;

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  population_kernel<<<blocks, threads>>>(d_sw, d_rw, d_mi, n_games, n_sym, rounds, forgetting,
                                         static_cast<unsigned long long>(seed));
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, e0, e1);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));

  std::vector<double> mi(static_cast<size_t>(n_games));
  cudaMemcpy(mi.data(), d_mi, static_cast<size_t>(n_games) * sizeof(double),
             cudaMemcpyDeviceToHost);

  const double Hmax = std::log2(static_cast<double>(n_sym));
  double sum = 0.0;
  int perfect = 0;
  for (double m : mi) { sum += m; if (m >= 0.99 * Hmax) ++perfect; }

  cudaFree(d_sw);
  cudaFree(d_rw);
  cudaFree(d_mi);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);

  PopulationResult res;
  res.n_games = n_games;
  res.rounds = rounds;
  res.mean_mi = sum / n_games;
  res.frac_perfect = static_cast<double>(perfect) / n_games;
  res.ms = static_cast<double>(ms);
  return res;
}

}  // namespace gary
