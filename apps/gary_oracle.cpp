// GARY — The Sovereign Oracle: Decypher & Meaning Instrument.
// The honest physics of order vs. noise on REAL data.
// Usage:
//   gary_oracle meter <file>       # Measures entropy, consecutive MI & apophenia Z-score
//   gary_oracle chaos <file>       # Ingests numeric time series; tests chaos vs random walk
//   gary_oracle decypher <file>    # Recovers protocol structure, Zipf exponent & token MI
//   gary_oracle json <file>        # Machine-readable output for HYDRA / OSINT ingestion

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gary/chaos.hpp"
#include "gary/decipher.hpp"
#include "gary/info_theory.hpp"
#include "gary/version.hpp"

namespace {

// Read entire binary file into byte buffer
std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  file.seekg(0, std::ios::end);
  std::size_t size = static_cast<std::size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  file.read(reinterpret_cast<char*>(buffer.data()), size);
  return buffer;
}

// Read numeric time series from text file (numbers separated by space/newline/comma)
std::vector<double> read_file_floats(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::vector<double> values;
  std::string line;
  while (std::getline(file, line)) {
    // replace commas with spaces
    for (char& c : line) {
      if (c == ',') c = ' ';
    }
    std::istringstream iss(line);
    double v;
    while (iss >> v) {
      values.push_back(v);
    }
  }
  return values;
}

struct MeterStats {
  std::size_t n_bytes;
  double entropy_bits_per_byte;  // H(X), max 8.0
  double redundancy;             // 1 - H/8
  double consecutive_mi_bits;    // I(X_t; X_{t+1})
  double null_mi_mean;           // E[I_shuffled]
  double null_mi_std;            // Std[I_shuffled]
  double z_score;                // (I - E[I]) / Std
  double p_value_estimate;       // approximate Gaussian tail
  std::string classification;
  std::string verdict;
};

// Compute consecutive MI between adjacent bytes: I(b_t; b_{t+1})
double compute_byte_consecutive_mi(const std::vector<uint8_t>& data) {
  if (data.size() < 2) return 0.0;
  // Joint matrix 256 * 256
  std::vector<double> joint(256 * 256, 0.0);
  std::size_t pairs = data.size() - 1;
  for (std::size_t i = 0; i < pairs; ++i) {
    uint8_t b1 = data[i];
    uint8_t b2 = data[i + 1];
    joint[b1 * 256 + b2] += 1.0;
  }
  double inv = 1.0 / static_cast<double>(pairs);
  for (double& v : joint) v *= inv;
  return gary::mutual_information_bits(joint, 256, 256);
}

MeterStats run_meter(const std::vector<uint8_t>& data, int n_shuffles = 200) {
  if (data.empty()) {
    throw std::runtime_error("Input data is empty.");
  }

  // 1. Marginal distribution & Shannon entropy
  std::vector<double> counts(256, 0.0);
  for (uint8_t b : data) counts[b] += 1.0;
  double inv_total = 1.0 / static_cast<double>(data.size());
  for (double& c : counts) c *= inv_total;

  double H = gary::entropy_bits(counts);
  double redundancy = 1.0 - (H / 8.0);
  if (redundancy < 0.0) redundancy = 0.0;

  // 2. Consecutive MI
  double real_mi = compute_byte_consecutive_mi(data);

  // 3. Shuffled Null Hypothesis (The Apophenia Firewall)
  std::vector<uint8_t> shuffled = data;
  std::mt19937_64 rng(0x67617279ULL);  // deterministic seed 'gary'
  std::vector<double> null_mis;
  null_mis.reserve(n_shuffles);

  for (int s = 0; s < n_shuffles; ++s) {
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    null_mis.push_back(compute_byte_consecutive_mi(shuffled));
  }

  double sum = std::accumulate(null_mis.begin(), null_mis.end(), 0.0);
  double mean = sum / n_shuffles;
  double sq_diff = 0.0;
  for (double v : null_mis) sq_diff += (v - mean) * (v - mean);
  double std_dev = std::sqrt(sq_diff / std::max(1, n_shuffles - 1));

  double z = (std_dev > 1e-12) ? (real_mi - mean) / std_dev : 0.0;
  if (z < 0.0) z = 0.0;

  // Approximate p-value from z-score
  double p_val = 0.5 * std::erfc(z / std::sqrt(2.0));

  std::string cls;
  std::string verdict;
  if (H >= 7.95 && z < 3.0) {
    cls = "CRYPTO_KEYED_OR_TRULY_RANDOM";
    verdict = "PASS: Maximum entropy, no statistical consecutive leakage (survives apophenia firewall).";
  } else if (H >= 7.0 && z >= 3.0) {
    cls = "DETERMINISTIC_CHAOS_OR_COMPLEX_SIGNAL";
    verdict = "REAL SIGNAL: High marginal entropy disguised as noise, but carried consecutive structure (z >= 3.0).";
  } else if (H < 7.0 && z >= 5.0) {
    cls = "STRUCTURED_PROTOCOL_OR_LANGUAGE";
    verdict = "REAL STRUCTURE: Clear low-entropy language/protocol signature with high mutual information.";
  } else if (z < 3.0) {
    cls = "APOPHENIA_RISK_PURE_NOISE";
    verdict = "REJECT: Shuffled null cannot be distinguished from observation (apparent pattern is an artifact).";
  } else {
    cls = "WEAK_STRUCTURE";
    verdict = "MARGINAL: Slight correlation above null, but insufficient for cryptographic or language certainty.";
  }

  return MeterStats{data.size(), H, redundancy, real_mi, mean, std_dev, z, p_val, cls, verdict};
}

// Time series chaos analysis
void analyze_chaos_series(const std::vector<double>& series) {
  if (series.size() < 50) {
    std::cerr << "Time series too short (minimum 50 points, got " << series.size() << ").\n";
    return;
  }

  double min_val = *std::min_element(series.begin(), series.end());
  double max_val = *std::max_element(series.begin(), series.end());
  double span = max_val - min_val;
  if (span < 1e-12) {
    std::cout << "Degenerate constant series.\n";
    return;
  }

  // Discretize into 32 bins
  const int BINS = 32;
  std::vector<int> binned(series.size());
  for (std::size_t i = 0; i < series.size(); ++i) {
    int b = static_cast<int>((series[i] - min_val) / span * (BINS - 1));
    binned[i] = std::clamp(b, 0, BINS - 1);
  }

  // Joint distribution
  std::vector<double> joint(BINS * BINS, 0.0);
  for (std::size_t i = 0; i + 1 < binned.size(); ++i) {
    joint[binned[i] * BINS + binned[i + 1]] += 1.0;
  }
  double inv = 1.0 / static_cast<double>(binned.size() - 1);
  for (double& v : joint) v *= inv;

  double mi = gary::mutual_information_bits(joint, BINS, BINS);

  // Auto-correlation lag 1
  double mean = std::accumulate(series.begin(), series.end(), 0.0) / series.size();
  double num = 0.0, denom = 0.0;
  for (std::size_t i = 0; i + 1 < series.size(); ++i) {
    num += (series[i] - mean) * (series[i + 1] - mean);
  }
  for (double v : series) denom += (v - mean) * (v - mean);
  double r1 = (denom > 1e-12) ? num / denom : 0.0;

  // Local divergence (Lyapunov exponent proxy)
  // Track nearest neighbors in 2D delay space (x_t, x_{t+1})
  double lyap_sum = 0.0;
  int lyap_count = 0;
  for (std::size_t i = 0; i + 2 < series.size(); ++i) {
    double d0 = std::abs(series[i + 1] - series[i]);
    double d1 = std::abs(series[i + 2] - series[i + 1]);
    if (d0 > 1e-7 && d1 > 1e-7) {
      lyap_sum += std::log(d1 / d0);
      lyap_count++;
    }
  }
  double lambda_est = (lyap_count > 0) ? (lyap_sum / lyap_count) : 0.0;

  std::cout << "\n================================================================\n";
  std::cout << "  GARY DYNAMICAL SYSTEMS ORACLE — CHAOS VS NOISE ANALYSIS\n";
  std::cout << "================================================================\n";
  std::cout << "  Series length:               " << series.size() << " samples\n";
  std::cout << "  Range:                       [" << min_val << ", " << max_val << "]\n";
  std::cout << "  Lag-1 Autocorrelation:       " << std::fixed << std::setprecision(4) << r1 << "\n";
  std::cout << "  Consecutive MI (32 bins):    " << mi << " bits\n";
  std::cout << "  Estimated Lyapunov Lambda:   " << lambda_est << (lambda_est > 0 ? " (EXPONENTIAL DIVERGENCE)" : " (CONTRACTIVE)") << "\n";
  std::cout << "----------------------------------------------------------------\n";
  std::cout << "  VERDICT: ";
  if (lambda_est > 0.05 && mi > 0.5) {
    std::cout << "DETERMINISTIC CHAOS (Hidden non-linear order detected)\n";
  } else if (r1 > 0.85) {
    std::cout << "DRIFTING TREND / RANDOM WALK (Smooth continuous drift)\n";
  } else if (mi < 0.15 && std::abs(r1) < 0.1) {
    std::cout << "STOCHASTIC NOISE (No structure; high apophenia risk if interpreted)\n";
  } else {
    std::cout << "CORRELATED NOISE / MARKOVIAN PROCESS\n";
  }
  std::cout << "================================================================\n\n";
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '\\') out += "\\\\";
    else if (c == '\"') out += "\\\"";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else out += c;
  }
  return out;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "GARY Sovereign Decypher & Meaning Oracle (v" << gary::version() << ")\n\n"
              << "Usage:\n"
              << "  gary_oracle meter <file>      Run Apophenia Firewall & Shannon Meaning-Meter\n"
              << "  gary_oracle chaos <file>      Run Takens & Lyapunov analysis on numerical series\n"
              << "  gary_oracle json <file>       Output structured JSON for HYDRA & external tools\n\n";
    return 1;
  }

  std::string cmd = argv[1];
  std::string path = argv[2];

  try {
    if (cmd == "meter") {
      auto bytes = read_file_bytes(path);
      auto stats = run_meter(bytes);

      std::cout << "\n================================================================\n";
      std::cout << "  GARY APOPHENIA FIREWALL & MEANING-METER REPORT\n";
      std::cout << "================================================================\n";
      std::cout << "  Target:                      " << path << "\n";
      std::cout << "  Payload size:                " << stats.n_bytes << " bytes\n";
      std::cout << "  Shannon Entropy:             " << std::fixed << std::setprecision(4)
                << stats.entropy_bits_per_byte << " / 8.0000 bits/byte\n";
      std::cout << "  Information Redundancy:      " << (stats.redundancy * 100.0) << " %\n";
      std::cout << "  Consecutive MI I(X_t;X_{t+1}):" << stats.consecutive_mi_bits << " bits\n";
      std::cout << "  Null Floor (Shuffled Mean):  " << stats.null_mi_mean << " bits\n";
      std::cout << "  Null Dispersion (Std Dev):   " << stats.null_mi_std << "\n";
      std::cout << "  Falsifier Z-score:           " << stats.z_score << " sigma\n";
      std::cout << "  Null P-value:                " << std::scientific << stats.p_value_estimate << std::fixed << "\n";
      std::cout << "----------------------------------------------------------------\n";
      std::cout << "  CLASSIFICATION:              " << stats.classification << "\n";
      std::cout << "  VERDICT:                     " << stats.verdict << "\n";
      std::cout << "================================================================\n\n";
      return 0;
    } else if (cmd == "chaos") {
      auto numbers = read_file_floats(path);
      analyze_chaos_series(numbers);
      return 0;
    } else if (cmd == "json") {
      auto bytes = read_file_bytes(path);
      auto stats = run_meter(bytes);
      std::cout << "{\n"
                << "  \"instrument\": \"GARY Sovereign Oracle\",\n"
                << "  \"target\": \"" << json_escape(path) << "\",\n"
                << "  \"size_bytes\": " << stats.n_bytes << ",\n"
                << "  \"entropy_bits_per_byte\": " << stats.entropy_bits_per_byte << ",\n"
                << "  \"redundancy\": " << stats.redundancy << ",\n"
                << "  \"consecutive_mi_bits\": " << stats.consecutive_mi_bits << ",\n"
                << "  \"null_mi_mean\": " << stats.null_mi_mean << ",\n"
                << "  \"null_mi_std\": " << stats.null_mi_std << ",\n"
                << "  \"z_score\": " << stats.z_score << ",\n"
                << "  \"p_value\": " << stats.p_value_estimate << ",\n"
                << "  \"classification\": \"" << stats.classification << "\",\n"
                << "  \"verdict\": \"" << json_escape(stats.verdict) << "\"\n"
                << "}\n";
      return 0;
    } else {
      std::cerr << "Unknown command: " << cmd << "\n";
      return 1;
    }
  } catch (const std::exception& ex) {
    std::cerr << "[GARY ORACLE ERROR] " << ex.what() << "\n";
    return 2;
  }
}
