# ADR-0002: C/C++-First Stack — Portable Core, Optional CUDA/Blackwell Backend, Raspberry-Capable

**Status:** Accepted
**Date:** 2026-06-25
**Deciders:** Antonio (QuantumDrizzy)
**Supersedes:** ADR-0001 §Stack (the Python-first lean)

## Context

GARY is a bare-metal systems project. Its heavy work — population-scale signaling games,
batched Monte Carlo, mutual information over large ensembles, dynamical-systems integration,
and cryptanalysis — is simulation + numerics + cryptography, all of which are C/C++ / CUDA
shaped, not Python-shaped. The owner's mandate is explicit: **C/C++ is the focus (100%)**;
Rust and Python only where they genuinely fit. The system should also run on a **Raspberry
Pi** (edge) where possible.

## Decision

- **C/C++ (C++17, headroom to C++20) is the primary language** — the *portable compute core*.
  CPU; builds and runs on x86 **and** ARM (Raspberry Pi). **No mandatory CUDA dependency in
  the core** — CUDA must never leak into core headers.
- **CUDA + tensor cores + Blackwell (sm_120)** = an **optional accelerated backend** for the
  scale tiers (populations of thousands, batched Monte Carlo, neural agents). Compiled in
  where available; the core runs fully without it.
- **Rust** = the **cryptography / PQC boundary** only (owner standard; ARM-portable, Pi-safe).
- **Python** = glue / plotting / one-off reference checks only ("donde toque") — never the core.
- **Build:** CMake (MSVC on Windows, gcc/clang on ARM/Pi), optional CUDA via
  `find_package(CUDAToolkit)`, optional tests.

This realizes the owner's tiering directly: **Pi = edge tier (portable C/C++ core),
Blackwell = compute tier (CUDA backend) — one codebase, two tiers.**

## Options Considered

### A: C/C++-first portable core + optional CUDA + Rust crypto + Python glue — CHOSEN
**Pros:** bare-metal control; portability Pi → Blackwell; no Python in hot loops; matches the
owner's identity and mandate. **Cons:** more dev time; manual memory management; slower
prototyping.

### B: Python-first with C++/CUDA kernels (ADR-0001's original lean)
**Pros:** fast prototyping. **Cons:** against the owner mandate; the heavy work is genuinely
systems work; Python is awkward as a Pi-resident core. **Rejected.**

### C: Rust-first
**Pros:** memory safety. **Cons:** owner wants C/C++ focus; the CUDA ecosystem is C/C++-native;
Rust is reserved for the crypto boundary. **Rejected.**

## Trade-off Analysis

C/C++ costs prototyping speed and manual memory safety vs. Python, and buys bare-metal
control, true portability (one core from Pi to Blackwell), and zero interpreter overhead in
the simulation/measurement loops. We accept the cost because performance, portability, and
the owner's focus *are* the point of the repo — not a convenience to optimise away later.

## Consequences

- **Easier:** a single portable core; a clean core ↔ CUDA-backend boundary; native Pi runs;
  honest GPU placement (the backend is visibly optional).
- **Harder:** CMake / toolchain complexity (MSVC + CUDA on Windows; gcc/clang on ARM); manual
  memory discipline; the Phase 0 seed is now C++, not a throwaway Python script.
- **Portability is a falsifiable claim:** the core must actually compile **and** run on ARM
  with CUDA absent — to be verified on real Raspberry hardware (owner's hardware era), not
  assumed.

## Action Items

1. **CMake skeleton:** portable `gary_core` (CPU) + `gary_seed` app; `GARY_BUILD_CUDA` option
   (default OFF → Pi/CPU path); `GARY_BUILD_TESTS` option. Must build with CUDA absent.
2. **Phase 0 — signaling-game seed in C++** (CPU, Pi-capable): N states / M signals /
   reinforcement; the mutual-information meter; the shuffled null control (apophenia tripwire).
3. **Phase 1 — CUDA / Blackwell backend** (batched populations), behind the optional flag.
4. **Crypto / PQC boundary in Rust** when that pillar is built.
