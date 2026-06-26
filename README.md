# GARY — a Decypher System

> The honest physics of meaning and order — telling the pattern that *is there*
> from the pattern *we put there*.

A GPU-native research bench that turns one question into instruments:

> **How do you distinguish the structure that is really there from the meaning you project onto it?**

Not a question to *believe* an answer to — a question to **measure**. The deliverable is the
firewall itself: real signal vs. apophenia, real order vs. teleology, emergent meaning vs.
imposed meaning. Everything below is measured, with a null/falsifier wired *before* the result.

**Status:** a working bench — 6 phases, all four lenses of [ADR-0001](docs/ADR-0001-the-spine.md)
plus a quantum substrate. C/C++ first (portable CPU core, optional CUDA backend, Raspberry-capable),
with the cryptography boundary in Rust.

## The four lenses (+ the quantum substrate)

| Lens | What it measures |
|------|------------------|
| **Information theory** | mutual information as the *meaning-meter* (0 → log₂N bits) |
| **Game theory** | meaning *emerging* from interaction (Lewis/Skyrms signaling); the dark forest |
| **Dynamical systems** | deterministic chaos that is *structured*, not random ("chaos is not disorder") |
| **Cryptography & decipherment** | the recoverability of meaning — emergent vs. keyed codes |
| **Quantum substrate** | correlation/coordination beyond any classical code (CHSH, QMI) |

## What's been measured

| Phase | App | Result (honest, reproducible) |
|-------|-----|-------------------------------|
| **0** Emergence | `gary_seed` | a code emerges from nothing → **perfect 3.00/3.00 bits** (forgetting escapes pooling) |
| **1** Scale | `gary_population` | thousands of games batched on Blackwell (sm_120) — **5–20× over 1 CPU thread** |
| **1b** FP32 | `gary_population` | *honest negative:* FP32 gives no real speedup (memory-bound) **and** fakes a perfect MI (flush-to-zero). FP64 stays the reference |
| **2** Dark forest | `gary_darkforest` | signaling under a catastrophic cost C → phase transition **communication → silence** (C\* ≈ 0.5–1.0) |
| **3** Lingua cosmica | `gary_lingua` | a third agent recovers a code's **structure** (MI) from ciphertext, but its **meaning** needs an anchor (behaviour/cribs). *MI ≠ meaning.* |
| **4** Quantum | `gary_quantum` | CHSH: entanglement **0.854 > 0.75** classical bound; QMI: a Bell state carries **2 bits** (double the classical max) |
| **5** Crypto | `crypto/` (Rust) | a weak code is broken by cribs; a **keyed code resists** (chance); **ML-DSA / FIPS-204** signs results (post-quantum provenance) |
| **6** Dynamical | `gary_chaos` | chaos vs. noise: same marginal entropy, but consecutive MI = **2.79 bits (chaos) vs ~0 (noise)**. Structure hiding in apparent randomness |

## Build & run

**C/C++ core + apps (CPU, portable):**
```sh
cmake -S . -B build -DGARY_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release          # core tests
./build/Release/gary_seed                  # and gary_darkforest, gary_lingua, gary_quantum, gary_chaos
```

**Optional CUDA / Blackwell backend** (the population scale tier): `scripts/build_cuda.bat`
(loads vcvars + Ninja; CUDA is off by default so the CPU/Raspberry path always builds).

**Cryptography boundary (Rust):**
```sh
cd crypto && cargo test && cargo run --release
```

## Honest notes (the firewall, applied to ourselves)

- **Verification-first:** every result has a null/falsifier built before it (shuffled controls
  read ~0; classical CHSH never exceeds 0.75; tampered signatures are rejected).
- **Distrust the too-good:** the FP32 "perfect" MI was caught as a precision artifact and
  documented as a [KNOWN_LIMIT], not shipped as a win.
- **GPU/quantum where they earn it:** the CUDA backend is optional; quantum is simulated and
  validated against known theoretical values (Tsirelson 0.8536, QMI 2 bits).
- **The keyed cipher in Phase 5 is a research stand-in, not production crypto.** The PQC piece
  (ML-DSA signing) uses the vetted `fips204` crate.

## Epistemics

Three tiers ([`docs/MANIFESTO.md`](docs/MANIFESTO.md)): **science** (measured) / **open**
(plausible, unresolved) / **inspiration** (fuel only, never cited as evidence). Default stance:
**mechanism, not mission.** A claim may move up a tier only by *measurement*, never by feeling.

Design: [`docs/ADR-0001`](docs/ADR-0001-the-spine.md) (the spine) · [`docs/ADR-0002`](docs/ADR-0002-cpp-first-stack.md) (C/C++-first stack).

## The name

`G.A.R.Y.` — a backronym; a *system*, not a person. **A Decypher System.**

## License

MIT © QuantumDrizzy — see [LICENSE](LICENSE). The bundled `fips204` crate is MIT OR Apache-2.0.
