# GARY — a Decypher System

> The honest physics of meaning and order — telling the pattern that *is there*
> from the pattern *we put there*.

You see a face in the clouds. You hear your name in the hum of a fan. You find a pattern in the
data and feel certain it means something. But does it? Is the structure *out there* — or did you
*put it there*?

You cannot settle that by looking harder. You need an instrument.

**GARY builds the instruments.** It is a research bench — a set of small, sharp, reproducible
experiments — that takes the oldest trap in human reasoning (seeing meaning that isn't there) and
turns it into something you can **measure**. Each experiment builds a device: a meaning-meter, a
game where agents invent their own language from nothing, a detector that separates structured
chaos from pure noise, a code you can or cannot break. And each device ships with a control wired
in *first* — a null, a falsifier — so a "pattern" only counts if it survives a test designed to
kill it. The deliverable isn't a result. It's the firewall: real signal vs. apophenia, real order
vs. teleology, meaning that *emerged* vs. meaning we *imposed*.

## What you find when you run it

Run the bench end to end and a story comes out of it — none of it hand-fed, all of it measured:

- Two agents with **no shared language**, playing a simple guessing game, invent a **perfect code
  from nothing** — three full bits of meaning out of thin air. But only once they learn to
  *forget*; without forgetting they get stuck speaking a muddled half-language forever.
- Make signaling **dangerous** — every message can be overheard by something that punishes you for
  sending it — and past a critical cost the agents **go silent**. Communication collapses into
  silence as the stable outcome: a game-theoretic answer to *why the sky might be quiet* (the dark
  forest). Push the same agents from lookup tables to **neural networks** and they go silent too —
  in fact sooner, because gradient-trained agents are more risk-averse.
- Let a third agent **eavesdrop** on two others' private code. It recovers *all* of the code's
  **structure** — the statistics line up perfectly — and still has **no idea what any of it
  means**. That gap, between decoding and understanding, is the whole point: an undeciphered
  script. *Mutual information measures structure, not meaning.* Meaning needs an anchor that
  structure alone never gives you.
- Hand the meaning-meter a **chaotic** signal and a stream of **pure noise**. By their histograms
  they look equally random. But the chaotic one carries **2.8 bits** of structure from one step to
  the next; the noise carries none. Order hides inside what looks like disorder — the exact inverse
  of seeing a face in static.
- Redo the foundational experiments with **neural agents** that learn by gradient descent
  (backprop written from scratch). The core results survive the jump to real networks — and where
  they *break* (a language collapsing under a transmission bottleneck; "compositionality" that
  turns out to lean on the network's wiring rather than the pressure to communicate), those
  failures are **on the record too**, not quietly dropped.

Every one of these is the same move — pulling apart the pattern that's *there* from the pattern we
*project* — made measurable. That's GARY.

## How it's organized — four lenses on one question

The bench attacks the question through four lenses, over a classical **and** a quantum substrate:

| Lens | The question it makes measurable |
|------|----------------------------------|
| **Information theory** | how much structure is *really* shared? — mutual information as the meaning-meter (0 → log₂N bits) |
| **Game theory** | can meaning *emerge* from interaction alone? — signaling games, the dark forest, compositionality |
| **Dynamical systems** | is this disorder, or order we can't see yet? — deterministic chaos with measurable shape |
| **Cryptography & decipherment** | who can recover this meaning, and at what cost? — emergent vs. keyed codes, post-quantum provenance |
| **Quantum substrate** | is there correlation *beyond* any classical code? — CHSH, quantum mutual information |

Two axes cut across all of it: **classical ↔ quantum**, and **tabular ↔ neural** agents. The 15
phases fill in that grid.

## The 15 phases, in full

Every row is reproducible from one app, with its null/falsifier built in. Honest negatives are kept
on the record, not hidden — that's the point of the bench.

**Foundations — the four lenses + quantum & crypto substrates (0–6)**

| Phase | App | Result (honest, reproducible) |
|-------|-----|-------------------------------|
| **0** Emergence | `gary_seed` | a code emerges from nothing → **perfect 3.00/3.00 bits** (forgetting escapes the pooling attractor); shuffled null reads 0 |
| **1** Scale | `gary_population` | thousands of games batched on Blackwell (sm_120) — **5–20× over 1 CPU thread** (FP64, kernel-only) |
| **1b** FP32 | `gary_population` | *honest negative:* FP32 gives no real speedup (memory-bound) **and** fakes a perfect MI (flush-to-zero). FP64 stays the reference |
| **2** Dark forest | `gary_darkforest` | signaling under a catastrophic cost C → phase transition **communication → silence** (C\* ≈ 0.5–1.0) |
| **3** Lingua cosmica | `gary_lingua` | a third agent recovers a code's **structure** (MI) from ciphertext, but its **meaning** needs an anchor. *MI ≠ meaning.* |
| **4** Quantum | `gary_quantum` | CHSH: entanglement **0.854 > 0.75** classical bound; QMI: a Bell state carries **2 bits** (double the classical max) |
| **5** Crypto | `crypto/` (Rust) | a weak code is broken by cribs; a **keyed code resists** (stays at chance); **ML-DSA / FIPS-204** signs results (post-quantum provenance) |
| **6** Dynamical | `gary_chaos` | chaos vs. noise: same marginal entropy, but consecutive MI = **2.79 bits (chaos) vs ~0 (noise)** |

**Going deeper — dynamical detail, cultural transmission, quantum learning (7–10)**

| Phase | App | Result (honest, reproducible) |
|-------|-----|-------------------------------|
| **7** Compositionality | `gary_compose` | iterated learning (Kirby): a bottleneck + compositional bias grows **topographic similarity 0 → 1.000**; holistic control stays ~0 |
| **8** Feigenbaum | `gary_bifurcation` | the period-doubling cascade, with **δ ≈ 4.57** — honestly *near* the true 4.669 (~2%, a finite-cascade estimate, not claimed exact) |
| **9** Quantum learning | `gary_qlearn` | agents *learn* a CHSH strategy: quantum **0.80 > 0.75** classical, but **below Tsirelson 0.854** — exploiting entanglement optimally is hard |
| **10** Hénon | `gary_henon` | a strange attractor: largest Lyapunov **+0.420** (chaotic), correlation dimension **D₂ ≈ 1.19** — chaos has a measurable, non-integer shape |

**Neural agents — the spine with gradient-trained networks (11–15)**

| Phase | App | Result (honest, reproducible) |
|-------|-----|-------------------------------|
| **11** Neural emergence | `gary_neural` | from-scratch MLP sender/receiver (REINFORCE) → success 0 → 83%, MI 0 → **2.25 / 2.585 bits (~87% of a perfect code)**; honest plateau, not 100% |
| **12** Neural compositionality | `gary_ncompose` | multi-head MLP on composite meanings → topo-sim **0.715**; *honest caveat:* the architecture (structured input + per-position heads) helps |
| **13** Disentanglement | `gary_disentangle` | the firewall on our own work: holistic input drops topo-sim **0.715 → 0.376** — the architecture drove most of phase 12, not pressure alone |
| **14** Neural iterated learning | `gary_niterate` | a bottleneck *alone* collapses the language (topo-sim → 0); adding expressivity pressure recovers only **weakly (0.05 → 0.22)** |
| **15** Neural dark forest | `gary_ndarkforest` | the transition **communication → silence emerges with neural agents too** — sharper, at a *lower* C\* (≈0.25–0.3) than tabular |

## Build & run

**C/C++ core + apps (CPU, portable — also builds on a Raspberry Pi):**
```sh
cmake -S . -B build -DGARY_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release          # core tests
./build/Release/gary_seed                  # then any of the 15 apps:
# gary_darkforest gary_lingua gary_quantum gary_chaos gary_compose
# gary_bifurcation gary_qlearn gary_henon gary_neural gary_ncompose
# gary_disentangle gary_niterate gary_ndarkforest
```

**Optional CUDA / Blackwell backend** (the population scale tier): `scripts/build_cuda.bat`
(loads vcvars + Ninja; CUDA is **off by default** so the CPU / Raspberry path always builds).

**Cryptography boundary (Rust):**
```sh
cd crypto && cargo test && cargo run --release
```

## Honest notes (the firewall, applied to ourselves)

- **Verification-first:** every result has a null/falsifier built before it (shuffled controls
  read ~0; classical CHSH never exceeds 0.75; tampered signatures are rejected).
- **Distrust the too-good:** the FP32 "perfect" MI (phase 1b) was caught as a precision artifact
  and documented as a `[KNOWN_LIMIT]`, not shipped as a win.
- **Negatives stay on the record:** the 87% neural plateau (11), δ ≈ 4.57 vs 4.669 (8), quantum
  learning below Tsirelson (9), and the weak neural compositionality (14) are reported as measured.
- **GPU/quantum where they earn it:** the CUDA backend is optional; quantum is *simulated* and
  validated against known theoretical values (Tsirelson 0.8536, QMI 2 bits).
- **The keyed cipher in phase 5 is a research stand-in, not production crypto.** The PQC piece
  (ML-DSA signing) uses the vetted `fips204` crate.

## Epistemics

Every claim is tagged **science** (measured, with nulls + falsifiers) or **open** (plausible but
unresolved, never asserted). Default stance: **mechanism, not narrative** — a claim moves up only
by *measurement*, never by feeling.

Design: [`docs/ADR-0001`](docs/ADR-0001-the-spine.md) (the spine) ·
[`docs/ADR-0002`](docs/ADR-0002-cpp-first-stack.md) (the C/C++-first stack).

## The name

`G.A.R.Y.` — a backronym; a *system*, not a person. **A Decypher System.**

## License

MIT © QuantumDrizzy — see [LICENSE](LICENSE). The bundled `fips204` crate is MIT OR Apache-2.0.
