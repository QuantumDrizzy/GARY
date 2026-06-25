# ADR-0001: The Spine — GARY's Thesis, Stack, and Build Sequence

**Status:** Proposed
**Date:** 2026-06-25
**Deciders:** Antonio (QuantumDrizzy)

## Context

The subject matter GARY lives in — contact, meaning, pattern, "the past", the alien
question — is saturated with *projected* meaning: myth, teleology, out-of-place artifacts,
adulterated data, confirmation bias, and wishful contact narratives that say more about us
than about anything out there.

Underneath all of it sits one honest, well-posed question:

> **How do you distinguish the structure that is really there from the meaning you project onto it?**

This is not a question to *believe* an answer to. It is a question to *measure*. The
deliverable of GARY is the firewall itself, turned into instruments: a way to separate
real signal from apophenia, real order from teleology, emergent meaning from imposed
meaning.

Constraints (from the owner):
- **One repository**, unified by a single spine — not a junk drawer of "math + language +
  vibration + patterns".
- **GPU-native** (CUDA / Blackwell) *where scale earns it* — not as decoration.
- **Engineering honesty is non-negotiable:** no inflation, no fake acceleration, nulls and
  falsifiers before claims.

## Decision

GARY is built on **one measurable spine** — separating real structure from projected
meaning — operationalised through three lenses that are facets of the same question:

1. **Information theory.** Mutual information `I(state; signal)` as the *meaning-meter*
   (0 bits = no shared meaning, up to `log2(N)` = perfect code). Real correlation vs.
   apophenia via shuffled / null controls. The degraded historical record modelled as a
   lossy channel (information decays; the past is low-SNR — thermodynamics of records).

2. **Emergent communication / game theory.** Lewis (1969) / Skyrms (2010) signaling games:
   meaning that *emerges from interaction* with nothing injected. The **Dark Forest** is
   the same game in the catastrophic-payoff regime (revealing your state is lethal →
   silence as equilibrium → a game-theoretic answer to Fermi). The *lingua cosmica*
   problem: can a third agent decode a code it never shared?

3. **Dynamical systems.** Deterministic chaos that is *structured* — strange attractors,
   fractal order, sensitive dependence != randomness. The "order from chaos" question and
   the honest claim that **chaos is not disorder**.

**Two substrate levels (the quantum / information axis).** The spine is realised at *both*
levels — the owner's "separate quantum from mathematics" intuition done right: not two
realities, but one structure realised on two substrates.
- *Classical:* Shannon mutual information, classical signaling games, classical chaos.
- *Quantum:* quantum mutual information / von Neumann entropy, **quantum games**
  (Eisert–Wilkens–Lewenstein, 1999), quantum chaos. **Simulated** quantum information on the
  GPU (state-vector / tensor-network) — *not* a QPU, and explicitly *not* "quantum
  mysticism": quantum-information + quantum-game-theory only.

Quantum is **first-class in the design, earned in the build**: sequenced *after* the classical
spine proves out — same discipline as "GPU where it earns it", never forced into Phase 0.

**AI is the learner, not a garnish.** The agents that evolve the codes climb a ladder:
Skyrms reinforcement → neural deep-RL agents (emergent communication) → (later) agents on the
quantum substrate. AI is *who plays the games*. (Honest footnote: an LLM is a hyper-human
mirror, not an alien — the truly alien mind shares no basis with us, which is exactly what
these agents model.)

**Epistemic hygiene — three tiers (the firewall, in the open).** Every statement carries a
tier: **science** (measured, with nulls + falsifiers; claims allowed) / **open** (plausible,
unresolved — panspermia, Fermi; hypotheses only) / **inspiration** (myth, fiction, wonder;
fuel only, never cited as evidence). Default stance: **mechanism, not mission** (mediocrity
principle; no universe-made-for-us).

**Stack — polyglot by domain.** Python for orchestration, ML, and information-theoretic
measurement; CUDA / C++ kernels for the scale tiers (populations, batched Monte Carlo,
neural agents, GPU quantum-information simulation). Not all-Python (the scale tiers need real
kernels); not Rust (this is
numerical / ML research — Python + CUDA is the honest fit; Rust is reserved for sovereign
boundaries, which GARY does not have).

## Options Considered

### Option A: One repo, spine-unified — CHOSEN
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Coherence | High — the spine *is* the thesis |
| Sprawl risk | Contained by the spine + tier tags |
| Owner fit | Matches "one repo" + "unify by the spine, not accumulation" |

**Pros:** one defensible thesis; the firewall is structural; scales cleanly to GPU.
**Cons:** requires discipline to keep every facet tied to the spine.

### Option B: Several small repos (one per facet)
| Dimension | Assessment |
|-----------|------------|
| Modularity | High |
| Coherence | Low — fragments the single question |

**Pros:** modular. **Cons:** the whole point is that these facets are *one* question;
splitting them destroys the thesis. **Rejected** (also against the explicit "un solo repo").

### Option C: Manifesto / philosophy repo (no engine)
**Pros:** fast to write. **Cons:** it would be an essay, not a science. The firewall must
be *code that measures*, not prose that asserts. **Rejected.**

## Trade-off Analysis

- **Wonder vs. rigor.** The wonder is kept as *fuel* (inspiration tier) but only the
  measured spine is "science". The topic's pull toward woo is countered by mandatory tier
  tags and falsifiers — not by amputating the wonder.
- **GPU honesty.** Forcing the GPU onto the toy seed would be inflation. The GPU is placed
  exactly where scale demands it (populations, neural agents, batched runs) and the seed is
  openly CPU. "GPU-accelerated" is a claim GARY must *earn*, like any other.
- **Breadth vs. focus.** "Vibration / patterns" (e.g. cymatics) is admitted only if it
  earns a place on the spine (as a signal medium / a structured-vs-random testbed);
  otherwise it stays in the inspiration tier. The spine is the gatekeeper.

## Consequences

- **Easier:** a coherent, defensible repo; the firewall is built in; a clean path to real
  GPU work; results are tier-tagged and reproducible.
- **Harder:** the discipline of keeping every facet on-spine and resisting sprawl; writing
  nulls / falsifiers *before* the satisfying "meaning emerged" plots.
- **To revisit:** whether the neural / population tier needs hand-written CUDA or torch
  suffices; whether the dynamical-systems facet stays in GARY or bridges to an existing
  substrate; whether GARY is ever made public (default: local).

## Action Items — Build Sequence

Verification-first: for each phase, the null / falsifier is built *before* the result.

1. **Phase 0 — Seed (CPU, ~1 day).** Lewis / Skyrms signaling game: N states, M signals,
   reinforcement learning. Measure `I(state; signal)` rising 0 → `log2(N)`; success rate;
   classify *signaling system* vs *pooling equilibrium*.
   *Falsifier:* MI must rise on real emergence **and stay ~0 on shuffled / null controls**.
   If projected patterns score as meaning, the meter is broken — fix it before anything else.
2. **Phase 1 — Populations (GPU).** Thousands of agent pairs co-evolving, batched on
   CUDA / Blackwell. Evolutionary dynamics of signaling; MI as an order parameter across a
   population. *Here Blackwell earns it.*
3. **Phase 2 — Dark Forest (game theory, hard).** Costly signals, deception, catastrophic
   payoffs. Does silence become the equilibrium? Tie to Fermi (open tier).
4. **Phase 3 — Lingua cosmica / decipherment.** Can a third agent decode the emergent code?
   The inverse problem (signal <-> meaning).
5. **Phase 4 — Neural agents.** Deep-RL emergent communication; test for *compositionality*
   (does grammar emerge?).
6. **Phase 5 — Dynamical-systems facet.** Real structural order vs projected meaning,
   measured on a chaotic substrate (strange attractors).

Cross-cutting: every result tagged science / open / inspiration; nulls + provenance + a
held-out check before any emergence claim.

## Falsifiers (the honest core)

GARY fails honestly if:
- the meaning-meter scores **projected / shuffled patterns as meaning** (apophenia leak), or
- an "emergence" result does not survive its null control, or
- a "GPU-accelerated" claim is not actually faster than the honest CPU baseline at the
  scale claimed.

Each of these is a tripwire to be wired *before* the corresponding feature.

## Name

**G.A.R.Y.** — a backronym. The project is a *system*, not a person (explicitly **not** Gary
McKinnon). Working expansion (pending the owner's final pick):

> **G**rounded · **A**pophenia-**R**esistant · **Y**ardstick
> — an honest instrument that refuses to read patterns that are not there.

The name carries the thesis (the firewall as a meter). Quantum and AI live in the
architecture, not in the four letters.
