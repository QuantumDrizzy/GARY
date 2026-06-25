# GARY

> The honest physics of meaning and order — telling the pattern that *is there*
> from the pattern *we put there*.

**Status:** Scaffold (architecture-first). Local. No code yet.

## The question

Strip the myth, the teleology, and the wishful contact narratives off the table and one
question remains underneath all of them:

> **How do you distinguish the structure that is really there from the meaning you project onto it?**

That question is not philosophy here — it is something you *measure*. GARY is a GPU-native
research engine that turns the firewall (real signal vs. projected meaning) into
instruments.

## The spine

Three measurable lenses on the one question:

- **Information theory** — mutual information as a *meaning-meter*; real correlation vs.
  apophenia; the degraded past as a lossy channel.
- **Emergent communication / game theory** — meaning that *arises from interaction*
  without being injected (Lewis / Skyrms signaling games); the **Dark Forest** as the
  catastrophic-payoff regime of that same game; can a third party decode a code it never
  shared? (the *lingua cosmica* problem).
- **Dynamical systems** — deterministic chaos that is *structured*, not random (strange
  attractors). "Chaos is not disorder."

The first genuine contact with a mind that shares no basis with us is, formally, the
emergent-communication problem. GARY models it.

## Three epistemic tiers (the firewall, in the open)

Every statement in this repo is tagged:

| Tier | Meaning | Rule |
|------|---------|------|
| **science** | measured, with nulls and falsifiers | claims allowed, must be reproducible |
| **open** | plausible, unresolved (panspermia, Fermi) | hypotheses only, never asserted |
| **inspiration** | myth, fiction, the wonder | fuel only — never cited as evidence |

Default stance: **mechanism, not mission.** No "universe made for us."

## Build sequence

See [`docs/ADR-0001-the-spine.md`](docs/ADR-0001-the-spine.md). The seed is a CPU
signaling game (validate the meaning-meter); the real work is GPU-native (populations,
neural agents, batched Monte Carlo) on CUDA / Blackwell — placed where scale earns it,
never as decoration.

## The name

`GARY` — see [ADR-0001 §Name](docs/ADR-0001-the-spine.md#name).
