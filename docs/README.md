# docs/

Entry point for everything beyond the top-level README.

| document | what's inside |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | Wire-format specification for the three tournament DOs. Tick request/response, snapshot manifest, packed-ternary layout, WebSocket push messages, threat model, defense semantics, reference implementations table, versioning policy. The thing a third-party client implementer reads. |
| [EMPIRICAL_STUDY.md](EMPIRICAL_STUDY.md) | Multi-seed convergence results. Variant comparison (vanilla / sharded / byzantine), attacker-count sweep (0..3 attackers, defense holds at 50% byzantine), sliding-window detection vs the patient attacker (Phase 14). All numbers reproducible from `scripts/empirical-study.mjs`. |
| [PAPER_DRAFT.md](PAPER_DRAFT.md) | arXiv-style 8-section writeup. Abstract → background → protocol → defense → evaluation → bandwidth scaling → related work → reproducibility → limitations. Working draft; not peer-reviewed. |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | What this work does *not* address: sybil resistance, patient-patient attackers, multi-coord federation, persistent DO state, attacker recovery, realistic model scale. Each entry has an open research direction. |
| [FUSEDX_INTEGRATION.md](FUSEDX_INTEGRATION.md) | Earlier integration plan (pre-Phase-1) for wiring `fusedx`'s gpt-gradfree engine as the worker scorer. Superseded in places by what was actually shipped, but documents the design intent. |

## How to read this

If you just want to **try the demo**, the top-level README is the entry point: `npm install && npx wrangler dev` plus the live URL.

If you want to **understand the protocol**, read `PROTOCOL.md`.

If you want to **see the numbers**, read `EMPIRICAL_STUDY.md`.

If you want to **cite this**, draw from `PAPER_DRAFT.md` (but note: working draft, not reviewed).

If you want to **extend it**, start with `OPEN_QUESTIONS.md` — it lists what's missing and approximately how to attack each one.
