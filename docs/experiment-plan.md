# JEPA BTC Forecasting — Prioritized Experiment Plan

> **Altitude:** This is the experiment/milestone **roadmap** — the sequence of experiments, each with a quantitative **gate** and a **decision**. It is not line-level code. Each phase below will get its own detailed TDD implementation plan (`docs/superpowers/plans/…`) when we execute it.
>
> **Companions:** [`jepa_btc_forecasting_spec.md`](../jepa_btc_forecasting_spec.md) (the design) and [`docs/literature-review.md`](literature-review.md) (the evidence). Section refs like "§5.4" point to the spec; "LR §3" points to the literature review.

**Goal:** Determine — as cheaply and honestly as possible — whether there is a cost-surviving short-horizon BTC signal, and whether CF-JEPA pretraining adds edge over a strong supervised baseline; ship whichever model wins.

**Prioritization principle:** Order experiments by *(decisiveness ÷ cost)*.
First ask whether one clean, liquid BTC market contains any own-venue signal after
costs. Additional instruments, exchanges, assets, and models are increments that
must beat that fixed baseline; they are not assumed to help. We do not build
CF-JEPA until a supervised baseline has proven there is signal to represent.

---

## Phase & gate map (read this first)

| Phase | Milestone | Decisive gate | Rough cost |
|---|---|---|---|
| **0** | Data integrity + measurement harness | Recon replay-equivalence is byte-identical (no lookahead) | Med (engineering) |
| **0S** | **Bounded single-venue signal screen** | **G0-BN:** Binance BTC-USDT perpetual own-book/trade model has stable OOS lift and positive net performance | Low |
| **1** | **Signal-existence gate** (baseline ladder) | **G1:** LightGBM clears net-of-cost PnL, DSR>0.95, acceptable PBO at some horizon | Low-Med |
| **2** | Validate the risky design decisions (on the cheap baseline) | G2 set: decoupling proven; Binance increment confirmed | Low |
| **3** | Supervised deep baseline | **G3:** supervised-deep ≥ LightGBM | Med |
| **4** | CF-JEPA pretraining + predictivity gate | **G4:** SSL representation passes the predictivity battery (not non-collapsed-but-empty) | High |
| **5** | Heads + the decisive comparison | **G5:** frozen-CF-JEPA beats same-arch-supervised-from-scratch, net-of-cost | Med |
| **6** | Walk-forward robustness + extensions | (only if G5 passes) | Med |

**Hard stops:** Fail **G0-BN** → do not acquire Coinbase/cross-venue,
multi-asset, full-archive, or JEPA data without a separately reviewed pivot.
`PREDICTIVE_NOT_TRADEABLE` may authorize only a documented fair-value/maker
experiment, not automatic expansion. Fail formal **G1** → stop or pivot; fail
**G3** → ship LightGBM; fail **G5** → ship the simpler model.

**Cross-cutting discipline (applies to every modeling screen/gate from Phase 0S on):**
- **Pre-register** labels, CV scheme, no-trade-band rule, and primary metric before touching the held-out OOS month. Every post-hoc tweak is a new "trial" and must enter the DSR trial count `N`. (LR §6)
- **Every predictivity claim is reported as a LIFT over a persistence/identity baseline**, OOS, under purged+embargoed CPCV. (LR §1, §3)
- **Stratify all results by spread/tick and volatility regime** — never report a single pooled number. (LR §4, LOBFrame)
- **Track effective `N`** (cluster correlated trials) for the Deflated Sharpe Ratio. (LR §6)

---

## Phase 0S — Bounded Binance single-venue signal gate

**Binding protocol:** [`docs/superpowers/plans/2026-07-10-staged-signal-acquisition.md`](superpowers/plans/2026-07-10-staged-signal-acquisition.md).

The first signal question uses one exchange, one instrument, and two source
products. It does not require Coinbase, Binance spot, auxiliary derivatives,
other assets, or a deep model.

1. **Source gate (#64):** certify Crypto Lake or CryptoHFTData for Binance
   BTC-USDT perpetual with independent, bounded reconstruction evidence.
2. **Bounded data (#68):** acquire and certify only L2 snapshots/deltas and
   trades for `2025-11-01..2026-01-31`.
3. **Producer/evaluator (#67):** emit explicit `binance_single_venue` manifests,
   Binance labels/costs, and a fixed-OOS workflow.
4. **G0-BN (#69):** use `2025-11-01..2025-12-31` for development/CPCV and
   `2026-01-01..2026-01-31` as untouched OOS. Before any forward read, drop a
   development row unless its complete guarded support ends before
   `2026-01-01T00:00:00Z`; January labels may not consume February.

The fixed baseline ladder is: persistence/no-change, microprice displacement,
penalized linear OFI, then LightGBM. Inputs are own-venue L2 book and trades;
2 s and 10 s are primary horizons and 60 s is a decay/control arm. All variants
share one immutable trial ledger.

Decision outcomes:

- **PASS / tradeable:** stable OOS persistence lift and positive preregistered
  net performance authorize the next incremental-data gate.
- **PREDICTIVE_NOT_TRADEABLE:** stable predictive lift but no taker-cost edge;
  stop broad expansion pending a human-approved fair-value/maker experiment.
- **FAIL:** no stable predictive lift; stop Coinbase, cross-venue, multi-asset,
  full-archive, and JEPA work.
- **INCONCLUSIVE:** data, leakage, cost, PBO, or reproducibility failure; repair
  without replacing or reusing OOS.

After PASS, test increments one at a time on fixed target rows: Binance spot,
then derivatives state, then low-cost Coinbase transfer/cross-venue data through
#65/#34/#47/#48, then other assets. Each added source must beat the preceding
rung OOS net-of-cost beyond preregistered uncertainty. April 2026 remains frozen
for the deferred cross-venue workflow; G0-BN never opens it.

---

## Phase 0 — Data integrity & the measurement harness

**Goal:** Build the substrate that makes every downstream number trustworthy. No forecasting model yet. This maps to spec §12 steps 1–4. **Rationale for going first:** the literature's single highest-severity risk is silent lookahead/leakage (LR §3, §6, §8); a wrong harness makes every later metric a fiction.

### E0.1 — Event-time reconstruction with a replay-equivalence test ⭐
- **Question:** Does our book-at-trade snapshot inject any lookahead, offline or live?
- **Setup:** One reconstruction function shared by train and live. Merge trades + L2 deltas on a single engine-time axis with a deterministic tiebreak; replay diffs in sequence/update-ID order (Binance `U/u/pu` continuity; re-snapshot on gap). Book-at-trade = apply-before-read with **strict `<`** at the trade boundary (never the post-trade book). Live: bounded-out-of-orderness watermark sized from measured stream skew.
- **GATE (E0.1):** A replay harness that feeds live-ordered (deliberately out-of-order) events produces **byte-identical features** to the offline reconstruction. Must be exact.
- **Decision:** Fail → fix before anything else. This test is non-negotiable and is the first thing to write.
- **Deliverable:** `recon/` module + `tests/test_replay_equivalence.py`.
- **Refs:** §5.3; LR §8.

### E0.2 — §4 sample verification (partly done)
- **Question:** Is `origin_time` populated in Crypto Lake Binance `book_delta_v2`? Where are the Coinbase gaps?
- **Setup:** Free-sample checks (already underway — see memory: Crypto Lake Coinbase ~80%/gappy; CoinAPI Flat Files chosen for Coinbase L2/L3). Confirm Binance `origin_time`; if empty, fall back to `received_time` (Tokyo capture makes this a tight proxy).
- **GATE (E0.2):** Reconstruct on exchange time if `origin_time` populated; else documented `received_time` fallback. Coinbase target-window gaps mapped and either backfilled or excluded.
- **Deliverable:** Data-source decision recorded; ingest writes raw Parquet partitioned by (exchange, symbol, day).
- **Refs:** §4; existing `probe_lake.py` / `diag_lake.py`.

### E0.3 — Notional bar sampler + the time-per-bar distribution
- **Question:** What threshold gives ~0.5–2s bars in active regimes, and how wide is the time-per-bar spread?
- **Setup:** Dollar bars off the trade stream, hybrid time cap, **adaptive threshold** (rolling 7–30d avg dollar-volume / target bars-per-day). Flag `emitted_by_time_cap`.
- **GATE (E0.3):** Median active-regime bar ≤ 2s (so the 2s horizon ≈ a few bars). Produce the **log-scale time-per-bar histogram** — this plot is itself the justification for §5.4 and belongs in the writeup.
- **Deliverable:** `bars/` sampler; the histogram artifact.
- **Refs:** §5.1–5.2; LR §5.

### E0.4 — Labels + purged/embargoed CPCV + uniqueness weighting
- **Setup:** Triple-barrier with **vol-scaled** horizontal barriers (EWMA of micro-window returns), vertical barrier = physical horizon, labels off **mid/microprice (never last-trade)**. Purge label **spans** (not rows); **embargo ≥ max(label horizon, longest feature look-back)**; CPCV for a distribution of OOS metrics; sample-uniqueness weighting / sequential bootstrap.
- **GATE (E0.4):** A deliberately-leaky control (random k-fold, no purge) must show inflated CV vs the purged/embargoed pipeline — proves the leakage controls actually bite.
- **Deliverable:** `data/` labels + CV module; leakage-control unit test.
- **Refs:** §8; LR §6.

### E0.5 — Cost model + no-trade-band PnL + DSR/PBO evaluator
- **Setup:** Net PnL charging **2×taker fee + half-spread + slippage**; no-trade band = round-trip cost + margin. Report **gross vs net side-by-side**. Add **MCC** (skill vs monetizability), **Deflated Sharpe Ratio**, **PBO via CSCV**. Honest taker fills (no passive-fill-at-mid assumption).
- **GATE (E0.5):** Evaluator reproduces a known-zero-edge synthetic series as DSR≈0 / PnL≤0 (sanity that it isn't manufacturing edge).
- **Deliverable:** `eval/` harness.
- **Refs:** §10; LR §4, §6.

---

## Phase 1 — The signal-existence gate (project-defining) ⭐

**Goal:** Confirm and expand only a signal already established by bounded G0-BN.
Formal G1 remains a later full-data confirmation with a new coverage-selected
holdout; it is not permission to ignore a failed G0-BN.

### E1.1 — Measure τ (the decay window)
- **Question:** At what horizon does microstructure directional predictability decay to noise?
- **Setup:** Fit OFI/imbalance → future-return predictive R² (and trade-sign/OFI ACF, impact-decay) vs horizon from 0.5s to 120s.
- **GATE/Output:** Empirical decay knee (literature expects ~10–30s; LR §5). **Set the horizon ladder from this** — confirm 2s/10s, add a ~20–30s rung, keep 60s as a decay/control arm.
- **Refs:** §5.4; LR §5.

### E1.2 — Feature engineering (the short list that carries signal)
- **Setup:** Start with Binance-perpetual multi-level/integrated OFI,
  microprice displacement, queue imbalance, spread/tick, signed trade flow/CVD,
  event intensity, VWAP-to-mid, book slope/depth, and intra-bar path. All are
  stationarized. Spot, funding/OI/liquidations, Coinbase, and other assets enter
  only through later explicit feature manifests; absent inputs are never
  zero-filled or inferred.
- **Deliverable:** `bars/` feature module.
- **Refs:** §6; LR §4, §7.

### E1.3 — Baseline ladder rungs 0–2
- **Setup:** Rung 0 = naive martingale / predict-zero. Rung 1 = penalized linear (Ridge/Elastic-Net) + DLinear/NLinear. Rung 2 = **LightGBM on the features**. All under purged/embargoed CPCV, net-of-cost, stratified by regime, with DSR + PBO.
- **GATE G1 (PROJECT GATE):** Does **LightGBM** clear **net-of-cost PnL with DSR > 0.95 and acceptable PBO** at **any** horizon? (And everything beats Rung 0.)
- **Decision:**
  - **PASS →** Proceed to Phase 2. Record the benchmark JEPA must beat.
  - **FAIL →** **STOP or pivot.** Options before abandoning: maker-execution economics, a different horizon near τ, richer features. Do **not** proceed to deep/JEPA — they will not conjure signal that LightGBM-on-OFI can't find. (LR §1)
- **Refs:** §10.1, §12.5; LR §1, §4.

---

## Phase 2 — Validate the risky design decisions (cheap, on the baseline)

**Goal:** Use the cheap LightGBM/feature pipeline to test the spec's *novel or contested* choices **before** investing in JEPA — some outcomes would change the whole design. Run only if G1 passed.

### E2.1 — The decoupling proof ⭐ (validates §5.4, our key contribution)
- **Question:** Does a fixed-bar-count horizon really produce wildly variable-predictability targets vs a fixed-physical-time horizon?
- **Setup:** Train identical LightGBM models on (a) fixed-bar-count target vs (b) fixed-physical-time target. Measure per-regime predictability (R²/MCC) variance across volatility regimes.
- **GATE (E2.1):** The fixed-bar-count target's predictability **variance across regimes is materially larger** than the fixed-physical-time target's; the fixed-physical-time target is at least as good net-of-cost.
- **Decision:** Confirms (or refutes) the spec's load-bearing §5.4 decision. No paper has shown this — it's our contribution either way.
- **Refs:** §5.4; LR §5 (novel/unvalidated).

### E2.2 — Bar clock ablation (Ané-Geman nuance)
- **Setup:** Dollar vs trade-count vs volume bars, compared on (i) return normality/homoscedasticity and (ii) downstream net-of-cost PnL.
- **GATE (E2.2):** Pick the clock that wins on downstream PnL; don't assume dollar wins on Gaussianity (it may not). Document.
- **Refs:** §5.1; LR §5.

### E2.3 — Binance→Coinbase increment ⭐ (conditional extension)
- **Question:** Does Binance signal add edge **over** a Coinbase-own-book model, net of latency + cost?
- **Setup:** Three models: Coinbase-own-book only; Binance-signal only; combined. Lag Binance features by realistic loop latency. **Re-estimate separately pre- and post-2024 spot-ETF.**
- **GATE (E2.3):** Combined beats Coinbase-own-book OOS net-of-cost by more than the bootstrap noise band.
- **Decision:** If Binance adds nothing over own-book after costs, the premise's marginal value is questionable → reconsider scope. (LR §7)
- **Prerequisite:** run only after G0-BN passes and #65 certifies an affordable
  Coinbase label/control contract. The deferred G0-XV pilot uses this shape as
  an acquisition screen; it does not satisfy the full pre/post-ETF claim.
- **Refs:** §1; LR §7.

### E2.4 — Perp-signal value (re-rank §6)
- **Setup:** Ablate funding / OI / liquidations vs OFI / basis as predictors at 2–60s.
- **GATE (E2.4):** Confirm OFI ≫ funding/OI/liquidations at these horizons (LR §7: funding R²≈0 at T+1). Demote weak signals to conditioners.
- **Refs:** §6; LR §7.

### E2.5 — Cross-venue lead-lag measurement
- **Setup:** Hayashi-Yoshida cross-correlation + VECM error-correction on tick/100ms returns, Binance vs Coinbase, current data.
- **GATE/Output:** Quantify the lead (expect sub-second; confirms "not mechanical arb"). Feeds the latency assumption in E2.3.
- **Refs:** §1; LR §7.

---

## Phase 3 — Supervised deep baseline (rung 3)

**Goal:** Does deep nonlinearity add anything over LightGBM at all, end-to-end, *without* SSL? This isolates "deep helps" from "pretraining helps." Run only if G1 passed.

### E3.1 — End-to-end supervised deep head
- **Setup:** A small supervised model on the same features — dilated-conv (the CF-JEPA backbone) and/or PatchTST/TiDE — trained end-to-end, purged CV, net-of-cost. Heads as triple-barrier classification.
- **GATE G3:** Is supervised-deep **at least competitive** with LightGBM net-of-cost?
- **Decision:**
  - **deep ≤ LightGBM →** SSL-frozen-deep almost certainly won't beat it either. **Ship LightGBM; deprioritize JEPA.** (LR §1)
  - **competitive/better →** Proceed to Phase 4.
- **Refs:** §9; LR §1.

---

## Phase 4 — CF-JEPA pretraining + the predictivity gate

**Goal:** Build CF-JEPA *correctly* (matching the real paper, not the spec's transformer substitution) with the anti-collapse fixes, and gate the SSL stage on **predictivity**, not just non-collapse. Run only if G3 passed. This is the expensive phase.

### E4.1 — Implement CF-JEPA (corrected)
- **Setup:** **Dilated depthwise-conv encoder** (revert the §7.2 PatchTST/transformer substitution — CNN backbones dominate TS-SSL; LR §2), EMA target encoder, near-identity **linear** forward predictors over short/mid/long zones, VICReg variance+covariance + multi-scale cross-crop invariance. Add the **physical-Δt predictor conditioning**: Time2Vec(Δt, elapsed_Δ) → FiLM (MetNet-style), randomized-horizon training (biased short), multi-horizon quantile head. (LR §8)
- **Deliverable:** `model/` + `train/` SSL loop.
- **Refs:** §7; LR §2, §8.

### E4.2 — Anneal ablation + the monitoring battery ⭐
- **Question:** Is the representation predictive, or non-collapsed-but-empty?
- **Setup:** Train with **anneal-to-floor (~5–15%) vs anneal-to-zero** prediction weight. Monitor every N steps on held-out OOS: **(A)** embedding variance (hard-collapse < 1e-7), RankMe (≥25k samples), **LiDAR**; **(B)** predicted-vs-actual future-embedding alignment **minus persistence**, linear-probe **lift over identity**, **temporal-shuffle control**, predictor degeneracy (‖W_p−I‖).
- **GATE G4 (SSL PREDICTIVITY GATE):** PASS only if A-metrics non-degenerate **AND** alignment strictly above persistence **AND** probe(z) beats raw-feature and identity baselines OOS **AND** that lift **survives the temporal-shuffle** (real ≫ shuffled).
- **Decision:**
  - **FAIL →** representation is non-collapsed-but-empty. Apply E4.3 fixes; if still failing, fall back (the anneal-to-zero design is the prime suspect — LR §3).
  - **PASS →** Proceed to heads.
- **Refs:** §7.2, §13; LR §3.

### E4.3 — Anti-collapse hardening (apply if G4 fails or warns)
- **Setup:** Add MTS-JEPA-style **soft codebook bottleneck** (anti-collapse + regime anchors); **harden the predictor** (higher LR than encoder, DirectPred-style eigenvalue floor, optional anti-identity penalty); raise the terminal prediction-weight floor.
- **GATE:** Re-run G4.
- **Refs:** LR §3 (MTS-JEPA arXiv:2602.04643; DirectPred arXiv:2102.06810).

---

## Phase 5 — Heads + the decisive comparison

**Goal:** Settle whether pretraining is worth it, and which encoder to route to. Run only if G4 passed.

### E5.1 — Heads per horizon
- **Setup:** Small MLP / linear heads on the pooled representation (last-token / attention-pool for the causal setup), per horizon, triple-barrier classification + return regression. Purged CPCV, net-of-cost.
- **Refs:** §7.3, §9.

### E5.2 — The decisive control ⭐ (frozen-SSL vs supervised-from-scratch)
- **Question:** Does *pretraining* add edge, or just deep learning?
- **Setup:** Compare three, net-of-cost under CPCV: (a) **frozen CF-JEPA + head**, (b) **the same conv architecture trained supervised from scratch**, (c) LightGBM.
- **GATE G5 (JEPA-WORTH-IT GATE):** Does (a) beat (b) by more than the bootstrap noise band?
- **Decision:**
  - **PASS →** Pretraining earns its complexity. Proceed to Phase 6.
  - **FAIL →** The edge (if any) is deep-learning, not pretraining. **Ship the simpler of (b)/(c).** (LR §1)
- **Refs:** §9, §10; LR §1.

### E5.3 — Dual-encoder routing
- **Setup:** For our (sharp-event) heads, compare the **online (higher-rank, discriminative)** encoder vs the **EMA target (smooth, low-rank)** encoder. The paper routes forecasting to the smooth EMA encoder, but sharp-event BTC detection is closer to anomaly/classification, which may favor the discriminative online encoder.
- **GATE/Output:** Route each head to whichever encoder wins OOS net-of-cost.
- **Refs:** §7.3; LR §2, §3.

---

## Phase 6 — Walk-forward robustness & extensions (only if G5 passes)

**Goal:** Confirm the edge is stable out-of-time and add the deferred extensions. Spec §12.8.

- **E6.1 Walk-forward:** Re-pretrain encoder monthly, retrain heads frequently; confirm the G5 result holds on the untouched ~1-month OOS and rolling forward. Confirm DSR survives the full trial count.
- **E6.2 Extensions (each gated on beating the current champion net-of-cost):** MTS-JEPA multi-resolution; frequency-domain auxiliary (FEI/TF-JEPA); imbalance/run bars; richer cross-venue context; SimTS as an SSL alternative arm.
- **Refs:** §7.1, §9, §12.8; LR §2, §3.

---

## Open questions this plan resolves (from LR §10)

1. τ (E1.1) · 2. decoupling proof (E2.1) · 3. bar clock (E2.2) · 4. pretraining value (E5.2) · 5. SSL predictivity (E4.2) · 6. Binance increment (E2.3) · 7. dual-encoder routing (E5.3) · 8. cost wall (E0.5, E1.3).

---

## Self-review (spec coverage)

- §1 premise → G0-BN first; G0-XV/E2.3/E2.5 only after PASS. §4 data →
  E0.1, E0.2, staged Phase 0S. §5 clock/horizon → E0.3, E1.1, E2.1,
  E2.2; recon → E0.1. §6 features → E1.2/E2.4. §7 model → E4.1–E4.3,
  E5.1/E5.3. §8 labels/CV → E0.4. §10 eval → E0.5, G0-BN, and later
  conditional gates. §13 pitfalls → replay/leakage/cost hard stops.
- **Coverage gap noted:** spec §11 (hardware/compute) is not an experiment — it's a provisioning note; no task needed.

---

*Sequence the spend: Phase 0/source gate → bounded Binance-perpetual data →
G0-BN → one-at-a-time source increments → conditional Coinbase/cross-venue
pilot → remaining archive/formal G1 → supervised-deep → only then JEPA. A
failed G0-BN ends broad acquisition before it becomes sunk cost.*
