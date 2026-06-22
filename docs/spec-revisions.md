# Spec Revisions — incorporating the literature review

**What this is.** Proposed replacement/added language for [`jepa_btc_forecasting_spec.md`](../jepa_btc_forecasting_spec.md), derived from [`docs/literature-review.md`](literature-review.md) (LR) and the [`docs/experiment-plan.md`](experiment-plan.md). Each item: a **CHANGE** summary, the **TEXT** to drop in (written in the spec's voice), and **Why + ref**. Items that reverse or modify a deliberate **Why:**-marked spec decision are marked **⚠ REVERSES A SPEC DECISION** and explain the reversal explicitly (per the spec's own governance rule).

**How to apply.** Review; then I apply as edits to the spec. Nothing here is applied yet.

---

## Summary of changes

| § | Change | Flag |
|---|---|---|
| 1 | Add Coinbase→Coinbase baseline + period-dependence caveat | — |
| 2 | Update key-decisions table (backbone, anti-collapse, features, eval, gates) | ⚠ (backbone) |
| 5 | Adaptive threshold; time-cap subpopulation; measure τ; add ~20–30s rung; prove decoupling by ablation | — |
| 6 | Re-rank features: OFI #1, microprice as target, spread regime gate; **downgrade funding/OI/liquidations** | ⚠ (perp signals) |
| 7 | **Revert transformer→dilated conv**; **anneal-to-floor not zero**; new collapse battery; codebook; predictor hardening; dual-encoder routing; physical-Δt FiLM | ⚠⚠ |
| 8 | Vol-scaled barriers; mid/microprice labels; purge spans; embargo≥lookback; CPCV; uniqueness weighting | — |
| 10 | LightGBM = gate **and** champion; MCC; honest taker fills; DSR/PBO; regime stratification; the decisive control | — |
| 12 | Baseline ladder + three hard-stop gates (G1/G3/G5) + SSL predictivity gate (G4) | — |
| 13 | Sharpen pitfalls (collapse diagnostics, anneal, purge-spans, adverse selection, period-dependence) | — |

---

## §1 — Objective & scope

**CHANGE:** Keep the Binance→Coinbase rationale; add a required own-book baseline and a period-dependence caveat.

**TEXT — append to §1, after the "Why Binance→Coinbase…" paragraph:**

> **Required baseline — Coinbase→Coinbase.** Before crediting Binance, model Coinbase's own book+flow predicting Coinbase's own move. The success criterion for the cross-venue thesis is explicit: **does Binance signal add edge *over* the Coinbase-own-book model, net of realistic loop latency and cost?** If it doesn't, the cross-venue premise's marginal value is unproven for our purposes.
>
> **Period-dependence (caveat).** The leading venue has changed over time. Sub-second/seconds price discovery for BTC was led by Binance/Huobi perps c.2021 (Albers et al. 2021), but the post-2024 spot-ETF complex shifted longer-horizon price discovery toward Coinbase/spot (Kia et al. 2026; Jang et al. 2025/26 find Binance leads *short-run*, Coinbase anchors the *long-run* equilibrium). Re-estimate the lead-lag separately pre/post-2024-ETF; do not assume a static leader.

**Why + ref:** LR §7. Albers (arXiv:2108.09750) supports the seconds-horizon premise; Kia 2026 / Jang 2025-26 show the regime shift; the own-book baseline is the only honest test of the increment.

---

## §2 — Key decisions at a glance

**CHANGE:** Update rows to reflect the corrected model and evaluation. **⚠ REVERSES A SPEC DECISION** on the encoder backbone (see §7).

**TEXT — replace/insert these table rows:**

| Decision | Choice | Section |
|---|---|---|
| Encoder backbone | **Dilated depthwise-conv (per the CF-JEPA paper), *not* a transformer/PatchTST** | §7 |
| Anti-collapse | EMA target + stop-grad + VICReg + multi-scale invariance, **plus a soft-codebook bottleneck, and the prediction loss annealed to a non-zero floor (not zero) gated on predictivity** | §7 |
| SSL success gate | **Predictivity battery (LiDAR + predicted-vs-persistence alignment + temporal-shuffle control + probe-lift-over-identity), not embedding-variance alone** | §7, §10 |
| Primary feature | **Multi-level / integrated OFI + microprice** (funding/OI/liquidations demoted to conditioners) | §6 |
| Labels | Triple-barrier (**vol-scaled barriers, labelled off mid/microprice**), purged + embargoed **CPCV**, sample-uniqueness weighting | §8 |
| Eval metric | Fees-included PnL w/ no-trade band, **+ MCC, Deflated Sharpe > 0.95, PBO via CSCV, regime-stratified** | §10 |
| Decisive model test | **Frozen CF-JEPA + head vs the same conv architecture trained supervised from scratch** (isolates the value of *pretraining*) | §10 |
| First milestone | Supervised baseline (LightGBM) — **the gate *and* the likely champion** | §10, §12 |

**Why + ref:** LR §0–§4. The backbone row reverses §2/§7.2's transformer choice (the paper uses dilated conv; see §7 below).

---

## §5 — Sampling / clock

**CHANGE:** Keep dollar bars and the clock/horizon decoupling (correct and load-bearing). Add operational refinements and an ablation to *prove* the novel decoupling.

**TEXT — add to §5.1:**
> **Adaptive threshold.** Set the dollar threshold from a rolling window (trailing ~7–30 days of avg dollar-volume / target bars-per-day), not a static constant — BTC dollar-volume grows over a training window, so a fixed threshold under-samples early and over-samples late.

**TEXT — add to §5.2:**
> **Time-capped bars are a distinct subpopulation.** A bar emitted by the time cap (quiet regime) is lower-information and calendar-clocked — it reintroduces exactly the heteroscedasticity the notional clock removes. Flag it (`emitted_by_time_cap`) and monitor whether its target statistics differ.

**TEXT — add to §5.4:**
> **Status: novel and unvalidated — prove it.** No published work states this clock/horizon decoupling; the closest crypto bars+DL work (Vlahavas et al., *Financial Innovation* 2025) still used a fixed *bar-count* vertical barrier. This is our key methodological contribution, so it must be **proven by ablation**: train identical models on a fixed-bar-count target vs a fixed-physical-time target and show the block-count target's predictability variance across volatility regimes is materially larger.
>
> **Cap the horizon at the *measured* decay window τ.** Estimate τ on our own data (decay of OFI→future-return predictive R² vs horizon; ACF crossing; impact-decay fit) rather than importing a constant. Literature puts crypto microstructure directional τ at ~3–30s, mostly gone by 30–60s. **Add a ~20–30s horizon rung** to the 2s/10s/60s ladder (the 10→60s gap is where decay actually happens; 60s becomes a decay/control arm).

**TEXT — add to §5.1 (ablation note):**
> **Bar-clock ablation.** Dollar bars are chosen for price-level robustness, *not* proven-best Gaussianity — Ané-Geman (2000) find trade-count can normalize returns better, and the dollar-vs-volume normality result is data-dependent. Run a dollar vs trade-count vs volume bar ablation on our own normality/homoscedasticity *and* downstream PnL; pick on downstream PnL.

**Why + ref:** LR §5.

---

## §6 — Feature engineering

**CHANGE:** Re-rank the per-bar vector so order-flow imbalance and microprice are first-class; demote the perp macro signals. **⚠ REVERSES A SPEC DECISION**: the spec lists funding/basis/OI/liquidations as "extra signal"; the evidence is that, *at seconds horizons*, only OFI (and basis) are predictive — funding/OI/liquidations are weak and are demoted to conditioners.

**TEXT — replace the §6 bullet list with this ordering (keep the existing "Why:" notes; add the new ones):**

> - **Order-flow imbalance (OFI) — the #1 feature.** Multi-level / **integrated** OFI computed the Cont way: *signed changes in depth at each level* (event-based, not static depth imbalance), summed/PCA-integrated over the top K levels. Add **cross-venue lagged OFI** (Binance perp & spot → Coinbase). **Why:** OFI is the single strongest short-horizon driver in both equities and crypto (Cont-Kukanov-Stoikov 2014; Cont-Cucuringu-Zhang 2021; top SHAP feature on Binance-perp 1s data, arXiv:2602.00776); lagged cross-asset OFI forecasts short-horizon returns (decaying fast) — the mechanism our cross-venue design exploits.
> - **Microprice — feature *and* candidate target/anchor.** Mid adjusted by spread + queue imbalance (Stoikov 2018); a martingale fair-value and a better short-horizon predictor than mid. Consider forecasting *microprice change* rather than raw mid.
> - **Spread, and spread/tick ratio — also a regime gate.** Predictability concentrates where the spread is tight (LOBFrame); use spread/tick to stratify and gate, not just as a feature.
> - **Book shape:** top-K level price offsets-from-mid + normalized sizes (both sides), depth imbalance, book slope.
> - **Trade-flow composition (within the bar):** trade count, signed volume / CVD increment, aggressor imbalance, largest print, VWAP−mid, trade-size-distribution moments. **Why (unchanged):** separates a single sweep from churn of identical notional. CVD is supporting, not standalone.
> - **Intra-bar path:** realized variance within the bar, max adverse excursion. (unchanged)
> - **Cross-venue:** full book+flow for both venues, plus Binance−Coinbase mid spread (basis). **Basis** carries moderate short-horizon information via error-correction.
> - **Perp state — conditioners, not primary signals.** Funding rate, basis, OI change, liquidation flags/intensity. **Why (revised):** at seconds-to-minutes these are weak return predictors — funding R²≈0 at T+1 for a single asset (Presto), OI direction is ambiguous, liquidations are a lagged/contrarian/volatility signal. Keep them as volatility/regime/cascade-risk conditioners; do **not** treat them as primary alpha. Confirm the demotion with the §6 ablation (E2.4).
> - **Time:** elapsed wall-clock duration of the bar, time-of-day encoding. (unchanged)
>
> **Label price:** label off **mid or microprice, never last-trade** — at a few seconds, bid-ask bounce dominates the semimartingale signal.

**Why + ref:** LR §4, §7.

---

## §7 — Model (CF-JEPA)

**⚠⚠ This section has the most changes, including two reversals of deliberate decisions.**

### 7.2 Architecture

**CHANGE — ⚠ REVERSES A SPEC DECISION:** The spec specifies a "small **causal Transformer**, PatchTST-style patching." The actual CF-JEPA paper (Lee & Sim 2026) uses a **dilated depthwise-conv (TCN-style) encoder** — there is no transformer or patching in the method. Revert to the paper's backbone; it is also better-supported (CNN backbones dominate time-series SSL).

**TEXT — replace the §7.2 "Context encoder" bullet:**
> - **Context encoder `f_θ`:** the CF-JEPA backbone — a stack of **D≈5 multi-scale dilated depthwise-conv (DWConv) blocks** (kernels {3,9,15}, dilation 2^i, BN→GELU→pointwise→residual), hidden ~256, representation dim ~320, ~5–15M params. Causal at inference by encoding only a left-padded prefix (the paper's forecasting protocol). **Why the change:** the spec's earlier "causal Transformer / PatchTST patching" was a substitution not present in the CF-JEPA paper; CNN/TCN backbones empirically dominate time-series SSL (TS2Vec, TempSSL), so adopt the paper's conv encoder. A transformer remains a later ablation, not the baseline.

**TEXT — replace the §7.2 "Loss" bullet and its Why:** (⚠ REVERSES the anneal behavior and the monitoring guidance)
> - **Loss:** smooth-L1/L2 between predicted and `sg(EMA-target)` embeddings over short/mid/long zones, **plus VICReg variance+covariance and the multi-scale cross-crop invariance term** (the paper's objective). The paper **anneals the prediction-loss weight to zero**; we instead **anneal to a non-zero floor (~5–15% of peak), adaptively** — reduce the weight only while predictivity (below) is non-decreasing.
>   - **Why anneal-to-floor, not zero:** in our near-martingale regime, once the prediction weight hits zero the only remaining curvature is VICReg's geometry, and SGD provably relaxes the representation onto that target-agnostic geometry — discarding predictive structure. A non-zero floor keeps a force that distinguishes "predict the future" from "encode the present."
>   - **Anti-collapse is *not* enough by itself, and variance is the wrong monitor.** A representation can pass every variance/rank check (RankMe, embedding variance) while encoding *no* predictive dynamics — these metrics are documented to be "fooled" by high-variance noise. Add a **soft-codebook bottleneck** (per MTS-JEPA, which found EMA+VICReg insufficient on weak-precursor series) and **harden the predictor against becoming identity** (higher LR than the encoder; DirectPred-style eigenvalue floor; optional anti-identity penalty).

**TEXT — add a new §7.2 bullet:**
> - **Physical-Δt predictor conditioning.** Condition the (linear, near-identity) forward predictors on continuous target horizon and per-bar elapsed time via **Time2Vec(Δt, elapsed_Δ) → FiLM** (MetNet-style), trained with **randomized target horizons (biased short)**; emit 2s/10s/60s in one shot (multi-horizon quantile head). Add a GRU-D-style decay-to-mean for long gaps. **Why:** cheap, proven, and implements "sharpen in bursts / regress to mean over long gaps." Skip Neural ODE/CDE/Hawkes — overkill and jump-unfriendly for a low-SNR signal.

### 7.3 Downstream heads

**TEXT — replace §7.3:**
> - After SSL pretraining, **freeze** the encoder, attach a small head per horizon/target on the pooled representation (**last-token / attention-pool** for the causal setup, not mean-pool). **Frame heads as triple-barrier classification** where possible — representation/SSL methods have a better track record on classification than on raw return regression.
> - **Dual-encoder routing (decide empirically).** CF-JEPA produces two encoders: the **online** (higher-rank, discriminative) and the **EMA target** (smooth, low-rank). The paper routes forecasting to the smooth EMA encoder — but sharp-event BTC detection is closer to anomaly/classification, which may favor the discriminative online encoder. Route each head to whichever wins OOS net-of-cost.
> - Heads are cheap → train several (2s/10s/20–30s/60s, return vs triple-barrier).

**Why + ref:** LR §2, §3, §8.

---

## §8 — Targets & labeling

**CHANGE:** Keep triple-barrier + purged/embargoed CV; sharpen the high-frequency specifics.

**TEXT — replace §8:**
> - **Triple-barrier** labels at the fixed physical horizon(s), with **horizontal barriers scaled by short-horizon realized volatility** (EWMA over a micro-window) — a static threshold is the dominant labeling failure mode. Vertical barrier = the physical horizon. Label off **mid/microprice, never last-trade**. Expect a heavy flat/"0" class at 2s; decide deliberately between 3-class {short,flat,long} and a two-stage {trade?/which side?}.
> - **Purged + embargoed CPCV** (López de Prado). **Purge label *spans* `[t₀,t₁]`, not rows** (at a 60s barrier a test block contaminates ~60s of neighbouring labels). **Embargo ≥ max(label horizon, longest feature look-back)** — a rolling feature crossing the test boundary leaks even with perfect label purging. Use **CPCV** for a *distribution* of OOS metrics (not a single walk-forward path), and **sample-uniqueness weighting / sequential bootstrap** (arguably more impactful than the CV scheme at our overlap). **Why:** overlapping HF labels leak badly; naive k-fold inflates results ~20%.
> - **Meta-labeling**, if used, is only a **size / no-trade gate on an exogenous signal** — it will *not* beat a well-tuned end-to-end model on the same features (replicated null results). Its framing is a natural fit for implementing the no-trade band as a learned gate.
> - Target values normalized (returns in bps), never raw prices. (unchanged)

**Why + ref:** LR §6.

---

## §10 — Validation & evaluation

**CHANGE:** Reframe the baseline's role; add the multiple-testing and honest-fill discipline; add the decisive model control and the SSL predictivity gate.

**TEXT — replace §10.1:**
> ### 10.1 Supervised baseline — the gate *and* the likely champion (milestone 0)
> Before any JEPA: build the bar pipeline + features, then fit **LightGBM** predicting forward return/triple-barrier from the features, with purged CPCV and the cost metric below. **Why:** on noisy financial/microstructure data, gradient boosting on good features is not a weak baseline — it is empirically the hardest model to beat (it matched/beat DeepLOB on BTC LOB; beat 14 foundation models on returns). It confirms whether *any* cost-surviving signal exists; representation learning multiplies signal, it does not manufacture it. **If LightGBM shows no edge after costs, stop or pivot — JEPA will not conjure one.**
>
> The baseline is the *signal-existence* gate; it does **not** test representation value. That is a separate, later control (§10.3).

**TEXT — replace §10.2 and add §10.3, §10.4:**
> ### 10.2 Metric
> **Fees-included PnL with a no-trade band**, band width set from the **cost distribution** (2×taker fee + half-spread + slippage + a noise-margin), not a round number. Report **gross vs net side-by-side — the gap is the finding.** **Model taker fills honestly** (no passive-fill-at-mid: fills are adversely selected — the orders that fill are the ones the market just moved against). Add **MCC** as a secondary skill metric (separates "has skill" from "monetizable"). **Stratify every result by spread/tick and volatility regime.** Guard against multiple testing: track effective `N` (cluster correlated trials), require **Deflated Sharpe Ratio > 0.95**, and compute **PBO via CSCV**. **Pre-register** labels/CV/band/metric before the held-out OOS month.
>
> ### 10.3 The decisive model control
> To decide whether *pretraining* (not just deep learning) earns its complexity, the headline comparison is **frozen CF-JEPA + head vs the *same conv architecture trained supervised from scratch* vs LightGBM**, net-of-cost under CPCV. Beating LightGBM shows "signal + some model helps"; beating supervised-from-scratch of equal capacity is the only thing that isolates the value of SSL pretraining.
>
> ### 10.4 SSL-stage predictivity gate
> Before attaching heads, gate the pretrained encoder on **predictivity, not just non-collapse**: pass only if (a) embedding geometry is non-degenerate (variance, RankMe, **LiDAR**), **and** (b) predicted-vs-actual future-embedding alignment is strictly above a persistence baseline, **and** (c) a linear probe beats raw-feature and identity baselines OOS, **and** (d) that lift **survives a temporal-shuffle control** (shuffle future targets — if predictivity survives, it was geometry/leakage, not dynamics). Report every predictivity number as a *lift over identity/persistence*.

**Why + ref:** LR §1, §3, §4, §6.

---

## §12 — Build order

**CHANGE:** Insert the baseline ladder and the explicit gates; front-load data integrity.

**TEXT — replace §12:**
> 1. **Ingest + archive** (Crypto Lake Binance + Coinbase via CoinAPI) → raw Parquet by (exchange, symbol, day). Verify §4 sample checks.
> 2. **Event-time reconstruction (`recon`)** — one function shared offline+live; **gate: byte-identical replay-equivalence test** (no lookahead).
> 3. **Notional-bar sampler + features (`bars`)** — §5 clock (adaptive threshold, time cap), §6 vector (OFI/microprice first). Output model-ready tensors; produce the time-per-bar histogram.
> 4. **Labels + purged/embargoed CPCV (`data`)** — §8.
> 5. **Baseline ladder (`eval`):** naive → penalized-linear → **LightGBM**, net-of-cost, DSR/PBO, regime-stratified. **GATE G1 (project gate): is there cost-surviving signal at any horizon?** Fail → stop/pivot.
> 6. **Design-validation ablations** (cheap, on the baseline): the decoupling proof (§5.4), bar-clock, **Binance-adds-edge-over-Coinbase-own-book**, perp-signal demotion.
> 7. **Supervised deep baseline.** **GATE G3:** does deep ≥ LightGBM? Fail → ship LightGBM.
> 8. **CF-JEPA pretraining (`model`/`train`)** — §7. **GATE G4: SSL predictivity battery** (§10.4). Fail → harden (codebook, predictor, anneal floor) or fall back.
> 9. **Heads + the decisive control.** **GATE G5:** frozen-CF-JEPA beats same-arch supervised-from-scratch net-of-cost? Pass → it earns its complexity.
> 10. **Walk-forward + extensions:** MTS-JEPA multi-resolution, frequency auxiliary, SimTS arm, richer cross-venue.
>
> **Three hard stops (G1, G3, G5) can end the project early and keep the most expensive work last.**

**Why + ref:** LR §0, experiment-plan.

---

## §13 — Pitfalls & gotchas

**CHANGE:** Sharpen four existing pitfalls and add two.

**TEXT — replace/augment these bullets:**
> - **Martingale collapse — *and* the subtler "non-collapsed but predictively empty".** A JEPA can pass embedding-variance/RankMe checks while encoding no dynamics; those metrics are documented to be fooled by high-variance noise. Diagnose with LiDAR + predicted-vs-persistence alignment + a **temporal-shuffle control** + probe-lift-over-identity (§10.4).
> - **Anneal-to-zero is the prime collapse suspect** on low-SNR data — it relaxes the encoder onto VICReg geometry. Anneal to a non-zero floor (§7.2).
> - **CV leakage — purge *spans*, not rows; embargo the longest feature look-back**, not just 0.01·T (§8).
> - **Forecasting ≠ tradeable — and fills are adversely selected.** Evaluate net-of-cost PnL with honest taker fills; never assume passive fills at mid (§10.2).
> - **(new) Period-dependence of the lead venue.** Binance's seconds-horizon lead is regime-dependent and shifted post-2024-ETF; re-estimate per period; keep a Coinbase-own-book baseline (§1).
> - **(new) Over-claiming SSL.** A frozen-JEPA win over LightGBM could be "deep learning helps," not "pretraining helps" — only the supervised-from-scratch control (§10.3) isolates pretraining value.

**Why + ref:** LR §3, §6, §7.

---

## §3 — Repo layout (minor)

**CHANGE:** None to the layout. Note only: `recon`/`bars` are Python-first now (correctness), Rust port deferred (spec §3's Rust rationale still holds for the throughput port; the replay-equivalence test becomes the cross-language conformance test). No spec text change required unless you want to record the Python-first decision.

---

*Apply order suggestion: §2 + §7 together (the backbone reversal touches both), then §6, §10, §12 (the gate-bearing changes), then §1/§5/§8/§13. I can apply all as edits on your go-ahead.*
