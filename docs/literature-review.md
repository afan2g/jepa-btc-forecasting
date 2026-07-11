# JEPA BTC Forecasting — Literature Review & Research Synthesis

**Purpose.** This document situates the design in [`jepa_btc_forecasting_spec.md`](../jepa_btc_forecasting_spec.md) against the published literature (incl. preprints through mid-2026). It records what the evidence supports, what it challenges, what is genuinely novel/unvalidated in our design, and concrete changes — each mapped to a spec section. It is the synthesis of an 8-thread parallel literature search.

**2026-07-11 scope note.** The literature findings remain evidence, but #66
changed the execution order: first test Binance BTC-USDT perpetual own-book and
trade-flow signal (`G0-BN`), then treat Binance spot/state, Coinbase transfer,
cross-venue context, and other assets as conditional increments. References
below to the Binance→Coinbase premise or a Coinbase-own-book baseline now inform
that later incremental gate; they are not prerequisites for G0-BN.

**How to read.** §0 is the executive summary (read this first). §§1–8 are the thematic findings, each ending with **→ Implications for our spec**. §9 consolidates recommendations by spec section. §10 is open questions to resolve on our own data. §11 is the master reference list.

**Verification status & honesty flags.**
- The **CF-JEPA paper itself** (Lee & Sim, *Knowledge-Based Systems*, May 2026; arXiv:2606.07031) did **not** surface in the research agents' web searches (too new / journal-indexed). However it was read directly from the PDF during design; its mechanics (dilated depthwise-conv encoder, EMA target, near-identity linear forward predictors over short/mid/long zones, VICReg variance+covariance + multi-scale cross-crop invariance, **prediction-loss weight annealed to zero**, dual-encoder asymmetry) are confirmed from source.
- Items marked **[abstract-only]** were read via abstract/secondary summary (publisher paywall/403).
- Items marked **[secondary]** come from practitioner/blog sources, not peer-reviewed.
- Items marked **[2026-preprint]** are very recent arXiv; treat specific numbers as preliminary/not-yet-replicated.
- No citation here is fabricated; anything an agent could not verify is flagged.

---

## 0. Executive summary

1. **LightGBM-on-good-features is not just the gate (§10.1) — it is likely the champion, and the bar for JEPA is higher than the spec implies.** On daily financial returns, CatBoost beat 14 time-series foundation models (which got *negative* R²); on BTC/USDT LOB, XGBoost and even logistic regression matched/beat DeepLOB. Tree models dominate noisy tabular/financial data for mechanistic reasons that are maximal at low SNR. **The honest test for JEPA is not "beats LightGBM" but "frozen CF-JEPA + head beats the same conv encoder trained supervised from scratch"** — the only comparison that isolates the value of *pretraining*. The spec lacks this control.

2. **"Forecasting power ≠ tradeable profit" is the field's clearest empirical result.** Our fees-included PnL + no-trade-band evaluation (§10.2) is correct and vindicated. Strengthen it: add MCC as a secondary skill metric, model taker fills honestly (fills are adversely selected), and stratify by spread/volatility regime.

3. **The collapse risk is real and the research sharpened it: *non-collapsed ≠ predictive*, and the standard diagnostics (embedding variance, RankMe) are explicitly known to be "fooled" by high-variance noise.** CF-JEPA's **anneal-to-zero prediction loss is the prime suspect** on near-martingale data — implicit-regularization theory predicts SGD relaxes the encoder onto VICReg's geometry once the prediction weight hits zero. Anneal to a non-zero floor and add a real predictivity gate (LiDAR + predicted-vs-persistence alignment + temporal-shuffle control + probe-lift-over-identity).

4. **The signal is a short list, and it is order flow.** Multi-level / integrated OFI (Cont et al.), microprice, queue imbalance, and spread-regime are the features that demonstrably carry short-horizon signal in both equities and crypto. Funding / OI / liquidations — our spec's "extra perp signals" — are weak at seconds horizons (funding R²≈0 at T+1). Re-rank §6 accordingly.

5. **τ (the microstructure decay window) ≈ a few seconds to ~30s.** The 2s/10s/60s ladder brackets it; add a ~20–30s rung and measure τ on our own data. The closest crypto paper targets 3s.

6. **Our clock/horizon *decoupling* (§5.4) is genuinely novel and unvalidated** — no paper states it, and the closest crypto bars+DL work still used fixed-bar-count horizons. It is our key contribution; prove it with a fixed-bar-count vs fixed-physical-time target ablation.

7. **The Binance→Coinbase premise (§1) is directionally sound at seconds but period-dependent** (post-2024 spot-ETF regime shifted price discovery toward Coinbase/spot) and cost-exposed (cross-venue taker PnL has been negative in the literature). Add a serious Coinbase→Coinbase baseline; define success as "Binance adds edge *over* the own-book model, net of cost/latency."

8. **The physical-Δt predictor has a clean, cheap, proven recipe:** Time2Vec(Δt, elapsed_Δ) → FiLM, MetNet-style, with randomized-horizon training and a multi-horizon quantile head. Skip Neural ODE/CDE/Hawkes as overkill for a low-SNR jumpy signal.

---

## 1. The core bet: does SSL pretraining help low-SNR financial forecasting?

**Simple models are hard to beat on noisy/financial data.**
- *Are Transformers Effective for Time Series Forecasting?* (Zeng et al., AAAI-23, arXiv:2205.13504): a one-layer **DLinear/NLinear** beats Informer/Autoformer/FEDformer on most long-horizon benchmarks. The counter-wave (PatchTST arXiv:2211.14730; iTransformer; TiDE; TimesNet; TimeMixer) only *marginally* beats a strong linear baseline, and on **clean, seasonal, high-SNR** series — not near-martingale returns.
- **Gradient boosting remains SOTA on tabular/feature data:** *Why do tree-based models still outperform deep learning on tabular data?* (Grinsztajn et al., NeurIPS 2022, arXiv:2207.08815) — three pathologies of NNs (bias toward over-smooth functions, rotational invariance, sensitivity to uninformative features) are **all maximal in low-SNR finance**. *Tabular Data: Deep Learning Is Not All You Need* (Shwartz-Ziv & Armon, 2022, arXiv:2106.03253) — XGBoost beats deep tabular models, including on the deep models' own datasets; an *ensemble* of XGBoost + DL beat XGBoost alone.

**Foundation models fail on financial returns.**
- *Re(Visiting) Time Series Foundation Models in Finance* (arXiv:2511.18578, Nov 2025) — the load-bearing skeptical result. Testing 14 TSFMs vs classical/boosting baselines on daily excess returns (~1.93B obs): **CatBoost R² −0.03%, Sharpe 6.79; Chronos-Large R² −1.37%; TimesFM-500M R² −2.80%, return −1.47%, directional acc ~51%.** Fine-tuning mostly made it worse. Only **pretraining-from-scratch on financial data** recovered economic gains (Sharpe up to 5.42) — and still lost to CatBoost on goodness-of-fit. Conclusion: *"generic time series pre-training does not directly transfer to financial domains."* (Caveat: daily, univariate — not our seconds regime.)
- *Empirical Asset Pricing via Machine Learning* (Gu, Kelly, Xiu, RFS 2020): even where deep nets win, monthly OOS R² tops out at **~0.33–0.40%**; the lesson is "edge lives at R² of fractions of a percent" with decades of data.

**Where SSL/JEPA does have a track record:** *latent-space prediction is more noise-robust than input reconstruction* (TS-JEPA, arXiv:2509.25449) — but TS-JEPA wins on **classification, not regression**. In-domain pretraining (what CF-JEPA does) sidesteps the *generic→finance* transfer failure, but not the question of whether *any* representation beats LightGBM-on-features at these horizons.

**→ Implications for our spec (§9, §10, §12):**
- Reframe §10.1: LightGBM is the **gate AND the likely champion**. Build a baseline ladder (§9) and require JEPA to beat *supervised-same-architecture-from-scratch*, not just LightGBM.
- Realistic expectation-setting: OOS R² of **0 to low single-digit bps** for 2–10s is a *success*; directional accuracy **55–60%+ should trigger a leakage hunt**, not celebration.
- Frame heads as **triple-barrier classification** (plays to JEPA's documented strength) rather than raw return regression.

---

## 2. SSL / JEPA for time series — situating CF-JEPA & prior art

**The paradigm verdict: forward-predictive latent SSL is the favored camp for forecasting**, and instance-discrimination contrastive is mismatched to it (converges across SimTS, TempSSL, TS-JEPA, "What Constitutes Good CL"). Random-mask reconstruction destroys temporal continuity (motivating CF-JEPA's mask-free design); but masking that *respects* temporal structure (SimMTM) is competitive.

**Closest prior art (must benchmark against):**
- **SimTS** (arXiv:2303.18205) — the near-identical predecessor: Siamese **forward prediction in latent space**, no negatives, "the forecasting task itself provides regularization." CF-JEPA ≈ SimTS + EMA target + VICReg + multi-zone predictors + loss annealing. **This is the honest "is the JEPA machinery worth it?" control.**
- **"What Constitutes Good Contrastive Learning in TS Forecasting?"** (arXiv:2306.12086) — found **end-to-end + joint self-supervised loss beats staged pretrain-then-finetune** for forecasting. Directly challenges our staged-SSL premise; test it head-to-head.
- **TS-JEPA / "Joint Embeddings Go Temporal"** (arXiv:2509.25449) — generic TS-JEPA (masking-based); noise-robust; **classification > forecasting**.
- **MTS-JEPA** (arXiv:2602.04643, [2026-preprint]) — multi-resolution JEPA for anomaly *prediction*; explicitly confronts collapse, needing a **soft-codebook bottleneck + dual-entropy** beyond EMA+variance. The most important paper for our collapse risk (see §3).
- **TF-JEPA** (ICLR 2026 TSALM, OpenReview 8bLa8PILyO) and **FEI** (AAAI 2025, arXiv:2412.20790) — frequency-domain non-contrastive variants; complementary auxiliary objective.
- **LaT-PFN** (arXiv:2405.10093) — JEPA + Prior-data-Fitted Network for in-context forecasting.
- Foundational: **I-JEPA** (arXiv:2301.08243); **C-JEPA** (arXiv:2410.19560) shows **JEPA + VICReg provably synergize to avoid collapse** — theoretical support for CF-JEPA's VICReg component.

**Strong general SSL-for-TS baselines that transfer to forecasting:** TS2Vec (arXiv:2106.10466; dilated-CNN + cropping — CF-JEPA's encoder ancestor), CoST (arXiv:2202.01575; forecasting-specific, seasonal-trend disentangle — weak at seconds), TimesURL (arXiv:2312.15709; wins on *short-term* forecasting), SimMTM (arXiv:2302.00861), T-Rep (arXiv:2310.04486; learnable time-embeddings — borrowable for our dollar-clock cadence). Classification-leaning (weak forecasting transfer): SoftCLT, TF-C, TimeMAE, TF-JEPA, FEI.

**→ Implications (§7):**
- Keep the **conv (dilated-DWConv) encoder** — CNN backbones dominate TS-SSL; the spec's PatchTST/transformer substitution (§7.2) is unsupported and off-paper. Revert it.
- Add **SimTS** and **end-to-end-joint** as first-class comparison arms.
- Consider a **frequency-domain auxiliary** (FEI/TF-JEPA-style) and **T-Rep-style cadence embeddings**.

---

## 3. Representational collapse — detection & prevention (the deepest payoff)

**Your core worry is a documented failure mode: a representation can pass every variance/rank check while being predictively useless — and the standard non-collapse diagnostics are explicitly known to be fooled by it.**

- **Two collapse modes** (standard terminology): *complete* (constant output; batch variance → 0; practitioner threshold mean var < ~1e-7) and *dimensional/rank* (embeddings in a low-dim subspace) — Jing et al., *Understanding Dimensional Collapse* (arXiv:2110.09348).
- **Why stop-grad + EMA + predictor work:** SimSiam (arXiv:2011.10566; stop-grad is load-bearing); **DirectPred** (Tian et al., arXiv:2102.06810) — the rigorous theory: removing stop-grad provably collapses; the predictor's eigenspaces align with the feature-correlation matrix; **weight decay** and **EMA-as-curriculum** are essential; **a higher predictor learning rate flattens the collapse basin** (a protective knob). 
- **VICReg** (arXiv:2105.04906): variance hinge (γ=1) prevents informational collapse, covariance prevents dimensional collapse — but it is a **pure geometry regularizer**, needing neither stop-grad nor a predictor. **This is exactly why annealing *toward* VICReg is risky: late training optimizes a target-agnostic geometry.** Barlow Twins (arXiv:2103.03230) and the contrastive/non-contrastive **duality** (arXiv:2206.02574) reinforce that these objectives shape geometry, carrying no explicit "predict-the-future-better-than-the-present" pressure.
- **The smoking gun (non-collapsed ≠ predictive):** **RankMe** (arXiv:2210.02885) and **LiDAR** (arXiv:2312.04000) papers warn that **a random mapping can have high effective rank with no downstream value** — RankMe "can be fooled... at full rank." LiDAR's stated motivation is precisely that covariance-rank fails to separate informative from uninformative high-variance features; it measures the rank of the *LDA matrix of the SSL task* instead.
- **Low-predictability regime:** CPC/InfoNCE (arXiv:1807.03748) gives the lens — the MI lower bound is near-zero when I(future; context) is tiny, so a predict-the-future objective degenerates toward predict-the-present at low SNR. **MTS-JEPA** (arXiv:2602.04643) empirically needed a codebook bottleneck beyond EMA+variance on weak-precursor series.
- **The anneal-to-zero risk:** no paper studies prediction-weight→0 on low-SNR data head-on (genuine gap), but implicit-regularization-drift theory (representational-drift work; arXiv:2302.02563) shows that once the task loss is zero-weighted, SGD noise performs a random walk minimizing the *remaining* (VICReg) regularizer — provably **relaxing the representation onto VICReg geometry, detached from predictive structure.**

**Monitoring battery (implement as the SSL-stage gate):**

*A. Collapse/geometry health (necessary, NOT sufficient):* batch embedding variance (hard-collapse < 1e-7); **RankMe** (use ≥~25k samples); singular-value spectrum; covariance off-diagonal energy; **LiDAR** (the best label-free "fooled-by-noise" detector — high RankMe + low LiDAR ⇒ variance is noise).

*B. Predictivity detectors (catch the exact failure — report all as LIFT over a persistence/identity baseline):*
- **Predicted-vs-actual future-embedding alignment minus persistence** (cosine(pred, target) − cosine(z_t, z_{t+k})); ≤ 0 ⇒ predictor learned identity.
- **Linear-probe lift over identity/current-state** (ΔR² / Δdirectional-acc / ΔIC, out-of-sample).
- **Temporal-shuffle control** (CRITICAL for martingale regime): re-run with future targets time-shuffled; if predictivity survives, it's geometry/leakage, not dynamics.
- **Predictor degeneracy** (‖W_p − I‖, singular values — is it becoming a copy?).
- **InfoNCE future-vs-distractor** (sets the predictability *ceiling*: if even this is at chance, the data's I(future;context) ≈ 0 — a data verdict, not a bug).

**SSL-stage gate:** PASS only if A-metrics non-degenerate **and** predicted-vs-actual alignment is strictly above persistence **and** probe(z) beats raw-feature *and* identity baselines OOS **and** that lift **survives the temporal-shuffle control**.

**→ Implications (§7.2, §13):**
- **Anneal the prediction weight to a non-zero floor (~5–15% of peak), not zero**; make the anneal **adaptive** (only reduce while predictivity lift is non-decreasing).
- **Replace "monitor embedding variance / VICReg terms"** with the battery above — variance/RankMe are the *wrong* instruments for our failure mode.
- **Borrow MTS-JEPA's codebook bottleneck** (anti-collapse + regime anchors).
- **Harden the predictor** against identity: higher LR than encoder; DirectPred-style eigenvalue floor; optional anti-identity penalty.
- Keep an **explicit persistence/identity baseline in the loop at all times**; every predictivity claim is a *lift over identity*.

---

## 4. LOB / crypto microstructure deep learning + features

**Architecture lineage** (FI-2010 equity benchmark): DeepLOB (CNN+Inception+LSTM, arXiv:1808.03668) → TransLOB → TABL/BiN-TABL (a 2-layer bilinear net rivals deep models) → Axial-LOB → HLOB → TLOB. **The SOTA-leapfrogging is small-margin and benchmark-fragile; depth is not the lever.** Beware: FI-2010 is pre-normalized and models overfit it and **fail to generalize** (arXiv:2308.01915).

**"Better inputs > deeper model" is the crypto consensus:**
- arXiv:2506.05764 (Bybit BTC/USDT, 100ms): **XGBoost ≈ logistic regression ≈ DeepLOB** when inputs are good; input smoothing (Savitzky-Golay) helped more than any architecture change. Accuracy ~0.50–0.53 at 100ms (barely above chance), rising only at longer horizons / deeper book.
- arXiv:2602.00776 (Binance Futures perp, **1s data, 3s target**, Jan-2022→Oct-2025, BTC + 4 alts): CatBoost + SHAP; **OFI, spread, VWAP-deviation are the top features**, effects matching microstructure theory, **strikingly stable across assets/tick-sizes**; taker fees-included backtest ⇒ **BTC ~13%/yr at modest information ratio** (a real but small edge). [2026-preprint]
- arXiv:2010.01241 (Coinbase BTC-USD spot, temporal CNN): **71% directional accuracy at 2s** — but **no PnL/fees** in the result.

**The decisive evaluation finding — LOBFrame** (*Deep LOB Forecasting: a microstructural guide*, arXiv:2403.09267, *Quantitative Finance* 2025): reports **MCC** (not accuracy) because of class imbalance; skill is **strongly conditional on spread-to-tick ratio** (large-tick MCC ~0.29 vs small-tick ~0.01–0.11 ≈ near-zero); introduces a "probability of forecasting a *complete (executable) transaction*" metric; core conclusion: **high statistical forecasting power does NOT correspond to actionable trading signals.** Effective horizon is **~2 price changes** (very short).

**Features that carry signal (the short list):**
1. **Order-flow imbalance (OFI), multi-level / integrated** — strongest single contemporaneous driver (Cont, Kukanov, Stoikov, arXiv:1011.6402: price change ≈ linear in best-level OFI, slope ∝ 1/depth; Cont-Cucuringu-Zhang, arXiv:2112.13213: PCA-**integrated** multi-level OFI raises explanatory power; **lagged cross-asset OFI forecasts short-horizon returns, decaying rapidly** — direct support for our cross-venue angle).
2. **Microprice** (Stoikov, SSRN 2970694) — mid adjusted by spread + queue imbalance; a martingale fair-value; better short-horizon predictor than mid. Candidate **feature AND prediction target/anchor**.
3. **Queue/volume imbalance**, **spread (and spread/tick as a regime selector)**, **signed trade flow / CVD** (supporting, not standalone), **VWAP-to-mid deviation**, **book slope/depth**, **intra-bar path**.

**Adverse selection (why naive backtests lie):** Moallemi & Yuan queue-value (2016); *The Market Maker's Dilemma* (arXiv:2502.18625) — **fill probability negatively correlates with post-fill return**: the orders that fill are the adversely-selected ones. Any passive-fill-at-mid assumption overstates PnL.

**SSL for LOB is a near-white-space:** LOBench (arXiv:2505.02139) and a contrastive-LOB manipulation-detection paper (arXiv:2508.17086) are early; **no paper yet demonstrates a tradeable, cost-surviving crypto edge from SSL pretraining.** Our CF-JEPA approach occupies genuinely under-explored territory (cuts both ways).

**→ Implications (§6, §10):**
- **Promote multi-level/integrated OFI to the #1 feature**; add **cross-venue lagged OFI** (Binance perp & spot → Coinbase). Use the *event-based* OFI definition (signed depth changes per level), not static depth imbalance.
- **Microprice as feature and candidate target.** Spread/tick as an explicit **regime gate**.
- Start with **gradient-boosted trees on engineered features** as a permanent baseline; escalate to deep only if it beats that net-of-cost.
- Add **MCC** secondary metric; **model taker fills honestly**; **stratify by spread/vol regime**.

---

## 5. Information bars, clocks, and the τ decay window

**Subordination / volume-clock (supports dollar bars):** Clark 1973 (subordinated BM, finite variance), Ané & Geman 2000 (**trade count may normalize returns better than volume** — a nuance against assuming *dollar* is best on Gaussianity), Easley-LdP-O'Hara *The Volume Clock* (2012) — returns over fixed-volume intervals are "significantly closer to i.i.d. Gaussian," and **"clock time per volume bucket varies dramatically between active and quiet markets"** (this is *our §5.4 premise, stated by the originators*). López de Prado info-driven bars (AFML Ch.2): **dollar > volume > tick** specifically for **price-level robustness** (our BTC-ranged-2x rationale; the closest crypto paper makes the identical $67M→$334M argument).

**Caveats:** "dollar bars give best normality" is **not robust** — one published test had volume beating dollar on normality; Ané-Geman favor trade-count. Honest claim: *information bars > time bars*, with **dollar chosen for price-level robustness, not proven-best Gaussianity**. **VPIN is heavily contested** (Andersen-Bondarenko: mechanically tied to volume/volatility, no incremental predictive power, didn't lead the Flash Crash) — use toxicity features only with volume/vol controls.

**The decay window τ (two distinct timescales — don't conflate):**
- *Directional* microstructure signal (relevant to us): **a few seconds to ~30s.** Queue imbalance predicts ~1–2 mid-changes then decays (Gould & Bonart, arXiv:1512.03492); OFI→price studied at ~10s scale, micro-price advantage "largely noise beyond ~30s" (Cont et al.); crypto secondary estimates ~3–10s; the Binance-perp paper *designs* for **3s**. **τ ≈ 3–30s, mostly gone by 30–60s.** Our 2s/10s/60s ladder brackets it (60s = decay/control arm).
- *Order-flow persistence* (long memory, H≈0.7 to ~10,000 trades; Lillo-Farmer) is **persistence of flow, not return predictability** — impact is transient (Bouchaud propagator). Don't read it as a long return-forecast horizon. (Also partly contested — Axioglou-Skouras.)

**The decoupling thesis is novel:** **no paper states** "fixed block-count horizon under an event clock yields variable-predictability targets → target fixed physical time instead." The closest, most relevant crypto paper (*Algorithmic crypto trading using information-driven bars, triple-barrier & DL*, Financial Innovation 2025) used **dollar/CUSUM bars + triple-barrier but a fixed-*bar-count* (24-bar) vertical barrier** — it did *not* take our decoupling step. This is our key contribution and our key risk.

**→ Implications (§5):**
- Keep dollar bars; **make the threshold adaptive** (rolling avg dollar-volume / target-bars-per-day, e.g. trailing 7–30d) — a static threshold under-samples early, over-samples late as volume grows.
- **Run a trade-count-bar ablation** (Ané-Geman) on our own normality/homoscedasticity metrics; don't assert dollar-bar Gaussianity as settled.
- **Treat time-capped bars as a labeled subpopulation** (flag `emitted_by_time_cap`); they reintroduce calendar-clock heteroscedasticity in quiet regimes.
- **Add a ~20–30s horizon rung**; **measure τ on our data** (decay of OFI→future-return R², ACF crossing, impact-decay fit) and set the cap just beyond the knee.
- **Prove the decoupling** with a fixed-bar-count vs fixed-physical-time target ablation (show predictability variance explodes for the former). Plot the **time-per-bar distribution** — it *is* the justification.

---

## 6. Labeling & validation

**Labeling:** Triple-barrier is the right shape for a barrier-like PnL eval, **but scale horizontal barriers by short-horizon realized vol** (static thresholds bury real moves — LdP's biggest failure mode), set the **vertical barrier = the physical horizon**, expect a heavy flat/"0" class at 2s, and **label off mid/microprice, never last-trade** (bid-ask bounce dominates the signal at a few seconds). **Trend-scanning** is a cheap secondary arm but data-hungry per event. **Meta-labeling won't beat a well-tuned end-to-end LightGBM on the same features** (replicated null results; even friendly studies found fixed-horizon meta-labeling *degraded* OOS) — use it only as a **size/no-trade gate on an exogenous signal**, which is a legitimate fit for our no-trade band.

**Validation — CPCV + purge + embargo + uniqueness weighting:**
- **Purge label *spans* [t₀, t₁], not rows** — the #1 mistake; at 60s barriers a test block contaminates ~60s of neighboring training labels.
- **Embargo must cover the longest feature look-back** (e.g. a 5-min rolling feature ⇒ ≥5-min embargo), not just LdP's 0.01·T — under-embargoing is a silent leak distinct from label leakage.
- **CPCV** for a *distribution* of OOS Sharpes (φ[N,k] paths), not a single walk-forward path — but don't let CPCV become a parameter search (select on inner folds; use outer paths only to estimate dispersion).
- **Sample-uniqueness weighting / sequential bootstrap** — arguably *more impactful than the CV scheme* at our overlap level; equal-weighting over-counts redundant near-duplicate labels.
- Empirical: naive k-fold inflated apparent performance up to ~20% in one HF study; leakage grows with autocorrelation/overlap.

**Evaluation — anti-self-deception:**
- Net PnL with a no-trade band sized from the **cost distribution** (taker fee×2 + half-spread + slippage + margin); **sweep band width, plot net-Sharpe vs turnover** (but count the sweep as trials). Report **gross vs net side-by-side — the gap is the finding.**
- **Multiple-testing controls:** track effective N (cluster correlated trials); require **Deflated Sharpe Ratio > 0.95** (False Strategy Theorem: ~1000 noise trials ⇒ expected max in-sample SR ≈ 3; ~45 trials on 5yr ⇒ fake SR≈1); compute **PBO via CSCV**. Harvey-Liu: the naive "50% haircut" is wrong; haircut depends on # and correlation of trials.
- **Pre-register** labels/CV/band/metric before touching the final test fold; every post-hoc tweak is another trial.

**→ Implications (§8, §10):** the gate (§12 step 5) should be **net-of-cost PnL with DSR > 0.95 and acceptable PBO under CPCV**, not just "is there signal." If LightGBM can't clear that at 2–60s, JEPA won't.

---

## 7. Cross-venue price discovery — the Binance→Coinbase premise (§1)

**Verdict: directionally sound at seconds, but period-dependent and cost-exposed.**

**Supporting:** *Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets* (Albers, Cucuringu, Howison, Shestopaloff, 2021, arXiv:2108.09750) — the closest analog to our spec: sub-second / 500ms returns, 14 BTC markets; **"Binance and Huobi perpetuals/futures are particularly strong leading markets"**; **trade-flow imbalance (signed volume) is the most powerful feature**; cross-venue models explain **10–37% of 500ms future returns**; the Binance USDT perp is *least predictable* (i.e., it leads). *Price Discovery in Fragmented Crypto Markets* (Jang et al., 2025/26, VECM, minute-level): **Binance/OKX dominate short-run leadership; Coinbase increasingly drives the long-run equilibrium, magnified by spot ETFs.** Perp-leads-spot supported by Alexander & Heck (2020) [abstract-only].

**Contradicting / complicating:** **CME futures lead spot** (arXiv:2506.08718); **post-2024 spot-ETF complex / Coinbase dominate** price discovery (~85%, Kia et al. 2026 [abstract-only]); noise-robust measures can flip the ranking to Bitfinex (Putniņš ILS); *Nothing but Noise?* (Dimpfl & Peter 2021 [abstract-only]) — cross-exchange rankings are biased by differential microstructure noise; *The Jury Is Out* (Frino et al. 2025 [abstract-only]) — spot-vs-futures leadership is explicitly unsettled and regime-dependent. **The lead venue has demonstrably changed over time** (FTX was major in Albers' 2021 data and is gone).

**The "NOT latency arb" framing is correct:** the mechanical lead is sub-second; after costs, ~91% of cross-venue gaps sit inside the no-trade band (Makarov-Schoar). We forecast the **common efficient price** via the most informative order flow — the right mental model; venues are cointegrated (~99.5% rank-1 weekly). **But Albers' cost-adjusted taker PnL was massively negative**; positive PnL appeared only for **maker** execution on low-fee venues (~1bp alpha). A taker strategy on Coinbase driven by Binance signal is exactly the config that usually loses.

**Which perp signals are actually predictive (re-ranks our §6 "extra signals"):** **OFI / signed volume = strongest** (and it's a *perp order-flow* signal, not funding/OI). **Basis (perp−spot) = moderate** via error-correction. **Funding = weak** (R²≈0 at T+1 single-asset; cross-sectional only). **OI = weak-moderate**, better as a vol/cascade-risk gate. **Liquidations = moderate but contrarian/vol**, reported with lag and exchange-capped.

**→ Implications (§1, §4, §6):**
- **Model Coinbase→Coinbase as a serious baseline.** Define success as: *does Binance signal add edge over the Coinbase own-book model, net of latency+cost?*
- **Measure our own Binance→Coinbase lead-lag** (Hayashi-Yoshida + VECM error-correction) on current data; **re-estimate pre/post-2024-ETF**.
- **Build the cost-adjusted PnL curve first** (decide taker vs maker upfront). Elevate **OFI** to first-class; downgrade funding/OI/liquidations to conditioners.

---

## 8. Horizon conditioning & async multi-stream alignment

**Conditioning the predictor on continuous Δt — recommended recipe:** **Time2Vec(Δt_target, elapsed_Δ)** (arXiv:1907.05321; embeds any continuous scalar, linear + learnable-sinusoid terms) → **FiLM** (arXiv:1709.07871; per-channel affine γ⊙h+β from a tiny generator), applied **MetNet-style** (arXiv:2003.12140 — single model conditioned on lead-time via FiLM; **randomized-horizon training acts as data augmentation**, biased-short sampling helps). This exactly implements the "sharpen in bursts / regress-to-mean over long gaps" behavior. Emit 2s/10s/60s in one shot via a **TFT-style multi-horizon quantile head** (arXiv:1912.09363).

**Irregular-sampling modeling — adopt vs overkill (for ~5–15M params, low SNR):**
- **Adopt:** pass elapsed Δ via Time2Vec (highest ROI); optionally an **mTAN**-style time-attention encoder (arXiv:2101.10318 — handles irregularity, no interpolation, no ODE solver); **GRU-D-style learnable decay-to-mean** (arXiv:1606.01865 — a few params that literally implement "regress to mean over long gaps"); optional **ALiBi/RoPE time-decay attention** in the trunk.
- **Skip (overkill + jump-unfriendly):** Neural ODE/ODE-RNN/Latent ODE (slow, unstable on discontinuities — exactly our data), Neural CDE (only the **online/causal rectilinear-interpolation variant**, arXiv:2106.11028, is live-safe; vanilla injects lookahead), ContiFormer, full hypernetworks (FiLM already *is* a tiny linear hypernetwork). Generative duration models (ACD/Neural Hawkes) are scope creep — conditioning on Δ subsumes the first-order effect; if ever wanted, use **Intensity-Free LogNormMix** (arXiv:1909.12127, closed-form).

**Async alignment / lookahead avoidance (§5.3):**
- **One event-time reconstruction function shared by train and live.** Merge trades + L2 deltas onto a single engine-time axis with a deterministic tiebreak; reconstruct the book by replaying diffs in **sequence/update-ID order** (Binance `U/u/pu` continuity rules for gap detection; re-snapshot on any gap), using IDs for *correctness* and timestamps for *axis placement*.
- **Book-at-trade WITHOUT lookahead = apply-before-read with strict `<`**: trade at index i sees only book deltas ordered strictly before it; never the post-trade book. (Use `<`, not `≤`, at the boundary.)
- **Live: bounded-out-of-orderness watermark** (Flink model) — hold each engine-time t until watermark ≥ t; late events are dropped/rerouted, never retro-injected. **Size the watermark from measured stream-skew**, per-horizon (a 60s predictor tolerates a larger watermark than a 2s one).
- **Replay-equivalence test:** assert byte-identical features offline vs. live-ordered replay. This is the single highest-severity correctness guard.

**→ Implications (§5.3, §5.4):** the spec's physical-Δt predictor and event-time recon both have concrete, proven implementations above; adopt the Time2Vec→FiLM + randomized-horizon recipe and the shared-recon-function + watermark + replay-equivalence discipline.

---

## 9. Consolidated recommendations by spec section

**§1 (Objective/premise):** Add a **Coinbase→Coinbase baseline**; success = Binance adds edge over own-book net of cost/latency. Re-estimate lead-lag pre/post-2024-ETF.

**§4 (Data):** Mixed-vendor decision stands. Elevate **OFI-grade** book/trade fields. Verify origin_time/sequence as planned; the recon-function discipline (§8) is the load-bearing correctness work.

**§5 (Clock/horizon):** Keep dollar bars; **adaptive threshold**; **trade-count ablation**; flag time-capped bars. **Add ~20–30s rung; measure τ.** **Prove the decoupling** (fixed-bars vs fixed-physical-time ablation). Shared recon function + strict `<` apply-before-read + watermark + replay-equivalence test.

**§6 (Features):** **#1 = multi-level/integrated OFI** (+ cross-venue lagged OFI); **microprice** (feature + candidate target); **spread/tick as regime gate**. **Downgrade funding/OI/liquidations to conditioners.** Keep intra-bar path, book shape, CVD (supporting).

**§7 (Model):** Keep **conv encoder** (revert the transformer substitution). **Anneal prediction loss to a non-zero floor**, adaptively gated. **New monitoring battery** (LiDAR + predicted-vs-persistence + temporal-shuffle + probe-lift), not variance/RankMe alone. **Codebook bottleneck** (MTS-JEPA). **Harden predictor** (higher LR, eigenvalue floor, anti-identity). **Dual-encoder routing:** test online (higher-rank) vs EMA-target (smooth) encoder for our heads — the paper routes forecasting to the smooth EMA encoder, but sharp-event BTC detection (closer to anomaly detection / classification) may favor the discriminative online encoder. Add **physical-Δt FiLM conditioning** (§8).

**§8 (Labels):** Triple-barrier with **vol-scaled barriers**, vertical = physical horizon, **mid/microprice labels**. Expect heavy flat class. Meta-labeling only as no-trade gate.

**§9 (Training):** Add the critical control: **frozen CF-JEPA vs same-arch supervised-from-scratch**. Consider end-to-end-joint (per arXiv:2306.12086) as an arm.

**§10 (Validation/eval):** Keep PnL + no-trade band; add **MCC**, **honest taker fills**, **regime stratification**, **DSR>0.95**, **PBO/CSCV**. Gate is net-of-cost + DSR, not just "signal exists."

**§12 (Build order):** Insert a **baseline ladder** (below) as steps 5a–5e before CF-JEPA.

**Baseline ladder (do not skip rungs; each must clear the prior OOS net-of-cost):**
0. Naive martingale / random-walk / predict-zero null.
1. Penalized linear (Ridge/Elastic-Net) + DLinear/NLinear.
2. **LightGBM on engineered OFI/microprice features — THE GATE (and likely champion).**
3. Small supervised deep head (MLP/TiDE/PatchTST), end-to-end, no SSL.
4. **CF-JEPA pretrain → freeze → head**, compared three ways: frozen+head vs **same-arch supervised-from-scratch** vs LightGBM.

---

## 10. Open questions to resolve on our own data

1. **τ measurement** — decay of OFI→future-return predictivity vs horizon; set cap from the knee.
2. **Decoupling proof** — fixed-bar-count vs fixed-physical-time target: does the former's predictability variance explode across regimes?
3. **Bar clock** — dollar vs trade-count vs volume on *our* normality/homoscedasticity; quantify the time-per-bar distribution spread.
4. **Pretraining value** — does frozen CF-JEPA beat same-architecture supervised-from-scratch? (The only test that isolates SSL value.)
5. **SSL predictivity gate** — does the representation pass the temporal-shuffle / probe-lift-over-identity battery, or is it non-collapsed-but-empty?
6. **Premise increment** — does Binance signal beat a Coinbase-own-book model net of latency+cost? Re-check pre/post-ETF.
7. **Dual-encoder routing** — online vs EMA-target encoder for our (sharp-event) heads.
8. **Cost wall** — build the cost-adjusted PnL curve first; decide taker vs maker; is any edge outside the no-trade band?

---

## 11. Master reference list

**CF-JEPA & JEPA core**
- CF-JEPA: Mask-free forward prediction with asymmetric encoder utilization — Lee & Sim — *Knowledge-Based Systems*, 2026 — arXiv:2606.07031 *(read from PDF; not web-indexed at search time)*
- I-JEPA — Assran et al. — CVPR 2023 — arXiv:2301.08243
- Connecting JEPA with Contrastive SSL (C-JEPA, VICReg synergy) — 2024 — arXiv:2410.19560
- VICReg — Bardes, Ponce, LeCun — ICLR 2022 — arXiv:2105.04906
- Barlow Twins — Zbontar et al. — 2021 — arXiv:2103.03230
- Duality between contrastive & non-contrastive SSL — Garrido et al. — 2022 — arXiv:2206.02574

**Predictive/forward SSL for time series (closest prior art)**
- SimTS: Rethinking Contrastive Representation Learning for TS Forecasting — Zheng et al. — 2023 — arXiv:2303.18205
- What Constitutes Good Contrastive Learning in TS Forecasting? — Zhang et al. — 2023 — arXiv:2306.12086
- TS-JEPA / Joint Embeddings Go Temporal — Ennadir et al. — NeurIPS 2024 TSALM — arXiv:2509.25449
- MTS-JEPA: Multi-Resolution JEPA for TS Anomaly Prediction — 2026 — arXiv:2602.04643 [2026-preprint]
- TF-JEPA — ICLR 2026 TSALM — OpenReview 8bLa8PILyO
- FEI: Frequency-Masked Embedding Inference — Fu & Hu — AAAI 2025 — arXiv:2412.20790
- LaT-PFN (JEPA + PFN, in-context forecasting) — 2024 — arXiv:2405.10093
- TempSSL: Rethinking SSL for TS Forecasting — Zhao et al. — *Knowledge-Based Systems* 305, 2024 — DOI 10.1016/j.knosys.2024.112652

**General SSL-for-TS**
- TS2Vec — Yue et al. — AAAI 2022 — arXiv:2106.10466
- CoST — Woo et al. — ICLR 2022 — arXiv:2202.01575
- TimesURL — Liu & Chen — AAAI 2024 — arXiv:2312.15709
- SoftCLT — Lee et al. — ICLR 2024 — arXiv:2312.16424
- T-Rep (learnable time-embeddings) — Fraikin et al. — ICLR 2024 — arXiv:2310.04486
- SimMTM — Dong et al. — NeurIPS 2023 — arXiv:2302.00861
- TimeMAE — Cheng et al. — 2023 — arXiv:2303.00320

**Collapse theory & diagnostics**
- SimSiam — Chen & He — 2020 — arXiv:2011.10566
- DirectPred (SSL dynamics without contrastive pairs) — Tian, Chen, Ganguli — 2021 — arXiv:2102.06810
- Understanding Dimensional Collapse (DirectCLR) — Jing et al. — 2021 — arXiv:2110.09348
- Understanding Collapse in Non-Contrastive Siamese SSL — Li et al. — ECCV 2022 — arXiv:2209.15007
- RankMe — Garrido et al. — ICML 2023 — arXiv:2210.02885
- LiDAR — Apple — 2023 — arXiv:2312.04000
- Alignment & Uniformity — Wang & Isola — ICML 2020 — arXiv:2005.10242
- CPC / InfoNCE — van den Oord et al. — 2018 — arXiv:1807.03748
- SGD-induced representational drift — 2023 — arXiv:2302.02563

**Forecasting baselines / foundation models / skeptic evidence**
- Are Transformers Effective for TS Forecasting? (DLinear) — Zeng et al. — AAAI 2023 — arXiv:2205.13504
- PatchTST — Nie et al. — ICLR 2023 — arXiv:2211.14730
- Why do tree-based models still outperform DL on tabular data — Grinsztajn et al. — NeurIPS 2022 — arXiv:2207.08815
- Tabular Data: Deep Learning Is Not All You Need — Shwartz-Ziv & Armon — 2022 — arXiv:2106.03253
- Re(Visiting) TS Foundation Models in Finance — 2025 — arXiv:2511.18578
- TS Foundation Models for Multivariate Financial TS — 2025 — arXiv:2507.07296
- Empirical Asset Pricing via ML — Gu, Kelly, Xiu — RFS 2020
- FinCast (finance foundation model) — 2025 — arXiv:2508.19609
- MOMENT — Goswami et al. — ICML 2024 — arXiv:2402.03885
- Lag-Llama — Rasul et al. — 2023 — arXiv:2310.08278

**LOB / crypto microstructure DL**
- DeepLOB — Zhang, Zohren, Roberts — 2019 — arXiv:1808.03668
- Multi-Horizon LOB Forecasting (Seq2Seq/Attention) — Zhang & Zohren — 2021 — arXiv:2105.10430
- TransLOB — Wallbridge — 2020 — arXiv:2003.00130
- TABL — Tran et al. — 2019 — arXiv:1712.00975; BiN — arXiv:2003.00598
- Axial-LOB — Kisiel & Gorse — 2022 — arXiv:2212.01807
- HLOB — Briola et al. — 2024 — arXiv:2405.18938
- TLOB (dual attention; BTC) — 2025 — arXiv:2502.15757
- T-KAN (alpha decay within seconds) — 2026 — arXiv:2601.02310 [2026-preprint]
- **LOBFrame — Deep LOB Forecasting: a microstructural guide** — Briola, Bartolucci, Aste — 2024 — arXiv:2403.09267
- LOB DL benchmark study (FI-2010 overfit) — 2023 — arXiv:2308.01915
- **Explainable Patterns in Crypto Microstructure (Binance perp, 1s, OFI/SHAP, taker backtest)** — 2026 — arXiv:2602.00776 [2026-preprint]
- **Better Inputs Matter More Than Stacking Another Hidden Layer (Bybit BTC)** — 2025 — arXiv:2506.05764
- Deep Learning for Digital Asset LOBs (Coinbase BTC, 2s, 71%) — 2020 — arXiv:2010.01241
- Representation Learning of LOB (LOBench, SSL) — 2025 — arXiv:2505.02139
- DeepVol (volatility, dilated causal conv) — Moreno-Pino & Zohren — 2022 — arXiv:2210.04797

**Order flow / microprice / adverse selection**
- The Price Impact of Order Book Events (OFI) — Cont, Kukanov, Stoikov — 2014 — arXiv:1011.6402
- Cross-Impact of OFI (integrated multi-level OFI) — Cont, Cucuringu, Zhang — 2021 — arXiv:2112.13213
- The Micro-Price — Stoikov — 2018 — SSRN 2970694
- Queue Imbalance as a One-Tick-Ahead Predictor — Gould & Bonart — 2016 — arXiv:1512.03492
- A Model for Queue Position Valuation — Moallemi & Yuan — 2016
- The Market Maker's Dilemma: Fill Probability vs Post-Fill Returns — Albers et al. — 2025 — arXiv:2502.18625

**Information bars / clocks / impact**
- A Subordinated Stochastic Process Model… — Clark — Econometrica 1973
- Order Flow, Transaction Clock, and Normality of Asset Returns — Ané & Geman — J. Finance 2000
- The Volume Clock — Easley, López de Prado, O'Hara — 2012
- Flow Toxicity and Liquidity in a HF World (VPIN) — Easley, LdP, O'Hara — RFS 2012
- VPIN and the Flash Crash (critique) — Andersen & Bondarenko — 2014 [abstract-level]
- The Long Memory of the Efficient Market — Lillo & Farmer — 2004 — arXiv:cond-mat/0311053
- Price Impact (propagator model) — Bouchaud — 2009 — arXiv:0903.2428
- Algorithmic crypto trading using information-driven bars, triple-barrier & DL — *Financial Innovation* 2025 — DOI 10.1186/s40854-025-00866-w
- Advances in Financial Machine Learning, Ch.2 — López de Prado — 2018

**Labeling / validation / overfitting**
- The 10 Reasons Most ML Funds Fail — López de Prado — 2018 (SSRN 3104816)
- Advances in Financial ML (TBM, meta-labeling, purged CV, CPCV) — López de Prado — 2018
- The Deflated Sharpe Ratio — Bailey & López de Prado — 2014 (SSRN 2460551)
- The Probability of Backtest Overfitting (CSCV/PBO) — Bailey, Borwein, LdP, Zhu — 2015 (SSRN 2326253)
- Pseudo-Mathematics and Financial Charlatanism (Min Backtest Length) — Bailey et al. — Notices AMS 2014
- Backtesting (haircut Sharpe) — Harvey & Liu — 2015 (SSRN 2345489); …and the Cross-Section (HLZ) — 2016
- Does Meta-Labeling Add to Signal Efficacy? — Singh & Joubert (Hudson & Thames) — JFDS [pro, with caveats]
- Why Meta-Labeling Is Not a Silver Bullet — QuantConnect forum [secondary, skeptical]
- Optimal Trading with Linear Costs (no-trade band) — Bouchaud et al. — 2012 — arXiv:1203.5957
- Hidden Leaks in Time Series Forecasting (k-fold leakage magnitude) — 2025 — arXiv:2512.06932

**Cross-venue price discovery**
- Fragmentation, Price Formation, Cross-Impact in Bitcoin Markets — Albers et al. — 2021 — arXiv:2108.09750
- Price Discovery in Fragmented Crypto Markets (Regulation/Institutions/On-Chain) — Jang et al. — 2025/26
- Price Discovery in Bitcoin: Impact of Unregulated Markets — Alexander & Heck — 2020 [abstract-level]
- Nothing but Noise? Price Discovery Across Crypto Exchanges — Dimpfl & Peter — 2021 [abstract-level]
- Price Discovery in Crypto Markets (CME leads) — 2025 — arXiv:2506.08718
- Price Discovery in the Bitcoin ETF Market — Kia et al. — 2026 [abstract-level]
- Price Discovery in Bitcoin Spot or Futures? The Jury Is Out — Frino et al. — 2025 [abstract-level]
- Trading and Arbitrage in Cryptocurrency Markets — Makarov & Schoar — JFE 2020
- Can Funding Rate Predict Price Change? — Presto Research [secondary]

**Horizon conditioning / continuous-time / alignment**
- Time2Vec — Kazemi et al. — 2019 — arXiv:1907.05321
- FiLM — Perez et al. — 2017 — arXiv:1709.07871
- MetNet (lead-time FiLM conditioning) — Sønderby et al. — 2020 — arXiv:2003.12140
- mTAN — Shukla & Marlin — ICLR 2021 — arXiv:2101.10318
- Latent ODEs / ODE-RNN — Rubanova, Chen, Duvenaud — NeurIPS 2019 — arXiv:1907.03907
- Neural CDEs — Kidger et al. — NeurIPS 2020 — arXiv:2005.08926
- Neural CDEs for Online Prediction (causal) — Morrill et al. — 2021 — arXiv:2106.11028
- GRU-D — Che et al. — 2016 — arXiv:1606.01865
- ContiFormer — Chen et al. — NeurIPS 2023 — arXiv:2402.10635
- ACD — Engle & Russell — Econometrica 1998
- Neural Hawkes Process — Mei & Eisner — NeurIPS 2017 — arXiv:1612.09328
- Intensity-Free TPP (LogNormMix) — Shchur et al. — ICLR 2020 — arXiv:1909.12127
- Temporal Fusion Transformer (multi-horizon quantile) — Lim et al. — 2019 — arXiv:1912.09363
- Binance — How To Manage A Local Order Book Correctly (U/u/pu rules)
- Apache Flink — Timely Stream Processing (watermarks / bounded-out-of-orderness)

---

*Synthesis of an 8-thread parallel literature search (SSL/JEPA, LOB/crypto microstructure, collapse diagnostics, information bars/clocks, labeling/validation, cross-venue price discovery, low-SNR baselines, horizon-conditioning/alignment). Load-bearing, repeatedly-cited conclusions: (1) LightGBM-on-OFI is the bar to beat; (2) forecasting power ≠ profit → evaluate net-of-cost; (3) non-collapsed ≠ predictive → the anneal-to-zero is the collapse suspect; (4) our clock/horizon decoupling is novel and must be proven by ablation.*
