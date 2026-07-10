# JEPA BTC/Coinbase Forecasting — Implementation Spec

**Purpose.** This is a design + implementation handoff for a coding agent. It captures *what* to build, *why* each choice was made, and the failure modes we already reasoned through so you don't reintroduce them. Where a decision is deliberate, it's marked **Why:** — do not "optimize" those away without flagging.

**Status.** Greenfield. Nothing built yet. Build order is in §12. Read §5 (clock) and §10 (validation) carefully — they contain the two most counterintuitive, project-defining decisions.

---

## 1. Objective & scope

- **Goal:** forecast the short-horizon (seconds → minutes) move of **Coinbase BTC-USD** mid price, using **Binance** (the leading, highest-information venue) as the primary signal source.
- **Output:** a normalized forward return (bps) and/or a triple-barrier label at a fixed physical horizon. Multiple horizons (e.g. 2s / 10s / 60s).
- **Use:** a statistical directional / fair-value signal. **Not** an HFT latency play. The same fair-value estimate can feed a separate BRTI / Kalshi-BTC workflow.
- **Explicitly out of scope:** sub-second latency arbitrage, co-location, order routing optimization, the execution/OMS layer (a thin paper-trading harness is enough for evaluation).

**Why Binance→Coinbase and not Coinbase→Coinbase:** at seconds-to-minutes the venues are cointegrated and any mechanical lead-lag has already reconciled, so this is *not* a lead-lag arb. Binance matters because it's the deepest, highest-volume, most information-rich BTC venue (spot + perp + funding + liquidations); its order flow carries information about where BTC is going next, and Coinbase moves with BTC. We forecast the common asset's move via the best information source.

---

## 2. Key decisions at a glance

| Decision | Choice | Section |
|---|---|---|
| JEPA variant | CF-JEPA (forward-prediction, mask-free) | §7 |
| Input sampling clock | Notional (dollar) bars, time-capped | §5 |
| Clock trigger stream | Trade stream, **not** book-update stream | §5 |
| Target horizon | Fixed **physical time**, decoupled from input clock | §5 |
| Cross-stream alignment | Event-time reconstruction on merged engine-time axis | §5 |
| Anti-collapse | EMA target + stop-grad **plus** VICReg | §7 |
| Labels | Triple-barrier, purged + embargoed CV | §8 |
| Training | SSL-pretrain encoder → freeze → small heads | §9 |
| First milestone | Supervised baseline (LightGBM) BEFORE any JEPA | §10, §12 |
| Eval metric | Fees-included PnL w/ no-trade band, not accuracy | §10 |
| Data: Binance | Crypto Lake `book_delta_v2` | §4 |
| Data: Coinbase | Tardis or self-capture (Crypto Lake too gappy) | §4 |
| Model size | ~5–15M params (start small) | §7 |

---

## 3. Repo / module layout (suggested)

```
ingest/      # vendor download + live WS capture, raw parquet archive
recon/       # event-time order-book reconstruction from deltas (Rust preferred)
bars/        # notional-bar sampler + feature engineering -> training tensors
data/        # dataset, windowing, purged/embargoed CV splits, labels
model/        # CF-JEPA encoder/target/predictor, losses, heads
train/        # SSL pretrain loop, head finetune loop, configs
eval/         # supervised baseline, backtest harness, PnL/no-trade-band metrics
```

**Why Rust for `recon`/`bars`:** book reconstruction + feature engineering over ~TB of ticks is the real wall-clock cost (not GPU training). It's CPU/IO-bound and embarrassingly parallel per (day, instrument). Python is fine for `model`/`train`/`eval`.

---

## 4. Data layer

### Venues & instruments
- **Signal:** Binance `BTCUSDT` **perpetual** (primary) + Binance `BTCUSDT` spot (secondary). Perp leads price discovery; funding/basis/OI/liquidations are extra signal.
- **Target:** Coinbase `BTC-USD` spot (the venue we label and trade).

### Vendor decision (verified against crypto-lake.com docs)
- **Binance → Crypto Lake.** Use `book_delta_v2` (true incremental L2, 1000+ levels, with `sequence_number`), `trades`, `funding`, `open_interest`, `liquidations`. Binance/Binance-futures book history starts **2022-11-14** (≥3 yr available; covers the 12–24 mo window). Crypto Lake captures in **AWS Tokyo**, next to Binance's engine, so `received_time` is a tight proxy for exchange time. $64/mo individual plan, Parquet on S3 + Python API. **Why:** structurally equivalent to Tardis for Binance, much cheaper, well-located.
- **Coinbase → Tardis.dev or self-capture.** Crypto Lake's Coinbase order book is only **~80% covered with large gaps** and isn't a first-class feed. For the venue we *label and trade against*, contiguous gaps fragment labels and bias regime coverage. Either buy Coinbase from Tardis (near-complete, history to 2019, exchange timestamps) or capture it live yourself off the free Coinbase Advanced Trade WS (`level2` + `market_trades`) going forward and backfill only what's needed from Tardis. **Mix vendors per venue** — don't pay all-Tardis for Binance data Crypto Lake already nails.

### Coinbase `level2` feed mechanics (for self-capture)
- `level2` sends a `snapshot` then `update` messages; each update is `{price_level, new_quantity, side, event_time}`.
- **`new_quantity` is the absolute size at that level, not a delta. `0` = remove the level.** Reconstruction is replace-keyed-on-price.
- Log local receipt time alongside `event_time` to measure/feed-lag and to reconstruct on exchange time.

### History span & split
- SSL pretrain: **12–24 months** (book dynamics are more stationary than alpha → more data helps the representation).
- Head finetune: recent **3–6 months**.
- Hold out a clean, contiguous, recent **~1 month** OOS, never touched; walk forward.
- **Why not >2–3 yr:** BTC microstructure drifts (perp dominance, ETF flows, fee changes); stale data can hurt. Weight recent data; use the long span for SSL only.

### Storage
- Raw archive: ~1–4 GB/day compressed across the stack → ~0.5–3 TB for 1–2 yr. Cheap SSD or S3.
- Processed bars+features: collapses to **GB-scale** → fits in RAM, no dataloader bottleneck.

### ⚠️ Verify on free samples before committing
1. Whether `origin_time` (exchange timestamp) is actually **populated** in Crypto Lake `book_delta_v2` for Binance — the column exists in the schema with real example values, but their Notes warn order-book `origin_time` is *often* empty (`0`/`-1`) and is feed-dependent. If populated, reconstruct on exchange time and capture location stops mattering. If empty, fall back to `received_time`.
2. The actual extent/locations of the Coinbase BTC-USD gaps in your target window.

---

## 5. Sampling / clock — THE critical subsystem

This is the most refined and most counterintuitive part. Read fully.

### 5.1 Input bars: notional (dollar) clock
Emit a bar each time cumulative **traded notional** crosses a threshold.
- **Clock off the *trade* stream, never the book-update stream. Why:** book-update events are dominated by quote churn and spoofing — the exact uninformative noise JEPA is meant to discard. Trades are realized aggression; they carry information.
- **Dollar, not volume. Why:** BTC ranges 2x+ across a training window; a fixed *volume* threshold carries very different information at $40k vs $100k. Dollar bars give the most homoscedastic, closest-to-Gaussian increments (subordination result), sample densely exactly when information arrives, and fight collapse (each bar = a roughly constant information quantum, so consecutive bars carry real change).
- Drive off Binance-perp notional (or combined Binance+Coinbase notional). Snapshot **both** books at each trigger.
- Tune the threshold for **~1 bar per 0.5–2s** of normal activity.

### 5.2 Hybrid time cap
Bars are "`$X` notional **or** `T` seconds, whichever first" (e.g. T ≈ 2–5s).
- **Why:** a pure notional clock can let a dead Sunday produce a single 45-minute bar. The cap bounds worst-case input heteroscedasticity without giving up notional pacing in active periods.

### 5.3 Event-time reconstruction (resolves the async-stream problem)
Trades and L2 updates arrive on **separate, asynchronous channels**. Do **not** snapshot the in-memory book at the instant a trade print arrives — the book may not have caught up.
- Merge trade + book-delta streams onto **one ordered engine-time axis** using `origin_time`/`sequence_number` (fall back to `received_time` only if origin_time is unpopulated — see §4 verification).
- Define the snapshot at engine-time `T` as the book inclusive of all events with timestamp ≤ `T`.
- Pick a **fixed convention**: is the snapshot *before* or *after* the triggering trade's book impact? Apply it identically in training and live.
- **Offline (training): trivial** — you have the complete ordered streams, so merge-sort by timestamp/sequence; there is no "lagging buffer."
- **Live:** add a few-ms watermark delay to wait for stragglers (free at this horizon).
- **Why this matters:** this is a general multi-stream alignment problem (affects *any* clock), and naive implementation silently injects lookahead/misalignment.

### 5.4 ⭐ DECOUPLE the input clock from the target horizon
The *input* is sampled in notional bars. The *forward target* is defined at a fixed **physical-time** offset (e.g. Δt = 10s), **not** at a fixed number of bars ahead.
- **Why (the core insight):** with a notional clock, a fixed *block-count* horizon spans wildly different physical time across regimes — ~50ms in a volatile cascade (highly predictable) vs ~45min on a quiet night (zero short-term predictability). That makes the target's predictability swing by orders of magnitude. A dataset full of unpredictable targets doesn't just waste capacity — it creates **collapse pressure** (the cheapest way to lower loss on unpredictable samples is to shrink the variance of their target embeddings, collapsing the quiet-regime subspace).
- Cap the target horizon near the empirical microstructure **decay window τ**, where predictability actually exists.
- **Condition the predictor on the physical Δt to target** (and on per-bar elapsed time), so it sharpens in bursts and regresses to the mean when the gap is long. This turns the variable horizon from a bug into a conditioning variable. (CF-JEPA's predictor is already horizon-conditioned; make that conditioning physical-time-aware, not block-count-aware.)
- Net: notional bars for *features* (homoscedastic, composition-rich inputs), physical-time-capped for the *label*.

---

## 6. Feature engineering — per-bar vector

Each bar is a feature vector (the encoder ingests a sequence of these, not raw book tensors). **All features stationarized:** prices as tick/bp offsets from mid; sizes log- or rolling-z-scored; targets as normalized returns.

- **Book shape:** top-K (e.g. K=10–20) level price offsets-from-mid + normalized sizes (both sides), spread, microprice, multi-level depth imbalance, book slope.
- **Trade-flow composition (within the bar):** trade count, signed volume / CVD increment, aggressor imbalance, largest print, VWAP−mid, a couple of trade-size-distribution moments.
  - **Why this set specifically:** a single $500k sweep and 1,000 × $500 retail flickers have identical notional but opposite alpha. They only "look identical" if you under-feature the bar. With trade_count, largest_print, and aggressor imbalance the two become maximally separable. A JEPA encoder mapping different compositions to different latents is the encoder doing its job — this is signal, not instability.
- **Intra-bar path:** realized variance within the bar, max adverse excursion. **Why:** a coarse bar can average away a sweep-then-refill into something that looks calm; path features recover that. (Also a reason to keep bars small.)
- **Cross-venue:** the full book+flow set for **both** Binance and Coinbase, plus the **Binance−Coinbase mid spread**.
- **Perp state:** funding rate, basis (perp−spot), OI change; optionally liquidation flags/intensity.
- **Time:** elapsed wall-clock duration of the bar, time-of-day encoding.

---

## 7. Model — CF-JEPA (forward-prediction)

### 7.1 Variant choice & rationale
- **I-JEPA** (vision) / **V-JEPA** (video): masked-region latent prediction; the original recipe. Not a forecasting setup.
- **TS-JEPA**: masking-based time-series JEPA (high mask ratio, patch tokens). Works, but masking inherits a continuity problem at mask boundaries that's awkward against a *causal* forecasting target.
- **CF-JEPA (chosen):** mask-free; replaces masking with **multi-horizon forward prediction** (predict future-window embeddings from a past-window context). **Why:** this *is* the forecasting use case, it's causal, and it avoids the masking continuity issue. Multi-horizon forward targets from one encoder map directly onto our 2s/10s/60s band.
- **MTS-JEPA** (multi-resolution): a good later extension (fine + coarse tokens) once the baseline works; not the starting point.

### 7.2 Architecture (start small — low SNR overfits)
- **Input:** sequence of per-bar feature vectors; context window **~128–256 bars**; patch PatchTST-style (**4–8 bars/token**) to cut sequence length.
- **Context encoder `f_θ`:** small **causal** Transformer — `d_model` 128–256, **4–6 layers**, 4–8 heads, **~5–15M params**. Causal/unidirectional (live, you only see the past).
- **Target encoder `f_θ̄`:** **EMA copy** of `f_θ` with **stop-gradient**. Encodes the future bar-window(s).
- **Predictor `g_φ`:** deliberately **lightweight** (narrower than the encoder); inputs = context embedding + **physical-Δt / horizon query**; outputs predicted target embedding(s), multi-horizon in one shot.
- **Loss:** smooth-L1 (or L2) between predicted and `sg(EMA-target)` embeddings, **plus VICReg variance + covariance regularization**.
  - **Why VICReg on top of EMA+stop-grad:** the standard I-JEPA anti-collapse (EMA target + stop-grad) is not enough in the near-martingale regime, where "copy the present" is a seductive degenerate solution, and where unpredictable quiet-regime samples create extra collapse pressure (§5.4). Variance reg forces the latents to spread. Starting coeffs: variance≈25, covariance≈1, prediction(inv)≈25 — tune.

### 7.3 Downstream heads
- After SSL pretraining, **freeze** the encoder (or lightly finetune), attach a **small MLP head** per horizon/target on the pooled representation.
- Heads are cheap → train several (2s/10s/60s, return vs triple-barrier, etc.).

---

## 8. Targets & labeling
- **Triple-barrier** labels (take-profit / stop / time barrier) at the fixed physical horizon(s). Gives directional labels and Kelly-sizing inputs.
- **Purged + embargoed cross-validation** (López de Prado). **Why:** bar labels at these horizons overlap, so naive CV leaks and inflates results badly.
- Target values normalized (returns in bps), never raw prices.

---

## 9. Training recipe
1. SSL-pretrain `f_θ` (CF-JEPA, §7) on the long **unlabeled** span (12–24 mo of bars).
2. Freeze `f_θ`.
3. Train small head(s) on the recent **labeled** window (3–6 mo) with purged/embargoed CV.
4. Walk-forward: re-pretrain encoder occasionally (e.g. monthly), retrain heads frequently (cheap) for drift.

---

## 10. Validation & evaluation — read before building the model

### 10.1 Supervised baseline FIRST (non-negotiable, milestone 0)
Before any JEPA: build the bar pipeline + features, then fit a **dead-simple supervised baseline** — LightGBM or a small supervised transformer predicting forward return from the same features, with purged CV and a fees-included PnL metric.
- **Why:** it confirms there's *any* signal at your horizon, sets the benchmark JEPA must beat, and shakes out the data/label/CV plumbing. **If the supervised baseline shows no edge, JEPA will not conjure one** — it adds representation quality + multi-horizon transfer on top of a real signal; it does not manufacture signal from noise.
- **Stage the data spend:** follow [`docs/superpowers/plans/2026-07-10-staged-signal-acquisition.md`](docs/superpowers/plans/2026-07-10-staged-signal-acquisition.md). Run a Coinbase-only preliminary screen, then a matched six-month Coinbase/Binance pilot before acquiring the remaining Binance archive. These are acquisition screens, not substitutes for the formal full-data G1. Weak Coinbase-own-book signal alone does not refute the primary Binance→Coinbase hypothesis.

### 10.2 Metric
- **Fees-included PnL with a no-trade band:** act only when `|forecast| > cost + margin`. The round-trip fee defines a no-trade band around the forecast.
- **Not** classification accuracy. **Why:** high forecasting power ≠ tradeable edge (the LOBFrame finding) — a model can nail next-tick direction and still not clear spread + fees + queue position.
- Coinbase Advanced fee context (for the cost model): taker ranges from ~1.20% (base) down to ~0.05% only at very high volume; maker is much lower (→0% at top tiers). At realistic solo volume, a patient-maker or selective-taker stance is the only economic option; treat fees as the minimum-signal filter.

---

## 11. Hardware & compute
- **Training:** RTX 3070 (8GB) is sufficient for a 5–15M model on GB-scale features (use grad accumulation, bf16, modest context). A single SSL run is **hours → a couple days** on the 3070 depending on model size / window overlap / epochs. Rent a single 24GB GPU (A10G/L4/4090/A100, ~$0.3–2/hr) for sweeps. **No multi-GPU / A100 cluster needed** — that's for models 10–100x larger.
- **Data pipeline:** the real wall-clock cost; CPU/IO-bound, parallel per (day, instrument), Rust.
- **Inference:** trivial; CPU-fine at seconds-to-minutes (sub-ms to low-ms forward pass). Run the live loop (Binance ingest → features → forward → signal → Coinbase order) on one small box, no GPU.
- **Heads:** minutes to train.

---

## 12. Suggested build order
1. **Ingest + staged archive:** verify bounded vendor samples and produce raw Parquet partitioned by (exchange, symbol, day). Complete the Coinbase pilot inputs first; acquire the predeclared six-month Binance pilot next; acquire the remaining 12–24-month Binance archive only after the cross-venue spend gate passes.
2. **Event-time reconstruction (`recon`):** merge trades + book deltas on engine-time; produce a consistent book-state-at-T API + replay.
3. **Notional-bar sampler + features (`bars`):** §5 clock (with time cap), §6 feature vector. Support explicit Coinbase-only and matched cross-venue datasets; never zero-fill an unavailable venue. Output model-ready tensors plus versioned feature manifests.
4. **Labels + purged/embargoed CV (`data`).**
5. **Supervised baseline (`eval`):** run the Coinbase-only preliminary screen, then the matched six-month cross-venue acquisition gate. After approved full-data acquisition, run formal G1: LightGBM on explicit features → forward return; purged CV; fees-included PnL + no-trade band. **Gate: is there signal?**
6. **CF-JEPA pretraining (`model`/`train`):** §7. Sanity-check for collapse (monitor embedding variance / VICReg terms).
7. **Heads + comparison:** does the SSL representation beat the supervised baseline **after costs**?
8. **Walk-forward + (later) extensions:** MTS-JEPA multi-resolution, imbalance/run bars, richer cross-venue context.

---

## 13. Pitfalls & gotchas (consolidated)
- **Stream async / lookahead:** never snapshot the live in-memory book on trade arrival; reconstruct on engine time (§5.3).
- **Martingale collapse:** consecutive fine-tick snapshots are near-identical → "copy the present" degenerate solution. Mitigate with notional bars + VICReg + physical-time targets.
- **Quiet-regime collapse pressure:** unpredictable long-physical-gap targets push the encoder to collapse that subspace (§5.4) — the physical-time-capped target + VICReg address this.
- **Composition blindness:** equal-notional bars hide sweep-vs-churn unless you add trade-flow composition + intra-bar path features (§6).
- **Volume vs dollar bars:** use dollar; volume bars drift with price level.
- **CV leakage:** overlapping labels → must purge + embargo.
- **Backtest lookahead:** if you ever add cross-venue timing, lag features by realistic loop latency; align on a single disciplined clock, not raw exchange timestamps from two venues.
- **Forecasting ≠ tradeable:** evaluate on fees-included PnL, not accuracy.
- **Over-provisioning compute:** resist the A100-cluster instinct; this is a small model.

---

## 14. Glossary / references
- **JEPA** — Joint-Embedding Predictive Architecture: predict in latent space, not input space; EMA target encoder + stop-grad; non-contrastive, no augmentations. (LeCun 2022; I-JEPA Assran et al. 2023.)
- **CF-JEPA** — Crop-based Forward JEPA: mask-free, multi-horizon forward prediction (chosen variant).
- **TS-JEPA / MTS-JEPA** — masking-based / multi-resolution time-series JEPA.
- **VICReg** — variance-invariance-covariance regularization (anti-collapse).
- **PatchTST** — patching scheme for time-series transformers.
- **Triple-barrier / purged CV** — López de Prado, *Advances in Financial Machine Learning*.
- **Subordination / volume clock** — Clark 1973; Ané & Geman 2000; dollar/information-driven bars (López de Prado).
- **Vendors** — Crypto Lake (crypto-lake.com): `book_delta_v2`, Tokyo capture, cheap; Tardis.dev: comprehensive incremental L2, better Coinbase coverage/timestamps.

---

*This spec is the synthesis of a design discussion. The clock decoupling (§5.4), event-time reconstruction (§5.3), and "supervised baseline first" (§10.1) are the load-bearing, hard-won decisions — preserve them.*
