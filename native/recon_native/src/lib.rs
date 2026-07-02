//! Native Crypto Lake `book_delta_v2` seed/reseed replay core (docs/data.md §5a-Recon).
//!
//! This is the throughput implementation of `recon.reseed._replay_seeded`. Python remains the
//! correctness oracle and orchestrator: it owns column resolution, side decoding, and — crucially —
//! snapshot parse/thin/**classification** (`snapshots_from_lake_book_df` + `classify_snapshot`). This
//! core consumes COMPACT columnar arrays with *precomputed* per-snapshot `reason_code` + `is_valid`
//! and must NOT reimplement snapshot-validation precedence (plan §"Native API"). It only:
//!   * replays the ordered delta stream against a tick-keyed order book,
//!   * runs the seed/reseed state machine off the precomputed `is_valid`/`reason_code`,
//!   * accumulates crossed/missing/thin metrics and (optionally) the top-K frame.
//!
//! Semantics preserved exactly (see `recon/reseed.py`):
//!   * deltas sort by `(engine_time, sequence_number, original_row_index)` — a STABLE sort by
//!     `(ts, seq)` reproduces NumPy `np.lexsort((seq, ts))` equal-key row order;
//!   * at equal timestamp a delta is applied BEFORE a snapshot (a same-ts snapshot is authoritative);
//!   * samples are "as of" `sample_ts` (all events with `event_ts <= sample_ts` reflected);
//!   * `size == 0.0` removes the level; sizes are absolute;
//!   * crossed-duration is only accounted once seeded and the trailing open run closes at
//!     `max(last_event_ts, final_sample_ts)`.
//!
//! The book is keyed by INTEGER TICKS (`round(price * price_scale)`) for deterministic ordering and
//! fast best-bid/ask, but each level carries its ORIGINAL source float price so emitted
//! `bid_i_price`/`ask_i_price` are byte-identical to Python (never reconstructed as `tick / scale`).

use std::collections::BTreeMap;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Number of snapshot reason codes; must match `recon.native.REASON_CODES`.
pub const N_REASONS: usize = 7;
/// Sentinel `seed_reason_code` meaning "no snapshots seen yet" (Python `"no_snapshots"`).
pub const NO_SNAPSHOTS: i64 = 255;
/// Result-dict ABI version; must match `recon.native._META_ABI` (bumped in lockstep whenever the
/// fields `reconstruct_seeded` returns change). v2: per-sample coverage — `present_first_idx`/
/// `present_last_idx`/`invalid_runs_idx` (partial-day fill plan doc, Task 3).
pub const META_ABI: i64 = 2;

// ------------------------------------------------------------------------- order book
/// Tick-keyed L2 book side value: the ORIGINAL source float price and the absolute size.
type Level = (f64, f64);

/// One side of the book, keyed by integer tick (`round(price*scale)`). `BTreeMap` gives O(log L)
/// best-bid/ask (first/last key) and ordered top-K iteration WITHOUT a full-book scan per event —
/// the algorithmic fix over the Python `max(dict)/min(dict)` touch scans.
#[derive(Default)]
struct Book {
    bids: BTreeMap<i64, Level>,
    asks: BTreeMap<i64, Level>,
    scale: f64,
}

impl Book {
    fn new(scale: f64) -> Self {
        Book { bids: BTreeMap::new(), asks: BTreeMap::new(), scale }
    }

    #[inline]
    fn tick(&self, price: f64) -> i64 {
        (price * self.scale).round() as i64
    }

    /// Apply a delta. `size == 0.0` removes the level (mirrors `OrderBook.apply`).
    #[inline]
    fn apply(&mut self, is_bid: bool, price: f64, size: f64) {
        let key = self.tick(price);
        let side = if is_bid { &mut self.bids } else { &mut self.asks };
        if size == 0.0 {
            side.remove(&key);
        } else {
            side.insert(key, (price, size));
        }
    }

    /// Replace the whole state from a validated snapshot (mirrors `OrderBook.reseed`).
    fn reseed(&mut self, bids: &[Level], asks: &[Level]) {
        self.bids.clear();
        self.asks.clear();
        for &(p, s) in bids {
            self.bids.insert(self.tick(p), (p, s));
        }
        for &(p, s) in asks {
            self.asks.insert(self.tick(p), (p, s));
        }
    }

    /// Best bid = highest tick (last key). Returns the source `(price, size)`.
    #[inline]
    fn best_bid(&self) -> Option<Level> {
        self.bids.iter().next_back().map(|(_, v)| *v)
    }

    /// Best ask = lowest tick (first key). Returns the source `(price, size)`.
    #[inline]
    fn best_ask(&self) -> Option<Level> {
        self.asks.iter().next().map(|(_, v)| *v)
    }
}

// ------------------------------------------------------------------------- input records
pub struct DeltaRec {
    ts: i64,
    seq: i64,
    is_bid: bool,
    price: f64,
    size: f64,
}

pub struct SnapRec {
    ts: i64,
    reason: u8,
    is_valid: bool,
    bids: Vec<Level>,
    asks: Vec<Level>,
}

// ------------------------------------------------------------------------- outcome
/// Replay result. Metric fields mirror `recon.reseed._replay_seeded`'s `metrics` dict; the frame
/// vectors are row-major (`n_samples x k`) and only populated when `frame_out` is true.
#[derive(Default)]
pub struct Outcome {
    pub seed_accepted: bool,
    pub seed_ts: Option<i64>,
    pub seed_reason_code: i64,
    pub reseed_count: i64,
    pub reseed_ts: Vec<i64>,
    pub reseed_blocked: i64,
    pub reason_counts: [i64; N_REASONS],
    pub n_samples: i64,
    pub crossed_samples: i64,
    pub crossed_sample_ts: Vec<i64>,
    pub missing_book_samples: i64,
    pub thin_depth_samples: i64,
    pub crossed_duration_ns: i64,
    /// Per-sample coverage (plan-doc Task 3): maximal half-open `[i0, i1)` sample-INDEX runs where
    /// the sample fails the shared stitch-policy validity predicate (both top-of-book prices
    /// present, non-NaN, `bid < ask` — `valid_mask_from_frame` at `min_levels_per_side=1`), plus
    /// the first/last sample indices where both tops are present (non-NaN). Index pairs, not
    /// timestamps: the replay does not know the grid step; Python converts against its grid.
    pub invalid_runs_idx: Vec<(i64, i64)>,
    pub present_first_idx: Option<i64>,
    pub present_last_idx: Option<i64>,
    pub frame_out: bool,
    pub mid: Vec<f64>,
    pub microprice: Vec<f64>,
    pub bid_px: Vec<f64>,
    pub bid_sz: Vec<f64>,
    pub ask_px: Vec<f64>,
    pub ask_sz: Vec<f64>,
}

// ------------------------------------------------------------------------- replay core
/// Pure-Rust seed/reseed replay. Consumes UNSORTED deltas (sorted here) + snapshots with
/// precomputed `reason`/`is_valid`, and `sample_ts` ascending. This is the exact analogue of
/// `recon.reseed._replay_seeded` merged with `reconstruct_lake_l2_at_samples_seeded`.
#[allow(clippy::too_many_arguments)]
pub fn replay(
    mut deltas: Vec<DeltaRec>,
    mut snaps: Vec<SnapRec>,
    sample_ts: &[i64],
    k: usize,
    scale: f64,
    frame_out: bool,
    reseed_enabled: bool,
    reseed_after_crossed_ns: i64,
) -> Outcome {
    // STABLE sort by (ts, seq) => equal (ts, seq) rows keep source order == NumPy lexsort behaviour.
    deltas.sort_by(|a, b| (a.ts, a.seq).cmp(&(b.ts, b.seq)));
    snaps.sort_by(|a, b| a.ts.cmp(&b.ts)); // stable; equal-ts snapshots keep input order

    let n = sample_ts.len();
    let mut book = Book::new(scale);

    let mut out = Outcome { n_samples: n as i64, frame_out, seed_reason_code: NO_SNAPSHOTS, ..Default::default() };
    if frame_out {
        out.mid.reserve(n);
        out.microprice.reserve(n);
        out.bid_px.reserve(n * k);
        out.bid_sz.reserve(n * k);
        out.ask_px.reserve(n * k);
        out.ask_sz.reserve(n * k);
    }

    let mut si = 0usize; // next sample index
    let mut seeded = false;
    let mut crossed_since: Option<i64> = None;
    let mut last_t: Option<i64> = None;
    let mut invalid_open: Option<i64> = None; // start index of the currently-open invalid run

    // emit sample `g`: count crossed/missing/thin, track per-sample coverage (`si` is the current
    // grid index at every call site), and (if frame_out) push the top-K row.
    macro_rules! emit {
        ($g:expr) => {{
            let g: i64 = $g;
            let bb = book.best_bid();
            let ba = book.best_ask();
            // Coverage predicates, mirroring `recon.reseed._replay_seeded`: `present` = both tops
            // set and non-NaN (the notna predicate behind lake_present_*); `valid` additionally
            // requires uncrossed (`bid < ask`, false on NaN in both languages) — exactly
            // `valid_mask_from_frame` at min_levels_per_side=1.
            let (present, valid) = match (bb, ba) {
                (Some((bp, _)), Some((ap, _))) => (!bp.is_nan() && !ap.is_nan(), bp < ap),
                _ => (false, false),
            };
            if valid {
                if let Some(s0) = invalid_open.take() {
                    out.invalid_runs_idx.push((s0, si as i64));
                }
            } else if invalid_open.is_none() {
                invalid_open = Some(si as i64);
            }
            if present {
                if out.present_first_idx.is_none() {
                    out.present_first_idx = Some(si as i64);
                }
                out.present_last_idx = Some(si as i64);
            }
            match (bb, ba) {
                (Some((bp, bs)), Some((ap, as_))) => {
                    if bp >= ap {
                        out.crossed_samples += 1;
                        out.crossed_sample_ts.push(g);
                    } else if book.bids.len() < k || book.asks.len() < k {
                        out.thin_depth_samples += 1;
                    }
                    if frame_out {
                        out.mid.push((bp + ap) / 2.0);
                        out.microprice.push((as_ * bp + bs * ap) / (bs + as_));
                    }
                }
                _ => {
                    out.missing_book_samples += 1;
                    if frame_out {
                        out.mid.push(f64::NAN);
                        out.microprice.push(f64::NAN);
                    }
                }
            }
            if frame_out {
                let mut nb = 0usize;
                for (_, &(p, s)) in book.bids.iter().rev().take(k) {
                    out.bid_px.push(p);
                    out.bid_sz.push(s);
                    nb += 1;
                }
                for _ in nb..k {
                    out.bid_px.push(f64::NAN);
                    out.bid_sz.push(f64::NAN);
                }
                let mut na = 0usize;
                for (_, &(p, s)) in book.asks.iter().take(k) {
                    out.ask_px.push(p);
                    out.ask_sz.push(s);
                    na += 1;
                }
                for _ in na..k {
                    out.ask_px.push(f64::NAN);
                    out.ask_sz.push(f64::NAN);
                }
            }
        }};
    }

    // update_crossed(t): accumulate established-book crossed DURATION (only once seeded).
    macro_rules! update_crossed {
        ($t:expr) => {{
            if seeded {
                let is_crossed = match (book.best_bid(), book.best_ask()) {
                    (Some((bp, _)), Some((ap, _))) => bp >= ap,
                    _ => false,
                };
                if is_crossed {
                    if crossed_since.is_none() {
                        crossed_since = Some($t);
                    }
                } else if let Some(cs) = crossed_since {
                    out.crossed_duration_ns += $t - cs;
                    crossed_since = None;
                }
            }
        }};
    }

    // Two-pointer merge of (ts,seq)-sorted deltas with ts-sorted snapshots; at equal ts the delta
    // is processed first (delta.ts <= snap.ts), matching `_merge_time_ordered`.
    let (nd, ns) = (deltas.len(), snaps.len());
    let (mut di, mut sj) = (0usize, 0usize);
    loop {
        let next_is_delta = match (di < nd, sj < ns) {
            (true, true) => deltas[di].ts <= snaps[sj].ts,
            (true, false) => true,
            (false, true) => false,
            (false, false) => break,
        };
        let t = if next_is_delta { deltas[di].ts } else { snaps[sj].ts };

        while si < n && sample_ts[si] < t {
            emit!(sample_ts[si]);
            si += 1;
        }

        if next_is_delta {
            let d = &deltas[di];
            book.apply(d.is_bid, d.price, d.size);
            di += 1;
        } else {
            let s = &snaps[sj];
            out.reason_counts[s.reason as usize] += 1;
            let usable = s.is_valid;
            if !seeded {
                if usable {
                    book.reseed(&s.bids, &s.asks);
                    seeded = true;
                    out.seed_ts = Some(t);
                    out.seed_accepted = true;
                    out.seed_reason_code = 0; // "ok"
                } else if out.seed_reason_code == NO_SNAPSHOTS {
                    out.seed_reason_code = s.reason as i64; // first rejection cause
                }
            } else if reseed_enabled
                && crossed_since.is_some()
                && t - crossed_since.unwrap() >= reseed_after_crossed_ns
            {
                if usable {
                    book.reseed(&s.bids, &s.asks);
                    out.reseed_count += 1;
                    out.reseed_ts.push(t);
                } else {
                    out.reseed_blocked += 1;
                }
            }
            sj += 1;
        }
        update_crossed!(t);
        last_t = Some(t);
    }

    while si < n {
        emit!(sample_ts[si]);
        si += 1;
    }

    // Close a still-open crossed run at the END of the observed window (grid end or last event).
    let end_ts = if n > 0 { Some(sample_ts[n - 1]) } else { None };
    let close_t = match (last_t, end_ts) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    if let (Some(cs), Some(ct)) = (crossed_since, close_t) {
        if ct > cs {
            out.crossed_duration_ns += ct - cs;
        }
    }
    // Close a trailing open invalid run at the grid end (half-open at n).
    if let Some(s0) = invalid_open {
        out.invalid_runs_idx.push((s0, n as i64));
    }

    out
}

// ------------------------------------------------------------------------- PyO3 marshaling
/// Slice a flat concatenated level array into `n` per-snapshot `Vec<Level>` using `counts`.
fn slice_levels(px: &[f64], sz: &[f64], counts: &[i64]) -> Vec<Vec<Level>> {
    let mut out = Vec::with_capacity(counts.len());
    let mut off = 0usize;
    for &c in counts {
        let c = c as usize;
        let mut v = Vec::with_capacity(c);
        for j in 0..c {
            v.push((px[off + j], sz[off + j]));
        }
        off += c;
        out.push(v);
    }
    out
}

/// Native seed/reseed reconstruction. All arrays are C-contiguous (the Python wrapper guarantees
/// this). Returns a dict whose keys the `recon.native` wrapper assembles into the Python-compatible
/// `(frame, meta)`. See `recon/native.py`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn reconstruct_seeded<'py>(
    py: Python<'py>,
    ts: PyReadonlyArray1<'py, i64>,
    seq: PyReadonlyArray1<'py, i64>,
    side_is_bid: PyReadonlyArray1<'py, bool>,
    price: PyReadonlyArray1<'py, f64>,
    size: PyReadonlyArray1<'py, f64>,
    sample_ts: PyReadonlyArray1<'py, i64>,
    k: usize,
    price_scale: f64,
    frame_out: bool,
    snap_ts: PyReadonlyArray1<'py, i64>,
    snap_bid_px: PyReadonlyArray1<'py, f64>,
    snap_bid_sz: PyReadonlyArray1<'py, f64>,
    snap_bid_n: PyReadonlyArray1<'py, i64>,
    snap_ask_px: PyReadonlyArray1<'py, f64>,
    snap_ask_sz: PyReadonlyArray1<'py, f64>,
    snap_ask_n: PyReadonlyArray1<'py, i64>,
    snap_reason: PyReadonlyArray1<'py, u8>,
    snap_is_valid: PyReadonlyArray1<'py, bool>,
    reseed_enabled: bool,
    reseed_after_crossed_ns: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let ts = ts.as_slice()?;
    let seq = seq.as_slice()?;
    let side = side_is_bid.as_slice()?;
    let price = price.as_slice()?;
    let size = size.as_slice()?;
    let sample_ts_s = sample_ts.as_slice()?;

    let n_deltas = ts.len();
    let mut deltas = Vec::with_capacity(n_deltas);
    for i in 0..n_deltas {
        deltas.push(DeltaRec { ts: ts[i], seq: seq[i], is_bid: side[i], price: price[i], size: size[i] });
    }

    let snap_ts_s = snap_ts.as_slice()?;
    let s_bid_px = snap_bid_px.as_slice()?;
    let s_bid_sz = snap_bid_sz.as_slice()?;
    let s_bid_n = snap_bid_n.as_slice()?;
    let s_ask_px = snap_ask_px.as_slice()?;
    let s_ask_sz = snap_ask_sz.as_slice()?;
    let s_ask_n = snap_ask_n.as_slice()?;
    let s_reason = snap_reason.as_slice()?;
    let s_valid = snap_is_valid.as_slice()?;

    let bid_levels = slice_levels(s_bid_px, s_bid_sz, s_bid_n);
    let ask_levels = slice_levels(s_ask_px, s_ask_sz, s_ask_n);
    let n_snaps = snap_ts_s.len();
    let mut snaps = Vec::with_capacity(n_snaps);
    for i in 0..n_snaps {
        snaps.push(SnapRec {
            ts: snap_ts_s[i],
            reason: s_reason[i],
            is_valid: s_valid[i],
            bids: bid_levels[i].clone(),
            asks: ask_levels[i].clone(),
        });
    }

    // The GIL is not needed during the pure-Rust hot loop; detach from it so this can overlap
    // Python work (e.g. day-level parallelism, a follow-on) and never blocks other threads.
    let out = py.detach(|| {
        replay(deltas, snaps, sample_ts_s, k, price_scale, frame_out, reseed_enabled, reseed_after_crossed_ns)
    });

    let d = PyDict::new(py);
    d.set_item("seed_accepted", out.seed_accepted)?;
    d.set_item("seed_ts", out.seed_ts)?;
    d.set_item("seed_reason_code", out.seed_reason_code)?;
    d.set_item("reseed_count", out.reseed_count)?;
    d.set_item("reseed_ts", out.reseed_ts)?;
    d.set_item("reseed_blocked", out.reseed_blocked)?;
    d.set_item("reason_counts", out.reason_counts.to_vec())?;
    d.set_item("n_samples", out.n_samples)?;
    d.set_item("crossed_samples", out.crossed_samples)?;
    d.set_item("crossed_sample_ts", out.crossed_sample_ts)?;
    d.set_item("missing_book_samples", out.missing_book_samples)?;
    d.set_item("thin_depth_samples", out.thin_depth_samples)?;
    d.set_item("crossed_duration_ns", out.crossed_duration_ns)?;
    d.set_item("invalid_runs_idx", out.invalid_runs_idx)?;
    d.set_item("present_first_idx", out.present_first_idx)?;
    d.set_item("present_last_idx", out.present_last_idx)?;
    if frame_out {
        d.set_item("mid", PyArray1::from_vec(py, out.mid))?;
        d.set_item("microprice", PyArray1::from_vec(py, out.microprice))?;
        d.set_item("bid_px", PyArray1::from_vec(py, out.bid_px))?;
        d.set_item("bid_sz", PyArray1::from_vec(py, out.bid_sz))?;
        d.set_item("ask_px", PyArray1::from_vec(py, out.ask_px))?;
        d.set_item("ask_sz", PyArray1::from_vec(py, out.ask_sz))?;
    }
    Ok(d)
}

#[pymodule]
fn recon_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Native Crypto Lake book_delta_v2 seed/reseed replay core (docs/data.md §5a-Recon).")?;
    m.add("N_REASONS", N_REASONS)?;
    m.add("NO_SNAPSHOTS", NO_SNAPSHOTS)?;
    m.add("META_ABI", META_ABI)?;
    m.add_function(wrap_pyfunction!(reconstruct_seeded, m)?)?;
    Ok(())
}

// ------------------------------------------------------------------------- pure-Rust unit tests
#[cfg(test)]
mod tests {
    use super::*;

    fn d(ts: i64, seq: i64, is_bid: bool, price: f64, size: f64) -> DeltaRec {
        DeltaRec { ts, seq, is_bid, price, size }
    }
    fn snap(ts: i64, valid: bool, reason: u8, bids: Vec<Level>, asks: Vec<Level>) -> SnapRec {
        SnapRec { ts, reason, is_valid: valid, bids, asks }
    }

    #[test]
    fn seed_makes_pre_delta_samples_visible() {
        let deltas = vec![d(100, 1, true, 100.0, 1.0)];
        let snaps = vec![snap(0, true, 0, vec![(100.0, 2.0)], vec![(101.0, 3.0)])];
        let out = replay(deltas, snaps, &[50, 150], 1, 100.0, true, true, 0);
        assert!(out.seed_accepted);
        assert_eq!(out.seed_ts, Some(0));
        // sample @50 reflects the seed: mid = 100.5
        assert_eq!(out.mid[0], 100.5);
    }

    #[test]
    fn equal_ts_delta_applied_before_snapshot() {
        // A delta and a snapshot share ts=10. The delta posts bid 105 (would strand/cross), then the
        // same-ts snapshot OVERWRITES the whole book to a clean 100/101. Sample @10 must be as-of both
        // => uncrossed 100/101 (snapshot wins), proving delta-before-snapshot ordering.
        let deltas = vec![d(0, 1, true, 90.0, 1.0), d(10, 2, true, 105.0, 1.0)];
        let snaps = vec![
            snap(0, true, 0, vec![(90.0, 1.0)], vec![(91.0, 1.0)]),
            snap(10, true, 0, vec![(100.0, 1.0)], vec![(101.0, 1.0)]),
        ];
        let out = replay(deltas, snaps, &[20], 1, 100.0, true, true, 0);
        assert_eq!(out.bid_px_at(0), 100.0);
        assert_eq!(out.ask_px_at(0), 101.0);
    }

    #[test]
    fn stable_equal_ts_seq_keeps_source_order() {
        // Two absolute-size updates to the SAME (ts,seq,side,price); final size depends on order.
        // Source order sets 5.0 then 9.0 => final 9.0. A stable sort must preserve that.
        let deltas = vec![
            d(0, 1, true, 100.0, 1.0),
            d(0, 1, false, 101.0, 1.0),
            d(5, 7, true, 100.0, 5.0),
            d(5, 7, true, 100.0, 9.0),
        ];
        let out = replay(deltas, vec![], &[10], 1, 100.0, true, false, 0);
        assert_eq!(out.bid_sz_at(0), 9.0);
    }

    #[test]
    fn reseed_recovers_stranded_book() {
        let deltas = vec![
            d(10, 1, true, 102.0, 1.0),   // strands ask 101 => crossed
            d(100, 2, false, 101.0, 0.0),
            d(100, 3, false, 103.0, 1.0),
        ];
        let snaps = vec![
            snap(0, true, 0, vec![(100.0, 1.0)], vec![(101.0, 1.0)]),
            snap(30, true, 0, vec![(102.0, 1.0)], vec![(103.0, 1.0)]),
        ];
        let out = replay(deltas, snaps, &[5, 20, 50, 150], 1, 100.0, true, true, 0);
        assert_eq!(out.reseed_count, 1);
        assert_eq!(out.reseed_ts, vec![30]);
        assert_eq!(out.crossed_samples, 1); // only @20 remained crossed
    }

    #[test]
    fn terminal_crossed_duration_reaches_grid_end() {
        let deltas = vec![d(10, 1, true, 102.0, 1.0)]; // crosses at last event, stays crossed
        let snaps = vec![snap(1, true, 0, vec![(100.0, 1.0)], vec![(101.0, 1.0)])];
        let out = replay(deltas, snaps, &[5, 10, 20, 30], 1, 100.0, false, false, 0);
        assert_eq!(out.crossed_samples, 3);
        assert_eq!(out.crossed_duration_ns, 20); // 30 (grid end) - 10 (onset)
    }

    #[test]
    fn coverage_clean_day_has_no_invalid_runs() {
        let snaps = vec![snap(0, true, 0, vec![(100.0, 1.0)], vec![(101.0, 1.0)])];
        let out = replay(vec![], snaps, &[5, 15, 25], 1, 100.0, false, true, 0);
        assert!(out.invalid_runs_idx.is_empty());
        assert_eq!(out.present_first_idx, Some(0));
        assert_eq!(out.present_last_idx, Some(2));
    }

    #[test]
    fn coverage_leading_missing_prefix_is_one_run() {
        // Seed lands at ts=30: samples 0/10/20 see an empty book (missing => invalid, not present).
        let snaps = vec![snap(30, true, 0, vec![(100.0, 1.0)], vec![(101.0, 1.0)])];
        let out = replay(vec![], snaps, &[0, 10, 20, 40], 1, 100.0, false, true, 0);
        assert_eq!(out.invalid_runs_idx, vec![(0, 3)]);
        assert_eq!(out.present_first_idx, Some(3));
        assert_eq!(out.present_last_idx, Some(3));
    }

    #[test]
    fn coverage_trailing_crossed_run_closes_at_grid_end() {
        // Crossed samples are invalid but PRESENT: presence reaches the grid end while the
        // trailing invalid run stays open until closed half-open at n.
        let deltas = vec![d(10, 1, true, 102.0, 1.0)]; // crosses at 10, never heals
        let snaps = vec![snap(0, true, 0, vec![(100.0, 1.0)], vec![(101.0, 1.0)])];
        let out = replay(deltas, snaps, &[5, 15, 25], 1, 100.0, false, false, 0);
        assert_eq!(out.invalid_runs_idx, vec![(1, 3)]);
        assert_eq!(out.present_first_idx, Some(0));
        assert_eq!(out.present_last_idx, Some(2));
    }

    #[test]
    fn coverage_internal_gap_is_internal_run() {
        // The reseed_recovers_stranded_book fixture: crossed only at sample index 1 (@20).
        let deltas = vec![
            d(10, 1, true, 102.0, 1.0),
            d(100, 2, false, 101.0, 0.0),
            d(100, 3, false, 103.0, 1.0),
        ];
        let snaps = vec![
            snap(0, true, 0, vec![(100.0, 1.0)], vec![(101.0, 1.0)]),
            snap(30, true, 0, vec![(102.0, 1.0)], vec![(103.0, 1.0)]),
        ];
        let out = replay(deltas, snaps, &[5, 20, 50, 150], 1, 100.0, false, true, 0);
        assert_eq!(out.invalid_runs_idx, vec![(1, 2)]);
        assert_eq!(out.present_first_idx, Some(0));
        assert_eq!(out.present_last_idx, Some(3));
    }

    #[test]
    fn coverage_one_sided_book_is_missing_and_never_present() {
        let deltas = vec![d(10, 1, true, 100.0, 1.0)]; // bid-only book, never two-sided
        let out = replay(deltas, vec![], &[5, 15, 25], 1, 100.0, false, false, 0);
        assert_eq!(out.invalid_runs_idx, vec![(0, 3)]);
        assert_eq!(out.present_first_idx, None);
        assert_eq!(out.present_last_idx, None);
    }

    #[test]
    fn coverage_empty_grid_is_empty() {
        let out = replay(vec![d(10, 1, true, 100.0, 1.0)], vec![], &[], 1, 100.0, false, false, 0);
        assert!(out.invalid_runs_idx.is_empty());
        assert_eq!(out.present_first_idx, None);
        assert_eq!(out.present_last_idx, None);
    }

    // Small accessors used only by the tests above.
    impl Outcome {
        fn bid_px_at(&self, row: usize) -> f64 {
            self.bid_px[row]
        }
        fn ask_px_at(&self, row: usize) -> f64 {
            self.ask_px[row]
        }
        fn bid_sz_at(&self, row: usize) -> f64 {
            self.bid_sz[row]
        }
    }
}
