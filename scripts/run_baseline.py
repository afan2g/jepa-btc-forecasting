"""Run the G1 study on a real ModelMatrix parquet (bars E0.3 + labels E0.4 output).

Usage: .venv/bin/python scripts/run_baseline.py model_matrix.parquet feature_manifest.json
Manifest JSON: {"feature_cols": [...], "max_lookback_ns": <int>, "embargo_ns": <int>,
 "gate": {"n_groups": 6, "k": 2, "min_trades": 30, "min_eff_trades": 10,
          "min_sample_sharpe": 0.0, "dsr_thresh": 0.95, "pbo_thresh": 0.5}}   # gate REQUIRED
"""
import sys, json, pathlib
# Run as a bare script (`python scripts/run_baseline.py ...`): Python puts this file's
# own dir (scripts/) on sys.path, not the repo root, so put the repo root first to make
# the `eval` package importable. Harmless when already importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from eval.runner import run_from_manifest

def main(matrix_path, manifest_path):
    m = pd.read_parquet(matrix_path)
    man = json.load(open(manifest_path))
    res = run_from_manifest(m, man)
    print(f"resolved gate: {res['gate']}")                # echo the EFFECTIVE (resolved) config
    for h, out in res["horizons"].items():
        status = "PASS" if out["g1_pass"] else ("INCONCLUSIVE" if out["g1_inconclusive"] else "FAIL")
        print(f"\n=== horizon {h} ===  G1: {status}  (winner={out['winner']}, pbo={out['pbo']:.3f})")
        for name, r in out["rungs"].items():
            print(f"  {name:9s} gross={r['gross_pnl']:.1f} net={r['net_pnl']:.1f} "
                  f"cost_wall={r['cost_wall']:.1f} trade_sr={r['trade_sharpe']:.3f} "
                  f"sample_sr={r['sample_sharpe']:.3f} dsr={r['dsr']:.3f} "
                  f"turnover={r['turnover']:.3f} mcc={r['mcc']:.3f} trades={r['n_trades']} "
                  f"pass={r['passes_solo']}")
        for reg, r in out["per_regime"].items():
            print(f"  regime {reg:6s}: net={r['net_pnl']:.1f} sample_sr={r['sample_sharpe']:.3f} n={r['n']}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
