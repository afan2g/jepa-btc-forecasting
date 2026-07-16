"""Run the G1 study on a real ModelMatrix parquet (bars E0.3 + labels E0.4 output).

Usage: .venv/bin/python scripts/run_baseline.py model_matrix.parquet feature_manifest.json
The manifest must be a v1 feature manifest (docs/feature-manifest.md) and must include
the pre-registered "gate" block. Legacy {feature_cols, embargo_ns, max_lookback_ns, gate}
dicts are NOT accepted here: write a v1 manifest and pre-register it.
"""
import sys, pathlib
# Run as a bare script (`python scripts/run_baseline.py ...`): Python puts this file's
# own dir (scripts/) on sys.path, not the repo root, so put the repo root first to make
# the `eval` package importable. Harmless when already importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from eval.guard import guarded_read_matrix, preflight_generic_manifest
from eval.manifest import load_manifest
from eval.runner import resolve_gate, run_from_manifest

def main(matrix_path, manifest_path):
    man = load_manifest(manifest_path)   # v1 schema-validated; fails before the parquet read
    preflight_generic_manifest(man)      # 67-D holdout guard (#90): refuses holdout-bound
    resolve_gate(man)                    # gate errors also surface before the parquet read
    m = guarded_read_matrix(matrix_path, man)   # guard re-runs at the read boundary itself
    res = run_from_manifest(m, man)
    ident = res["manifest"]
    print(f"manifest: {ident['dataset_id']} / {ident['build_id']} "
          f"({len(ident['feature_cols'])} features)")
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
