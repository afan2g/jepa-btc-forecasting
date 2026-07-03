import pathlib, pytest
import pandas as pd
from eval.manifest import load_manifest
from eval.runner import run_from_manifest

MATRIX = pathlib.Path("data/processed/model_matrix.parquet")
MANIFEST = pathlib.Path("data/processed/feature_manifest.json")

pytestmark = pytest.mark.skipif(not (MATRIX.exists() and MANIFEST.exists()),
    reason="needs real ModelMatrix + manifest from bars (E0.3) + labels (E0.4)")

def test_real_matrix_runs_through_manifest():
    m = pd.read_parquet(MATRIX)
    man = load_manifest(MANIFEST)                        # the real artifact must be v1
    res = run_from_manifest(m, man)                      # same path the CLI uses
    assert res["gate"] and res["horizons"]
    assert res["manifest"]["dataset_id"]                 # identity echoed for reproducibility
    any_h = next(iter(res["horizons"].values()))
    assert "g1_pass" in any_h and any_h["per_regime"]
