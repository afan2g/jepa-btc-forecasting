from eval.manifest import validate_frame, validate_manifest
from eval.synthetic import FEATURES, make_manifest, make_matrix


def test_make_manifest_is_schema_valid():
    validate_manifest(make_manifest(list(FEATURES), 10_000_000_000))


def test_make_manifest_matches_make_matrix_frame():
    df, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    validate_frame(df, make_manifest(feats, lb))  # helper mirrors the generator's contract
