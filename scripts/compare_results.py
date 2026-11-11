"""Compare a rerun's results with the published results.

    python scripts/compare_results.py [--reference work/published_results] [--results results]

Every file of the reference folder is compared with the file of the same name in the results folder:
identical bytes; or, for CSV and NPZ files, the largest difference of any numeric value. A value counts as
matching when |rerun - published| <= atol + rtol * |published| (defaults 1e-3 and 1e-4: the CSVs are
rounded to 3 to 5 decimals, so a rerun on other hardware can flip a last digit). Text columns must match
exactly. The reference files are first checked against the checksums in results/manifest.json, so a
reference copy that was itself overwritten by a rerun is caught.

Exit status 0 when every file matches, 1 otherwise.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_csv(ref, new, atol, rtol):
    a = pd.read_csv(ref, keep_default_na=False, na_values=[""], low_memory=False)
    b = pd.read_csv(new, keep_default_na=False, na_values=[""], low_memory=False)
    if list(a.columns) != list(b.columns):
        return False, f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if a.shape != b.shape:
        return False, f"shape {b.shape} instead of {a.shape}"
    worst, where = 0.0, ""
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x, y = x.to_numpy(float), y.to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                return False, f"missing values differ in column {col}"
            ok = ~np.isnan(x)
            excess = np.abs(y[ok] - x[ok]) - (atol + rtol * np.abs(x[ok]))
            if excess.size and excess.max() > worst:
                i = int(np.argmax(excess))
                worst, where = float(excess.max()), f"column {col}: {y[ok][i]} instead of {x[ok][i]}"
            continue
        if not x.astype(str).equals(y.astype(str)):
            i = int(np.flatnonzero(x.astype(str).to_numpy() != y.astype(str).to_numpy())[0])
            return False, f"text differs in column {col}, row {i + 2}: {y.iloc[i]!r} instead of {x.iloc[i]!r}"
    if worst > 0:
        return False, "beyond tolerance, largest in " + where
    return True, "within tolerance"


def compare_npz(ref, new, atol, rtol):
    with np.load(ref, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return False, f"arrays differ: {sorted(set(a.files) ^ set(b.files))}"
        for key in a.files:
            x, y = a[key], b[key]
            if x.shape != y.shape:
                return False, f"{key}: shape {y.shape} instead of {x.shape}"
            if x.dtype.kind in "fiu" and not np.allclose(y, x, atol=atol, rtol=rtol, equal_nan=True):
                return False, f"{key}: largest difference {np.nanmax(np.abs(y.astype(float) - x)):.3g}"
            if x.dtype.kind not in "fiu" and not np.array_equal(x, y):
                return False, f"{key}: values differ"
    return True, "within tolerance"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=ROOT / "work" / "published_results")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text())["results"]
    refs = sorted(p for p in args.reference.iterdir() if p.suffix in (".csv", ".npz"))
    if not refs:
        sys.exit(f"No reference files in {args.reference}")
    failures = 0
    for ref in refs:
        published = manifest.get(f"results/{ref.name}")
        new = args.results / ref.name
        if published is not None and sha(ref) != published:
            status, ok = "reference copy is not the published file (checksum differs from the manifest)", False
        elif not new.exists():
            status, ok = "missing from the rerun", False
        elif sha(ref) == sha(new):
            status, ok = "identical", True
        elif ref.suffix == ".csv":
            ok, status = compare_csv(ref, new, args.atol, args.rtol)
        elif ref.suffix == ".npz":
            ok, status = compare_npz(ref, new, args.atol, args.rtol)
        else:
            status, ok = "differs", False
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {ref.name:32s} {status}")
    print(f"\n{len(refs) - failures} of {len(refs)} files match the published results")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
