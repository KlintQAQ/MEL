from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from data.features import build_one
from model.mel import (
    FeatureSet, MatchedFlatMLP, RIDGE_ALPHAS, avg_corr, build_feature_cache,
    corr_per_roi, fit_ridge, set_seed, train_neural,
)

REPRESENTATIONS = {
    "canonical": ("Classical lagged bandpower / MEL-flat", "tokens_z"),
    "uniform": ("Uniform STFT bins", "stft_uniform"),
    "whole_history": ("Flattened PSD", "psd"),
    "morlet": ("Morlet wavelet energy", "morlet"),
}


def load_representation(canonical: Path, extra: Path, representation: str) -> FeatureSet:
    suffix = REPRESENTATIONS[representation][1]
    source = canonical if representation == "canonical" else extra
    with np.load(source, allow_pickle=False) as data:
        arrays = [data[f"{split}_{suffix}"].astype(np.float32) for split in ("train", "val", "test")]
    for array in arrays:
        if array.ndim < 2 or int(np.prod(array.shape[1:])) != 1040 or not np.isfinite(array).all():
            raise ValueError(f"Invalid 1040-dimensional features: {source.name}/{representation}")
        expected = (26, 40) if representation == "whole_history" else (8, 26, 5)
        if array.shape[1:] != expected:
            raise ValueError(f"Wrong coordinate layout for {representation}: {array.shape[1:]}, expected {expected}")
    return FeatureSet(*arrays, representation != "whole_history")


def score_row(scan: str, name: str, family: str, seed: int, val_pred: np.ndarray,
              test_pred: np.ndarray, val_y: np.ndarray, test_y: np.ndarray, **metadata) -> dict:
    per_roi = corr_per_roi(test_pred, test_y)
    return {
        "scan": scan, "representation": name, "family": family, "seed": seed,
        "val_avgR": avg_corr(val_pred, val_y), "test_avgR": float(np.nanmean(per_roi)),
        **{f"roi_{i}_R": float(value) for i, value in enumerate(per_roi)}, **metadata,
    }


def run_scan(scan: str, args: argparse.Namespace, config: dict, output: Path) -> list[dict]:
    canonical = args.canonical_dir / f"{scan}.npz"
    extra = args.feature_dir / f"{scan}.npz"
    if not canonical.exists():
        built = build_one(scan, args.aligned_dir, args.canonical_dir, fs=200, bin_sec=2)
        if built["status"] == "missing":
            raise FileNotFoundError(f"Missing aligned EEG cache for {scan}. Run data/prepare.py first.")
    if any(name != "canonical" for name in args.representations) and not extra.exists():
        build_feature_cache(scan, args.aligned_dir, args.canonical_dir, args.feature_dir, False)
    with np.load(canonical, allow_pickle=False) as data:
        train_y, val_y = [data[f"{split}_y"].astype(np.float32) for split in ("train", "val")]
    if train_y.ndim != 2 or train_y.shape[1] != 7 or val_y.shape[1] != 7:
        raise ValueError("The matched protocol requires seven target columns")
    selections, pending = [], []
    for representation in args.representations:
        name = REPRESENTATIONS[representation][0]
        features = load_representation(canonical, extra, representation)
        if len(features.train) != len(train_y) or len(features.val) != len(val_y):
            raise ValueError(f"Feature/target row mismatch for {scan}/{representation}")
        if "ridge" in args.readouts:
            val_pred, test_pred, alphas, state = fit_ridge(features, train_y, val_y, None)
            pending.append((name, "Ridge", 0, val_pred, test_pred, {"best_epoch": -1}))
            selections.append({"representation": name, "family": "Ridge", "alphas": alphas})
            np.savez_compressed(output / f"{representation}_ridge_state.npz", **state)
        if "mlp" in args.readouts:
            predictions = []
            for seed in args.seeds:
                set_seed(seed)
                model = MatchedFlatMLP(1040, 7, config["hidden_dim"], config["dropout"])
                val_pred, test_pred, epoch, count, val_score = train_neural(
                    model, features, train_y, val_y, None, torch.device(args.device), seed,
                    config["epochs"], config["patience"], config["learning_rate"],
                    config["weight_decay"], config["batch_size"],
                )
                pending.append((name, "Matched MLP", seed, val_pred, test_pred,
                                {"best_epoch": epoch, "n_parameters": count}))
                selections.append({"representation": name, "family": "Matched MLP", "seed": seed,
                                   "best_epoch": epoch, "validation_mean_R": val_score})
                predictions.append((val_pred, test_pred))
                torch.save(model.state_dict(), output / f"{representation}_mlp_seed{seed}.pt")
            if len(predictions) > 1:
                pending.append((name, "Matched MLP Ensemble", -1,
                                np.mean([item[0] for item in predictions], axis=0),
                                np.mean([item[1] for item in predictions], axis=0),
                                {"best_epoch": -1, "n_models": len(predictions)}))
    (output / "selection.json").write_text(json.dumps({"scan": scan, "config": config,
        "selection": "validation correlation only", "models": selections}, indent=2) + "\n", encoding="utf-8")
    with np.load(canonical, allow_pickle=False) as data:
        test_y = data["test_y"].astype(np.float32)
    rows = []
    for index, (name, family, seed, val_pred, test_pred, metadata) in enumerate(pending):
        if test_pred.shape != test_y.shape or not np.isfinite(test_pred).all():
            raise ValueError(f"Invalid prediction shape/values in {scan}/{name}/{family}")
        rows.append(score_row(scan, name, family, seed, val_pred, test_pred, val_y, test_y, **metadata))
        np.savez_compressed(output / f"prediction_{index:02d}.npz", validation=val_pred, test=test_pred)
    return rows


def compare_reference(frame: pd.DataFrame, output: Path, config: dict) -> dict:
    reference = pd.read_csv(ROOT / "reference_results/readout_scan_scores.csv")
    columns = ["val_avgR", "test_avgR", *[f"roi_{i}_R" for i in range(7)]]
    keys = ["scan", "family", "representation"]
    merged = frame.merge(reference[keys + columns], on=keys, suffixes=("", "_reference"), validate="one_to_one")
    for source, scores in (("result", frame[columns]),
                           ("reference", merged[[column + "_reference" for column in columns]])):
        invalid = ~np.isfinite(scores.to_numpy(dtype=np.float64))
        if invalid.any():
            invalid_columns = ", ".join(scores.columns[invalid.any(axis=0)])
            raise ValueError(f"Non-finite {source} scores in {invalid_columns}; reference verification failed")
    roi_tolerance = float(config["reference_ridge_roi_tolerance"])
    mean_tolerance = float(config["reference_ridge_mean_tolerance"])
    if not np.isfinite([roi_tolerance, mean_tolerance]).all() or min(roi_tolerance, mean_tolerance) < 0:
        raise ValueError("Reference tolerances must be finite and non-negative")
    records = []
    for _, row in merged.iterrows():
        maximum = max(abs(float(row[column]) - float(row[column + "_reference"])) for column in columns)
        mean_difference = abs(float(row.test_avgR) - float(row.test_avgR_reference))
        val_difference = abs(float(row.val_avgR) - float(row.val_avgR_reference))
        ridge_match = (maximum <= roi_tolerance
                       and max(mean_difference, val_difference) <= mean_tolerance)
        records.append({**{key: row[key] for key in keys}, "max_absolute_score_difference": maximum,
                        "test_mean_difference": mean_difference, "validation_mean_difference": val_difference,
                        "ridge_reference_match": bool(ridge_match) if row.family == "Ridge" else None,
                        "note": "declared mean/ROI tolerances; inspect numerical deltas" if row.family == "Ridge" else
                        "not certified as reference reproduction; CPU/GPU stochastic training differs"})
    pd.DataFrame(records).to_csv(output / "reference_comparison.csv", index=False)
    ridge = [item for item in records if item["family"] == "Ridge"]
    if len(ridge) != int((frame.family == "Ridge").sum()):
        raise ValueError("Some Ridge results have no bundled reference row")
    if any(not item["ridge_reference_match"] for item in ridge):
        raise RuntimeError("Ridge reproduction differs from recorded scores; inspect reference_comparison.csv")
    mlp = [item for item in records if item["family"] != "Ridge"]
    if mlp:
        print("[reference check] MLP runs completed, but CPU/CUDA score equivalence is not certified.", flush=True)
    return {"ridge_checked_rows": len(ridge), "ridge_within_declared_tolerances": True,
            "ridge_mean_tolerance": mean_tolerance,
            "ridge_roi_tolerance": roi_tolerance,
            "mlp_reference_equivalence": "not certified" if mlp else "not evaluated",
            "maximum_mlp_score_difference": max((item["max_absolute_score_difference"] for item in mlp), default=None)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper's four representations with matched Ridge/MLP readouts.")
    parser.add_argument("--scans", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/matched_readouts.json")
    parser.add_argument("--aligned-dir", type=Path, default=ROOT / "data/cache/vu_aligned_eeg")
    parser.add_argument("--canonical-dir", type=Path, default=ROOT / "data/cache/vu_bandpower_tokens")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "data/cache/vu_spectral_features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/matched_readouts")
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--readouts", nargs="+", choices=["ridge", "mlp"], default=["ridge", "mlp"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    for name in ("scans", "representations", "readouts"):
        items = getattr(args, name)
        if len(items) != len(set(items)):
            parser.error(f"Duplicate {name} are not allowed")
    if any(re.fullmatch(r"sub\d+-scan\d+", scan) is None for scan in args.scans):
        parser.error("Scan identifiers must use the form sub01-scan01")
    if args.threads < 1:
        parser.error("--threads must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.seeds = args.seeds or config["seeds"]
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("Use distinct fixed seeds")
    expected_alphas = np.logspace(config["ridge_alpha_min_log10"], config["ridge_alpha_max_log10"], config["ridge_alpha_count"])
    if not np.array_equal(expected_alphas, RIDGE_ALPHAS):
        parser.error("This entry point requires the matched 33-penalty grid")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    rows = []
    with threadpool_limits(limits=args.threads):
        for scan in args.scans:
            print(f"[matched readouts] {scan}", flush=True)
            scan_output = args.output_dir / scan
            scan_output.mkdir()
            rows.extend(run_scan(scan, args, config, scan_output))
    runs = pd.DataFrame(rows)
    runs.to_csv(args.output_dir / "per_run.csv", index=False)
    columns = ["val_avgR", "test_avgR", *[f"roi_{i}_R" for i in range(7)]]
    scans = runs.groupby(["scan", "family", "representation"], as_index=False)[columns].mean()
    scans.to_csv(args.output_dir / "per_scan.csv", index=False)
    versions = {package: importlib.metadata.version(package) for package in
                ("numpy", "scipy", "scikit-learn", "pandas", "torch")}
    manifest = {"protocol": config["protocol"], "config": config, "seeds": args.seeds,
                "device": args.device, "scans": args.scans, "readouts": args.readouts,
                "representations": args.representations, "threads": args.threads,
                "target_column_order": config["target_column_order"],
                "versions": versions, "elapsed_seconds": time.monotonic() - start,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "scope": "Reproduction on previously inspected retrospective splits, not fresh validation"}
    if args.check_reference:
        try:
            manifest["reference_check"] = compare_reference(scans, args.output_dir, config)
        except Exception as error:
            manifest["reference_check"] = {"status": "failed", "error": str(error)}
            raise
        finally:
            (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(scans[["scan", "family", "representation", "test_avgR"]].to_string(index=False))


if __name__ == "__main__":
    main()
