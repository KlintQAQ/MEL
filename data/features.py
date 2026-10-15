from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]
COPY_KEYS = [
    "target_epoch",
    "y",
    "epoch_index",
    "fmri_index",
    "scan_index",
    "subject_index",
]


def discover_scans(cache_dir: Path) -> list[str]:
    manifest = cache_dir / "cache_manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        return sorted(df["scan_id"].drop_duplicates().tolist())
    return sorted(path.stem for path in cache_dir.glob("*.npz"))


def bandpower_tokens(eeg: np.ndarray, fs: int = 200, bin_sec: int = 2) -> np.ndarray:
    if eeg.ndim != 3:
        raise ValueError(f"Expected EEG [N, C, T], got {eeg.shape}")
    n_epoch, n_chan, n_time = eeg.shape
    bin_samples = int(fs * bin_sec)
    n_bins = n_time // bin_samples
    usable = n_bins * bin_samples
    if n_bins <= 0:
        raise ValueError(f"EEG window {n_time} is shorter than one {bin_sec}s bin.")

    x = eeg[:, :, :usable].reshape(n_epoch, n_chan, n_bins, bin_samples).astype(np.float32)
    window = np.hanning(bin_samples).astype(np.float32)
    x = x * window[None, None, None, :]
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(bin_samples, d=1.0 / fs)

    band_features = []
    for _, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            raise ValueError(f"No FFT bins found for band {lo}-{hi}Hz.")
        band_features.append(spec[..., mask].mean(axis=-1))
    tokens = np.stack(band_features, axis=-1).transpose(0, 2, 1, 3)
    return np.log1p(tokens).astype(np.float32)


def zscore_by_train(tokens: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    train = tokens["train"]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return {split: ((value - mean) / std).astype(np.float32) for split, value in tokens.items()}, mean, std


def copy_split_arrays(source: np.lib.npyio.NpzFile, split: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for suffix in COPY_KEYS:
        key = f"{split}_{suffix}"
        if key in source.files:
            out[key] = source[key]
    return out


def build_one(scan: str, input_dir: Path, output_dir: Path, fs: int, bin_sec: int, force: bool = False) -> dict[str, Any]:
    src_path = input_dir / f"{scan}.npz"
    dst_path = output_dir / f"{scan}.npz"
    meta_path = output_dir / f"{scan}.json"
    if dst_path.exists() and meta_path.exists() and not force:
        return {"scan": scan, "status": "cached", "output_path": str(dst_path)}
    if not src_path.exists():
        return {"scan": scan, "status": "missing", "source_path": str(src_path)}

    source = np.load(src_path)
    raw_tokens = {
        split: bandpower_tokens(source[f"{split}_eeg"], fs=fs, bin_sec=bin_sec)
        for split in ("train", "val", "test")
    }
    z_tokens, mean, std = zscore_by_train(raw_tokens)

    output: dict[str, np.ndarray] = {
        "token_mean_train": mean.astype(np.float32),
        "token_std_train": std.astype(np.float32),
    }
    for split in ("train", "val", "test"):
        output[f"{split}_tokens"] = raw_tokens[split]
        output[f"{split}_tokens_z"] = z_tokens[split]
        output[f"{split}_tokens_flat_z"] = z_tokens[split].reshape(len(z_tokens[split]), -1).astype(np.float32)
        output.update(copy_split_arrays(source, split))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **output)

    metadata = {
        "dataset_id": "VU",
        "scan_id": scan,
        "source_cache": str(src_path),
        "output_cache": str(dst_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fs_hz": fs,
        "bin_sec": bin_sec,
        "window_sec": int(raw_tokens["train"].shape[1] * bin_sec),
        "bands": [{"name": name, "low_hz": lo, "high_hz": hi} for name, lo, hi in BANDS],
        "token_shape": {
            split: list(value.shape)
            for split, value in raw_tokens.items()
        },
        "flat_feature_dim": int(np.prod(raw_tokens["train"].shape[1:])),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "scan": scan,
        "status": "built",
        "output_path": str(dst_path),
        "train_shape": list(raw_tokens["train"].shape),
        "val_shape": list(raw_tokens["val"].shape),
        "test_shape": list(raw_tokens["test"].shape),
        "flat_feature_dim": metadata["flat_feature_dim"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached multi-band EEG neural-state tokens from aligned VU EEG epochs.")
    parser.add_argument("--input-dir", default="data/cache/vu_aligned_eeg")
    parser.add_argument("--output-dir", default="data/cache/vu_bandpower_tokens")
    parser.add_argument("--scans", nargs="*", default=None)
    parser.add_argument("--all-scans", action="store_true")
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--bin-sec", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    scans = discover_scans(input_dir) if args.all_scans or not args.scans else args.scans
    rows = []
    for scan in scans:
        print(f"[bandpower-cache] {scan}")
        rows.append(build_one(scan, input_dir, output_dir, fs=args.fs, bin_sec=args.bin_sec, force=args.force))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "cache_manifest.csv", index=False)
    summary = {
        "dataset_id": "VU",
        "n_scans_requested": len(scans),
        "n_scans_built_or_cached": int(manifest["status"].isin(["built", "cached"]).sum()),
        "n_missing": int((manifest["status"] == "missing").sum()),
        "output_dir": str(output_dir),
        "bands": [{"name": name, "low_hz": lo, "high_hz": hi} for name, lo, hi in BANDS],
        "token_layout": "[epoch, time_bin_2s, eeg_channel, frequency_band]",
    }
    (output_dir / "cache_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
