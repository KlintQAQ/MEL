from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROI_COLUMNS = [
    "Thalamus", "Putamen", "Cuneus", "Heschl\u2019s gyrus",
    "Middle frontal gyrus anterior", "Precuneus anterior", "global signal clean",
]


def prepare_scan(data_root: Path, output_dir: Path, scan: str, event: str) -> dict:
    from data.loader import _load_intrascan_epoch_data

    if re.fullmatch(r"sub\d+-scan\d+", scan) is None:
        raise ValueError(f"Invalid scan identifier: {scan}")
    output = output_dir / f"{scan}.npz"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory.")
    args = SimpleNamespace(
        dataset_root=str(data_root), dataname=scan, TR=2.1,
        mri_sync_event=event, labels_roi="|".join(ROI_COLUMNS), seed=16,
    )
    splits, split_info = _load_intrascan_epoch_data(args)
    payload = {}
    for index, split in enumerate(("train", "val", "test")):
        eeg = np.asarray(splits[2 * index], dtype=np.float32)
        targets = np.asarray(splits[2 * index + 1], dtype=np.float32)
        if eeg.ndim != 3 or eeg.shape[1:] != (26, 3200):
            raise ValueError(f"{scan}/{split}: expected [N,26,3200], got {eeg.shape}")
        if targets.shape != (len(eeg), 7) or len(eeg) < 2:
            raise ValueError(f"{scan}/{split}: invalid target shape {targets.shape}")
        if not np.isfinite(eeg).all() or not np.isfinite(targets).all():
            raise ValueError(f"Nonfinite input in {scan}/{split}")
        payload[f"{split}_eeg"] = eeg
        payload[f"{split}_y"] = targets
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata = {
        "scan": scan, "split_info": split_info, "roi_columns": ROI_COLUMNS,
        "mri_sync_event": event, "cache_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "protocol": "Inherited whole-scan BOLD filtering and normalization before chronological splitting",
        "backbone_required": False,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export aligned VU EEG/target arrays without training a neural backbone.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cache/vu_aligned_eeg")
    parser.add_argument("--scans", nargs="+", required=True)
    parser.add_argument("--mri-sync-event", default="R149")
    args = parser.parse_args()
    if len(args.scans) != len(set(args.scans)):
        parser.error("Duplicate scan identifiers are not allowed")
    for scan in args.scans:
        metadata = prepare_scan(args.data_root, args.output_dir, scan, args.mri_sync_event)
        print(json.dumps(metadata["split_info"]), flush=True)


if __name__ == "__main__":
    main()
