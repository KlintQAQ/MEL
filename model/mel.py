from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FS = 200
BIN_SEC = 2
N_BINS = 8
N_CHANNELS = 26
N_BANDS = 5
UNIFORM_BANDS = np.linspace(0.5, 45.0, N_BANDS + 1)
RIDGE_ALPHAS = np.logspace(-1, 7, 33)

@dataclass(frozen=True)
class FeatureSet:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    structured: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def corr_per_roi(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for roi in range(target.shape[1]):
        x = np.asarray(prediction[:, roi], dtype=np.float64)
        y = np.asarray(target[:, roi], dtype=np.float64)
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            values.append(float("nan"))
        else:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return np.asarray(values)


def avg_corr(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.nanmean(corr_per_roi(prediction, target)))


def zscore_from_train(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
    )


def _binned_periodogram(
    eeg: np.ndarray,
    edges: np.ndarray,
    fs: int = FS,
    bin_sec: int = BIN_SEC,
) -> np.ndarray:
    n, channels, samples = eeg.shape
    bin_samples = fs * bin_sec
    bins = samples // bin_samples
    segments = eeg[:, :, : bins * bin_samples].reshape(n, channels, bins, bin_samples)
    segments = segments * np.hanning(bin_samples)[None, None, None, :]
    spectrum = np.abs(np.fft.rfft(segments, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(bin_samples, d=1.0 / fs)
    features = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        features.append(spectrum[..., mask].mean(axis=-1))
    return np.log1p(np.stack(features, axis=-1).transpose(0, 2, 1, 3)).astype(np.float32)


def _flattened_psd(eeg: np.ndarray, fs: int = FS, n_frequency_bins: int = 40) -> np.ndarray:
    window = np.hanning(eeg.shape[-1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(eeg * window[None, None, :], axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(eeg.shape[-1], d=1.0 / fs)
    edges = np.linspace(0.5, 45.0, n_frequency_bins + 1)
    features = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        features.append(spectrum[..., mask].mean(axis=-1))
    return np.log1p(np.stack(features, axis=-1)).astype(np.float32)


def _morlet_kernel(center_hz: float, fs: int, cycles: float) -> np.ndarray:
    sigma_t = cycles / (2.0 * np.pi * center_hz)
    radius = max(8, int(math.ceil(3.5 * sigma_t * fs)))
    time_axis = np.arange(-radius, radius + 1, dtype=np.float64) / fs
    carrier = np.exp(2j * np.pi * center_hz * time_axis)
    envelope = np.exp(-(time_axis**2) / (2.0 * sigma_t**2))
    kernel = carrier * envelope
    kernel = kernel - kernel.mean()
    kernel = kernel / np.sqrt(np.sum(np.abs(kernel) ** 2) + 1e-12)
    return kernel.astype(np.complex64)


def _morlet_band_energy(eeg: np.ndarray, fs: int = FS, batch_size: int = 32) -> np.ndarray:
    centers = [2.0, 6.0, 10.5, 21.5, 37.5]
    cycles = [3.0, 4.0, 5.0, 7.0, 9.0]
    n, channels, samples = eeg.shape
    bin_samples = fs * BIN_SEC
    result = np.empty((n, N_BINS, channels, len(centers)), dtype=np.float32)
    for band, (center, cycle) in enumerate(zip(centers, cycles)):
        kernel = _morlet_kernel(center, fs, cycle)
        full_length = samples + len(kernel) - 1
        fft_length = 1 << int(math.ceil(math.log2(full_length)))
        kernel_fft = np.fft.fft(kernel, n=fft_length)
        crop_start = (len(kernel) - 1) // 2
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = eeg[start:stop].astype(np.float32, copy=False)
            transformed = np.fft.ifft(
                np.fft.fft(batch, n=fft_length, axis=-1) * kernel_fft[None, None, :],
                axis=-1,
            )
            same = transformed[..., crop_start : crop_start + samples]
            power = np.abs(same) ** 2
            power = power.reshape(stop - start, channels, N_BINS, bin_samples).mean(axis=-1)
            result[start:stop, :, :, band] = np.log1p(power.transpose(0, 2, 1))
    return result


def build_feature_cache(
    scan: str,
    aligned_dir: Path,
    mel_dir: Path,
    feature_dir: Path,
    force: bool,
) -> Path:
    output_path = feature_dir / f"{scan}.npz"
    if output_path.exists() and not force:
        return output_path
    aligned = np.load(aligned_dir / f"{scan}.npz")
    mel = np.load(mel_dir / f"{scan}.npz")
    payload: dict[str, np.ndarray] = {}
    for key in ("train_y", "val_y", "test_y"):
        payload[key] = mel[key].astype(np.float32)

    raw_features: dict[str, dict[str, np.ndarray]] = {
        "stft_uniform": {},
        "morlet": {},
        "psd": {},
    }
    for split in ("train", "val", "test"):
        eeg = aligned[f"{split}_eeg"].astype(np.float32)
        raw_features["stft_uniform"][split] = _binned_periodogram(eeg, UNIFORM_BANDS)
        raw_features["morlet"][split] = _morlet_band_energy(eeg)
        raw_features["psd"][split] = _flattened_psd(eeg)

    for name, splits in raw_features.items():
        train, val, test = zscore_from_train(splits["train"], splits["val"], splits["test"])
        payload[f"train_{name}"] = train
        payload[f"val_{name}"] = val
        payload[f"test_{name}"] = test

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def _corr_1d(prediction: np.ndarray, target: np.ndarray) -> float:
    if np.std(prediction) < 1e-8 or np.std(target) < 1e-8:
        return float("-inf")
    return float(np.corrcoef(prediction, target)[0, 1])


def fit_ridge(
    features: FeatureSet,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[float], dict[str, np.ndarray]]:
    train_x = features.train.reshape(len(features.train), -1)
    val_x = features.val.reshape(len(features.val), -1)
    test_x = features.test.reshape(len(features.test), -1)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)
    val_prediction = np.zeros_like(val_y)
    test_prediction = np.zeros((len(test_x), train_y.shape[1]), dtype=train_y.dtype)
    selected: list[float] = []
    coefficients = np.zeros((train_y.shape[1], train_x.shape[1]), dtype=np.float32)
    intercepts = np.zeros(train_y.shape[1], dtype=np.float32)
    for roi in range(train_y.shape[1]):
        best_score = -float("inf")
        best_model: Ridge | None = None
        best_alpha = float(RIDGE_ALPHAS[0])
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=float(alpha)).fit(train_x, train_y[:, roi])
            score = _corr_1d(model.predict(val_x), val_y[:, roi])
            if score > best_score:
                best_score = score
                best_model = model
                best_alpha = float(alpha)
        if best_model is None:
            raise ValueError(f"No finite validation correlation for target column {roi}; check validation responses.")
        val_prediction[:, roi] = best_model.predict(val_x)
        test_prediction[:, roi] = best_model.predict(test_x)
        coefficients[roi] = np.asarray(best_model.coef_, dtype=np.float32)
        intercepts[roi] = float(best_model.intercept_)
        selected.append(best_alpha)
    state = {
        "mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scale": np.asarray(scaler.scale_, dtype=np.float32),
        "weight": coefficients,
        "bias": intercepts,
    }
    return val_prediction, test_prediction, selected, state


@torch.no_grad()
def predict_torch(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(features).to(device)).detach().cpu().numpy()


def train_neural(
    model: nn.Module,
    features: FeatureSet,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    include_initial: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features.train), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val = -float("inf")
    best_epoch = -1
    remaining = patience
    if include_initial:
        initial_prediction = predict_torch(model, features.val, device)
        best_val = avg_corr(initial_prediction, val_y)
        best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_prediction = predict_torch(model, features.val, device)
        val_score = avg_corr(val_prediction, val_y)
        if val_score > best_val + 1e-6:
            best_val = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining = patience
        else:
            remaining -= 1
        if remaining <= 0:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected.")
    model.load_state_dict(best_state)
    val_prediction = predict_torch(model, features.val, device)
    test_prediction = predict_torch(model, features.test, device)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return val_prediction, test_prediction, best_epoch, parameters, best_val


class MatchedFlatMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 7,
        hidden_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.reshape(features.shape[0], -1))
