from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from data.features import bandpower_tokens, zscore_by_train
from model.run import REPRESENTATIONS, load_representation
from data.prepare import ROI_COLUMNS
from model.mel import (
    FeatureSet, MatchedFlatMLP, _binned_periodogram, _flattened_psd,
    _morlet_band_energy, _morlet_kernel, build_feature_cache, fit_ridge,
    set_seed, train_neural, zscore_from_train,
)

class SpectralContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eeg = np.random.default_rng(7).normal(size=(3, 26, 3200)).astype(np.float32)

    def test_equal_dimensional_estimators(self):
        arrays = [bandpower_tokens(self.eeg), _binned_periodogram(self.eeg, np.linspace(0.5, 45, 6)),
                  _flattened_psd(self.eeg), _morlet_band_energy(self.eeg, batch_size=2)]
        for value in arrays:
            self.assertEqual(value.reshape(3, -1).shape, (3, 1040))
            self.assertTrue(np.isfinite(value).all())

    def test_canonical_matches_general_fourier_estimator(self):
        direct = bandpower_tokens(self.eeg)
        general = _binned_periodogram(self.eeg, np.array([0.5, 4, 8, 13, 30, 45]))
        np.testing.assert_allclose(direct, general, atol=2e-6, rtol=2e-6)

    def test_morlet_batch_size_does_not_change_values(self):
        np.testing.assert_array_equal(_morlet_band_energy(self.eeg, batch_size=1),
                                      _morlet_band_energy(self.eeg, batch_size=3))

    def test_morlet_kernels_zero_mean_unit_energy(self):
        for center, cycles in zip([2, 6, 10.5, 21.5, 37.5], [3, 4, 5, 7, 9]):
            kernel = _morlet_kernel(center, 200, cycles)
            self.assertLess(abs(kernel.mean()), 1e-7)
            self.assertAlmostEqual(float(np.sum(abs(kernel) ** 2)), 1.0, places=6)

    def test_train_only_calibration(self):
        rng = np.random.default_rng(8)
        data = {name: rng.normal(size=(4, 8, 26, 5)).astype(np.float32)
                for name in ("train", "val", "test")}
        normalized, mean, scale = zscore_by_train(data)
        modified = {**data, "val": data["val"] + 1000, "test": data["test"] - 1000}
        normalized_2, mean_2, scale_2 = zscore_by_train(modified)
        np.testing.assert_array_equal(mean, mean_2)
        np.testing.assert_array_equal(scale, scale_2)
        np.testing.assert_array_equal(normalized["train"], normalized_2["train"])

    def test_constant_coordinate_calibration_is_finite(self):
        values = np.ones((4, 8, 26, 5), dtype=np.float32)
        for value in zscore_from_train(values, values, values):
            np.testing.assert_array_equal(value, np.zeros_like(values))

    def test_feature_build_does_not_require_neurobolt_predictions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ["aligned", "canonical", "extra"]:
                (root / directory).mkdir()
            aligned, canonical = {}, {}
            for split in ("train", "val", "test"):
                aligned[f"{split}_eeg"] = self.eeg
                canonical[f"{split}_y"] = np.zeros((3, 7), dtype=np.float32)
            np.savez_compressed(root / "aligned/demo.npz", **aligned)
            np.savez_compressed(root / "canonical/demo.npz", **canonical)
            path = build_feature_cache("demo", root / "aligned", root / "canonical", root / "extra", False)
            with np.load(path, allow_pickle=False) as result:
                self.assertIn("train_morlet", result)
                self.assertNotIn("test_y_hat", result)

    def test_mismatched_feature_dimension_is_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong.npz"
            np.savez(path, **{f"{split}_tokens_z": np.zeros((3, 99), np.float32)
                              for split in ("train", "val", "test")})
            with self.assertRaises(ValueError):
                load_representation(path, path, "canonical")


class ReadoutContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        rng = np.random.default_rng(21)
        self.features = FeatureSet(*[rng.normal(size=(n, 8)).astype(np.float32)
                                    for n in (25, 12, 10)], False)
        weights = rng.normal(size=(8, 7)).astype(np.float32)
        self.train_y = self.features.train @ weights
        self.val_y = self.features.val @ weights
        self.test_y = self.features.test @ weights

    def test_paper_mlp_parameter_count(self):
        model = MatchedFlatMLP(1040, 7, 16, 0.1)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 18855)
        self.assertEqual(tuple(model(torch.zeros(2, 8, 26, 5)).shape), (2, 7))

    def test_ridge_choices_do_not_depend_on_test_responses(self):
        with threadpool_limits(limits=1):
            first = fit_ridge(self.features, self.train_y, self.val_y, self.test_y)
            second = fit_ridge(self.features, self.train_y, self.val_y, -self.test_y)
        self.assertEqual(first[2], second[2])
        np.testing.assert_array_equal(first[1], second[1])

    def test_shared_permutation_preserves_ridge_prediction(self):
        permutation = np.arange(8)[::-1]
        permuted = FeatureSet(*[value[:, permutation] for value in
                               (self.features.train, self.features.val, self.features.test)], False)
        with threadpool_limits(limits=1):
            first = fit_ridge(self.features, self.train_y, self.val_y, None)
            second = fit_ridge(permuted, self.train_y, self.val_y, None)
        np.testing.assert_allclose(first[1], second[1], atol=2e-5, rtol=2e-5)

    def test_mlp_checkpoint_does_not_depend_on_test_responses(self):
        outputs = []
        for test_y in (self.test_y, -self.test_y):
            set_seed(3)
            model = MatchedFlatMLP(8, 7, 4, 0.1)
            outputs.append(train_neural(model, self.features, self.train_y, self.val_y, test_y,
                                       torch.device("cpu"), 3, 3, 2, 1e-3, 1e-3, 8))
        self.assertEqual(outputs[0][2], outputs[1][2])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])

    def test_paper_representation_names_are_exact(self):
        self.assertEqual(set(REPRESENTATIONS), {"canonical", "uniform", "whole_history", "morlet"})

    def test_historical_target_storage_order_is_explicit(self):
        self.assertEqual(ROI_COLUMNS[:3], ["Thalamus", "Putamen", "Cuneus"])
        self.assertEqual(ROI_COLUMNS[6], "global signal clean")


if __name__ == "__main__":
    unittest.main()
