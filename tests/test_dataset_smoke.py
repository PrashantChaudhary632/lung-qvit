"""
Smoke tests that run entirely on synthetic data — no LUNA16 download needed.
Verifies the preprocessing pipeline works correctly before downloading real data.

Run with: pytest tests/test_dataset_smoke.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "preprocessing"))

from preprocessing_utils import (
    load_mhd_image,
    resample_to_spacing,
    apply_lung_window,
    extract_patch,
    world_to_voxel,
)
from labels import diameter_heuristic_label, malignancy_score_to_label, DiagnosisLabel


@pytest.fixture
def synthetic_mhd(tmp_path):
    """Write a small synthetic .mhd/.raw volume that mimics a LUNA16 scan."""
    rng = np.random.default_rng(0)
    array = rng.integers(low=-1000, high=400, size=(40, 80, 80)).astype(np.int16)  # (Z, Y, X)

    itk_image = sitk.GetImageFromArray(array)
    itk_image.SetSpacing((0.7, 0.7, 2.5))   # (x, y, z) — typical anisotropic LUNA16 spacing
    itk_image.SetOrigin((-150.0, -150.0, -300.0))

    out_path = tmp_path / "synthetic_series.mhd"
    sitk.WriteImage(itk_image, str(out_path))
    return str(out_path)


def test_load_mhd_image(synthetic_mhd):
    vol = load_mhd_image(synthetic_mhd)
    assert vol.array.shape == (40, 80, 80)
    assert vol.spacing.shape == (3,)
    assert vol.origin.shape == (3,)


def test_resample_to_isotropic(synthetic_mhd):
    vol = load_mhd_image(synthetic_mhd)
    resampled = resample_to_spacing(vol, new_spacing=(1.0, 1.0, 1.0))
    # z-spacing was coarsest (2.5mm) so z-dimension should grow after resampling
    assert resampled.array.shape[0] > vol.array.shape[0]
    assert np.allclose(resampled.spacing, [1.0, 1.0, 1.0])


def test_lung_window_normalizes_to_unit_range():
    hu = np.array([-2000, -1350, -600, 150, 2000], dtype=np.float32)
    windowed = apply_lung_window(hu)
    assert windowed.min() >= 0.0
    assert windowed.max() <= 1.0
    assert windowed[1] < windowed[2] < windowed[3]


def test_world_to_voxel_roundtrip():
    origin = np.array([-150.0, -150.0, -300.0])
    spacing = np.array([0.7, 0.7, 2.5])
    world_point = np.array([-150.0 + 7.0, -150.0 + 7.0, -300.0 + 25.0])
    voxel = world_to_voxel(world_point, origin, spacing)
    assert np.array_equal(voxel, [10, 10, 10])


def test_extract_patch_correct_size():
    volume = np.arange(40 * 80 * 80).reshape(40, 80, 80).astype(np.float32)
    patch = extract_patch(volume, center_voxel_zyx=(20, 40, 40), patch_size=(16, 32, 32))
    assert patch.shape == (16, 32, 32)


def test_extract_patch_handles_edge_padding():
    volume = np.ones((40, 80, 80), dtype=np.float32)
    patch = extract_patch(volume, center_voxel_zyx=(0, 0, 0), patch_size=(16, 32, 32))
    assert patch.shape == (16, 32, 32)
    assert patch.min() == 0.0  # padded region
    assert patch.max() == 1.0  # real data region


def test_diameter_heuristic_label():
    assert diameter_heuristic_label(2.0) == DiagnosisLabel.BENIGN
    assert diameter_heuristic_label(10.0) == DiagnosisLabel.MALIGNANT


def test_malignancy_score_to_label_boundaries():
    assert malignancy_score_to_label(1.5) == DiagnosisLabel.BENIGN
    assert malignancy_score_to_label(4.5) == DiagnosisLabel.MALIGNANT
    with pytest.raises(ValueError):
        malignancy_score_to_label(3.0)


def test_full_pipeline_end_to_end(synthetic_mhd):
    """load -> resample -> window -> locate nodule -> extract patch."""
    vol = load_mhd_image(synthetic_mhd)
    resampled = resample_to_spacing(vol, new_spacing=(1.0, 1.0, 1.0))
    windowed = apply_lung_window(resampled.array)

    world_xyz = vol.origin + (vol.spacing * np.array([40, 40, 20]))
    voxel_xyz = world_to_voxel(world_xyz, resampled.origin, resampled.spacing)
    voxel_zyx = voxel_xyz[::-1]

    patch = extract_patch(windowed, voxel_zyx, patch_size=(32, 64, 64))
    assert patch.shape == (32, 64, 64)
    assert 0.0 <= patch.min() and patch.max() <= 1.0