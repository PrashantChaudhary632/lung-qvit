"""
PyTorch Dataset for LUNA16, with patient-level grouping baked in so that
downstream patient-stratified k-fold splitting (Stage 6) is straightforward
and leakage-safe from the start.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing_utils import (
    load_mhd_image,
    resample_to_spacing,
    apply_lung_window,
    extract_patch,
    world_to_voxel,
)
from labels import diameter_heuristic_label


class LUNA16Dataset(Dataset):
    """
    Parameters
    ----------
    data_dir : str
        Directory containing the .mhd/.raw files (e.g. data/luna16_subset0).
    annotations_csv : str
        Path to LUNA16's annotations.csv (seriesuid, coordX/Y/Z, diameter_mm).
    patch_size : tuple[int, int, int]
        (Z, Y, X) size of extracted patches, in voxels post-resampling.
    target_spacing : tuple[float, float, float]
        Isotropic spacing (mm) to resample every volume to.
    label_fn : callable
        Function mapping a nodule row -> DiagnosisLabel. Defaults to the
        diameter heuristic for smoke-testing; swap for the pylidc-based
        function once malignancy labels are wired up (see labels.py).
    """

    def __init__(self, data_dir, annotations_csv, patch_size=(32, 64, 64),
                 target_spacing=(1.0, 1.0, 1.0), label_fn=None):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.label_fn = label_fn or (lambda row: diameter_heuristic_label(row["diameter_mm"]))

        annotations = pd.read_csv(annotations_csv)

        # Only keep annotations for series we actually have .mhd files for
        # (Subset 0 will only cover a fraction of the full annotations.csv).
        available_uids = {p.stem for p in self.data_dir.glob("*.mhd")}
        self.annotations = annotations[annotations["seriesuid"].isin(available_uids)].reset_index(drop=True)

        if len(self.annotations) == 0:
            raise ValueError(
                f"No matching series found between {annotations_csv} and "
                f".mhd files in {data_dir}. Check that both point at the "
                f"same LUNA16 subset."
            )

        # patient_id == seriesuid in LUNA16 (one series per patient scan) —
        # exposed explicitly here so Stage 6's patient-stratified splitter
        # can group on it directly.
        self.annotations["patient_id"] = self.annotations["seriesuid"]

    def __len__(self):
        return len(self.annotations)

    def patient_ids(self):
        """All patient IDs, for building patient-stratified CV folds."""
        return self.annotations["patient_id"].unique().tolist()

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        mhd_path = self.data_dir / f"{row['seriesuid']}.mhd"

        volume = load_mhd_image(str(mhd_path))
        volume = resample_to_spacing(volume, new_spacing=self.target_spacing)

        world_xyz = np.array([row["coordX"], row["coordY"], row["coordZ"]])
        voxel_xyz = world_to_voxel(world_xyz, volume.origin, volume.spacing)
        voxel_zyx = voxel_xyz[::-1]  # array is indexed (z, y, x)

        patch = extract_patch(volume.array, voxel_zyx, self.patch_size)
        patch = apply_lung_window(patch)

        label = int(self.label_fn(row))

        return {
            "image": torch.from_numpy(patch).unsqueeze(0).float(),  # (1, Z, Y, X)
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": row["patient_id"],
            "diameter_mm": float(row["diameter_mm"]),
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python luna16_dataset.py <data_dir> <annotations_csv>")
        sys.exit(1)

    ds = LUNA16Dataset(data_dir=sys.argv[1], annotations_csv=sys.argv[2])
    print(f"Dataset size: {len(ds)} nodules across {len(ds.patient_ids())} patients")
    sample = ds[0]
    print(f"Sample image shape: {sample['image'].shape}, label: {sample['label']}")