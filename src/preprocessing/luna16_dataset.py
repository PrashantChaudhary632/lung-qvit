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
    pylidc_centroid_to_world,
    load_dicom_slice_metadata,
)
from labels import diameter_heuristic_label, get_labels_via_pylidc


class LUNA16Dataset(Dataset):
    """
    Parameters
    ----------
    data_dir : str
        Directory containing the .mhd/.raw files (e.g. data/luna16_subset0).
    annotations_csv : str
        Path to LUNA16's annotations.csv. Only used when label_source="diameter".
    patch_size : tuple[int, int, int]
        (Z, Y, X) size of extracted patches, in voxels post-resampling.
    target_spacing : tuple[float, float, float]
        Isotropic spacing (mm) to resample every volume to.
    label_source : str
        "diameter" (default, smoke-testing only) or "pylidc" (real
        radiologist-consensus malignancy labels, requires local DICOM data
        under dicom_root and pylidc's bundled database).
    dicom_root : str
        Directory containing LIDC-IDRI/<patient_id>/ DICOM folders. Only
        used when label_source="pylidc".
    """

    def __init__(self, data_dir, annotations_csv=None, patch_size=(32, 64, 64),
                 target_spacing=(1.0, 1.0, 1.0), label_source="diameter",
                 dicom_root="data/LIDC-IDRI"):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.label_source = label_source
        self.dicom_root = Path(dicom_root)

        available_uids = sorted(p.stem for p in self.data_dir.glob("*.mhd"))
        if len(available_uids) == 0:
            raise ValueError(f"No .mhd files found in {data_dir}")

        if label_source == "pylidc":
            self.annotations = self._build_from_pylidc(available_uids)
        elif label_source == "diameter":
            if annotations_csv is None:
                raise ValueError("annotations_csv is required when label_source='diameter'")
            self.annotations = self._build_from_csv(annotations_csv, available_uids)
        else:
            raise ValueError(f"Unknown label_source: {label_source}")

        if len(self.annotations) == 0:
            raise ValueError("No usable labeled nodules found -- check data_dir and label_source.")

    def _build_from_csv(self, annotations_csv, available_uids):
        annotations = pd.read_csv(annotations_csv)
        annotations = annotations[annotations["seriesuid"].isin(available_uids)].reset_index(drop=True)
        annotations["patient_id"] = annotations["seriesuid"]
        annotations["label"] = annotations["diameter_mm"].apply(
            lambda d: int(diameter_heuristic_label(d))
        )
        return annotations

    def _build_from_pylidc(self, available_uids):
        rows = []
        for uid in available_uids:
            patient_dicom_dirs = list(self.dicom_root.glob(f"*/{uid}"))
            if not patient_dicom_dirs:
                continue  # no local DICOM for this patient, skip
            patient_dicom_dir = patient_dicom_dirs[0]
            patient_id = patient_dicom_dir.parent.name

            nodules = get_labels_via_pylidc(uid)
            if not nodules:
                continue

            slice_metadata = load_dicom_slice_metadata(patient_dicom_dir)  # load once per patient
            for nodule in nodules:
                world_xyz = pylidc_centroid_to_world(nodule["centroid_xyz"], slice_metadata)
                rows.append({
                    "seriesuid": uid,
                    "patient_id": patient_id,
                    "coordX": world_xyz[0],
                    "coordY": world_xyz[1],
                    "coordZ": world_xyz[2],
                    "diameter_mm": np.nan,  # not computed for pylidc-sourced nodules
                    "label": int(nodule["label"]),
                    "mean_malignancy": nodule["mean_malignancy"],
                    "n_readers": nodule["n_readers"],
                })
        return pd.DataFrame(rows)

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

        return {
            "image": torch.from_numpy(patch).unsqueeze(0).float(),  # (1, Z, Y, X)
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "patient_id": row["patient_id"],
            "diameter_mm": float(row["diameter_mm"]) if pd.notna(row["diameter_mm"]) else None,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python luna16_dataset.py <data_dir> <annotations_csv>")
        sys.exit(1)

    ds = LUNA16Dataset(data_dir=sys.argv[1], annotations_csv=sys.argv[2], label_source="diameter")
    print(f"Dataset size: {len(ds)} nodules across {len(ds.patient_ids())} patients")
    sample = ds[0]
    print(f"Sample image shape: {sample['image'].shape}, label: {sample['label']}")