"""
Empirically determines the correct way to convert pylidc's index-based
nodule centroids into real-world mm coordinates, by testing against
patients where we already know the correct answer from annotations.csv.
"""

import sys
sys.path.insert(0, "src/preprocessing")

from pathlib import Path
import numpy as np
if not hasattr(np, "int"):
    np.int = int

import pandas as pd
import pydicom
import pylidc as pl

annotations = pd.read_csv("data/annotations.csv")
uids = Path("data/subset0_series_uids.txt").read_text().strip().split("\n")

# Find a patient with exactly one nodule in annotations.csv, for a clean 1:1 test
for uid in uids:
    matches = annotations[annotations["seriesuid"] == uid]
    if len(matches) != 1:
        continue

    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
    if scan is None:
        continue
    nodules = scan.cluster_annotations(verbose=False)
    nodules = [n for n in nodules if len(n) <= 4]
    if len(nodules) != 1:
        continue

    known = matches.iloc[0]
    known_xyz = np.array([known["coordX"], known["coordY"], known["coordZ"]])
    print(f"Testing patient {scan.patient_id} (series {uid[:20]}...)")
    print(f"  Known nodule location (annotations.csv): {known_xyz}")

    centroid_ijk = nodules[0][0].centroid  # use first annotation in cluster
    print(f"  pylidc centroid (index space): {centroid_ijk}")

    # Load our local DICOM files for this patient, sorted by z ascending
    dicom_dir = Path("data/LIDC-IDRI") / scan.patient_id
    dcm_paths = list(dicom_dir.rglob("*.dcm"))
    dcm_files = [pydicom.dcmread(str(p)) for p in dcm_paths]
    dcm_files.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    k = int(round(centroid_ijk[2]))
    slice_dcm = dcm_files[k]
    origin_xy = np.array(slice_dcm.ImagePositionPatient[:2])
    pixel_spacing = np.array(slice_dcm.PixelSpacing)
    z_mm = float(slice_dcm.ImagePositionPatient[2])

    i, j = centroid_ijk[0], centroid_ijk[1]

    # Try both axis conventions
    option_a = origin_xy + np.array([i, j]) * pixel_spacing  # (i=x, j=y)
    option_b = origin_xy + np.array([j, i]) * pixel_spacing  # (i=y, j=x) -- row/col swapped

    world_a = np.array([option_a[0], option_a[1], z_mm])
    world_b = np.array([option_b[0], option_b[1], z_mm])

    dist_a = np.linalg.norm(world_a - known_xyz)
    dist_b = np.linalg.norm(world_b - known_xyz)

    print(f"  Option A (i=x,j=y): {world_a}, distance from known: {dist_a:.1f}mm")
    print(f"  Option B (i=y,j=x): {world_b}, distance from known: {dist_b:.1f}mm")
    print(f"  --> {'Option A' if dist_a < dist_b else 'Option B'} is correct")
    break
else:
    print("No suitable single-nodule patient found for testing")