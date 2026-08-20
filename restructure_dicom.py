"""
Restructures downloaded DICOM data from series-UID-named folders into the
patient-ID-named structure pylidc expects: LIDC-IDRI/LIDC-IDRI-dddd/...

pylidc recursively searches within each patient folder for matching
series/study UIDs, so we don't need to preserve any particular structure
inside each patient folder -- we just need the top-level folder name to
be the correct LIDC-IDRI-dddd patient ID.
"""

import shutil
from pathlib import Path
import pydicom

src_root = Path("data/lidc_dicom")
dst_root = Path("data/LIDC-IDRI")
dst_root.mkdir(parents=True, exist_ok=True)

series_folders = [p for p in src_root.iterdir() if p.is_dir()]
print(f"Found {len(series_folders)} series folders to restructure")

moved = 0
for series_folder in series_folders:
    dcm_files = list(series_folder.rglob("*.dcm"))
    if not dcm_files:
        print(f"  WARNING: no .dcm files found in {series_folder.name}, skipping")
        continue

    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    patient_id = str(ds.PatientID).strip()

    dst_patient_dir = dst_root / patient_id
    dst_patient_dir.mkdir(parents=True, exist_ok=True)

    dst_series_dir = dst_patient_dir / series_folder.name
    if dst_series_dir.exists():
        print(f"  Already restructured: {patient_id}")
        continue

    shutil.move(str(series_folder), str(dst_series_dir))
    moved += 1
    print(f"  {series_folder.name[:20]}... -> {patient_id}")

print(f"\nRestructured {moved} series into {dst_root}")