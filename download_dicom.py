"""
Downloads DICOM images + bundled XML annotations for the 89 LUNA16 subset0
patients directly from TCIA, using the official tcia_utils package.
This sidesteps the NBIA Data Retriever manifest file entirely.
"""

from pathlib import Path
from tcia_utils import nbia

uids_path = Path("data/subset0_series_uids.txt")
uids = uids_path.read_text().strip().split("\n")
print(f"Downloading {len(uids)} series from TCIA...")

out_dir = Path("data/lidc_dicom")
out_dir.mkdir(parents=True, exist_ok=True)

nbia.downloadSeries(
    series_data=uids,
    input_type="list",
    path=str(out_dir),
)

print("Download complete.")