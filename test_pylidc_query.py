"""
Verifies pylidc's pre-built database can be queried directly using our
known LUNA16 subset0 series UIDs -- no custom population needed.
"""

from pathlib import Path
import pylidc as pl

uids = Path("data/subset0_series_uids.txt").read_text().strip().split("\n")
print(f"Checking {len(uids)} known series UIDs against pylidc's database...")

found = 0
for uid in uids[:5]:  # just check the first 5 as a quick test
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
    if scan:
        found += 1
        print(f"  FOUND: {scan.patient_id}, {len(scan.annotations)} annotations")
        for ann in scan.annotations[:2]:
            print(f"    nodule malignancy score: {ann.malignancy}")
    else:
        print(f"  NOT FOUND: {uid[:30]}...")

print(f"\n{found}/5 tested UIDs matched.")