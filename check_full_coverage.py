from pathlib import Path
import pylidc as pl

uids = Path("data/subset0_series_uids.txt").read_text().strip().split("\n")
missing = []
for uid in uids:
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
    if not scan:
        missing.append(uid)

print(f"{len(uids) - len(missing)}/{len(uids)} series found in pylidc's database")
if missing:
    print("Missing:")
    for uid in missing:
        print(f"  {uid}")