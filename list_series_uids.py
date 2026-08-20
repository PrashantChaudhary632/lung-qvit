"""
Extracts the 89 series UIDs from LUNA16 subset0 filenames — these are the
same UIDs used by TCIA/LIDC-IDRI, needed to build an NBIA download manifest.
"""

from pathlib import Path

data_dir = Path("data/luna16_subset0")
uids = sorted(p.stem for p in data_dir.glob("*.mhd"))

print(f"Found {len(uids)} series UIDs")

out_path = Path("data/subset0_series_uids.txt")
out_path.write_text("\n".join(uids))
print(f"Saved to {out_path}")

# Show first 3 as a sanity check
for uid in uids[:3]:
    print(f"  {uid}")
    