import sys
sys.path.insert(0, "src/preprocessing")

from pathlib import Path
from labels import get_labels_via_pylidc

uids = Path("data/subset0_series_uids.txt").read_text().strip().split("\n")

total_nodules = 0
total_indeterminate_skipped = 0

for uid in uids[:10]:  # first 10 patients as a sample
    nodules = get_labels_via_pylidc(uid)
    print(f"{uid[:30]}...: {len(nodules)} labeled nodules")
    for n in nodules:
        print(f"    malignancy={n['mean_malignancy']:.1f}, label={n['label'].name}, "
              f"readers={n['n_readers']}, centroid={tuple(round(c,1) for c in n['centroid_xyz'])}")
    total_nodules += len(nodules)

print(f"\nTotal labeled nodules across 10 patients: {total_nodules}")
