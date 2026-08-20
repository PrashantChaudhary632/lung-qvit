import sys
sys.path.insert(0, "src/preprocessing")

from pathlib import Path
from collections import Counter
from labels import get_labels_via_pylidc, DiagnosisLabel

uids = Path("data/subset0_series_uids.txt").read_text().strip().split("\n")

total_nodules = 0
patients_with_zero = 0
label_counts = Counter()
reader_counts = Counter()

for uid in uids:
    nodules = get_labels_via_pylidc(uid)
    if len(nodules) == 0:
        patients_with_zero += 1
    for n in nodules:
        total_nodules += 1
        label_counts[n["label"].name] += 1
        reader_counts[n["n_readers"]] += 1

print(f"Patients: {len(uids)}")
print(f"Patients with 0 labeled nodules: {patients_with_zero}")
print(f"Total labeled nodules: {total_nodules}")
print()
print("Label distribution:")
for label, count in label_counts.most_common():
    pct = 100 * count / total_nodules
    print(f"  {label}: {count} ({pct:.1f}%)")
print()
print("Reader count distribution (how many radiologists agreed per nodule):")
for readers, count in sorted(reader_counts.items()):
    print(f"  {readers} reader(s): {count} nodules")