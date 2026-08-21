import sys
sys.path.insert(0, "src/preprocessing")

from collections import Counter
from luna16_dataset import LUNA16Dataset

ds = LUNA16Dataset(
    data_dir="data/luna16_subset0",
    label_source="pylidc",
    normal_csv="data/normal_candidates.csv",
)

print(f"Total dataset size: {len(ds)} examples across {len(ds.patient_ids())} patients")

label_names = {0: "NORMAL", 1: "BENIGN", 2: "MALIGNANT"}
labels = [int(ds.annotations.iloc[i]["label"]) for i in range(len(ds))]
counts = Counter(labels)
print("\nClass distribution:")
for label_val, name in label_names.items():
    count = counts.get(label_val, 0)
    pct = 100 * count / len(ds)
    print(f"  {name}: {count} ({pct:.1f}%)")

# Pull one real sample from each class to confirm end-to-end extraction works
print("\nSample check (one per class):")
seen = set()
for i in range(len(ds)):
    label_val = labels[i]
    if label_val in seen:
        continue
    sample = ds[i]
    print(f"  {label_names[label_val]}: shape={sample['image'].shape}, "
          f"patient={sample['patient_id']}, pixel range=[{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
    seen.add(label_val)
    if len(seen) == 3:
        break