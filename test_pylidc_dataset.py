import sys
sys.path.insert(0, "src/preprocessing")

from luna16_dataset import LUNA16Dataset

ds = LUNA16Dataset(
    data_dir="data/luna16_subset0",
    label_source="pylidc",
)

print(f"Dataset size: {len(ds)} nodules across {len(ds.patient_ids())} patients")

sample = ds[0]
print(f"\nSample 0:")
print(f"  image shape: {sample['image'].shape}")
print(f"  label: {sample['label'].item()}")
print(f"  patient_id: {sample['patient_id']}")

sample = ds[10]
print(f"\nSample 10:")
print(f"  image shape: {sample['image'].shape}")
print(f"  label: {sample['label'].item()}")
print(f"  patient_id: {sample['patient_id']}")