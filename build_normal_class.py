"""
Samples Normal-class (no nodule) candidate locations from LUNA16's
candidates_V2.csv, restricted to our 89 subset0 patients.

Safety filter: even though class=0 already means "not flagged as a true
nodule" by LUNA16's own candidate generator, we additionally require each
sampled point to be at least MIN_DISTANCE_MM away from every known
annotated nodule (from annotations.csv) in that same patient -- a second
safety net against sampling tissue that overlaps real nodule margins.
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # reproducible sampling

TARGET_COUNT = 144  # match our real nodule count
MIN_DISTANCE_MM = 20.0

uids = open("data/subset0_series_uids.txt").read().strip().split("\n")

candidates = pd.read_csv("data/candidates_V2.csv")
candidates = candidates[candidates["seriesuid"].isin(uids) & (candidates["class"] == 0)].reset_index(drop=True)

annotations = pd.read_csv("data/annotations.csv")
annotations = annotations[annotations["seriesuid"].isin(uids)].reset_index(drop=True)

print(f"Candidate pool (class=0, our patients): {len(candidates)}")
print(f"Known nodules to filter against: {len(annotations)}")

# Build per-patient nodule location lookup for fast distance filtering
nodule_locations = {}
for uid, group in annotations.groupby("seriesuid"):
    nodule_locations[uid] = group[["coordX", "coordY", "coordZ"]].to_numpy()

def min_distance_to_nodules(row):
    uid = row["seriesuid"]
    if uid not in nodule_locations:
        return np.inf  # no known nodules for this patient, always safe
    point = np.array([row["coordX"], row["coordY"], row["coordZ"]])
    dists = np.linalg.norm(nodule_locations[uid] - point, axis=1)
    return dists.min()

print("Computing distances to known nodules (this may take a moment)...")
candidates["min_dist_to_nodule"] = candidates.apply(min_distance_to_nodules, axis=1)

safe_candidates = candidates[candidates["min_dist_to_nodule"] >= MIN_DISTANCE_MM].reset_index(drop=True)
print(f"Safe candidates (>= {MIN_DISTANCE_MM}mm from any nodule): {len(safe_candidates)}")

# Sample roughly evenly across patients, capped, then trim to exact target
per_patient_cap = max(1, TARGET_COUNT // safe_candidates["seriesuid"].nunique() + 2)
sampled = (
    safe_candidates.groupby("seriesuid", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), per_patient_cap), random_state=42))
)

if len(sampled) > TARGET_COUNT:
    sampled = sampled.sample(TARGET_COUNT, random_state=42)

sampled = sampled.reset_index(drop=True)
print(f"\nFinal sampled Normal examples: {len(sampled)}")
print(f"Spread across {sampled['seriesuid'].nunique()} patients")
print(sampled.groupby("seriesuid").size().describe())

out_path = "data/normal_candidates.csv"
sampled[["seriesuid", "coordX", "coordY", "coordZ"]].to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")