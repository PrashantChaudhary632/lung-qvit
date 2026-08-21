"""
Patient-stratified train/val/test splitting. Splits are done at the patient
level (not the example level) so no single patient's nodules/normals leak
across splits -- critical given some patients contribute multiple examples.

Stratification also tries to balance class distribution across splits,
important given our 50/31/19 NORMAL/BENIGN/MALIGNANT imbalance.
"""

import numpy as np
import pandas as pd


def patient_stratified_split(annotations_df, train_frac=0.70, val_frac=0.15,
                              seed=42):
    """
    Split patients (not examples) into train/val/test sets.

    Stratifies by each patient's *majority* label, so patients dominated by
    malignant nodules aren't all accidentally dumped into one split. This is
    an approximation -- with only 88 patients, perfect stratification across
    3 classes and 3 splits isn't achievable, and that should be acknowledged
    as a limitation given the small dataset size.

    Returns: (train_patient_ids, val_patient_ids, test_patient_ids)
    """
    rng = np.random.default_rng(seed)

    patient_majority_label = (
        annotations_df.groupby("patient_id")["label"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    train_ids, val_ids, test_ids = [], [], []

    for label_val in patient_majority_label.unique():
        patients_this_label = patient_majority_label[
            patient_majority_label == label_val
        ].index.to_numpy()
        rng.shuffle(patients_this_label)

        n = len(patients_this_label)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        train_ids.extend(patients_this_label[:n_train])
        val_ids.extend(patients_this_label[n_train:n_train + n_val])
        test_ids.extend(patients_this_label[n_train + n_val:])

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def summarize_split(annotations_df, patient_ids, name):
    subset = annotations_df[annotations_df["patient_id"].isin(patient_ids)]
    print(f"{name}: {len(patient_ids)} patients, {len(subset)} examples")
    print(f"  {subset['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src/preprocessing")
    from luna16_dataset import LUNA16Dataset

    ds = LUNA16Dataset(
        data_dir="data/luna16_subset0",
        label_source="pylidc",
        normal_csv="data/normal_candidates.csv",
    )

    train_ids, val_ids, test_ids = patient_stratified_split(ds.annotations)

    summarize_split(ds.annotations, train_ids, "Train")
    summarize_split(ds.annotations, val_ids, "Val")
    summarize_split(ds.annotations, test_ids, "Test")