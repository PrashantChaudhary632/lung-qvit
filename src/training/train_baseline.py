"""
Trains the classical CNN baseline on the patient-stratified split.
Reports per-class F1 and a confusion matrix, not just accuracy -- accuracy
alone would be misleading given the 50/31/19 class imbalance.
"""

import sys
sys.path.insert(0, "src/preprocessing")
sys.path.insert(0, "src/models/cnn")
sys.path.insert(0, "src/training")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix

from luna16_dataset import LUNA16Dataset
from splits import patient_stratified_split
from baseline_cnn import BaselineCNN3D

DEVICE = torch.device("cpu")
EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-3


def build_index_subset(dataset, patient_ids):
    """Row indices in dataset.annotations belonging to the given patients."""
    mask = dataset.annotations["patient_id"].isin(patient_ids)
    return dataset.annotations[mask].index.tolist()


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def main():
    print("Loading dataset...")
    ds = LUNA16Dataset(
        data_dir="data/luna16_subset0",
        label_source="pylidc",
        normal_csv="data/normal_candidates.csv",
    )

    train_ids, val_ids, test_ids = patient_stratified_split(ds.annotations)

    train_subset = Subset(ds, build_index_subset(ds, train_ids))
    val_subset = Subset(ds, build_index_subset(ds, val_ids))
    test_subset = Subset(ds, build_index_subset(ds, test_ids))

    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = BaselineCNN3D(num_classes=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Class-weighted loss to counter the 50/31/19 imbalance
    label_counts = ds.annotations.iloc[
        build_index_subset(ds, train_ids)
    ]["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        [1.0 / label_counts.get(i, 1) for i in range(3)], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * 3
    print(f"Class weights (to counter imbalance): {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_train_loss = total_loss / len(train_subset)

        val_labels, val_preds = evaluate(model, val_loader)
        val_acc = np.mean(np.array(val_labels) == np.array(val_preds))

        print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss={avg_train_loss:.4f} | val_acc={val_acc:.3f}")

    print("\n=== Final Test Set Evaluation ===")
    test_labels, test_preds = evaluate(model, test_loader)
    print(classification_report(
        test_labels, test_preds, target_names=["NORMAL", "BENIGN", "MALIGNANT"], zero_division=0
    ))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(test_labels, test_preds))

    torch.save(model.state_dict(), "results/baseline_cnn.pt")
    print("\nModel saved to results/baseline_cnn.pt")


if __name__ == "__main__":
    main()