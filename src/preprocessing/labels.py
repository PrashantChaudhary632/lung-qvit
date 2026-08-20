"""
Malignancy label extraction for the 3-class Normal / Benign / Malignant task.

pylidc ships with a pre-built SQLite database containing all LIDC-IDRI
malignancy annotations -- no custom database population is needed. It's
queryable directly by series_instance_uid, which matches LUNA16's own
patient identifiers.
"""

from enum import IntEnum

import numpy as np
if not hasattr(np, "int"):
    np.int = int  # compatibility shim: pylidc 0.2.3 uses the deprecated np.int alias


class DiagnosisLabel(IntEnum):
    NORMAL = 0
    BENIGN = 1
    MALIGNANT = 2


def malignancy_score_to_label(mean_malignancy: float) -> DiagnosisLabel:
    """Map a mean radiologist malignancy score (1-5) to a 3-class label.

    LIDC-IDRI malignancy scale: 1=highly unlikely, 3=indeterminate, 5=highly
    suspicious. This threshold (<3 benign, >3 malignant, ==3 raises) is a
    common convention in LIDC-IDRI literature but should be cited and
    justified explicitly in your methodology, not treated as self-evident.
    """
    if mean_malignancy < 3:
        return DiagnosisLabel.BENIGN
    elif mean_malignancy > 3:
        return DiagnosisLabel.MALIGNANT
    else:
        raise ValueError(
            "Malignancy score == 3 (indeterminate) -- decide explicitly how "
            "to handle these (drop from dataset vs. separate 'indeterminate' "
            "class vs. manual review) rather than defaulting silently."
        )


def get_labels_via_pylidc(series_uid: str):
    """Fetch per-nodule malignancy consensus for one series via pylidc.

    Returns a list of dicts: [{"centroid_xyz": (x,y,z), "mean_malignancy": float,
    "label": DiagnosisLabel, "n_readers": int}, ...]

    Multiple radiologists independently annotate the same physical nodule in
    LIDC-IDRI. cluster_annotations() groups those independent annotations
    together so we get one consensus per nodule rather than one row per
    radiologist reading. mean_malignancy averages the 1-5 malignancy scores
    across the radiologists who annotated that nodule.
    """
    import pylidc as pl

    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
    if scan is None:
        return []

    nodules = scan.cluster_annotations(verbose=False)

    results = []
    for nodule_annotations in nodules:
        # LIDC-IDRI has at most 4 radiologist readers per scan. A cluster
        # larger than 4 means cluster_annotations() likely merged two
        # spatially-close but physically distinct nodules -- excluding
        # these rather than averaging across a corrupted group. This is a
        # documented known limitation, not a silent default.
        if len(nodule_annotations) > 4:
            continue

        malignancy_scores = [a.malignancy for a in nodule_annotations]
        mean_malignancy = sum(malignancy_scores) / len(malignancy_scores)

        centroids = [a.centroid for a in nodule_annotations]
        mean_centroid = tuple(
            sum(c[i] for c in centroids) / len(centroids) for i in range(3)
        )

        try:
            label = malignancy_score_to_label(mean_malignancy)
        except ValueError:
            # mean == 3, indeterminate -- skip rather than silently guess
            continue

        results.append({
            "centroid_xyz": mean_centroid,
            "mean_malignancy": mean_malignancy,
            "label": label,
            "n_readers": len(nodule_annotations),
        })

    return results


def diameter_heuristic_label(diameter_mm: float) -> DiagnosisLabel:
    """Crude diameter-based fallback label for smoke-testing ONLY.

    NOT a valid label source for reported experiments -- nodule size is
    correlated with but far from determinative of malignancy. Use this only
    to verify the data pipeline runs end-to-end before pylidc is set up.
    """
    if diameter_mm < 4:
        return DiagnosisLabel.BENIGN
    return DiagnosisLabel.MALIGNANT