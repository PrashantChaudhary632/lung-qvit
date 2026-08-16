"""
Malignancy label extraction for the 3-class Normal / Benign / Malignant task.

LUNA16's own annotations.csv gives nodule *location*, not malignancy.
Malignancy scores (1-5 scale, per-radiologist) live in the original LIDC-IDRI
XML annotation files. `pylidc` parses those XMLs and links them to LUNA16
series UIDs via a local SQLite DB it builds from the raw LIDC-IDRI download.

pylidc setup (one-time, once you have LIDC-IDRI XML data):
    1. Create a .pylidcrc config file pointing pylidc at your DICOM directory.
    2. pylidc needs the DICOM directory structure to build its index the
       first time — see pylidc docs for the lightest-weight way to get this.
"""

from enum import IntEnum


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
            "Malignancy score == 3 (indeterminate) — decide explicitly how "
            "to handle these (drop from dataset vs. separate 'indeterminate' "
            "class vs. manual review) rather than defaulting silently."
        )


def get_labels_via_pylidc(series_uid: str):
    """Fetch per-nodule malignancy consensus for one series via pylidc.

    NOT YET IMPLEMENTED — stub to fill in once pylidc + LIDC-IDRI XML data
    is set up locally. We'll implement this together once you have that
    data, rather than guessing at correct nodule-to-series matching now.
    """
    raise NotImplementedError(
        "Set up pylidc + LIDC-IDRI XML annotations first, then implement "
        "using pylidc.query(pylidc.Scan) and cluster_annotations() to get "
        "per-nodule consensus malignancy."
    )


def diameter_heuristic_label(diameter_mm: float) -> DiagnosisLabel:
    """Crude diameter-based fallback label for smoke-testing ONLY.

    NOT a valid label source for reported experiments — nodule size is
    correlated with but far from determinative of malignancy. Use this only
    to verify the data pipeline runs end-to-end before pylidc is set up.
    """
    if diameter_mm < 4:
        return DiagnosisLabel.BENIGN
    return DiagnosisLabel.MALIGNANT