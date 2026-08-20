import sys
sys.path.insert(0, "src/preprocessing")

from pathlib import Path
import numpy as np
if not hasattr(np, "int"):
    np.int = int

import pylidc as pl

uids = Path("data/subset0_series_uids.txt").read_text().strip().split("\n")

for uid in uids:
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
    if scan is None:
        continue
    nodules = scan.cluster_annotations(verbose=False)
    for cluster in nodules:
        if len(cluster) > 4:
            print(f"ANOMALY: {scan.patient_id}, cluster size {len(cluster)}")
            for ann in cluster:
                print(f"  annotation id={ann.id}, malignancy={ann.malignancy}, centroid={ann.centroid}")