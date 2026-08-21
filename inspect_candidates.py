import pandas as pd

df = pd.read_csv("data/candidates_V2.csv")
print(f"Total rows: {len(df)}")
print(df["class"].value_counts())
print()

available_uids = set(pd.read_csv("data/subset0_series_uids.txt", header=None)[0]) if False else None
uids = open("data/subset0_series_uids.txt").read().strip().split("\n")
subset0_df = df[df["seriesuid"].isin(uids)]
print(f"Rows matching our 89 patients: {len(subset0_df)}")
print(subset0_df["class"].value_counts())