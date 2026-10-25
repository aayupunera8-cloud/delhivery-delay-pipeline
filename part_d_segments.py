"""Part D (1/3): hub profile, K-Means segments, separation evidence and tier-vs-segment check."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile (groupby sorts by ascending hub_id)
profile = table.groupby("hub_id").agg(
    shipments=("shipment_id", "count"), mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")).reset_index().sort_values("hub_id").reset_index(drop=True)
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. K-Means on the scaled four profile columns (hubs in ascending hub_id order)
COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[COLS])
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)


# 3. separation evidence for K = 2..5
def across_over_within(Z, labels):
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(len(Z), k=1)
    dist, same = d[iu], (labels[:, None] == labels[None, :])[iu]
    return dist[~same].mean() / dist[same].mean()


sep = []
for K in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z).labels_
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": K, "across_over_within": across_over_within(Z, lab),
                "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
pd.DataFrame(sep).round(4).to_csv("outputs/part_d_separation.csv", index=False)

# 4. segment names by mean delay rate of the cluster's hubs
profile["cluster"] = km.labels_
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

seg = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

seg_prof = (profile.groupby("segment").agg(hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                                           delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                                           mean_n_stops=("mean_n_stops", "mean")).reset_index())
seg_prof["segment"] = pd.Categorical(seg_prof["segment"], ["Standard", "Watch", "Escalate"])
seg_prof = seg_prof.sort_values("segment")
seg_prof.round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
seg[seg["segment"] != seg["hub_tier"].map(implied)].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print(open("outputs/part_d_separation.csv").read())
