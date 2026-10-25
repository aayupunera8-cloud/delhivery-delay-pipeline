"""Synthetic shipment data for the Delay Intelligence capstone.
Run once:  python generate_data.py   (writes data/hubs.csv, data/raw_shipments.csv, data/new_shipments.csv)
Do not edit any constant: every number in the brief was measured on this exact output."""
import os
import numpy as np
import pandas as pd

HUBS = [
    ("HUB01", "Mumbai", "West", 1, 1500), ("HUB02", "Delhi", "North", 1, 1600),
    ("HUB03", "Bengaluru", "South", 1, 1450), ("HUB04", "Hyderabad", "South", 1, 1350),
    ("HUB05", "Chennai", "South", 1, 1300), ("HUB06", "Pune", "West", 1, 1250),
    ("HUB07", "Kolkata", "East", 1, 1200), ("HUB08", "Ahmedabad", "West", 1, 1150),
    ("HUB09", "Jaipur", "North", 2, 950), ("HUB10", "Lucknow", "North", 2, 900),
    ("HUB11", "Chandigarh", "North", 2, 850), ("HUB12", "Indore", "West", 2, 820),
    ("HUB13", "Nagpur", "West", 2, 780), ("HUB14", "Coimbatore", "South", 2, 740),
    ("HUB15", "Kochi", "South", 2, 700), ("HUB16", "Surat", "West", 2, 650),
    ("HUB17", "Patna", "East", 3, 520), ("HUB18", "Ranchi", "East", 3, 480),
    ("HUB19", "Guwahati", "East", 3, 450), ("HUB20", "Raipur", "East", 3, 420),
    ("HUB21", "Bhubaneswar", "East", 3, 390), ("HUB22", "Dehradun", "North", 3, 350),
    ("HUB23", "Siliguri", "East", 3, 310), ("HUB24", "Agartala", "East", 3, 270),
]
DIST_MU = [180, 320, 480]      # mean lane distance (km) for hub tier 1 / 2 / 3
DIST_SD = [50, 90, 150]
STOPS_MU = [55, 70, 85]        # mean stops on the rider route for hub tier 1 / 2 / 3
HUB_EFFECT = [0, 2, 4]         # extra hours for hub tier 1 / 2 / 3
# How a hub really operates (1 = metro-like, 2 = regional, 3 = stretched). Four hubs operate unlike their tier label.
OPERATING_TIER = {"HUB07": 2, "HUB13": 3, "HUB15": 3, "HUB22": 2}
BASE_HOURS = {"Express": 10, "Standard": 28, "Economy": 58}


def simulate(n_per_hub, seed):
    """Clean shipments, n_per_hub rows for each of the 24 hubs, in hub order."""
    np.random.seed(seed)
    hubs = pd.DataFrame(HUBS, columns=["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"])
    n = n_per_hub * len(hubs)
    op_tier = [OPERATING_TIER.get(h, t) for h, t in zip(hubs["hub_id"], hubs["hub_tier"])]
    tier = np.repeat(op_tier, n_per_hub) - 1
    hub_id = np.repeat(hubs["hub_id"].values, n_per_hub)
    distance = np.clip(np.random.normal(np.array(DIST_MU)[tier], np.array(DIST_SD)[tier]), 15, 1500).round(1)
    stops = np.clip(np.random.normal(np.array(STOPS_MU)[tier], 10), 20, 140).round().astype(int)
    weight = np.clip(np.random.lognormal(0.6, 0.7, n), 0.1, 40).round(2)
    value = np.random.lognormal(7.0, 0.8, n).round().astype(int)
    service = np.random.choice(["Express", "Standard", "Economy"], n, p=[0.2, 0.55, 0.25])
    pay = np.random.choice(["Prepaid", "COD"], n, p=[0.6, 0.4])
    rain = (np.random.rand(n) < 0.18).astype(int)
    day = np.random.randint(0, 90, n)
    hour = np.random.randint(6, 22, n)
    minute = np.random.randint(0, 60, n)
    ts = pd.Timestamp("2026-07-01") + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour * 60 + minute, unit="m")
    sale = ((ts >= pd.Timestamp("2026-09-15")) & (ts < pd.Timestamp("2026-09-29"))).astype(int)
    weekend = (ts.dayofweek >= 5).astype(int)
    hours = (np.array([BASE_HOURS[s] for s in service]) + 0.03 * distance + 0.10 * stops
             + np.array(HUB_EFFECT)[tier] + 0.12 * weight + 4 * rain + 7 * sale
             + 3 * (pay == "COD") + 2 * weekend + 3 * (hour >= 18)
             + 12 * sale * (service == "Express") + 8 * (distance > 400) * rain
             + np.random.normal(0, 4, n)).round(2)
    return pd.DataFrame({
        "hub_id": hub_id, "service_type": service, "payment_mode": pay,
        "parcel_weight_kg": weight, "distance_km": distance, "declared_value": value,
        "n_stops": stops, "rain_flag": rain, "sale_event_flag": sale, "dispatch_time": ts,
        "label_print_batch": np.random.randint(1000, 9999, n),
        "scanner_build_no": np.random.randint(1, 40, n),
        "tape_roll_no": np.random.randint(1, 500, n),
        "sticker_serial_mod": np.random.uniform(0, 1, n).round(3),
        "shelf_slot_no": np.random.randint(1, 300, n),
        "delivery_hours": hours,
    })


def make_messy(df, seed, n_missing, n_case, n_sentinel, n_dupes, id_start):
    """Turn a clean table into a raw export: text columns, gaps, duplicates, a sentinel."""
    np.random.seed(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    df.insert(0, "shipment_id", ["SHP" + str(id_start + i) for i in range(n)])
    df["dispatch_time"] = df["dispatch_time"].dt.strftime("%Y-%m-%d %H:%M")
    df["declared_value"] = df["declared_value"].map(lambda v: "\u20b9" + format(v, ","))
    for col, k in [("distance_km", n_missing[0]), ("parcel_weight_kg", n_missing[1]),
                   ("declared_value", n_missing[2]), ("payment_mode", n_missing[3])]:
        df.loc[np.random.choice(n, k, replace=False), col] = np.nan
    pick = np.random.choice(n, n_case, replace=False)
    shapes = np.random.randint(0, 3, n_case)
    for i, s in zip(pick, shapes):
        v = df.at[i, "service_type"]
        df.at[i, "service_type"] = [v.upper(), v.lower(), "  " + v + " "][s]
    if n_sentinel:
        df.loc[np.random.choice(n, n_sentinel, replace=False), "parcel_weight_kg"] = 999.0
    if n_dupes:
        df = pd.concat([df, df.iloc[np.random.choice(n, n_dupes, replace=False)]], ignore_index=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    hubs = pd.DataFrame(HUBS, columns=["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"])
    hubs.to_csv("data/hubs.csv", index=False)
    raw = make_messy(simulate(50, 2604), seed=7, n_missing=(36, 24, 12, 18),
                     n_case=120, n_sentinel=5, n_dupes=24, id_start=10001)
    raw.to_csv("data/raw_shipments.csv", index=False)
    new = make_messy(simulate(2, 99), seed=8, n_missing=(3, 2, 1, 2),
                     n_case=6, n_sentinel=0, n_dupes=0, id_start=90001)
    new = new.drop(columns=["delivery_hours"])
    new.to_csv("data/new_shipments.csv", index=False)
    print(len(hubs), len(raw), len(new))
