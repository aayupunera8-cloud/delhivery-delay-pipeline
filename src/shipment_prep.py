"""Shared constants and preparation functions used by every part of the pipeline."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians)."""
    df = raw.copy()
    # 1. drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    # 2. service_type: strip + title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    # 3. payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    # 4. declared_value: remove rupee sign and commas -> declared_value_inr
    cleaned = (df["declared_value"].astype("string")
               .str.replace("\u20b9", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    df["declared_value_inr"] = pd.to_numeric(cleaned, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])
    # 5. dispatch_time -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    # 6. sentinel weights (> 100) -> missing
    df["parcel_weight_kg"] = df["parcel_weight_kg"].astype(float)
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    # 7. median fill
    if medians is None:
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])
    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and derive model features. Never uses delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    out = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    out["promised_hours"] = out["service_type"].map(PROMISED_HOURS).astype(int)
    out["is_cod"] = (out["payment_mode"] == "COD").astype(int)
    out["is_weekend"] = (out["dispatch_time"].dt.dayofweek >= 5).astype(int)
    out["dispatch_hour"] = out["dispatch_time"].dt.hour.astype(int)
    out["log_declared_value"] = np.log(out["declared_value_inr"])
    return out
