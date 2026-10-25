"""Part B: delivery-hours regression (split first, scale on training rows only)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["delivery_hours"]

# 1. split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaling without leakage
scaler = StandardScaler().fit(X_train)
train_scaled, test_scaled = scaler.transform(X_train), scaler.transform(X_test)
pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [len(X_train), len(X_val), len(X_test),
              np.abs(train_scaled.mean(axis=0)).max(), np.abs(test_scaled.mean(axis=0)).max()],
}).round(4).to_csv("outputs/part_b_checks.csv", index=False)
# (the 4-decimal rounding shows 0.0 for the train value; the raw value is < 1e-9)
print("train_scaled_max_abs_mean (unrounded):", np.abs(train_scaled.mean(axis=0)).max())


def scores(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"MAE": mean_absolute_error(y_true, pred), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, pred)}


# 3. baseline
rows = [{"model": "mean_baseline", "split": "val", **scores(y_val, np.full(len(y_val), y_train.mean()))}]

# 4. tree depth diagnostic
depth_rows, best_depth, best_val = [], None, np.inf
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tr = np.sqrt(mean_squared_error(y_train, t.predict(X_train)))
    va = np.sqrt(mean_squared_error(y_val, t.predict(X_val)))
    depth_rows.append({"max_depth": "unlimited" if d is None else d, "train_rmse": tr, "val_rmse": va})
    if va < best_val - 1e-12:           # strict: ties keep the smaller depth
        best_val, best_depth = va, d
pd.DataFrame(depth_rows).round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. four candidates
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, m in candidates.items():
    m.fit(X_train, y_train)
    s = scores(y_val, m.predict(X_val))
    val_rmse[name] = s["RMSE"]
    rows.append({"model": name, "split": "val", **s})

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": np.sqrt(mean_squared_error(y_val, p)), "val_r2": r2_score(y_val, p)})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. cross-validation of the two best, shared KFold, then one test score
ranking = sorted(val_rmse, key=val_rmse.get)
top2 = ranking[:2]
X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in top2:
    r2s = cross_val_score(candidates[name], X_tv, y_tv, cv=kf)
    cv_rows.append({"model": name, "mean_r2": r2s.mean(), "std_r2": r2s.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranking[0]
rows.append({"model": final_name, "split": "test", **scores(y_test, candidates[final_name].predict(X_test))})
pd.DataFrame(rows)[["model", "split", "MAE", "MSE", "RMSE", "R2"]].round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print("best depth:", best_depth, "| ranking:", ranking, "| final:", final_name)
