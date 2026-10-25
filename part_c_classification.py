"""Part C: delay classifier, feature signal, tuning and alert threshold."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]

# 1. stratified split (two steps)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "| delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_row(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0)}


# 2. baseline + three classifiers on validation (0.5 rule)
model_rows = [{"model": "blind_all_on_time", **cm_row(y_val, np.zeros(len(y_val), dtype=int))}]
logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **cm_row(y_val, logit.predict(X_val))})

best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1 + 1e-12:               # strict: ties keep the smaller depth
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
model_rows.append({"model": f"decision_tree_depth_{best_d}", **cm_row(y_val, tree.predict(X_val))})
rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **cm_row(y_val, rf_default.predict(X_val))})
pd.DataFrame(model_rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. feature signal
sk_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sk_all.scores_,
                   "rf_importance": rf_default.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
noise_share = fs.loc[fs["is_noise"], "rf_importance"].sum()
print("noise share of RF importance:", round(noise_share, 4))

sel_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
    p.fit(X_train, y_train)
    sel_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(X_val))})
pd.DataFrame(sel_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning (selector fitted on training rows only; test rows untouched)
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xt_train, Xt_val = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xt_train, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xt_train, y_train)
default7 = RandomForestClassifier(random_state=42).fit(Xt_train, y_train)

DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
cands = {
    "grid_search": (gs.best_params_, gs.best_score_, len(gs.cv_results_["params"]), 5, f1_score(y_val, gs.best_estimator_.predict(Xt_val))),
    "randomized_search": (rs.best_params_, rs.best_score_, len(rs.cv_results_["params"]), 5, f1_score(y_val, rs.best_estimator_.predict(Xt_val))),
    "default_forest_same_7_features": (DEFAULT_PARAMS, np.nan, 1, 0, f1_score(y_val, default7.predict(Xt_val))),
}
tune_rows = []
for name, (params, cv_acc, n_cand, folds, vf1) in cands.items():
    tune_rows.append({"search": name, "candidates": n_cand, "cv_folds": folds, "total_fits": (n_cand * folds) if folds else 1,
                      "best_params": json.dumps({k: (None if params[k] is None else int(params[k])) for k in sorted(params)}),
                      "best_cv_accuracy": cv_acc, "val_f1": vf1})
tuning = pd.DataFrame(tune_rows).round(4)
tuning.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. final configuration: highest val_f1 (as written in the CSV); ties -> default, grid, randomized
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
best_val = tuning["val_f1"].max()
chosen = next(n for n in priority if tuning.loc[tuning["search"] == n, "val_f1"].iloc[0] == best_val)
params = cands[chosen][0]
final_model = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                        ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# 6. alert threshold on validation probabilities
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
p_val = final_model.predict_proba(X_val)[:, 1]
th_rows = [{"threshold": t, **cm_row(y_val, (p_val >= t).astype(int))} for t in thresholds]
th = pd.DataFrame(th_rows).round(4)
th.to_csv("outputs/part_c_thresholds.csv", index=False)
ok = th.loc[th["recall"] >= 0.80, "threshold"]
threshold = float(ok.max()) if len(ok) else 0.1

# 7. one test evaluation
p_test = final_model.predict_proba(X_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **cm_row(y_test, (p_test >= threshold).astype(int))}]).round(4) \
    .to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen:", chosen, params, "threshold:", threshold)
