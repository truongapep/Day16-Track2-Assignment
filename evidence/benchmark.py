import json, os, platform, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score)

SEED = 42
DATA = "/home/ubuntu/ml-benchmark/creditcard.csv"
OUT = "/home/ubuntu/ml-benchmark/benchmark_result.json"
THRESHOLD = 0.5

# 1. Load data
t0 = time.perf_counter()
df = pd.read_csv(DATA)
load_time = time.perf_counter() - t0

X = df.drop(columns=["Class"])
y = df["Class"]
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=SEED)

# 2. Train
model = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=31,
    reg_lambda=10, min_child_samples=100, min_child_weight=1,
    random_state=SEED, n_jobs=os.cpu_count(), verbose=-1)
t0 = time.perf_counter()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
train_time = time.perf_counter() - t0
best_iter = int(model.best_iteration_)

# 3. Evaluate on the test set
proba = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
metrics = {
    "auc_roc": float(roc_auc_score(y_test, proba)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "f1_score": float(f1_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred)),
}

# 4. Inference latency (1 row)
row = X_test.iloc[[0]]
for _ in range(20):
    model.predict_proba(row, num_iteration=best_iter)
lat = []
for _ in range(200):
    t0 = time.perf_counter()
    model.predict_proba(row, num_iteration=best_iter)
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)

# 5. Inference throughput (1000 rows)
batch = X_test.iloc[:1000]
model.predict_proba(batch, num_iteration=best_iter)
bt = []
for _ in range(20):
    t0 = time.perf_counter()
    model.predict_proba(batch, num_iteration=best_iter)
    bt.append(time.perf_counter() - t0)
bt_mean = float(np.mean(bt))

result = {
    "load_time_sec": round(load_time, 4),
    "train_time_sec": round(train_time, 4),
    "best_iteration": best_iter,
    **{k: round(v, 6) for k, v in metrics.items()},
    "inference_latency_1row_ms": {
        "mean": round(float(lat.mean()), 4),
        "median": round(float(np.median(lat)), 4),
        "p95": round(float(np.percentile(lat, 95)), 4),
        "runs": 200,
    },
    "inference_throughput_1000rows": {
        "batch_time_ms_mean": round(bt_mean * 1000, 4),
        "rows_per_sec": round(1000 / bt_mean, 2),
        "runs": 20,
    },
    "setup": {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "fraud_in_test": int(y_test.sum()),
        "seed": SEED,
        "threshold": THRESHOLD,
        "hyperparams": {"n_estimators_max": 1000, "learning_rate": 0.05, "num_leaves": 31, "reg_lambda": 10, "min_child_samples": 100, "min_child_weight": 1, "early_stopping_rounds": 50, "early_stopping_metric": "binary_logloss"},
        "n_jobs": os.cpu_count(),
        "lightgbm": lgb.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
    },
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
print("Saved:", OUT)
