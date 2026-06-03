import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. BACA DATASET
# ============================================================
df = pd.read_csv("tangerang_2005_2025.csv")
print("=" * 60)
print("DATA AWAL")
print("=" * 60)
print(df.head())
print(f"\nShape: {df.shape}")

# ============================================================
# 2. UBAH TANGGAL & SORT ASCENDING (penting untuk time split)
# ============================================================
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df = df.sort_values("tanggal").reset_index(drop=True)

# ============================================================
# 3. FITUR WAKTU
# ============================================================
df["bulan"] = df["tanggal"].dt.month
df["hari"]  = df["tanggal"].dt.day

# ============================================================
# 4. TARGET HUJAN
# ============================================================
df["target_hujan"] = (df["curah_hujan"] > 0).astype(int)

# ============================================================
# 5. FITUR LAG (suhu, kelembapan, hujan kemarin)
# ============================================================
df["suhu_kemarin"]       = df["suhu_rata"].shift(1)
df["kelembapan_kemarin"] = df["kelembapan"].shift(1)
df["hujan_kemarin"]      = df["target_hujan"].shift(1)

# Lag 2 hari (opsional, bisa memperkuat pola)
df["suhu_2halu"]         = df["suhu_rata"].shift(2)
df["kelembapan_2halu"]   = df["kelembapan"].shift(2)

# ============================================================
# 6. HAPUS DATA KOSONG
# ============================================================
print("\nMissing Values:")
print(df.isnull().sum())
print(df.describe())
df = df.dropna().reset_index(drop=True)

print(f"\nShape setelah dropna + lag: {df.shape}")

# ============================================================
# 7. DEFINISI FITUR & TARGET
# ============================================================
FEATURES = [
    # Fitur utama
    "suhu_rata",
    "kelembapan",
    "lama_penyinaran",
    "angin",
    # Fitur waktu
    "bulan",
    "hari",
    # Fitur lag
    "suhu_kemarin",
    "kelembapan_kemarin",
    "hujan_kemarin",
    "suhu_2halu",
    "kelembapan_2halu",
]

X = df[FEATURES]
y = df["target_hujan"]

print(f"\nJumlah fitur: {len(FEATURES)}")
print(f"Distribusi target → Tidak Hujan: {(y==0).sum()} | Hujan: {(y==1).sum()}")

# ============================================================
# 8. TIME-BASED SPLIT (80% train, 20% test)
# ============================================================
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTime-based Split:")
print(f"  Train : {len(X_train)} sampel ({df['tanggal'].iloc[0].date()} → {df['tanggal'].iloc[split_idx-1].date()})")
print(f"  Test  : {len(X_test)} sampel  ({df['tanggal'].iloc[split_idx].date()} → {df['tanggal'].iloc[-1].date()})")

# ============================================================
# 9. HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

param_grid = {
    "n_estimators" : [200, 500],
    "max_depth"    : [10, 15, 20],
    "min_samples_split": [5, 10],
    "min_samples_leaf" : [2, 4],
}

base_rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Gunakan TimeSeriesSplit untuk GridSearch agar konsisten dengan time split
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest Params : {grid_search.best_params_}")
print(f"Best F1 CV  : {grid_search.best_score_:.4f}")

# ============================================================
# 10. MODEL TERBAIK
# ============================================================
model = grid_search.best_estimator_

# ============================================================
# 11. CROSS-VALIDATION (k-fold, TimeSeriesSplit)
# ============================================================
print("\n" + "=" * 60)
print("CROSS-VALIDATION (TimeSeriesSplit, k=5)")
print("=" * 60)

cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="accuracy", n_jobs=-1)

print(f"CV Scores per fold : {[f'{s:.4f}' for s in cv_scores]}")
print(f"CV Mean Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 12. EVALUASI FINAL DI TEST SET
# ============================================================
print("\n" + "=" * 60)
print("EVALUASI FINAL (Test Set)")
print("=" * 60)

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

acc_train = accuracy_score(y_train, model.predict(X_train))
acc_test  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"Train Accuracy : {acc_train:.4f}")
print(f"Test  Accuracy : {acc_test:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")

gap = acc_train - acc_test
if gap > 0.10:
    print(f"⚠️  Overfitting terdeteksi! Gap: {gap:.4f}")
else:
    print(f"✅ Model generalisasi baik. Gap: {gap:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Tidak Hujan", "Hujan"]))

# ============================================================
# 13. VISUALISASI
# ============================================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Random Forest – Evaluasi Model Cuaca", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- (A) Feature Importance ---
ax1 = fig.add_subplot(gs[0, :])
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
colors = ["#2196F3" if f not in ["hujan_kemarin","suhu_kemarin","kelembapan_kemarin","suhu_2halu","kelembapan_2halu"]
          else "#FF9800" for f in feat_imp.index]
bars = ax1.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="white", height=0.6)
ax1.set_xlabel("Importance Score")
ax1.set_title("Feature Importance (Biru = Fitur Utama, Oranye = Fitur Lag)", fontweight="bold")
ax1.axvline(feat_imp.mean(), color="red", linestyle="--", alpha=0.7, label=f"Rata-rata: {feat_imp.mean():.3f}")
ax1.legend()
for bar, val in zip(bars, feat_imp.values):
    ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8)

# --- (B) Confusion Matrix ---
ax2 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred)
im = ax2.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax2)
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Tidak Hujan", "Hujan"])
ax2.set_yticklabels(["Tidak Hujan", "Hujan"])
ax2.set_xlabel("Prediksi"); ax2.set_ylabel("Aktual")
ax2.set_title("Confusion Matrix", fontweight="bold")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                 fontsize=14, color="white" if cm[i, j] > cm.max()/2 else "black")

# --- (C) CV Scores per Fold ---
ax3 = fig.add_subplot(gs[1, 1])
folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
bar_colors = ["#4CAF50" if s >= cv_scores.mean() else "#F44336" for s in cv_scores]
ax3.bar(folds, cv_scores, color=bar_colors, edgecolor="white")
ax3.axhline(cv_scores.mean(), color="navy", linestyle="--", label=f"Mean: {cv_scores.mean():.4f}")
ax3.set_ylim(0, 1)
ax3.set_ylabel("Accuracy")
ax3.set_title("Cross-Validation Scores per Fold", fontweight="bold")
ax3.legend()
for i, (fold, score) in enumerate(zip(folds, cv_scores)):
    ax3.text(i, score + 0.01, f"{score:.3f}", ha="center", fontsize=10)

plt.savefig("hasil_evaluasi_model.png", dpi=150, bbox_inches="tight")
print("\n✅ Visualisasi disimpan: hasil_evaluasi_model.png")
plt.show()

# ============================================================
# 14. SIMPAN MODEL (opsional)
# ============================================================
import joblib
joblib.dump(model, "model_cuaca_rf.pkl")
print("✅ Model disimpan: model_cuaca_rf.pkl")

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

from sklearn.metrics import roc_auc_score

roc = roc_auc_score(y_test, y_pred_prob[:,1])

print(f"ROC AUC Score : {roc:.4f}")