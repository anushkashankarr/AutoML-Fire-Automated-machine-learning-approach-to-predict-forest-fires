#!/usr/bin/env python3
"""
Ecozone-aware AutoML wildfire detection system
- DBSCAN + macro clustering
- Zone-wise models
- Optional Optuna tuning
- Per-zone reports
"""
import copy
import os
import json
import time
import warnings
import argparse
import shap
import seaborn as sns


import sys
import smogn
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")

# == Core numerics ==
import numpy as np
import pandas as pd


# == Visualization ==
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# == SKLEARN core ==
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score
)
from sklearn.base import clone

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    average_precision_score,
    fbeta_score,
    r2_score,
)

import geopandas as gpd
from shapely.geometry import Point

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC

# Parallel execution
from concurrent.futures import ProcessPoolExecutor, as_completed

# Persistence
import joblib

# Density plot (zone map)
import seaborn as sns
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True

    #  CPU SAFETY ----------
    try:
        torch.set_num_threads(1) # Torch thread limit set to 1 for safer CPU usage"
    except:
        pass

except Exception:
    TORCH_AVAILABLE = False
    print("[!] Torch unavailable — DL model search disabled")



try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True

    # Silent logging — disables LightGBM warnings/info
    try:
        lgb.basic._config.set("verbosity", -1)
    except Exception:
        pass

    class SilentLogger:
        def info(self, msg): pass
        def warning(self, msg): pass

    try:
        lgb.register_logger(SilentLogger())
    except Exception:
        pass

except Exception as e:
    LIGHTGBM_AVAILABLE = False
    print("[!] LightGBM not available:", e)



import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
import os
os.environ["LIGHTGBM_VERBOSE"] = "0"

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN

class CFG:
    target_class = 'fire_detected'
    target_reg = 'fire_intensity'
    drop_cols = ['system:index', 'geometry_json', '.geo', 'date']

    eps = 25
    min_samples = 120
    test_size = 0.2
    random_state = 42

    optuna_trials = 30
    ensemble_n = 3
    min_zone_rows = 200

def save_fig(fig, fname):
    try:
        fig.tight_layout()
        fig.savefig(fname, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print(f"[i] Saved figure : {fname}")
    except Exception as e:
        print("[!] save_fig error:", e, fname)

def load_data(path):
    print('[i] load_data: path =', path)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Print raw column names
    print("[i] Raw columns:", list(df.columns))
    print(f"[i] Data loaded : shape = {df.shape}")

    # Drop unwanted columns
    df = df.drop(columns=[c for c in CFG.drop_cols if c in df.columns], errors='ignore')

    # Print columns after dropping unused
    print("[i] After dropping CFG.drop_cols:", list(df.columns))

    return df

def engineer_features(df):
    print('[i] engineer_features: start')

    # --- FIX: Clean brackets from dirty CSV data ---
    # This handles values like "[0.5]", "[5E-1]", or "['12.3']"
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to clean brackets if they exist
            try:
                # Check if it looks like a string number with brackets
                mask = df[col].astype(str).str.contains(r'\[.*\]')
                if mask.any():
                    print(f"[!] Cleaning brackets in column: {col}")
                    df[col] = df[col].astype(str).str.replace(r'[\[\]]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                pass 
                
    # Force target columns to numeric explicitly
    if CFG.target_reg in df.columns:
        df[CFG.target_reg] = pd.to_numeric(df[CFG.target_reg], errors='coerce')
        # Fill NaNs in target if any resulted from bad parsing (optional but safer)
        if df[CFG.target_reg].isna().sum() > 0:
            print(f"[!] Found NaNs in target {CFG.target_reg} after cleaning. Filling with median.")
            df[CFG.target_reg] = df[CFG.target_reg].fillna(df[CFG.target_reg].median())

    df = df.copy()

    if 'temperature_2m' in df.columns:
        df['temperature_2m'] = df['temperature_2m'].replace(0, np.nan)
        df['temperature_2m'] = df['temperature_2m'].fillna(df['temperature_2m'].median())
        df['temp_c'] = df['temperature_2m'] - 273.15

    if 'NDVI' in df.columns:
        # Ensure NDVI is numeric before scaling
        df['NDVI'] = pd.to_numeric(df['NDVI'], errors='coerce').fillna(0)
        df['ndvi_scaled'] = df['NDVI'] / 10000.0

    if 'vapor_pressure_deficit' in df.columns and 'wind_speed' in df.columns:
        # Ensure inputs are numeric
        df['vapor_pressure_deficit'] = pd.to_numeric(df['vapor_pressure_deficit'], errors='coerce').fillna(0)
        df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce').fillna(0)
        df['vpd_wind_idx'] = df['vapor_pressure_deficit'] * df['wind_speed']

    if 'latitude' in df.columns and 'longitude' in df.columns:
        df[['latitude', 'longitude']] = df[['latitude','longitude']].fillna(method='ffill').fillna(0)

    # Print new columns and total count
    print("[i] After Feature Engineering columns:", list(df.columns))
    print('[i] engineer_features: done : columns =', len(df.columns))

    return df

def extract_ecozones(df, eps=None, min_samples=None):
    print("[i] hybrid clustering (probabilistic routing)...")

    # ========= 1) COARSE CLUSTERING (fast) ==========
    k = max(9, min(15, int(np.sqrt(len(df)/40000))))
    coords = df[['latitude', 'longitude']].values

    km = KMeans(n_clusters=k, random_state=42)
    df['macro_zone'] = km.fit_predict(coords)
    print(f"[i] KMeans macro clusters = {k}")

    zones_full = np.full(len(df), -1)
    zone_counter = 0

    # ========= 2) DBSCAN inside each macro region ==========
    for mz in range(k):
        sub = df[df['macro_zone'] == mz]

        if len(sub) < 2000:
            continue

        coords_rad = np.radians(sub[['latitude', 'longitude']])

        radius_km = eps if eps is not None else 15
        min_pts = min_samples if min_samples is not None else 80

        db = DBSCAN(
            eps=radius_km / 6371.0,
            min_samples=min_pts,
            metric="haversine"
        ).fit(coords_rad)

        labels = db.labels_
        if labels is None:
            continue

        for cid in np.unique(labels):
            if cid == -1:  # skip noise
                continue

            mask = (labels == cid)
            if mask.sum() < 150:  # small ignored
                continue
            zones_full[sub.index[mask]] = zone_counter
            zone_counter += 1

    print(f"[i] DBSCAN refined ecozones =", len(np.unique(zones_full[zones_full != -1])))

    # ========= 3) Assign remaining using probabilistic router ==========
    valid = df[zones_full != -1]

    if len(valid) == 0:
        print("[!] No DBSCAN ecozones — falling back to macro KNN clusters.")
        df['ecozone'] = df['macro_zone']
        final_count = df['ecozone'].nunique()
        print(f"[i] Final ecozone count = {final_count}")
        return df, None, None

    # ===== Spatially-corrected coordinates (longitude shrinkage) =====
    valid_xy = valid.copy()
    valid_xy['x'] = valid['longitude'] * np.cos(np.radians(valid['latitude']))
    valid_xy['y'] = valid['latitude']

    df_xy = df.copy()
    df_xy['x'] = df['longitude'] * np.cos(np.radians(df['latitude']))
    df_xy['y'] = df['latitude']

    # ========= routing remaining points using kNN ==========
    knn = KNeighborsClassifier(n_neighbors=40, weights="distance")

    valid_labels = zones_full[valid.index]
    knn.fit(valid_xy[['x', 'y']], valid_labels)

    df['ecozone'] = knn.predict(df_xy[['x', 'y']])

    print("[i] Final ecozone count =", df['ecozone'].nunique())

    return df, None, knn

def __init__(self, model_type, input_dim, hidden_dim, dropout):
  super().__init__()

  self.model_type = model_type

  if model_type == "lstm":
      self.seq = nn.LSTM(input_dim, hidden_dim, batch_first=True)
  elif model_type == "gru":
      self.seq = nn.GRU(input_dim, hidden_dim, batch_first=True)
  else:
      layer = nn.TransformerEncoderLayer(
          d_model=input_dim, nhead=2, dropout=dropout
      )
      self.seq = nn.TransformerEncoder(layer, num_layers=2)

  self.head = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim // 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim // 2, 1),
  )

def forward(self, x):
    if self.model_type in {"lstm", "gru"}:
        out, _ = self.seq(x)
    else:
        out = self.seq(x)

    z = out[:, -1, :]  # last timestep embedding
    return torch.sigmoid(self.head(z))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: sigmoid outputs
        targets: smoothed / true labels
        """

        bce = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce)

        return ((1-pt) ** self.gamma * bce).mean()

def make_sequence(X, seq_len):
    n = len(X)
    if n < seq_len: 
        seq_len = n
    xs = []
    for i in range(n):
        start = max(0, i-seq_len+1)
        win = X[start:i+1]
        if len(win) < seq_len:
            pad = np.zeros((seq_len-len(win), X.shape[1]))
            win = np.vstack((pad, win))
        xs.append(win)
    return np.stack(xs)

class FireDL(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, dropout):
        super().__init__()

        self.model_type = model_type

        if model_type == "lstm":
            self.seq = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            core_out_dim = hidden_dim

        elif model_type == "gru":
            self.seq = nn.GRU(input_dim, hidden_dim, batch_first=True)
            core_out_dim = hidden_dim

        else:  # transformer encoder
            n_heads = 2
            d_model = input_dim

            # Ensure divisible dimension
            if d_model % n_heads != 0:
                d_model = n_heads * ((d_model // n_heads) + 1)
                self.proj = nn.Linear(input_dim, d_model)
            else:
                self.proj = None

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout,
                batch_first=False  # transformer default
            )
            self.seq = nn.TransformerEncoder(layer, num_layers=2)
            core_out_dim = d_model

        self.head = nn.Sequential(
            nn.Linear(core_out_dim, core_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(core_out_dim // 2, 1),
        )

    def forward(self, x):
        if self.model_type in {"lstm", "gru"}:
            out, _ = self.seq(x)  # (batch, seq, hidden)
        else:
            # apply embedding projection if created
            if hasattr(self, "proj") and self.proj is not None:
                x = self.proj(x)

            # transformer expects (seq_len, batch, dim)
            x = x.permute(1, 0, 2)

            out = self.seq(x)  # (seq, batch, dim)

            # back to (batch, seq, dim)
            out = out.permute(1, 0, 2)

        z = out[:, -1, :]  # last timestep
        return torch.sigmoid(self.head(z))

class DLClassifierWrapper:
    """
    Sklearn-style wrapper around FireDL so that:
    - predict_proba(X) available
    - registry + predict_zone code same reh sakta hai
    """
    def __init__(self, dl_model, seq_len, device="cpu"):
        self.model = dl_model.to(device)
        self.seq_len = seq_len
        self.device = device

    def _prepare_sequences(self, X):
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)
        Xs = make_sequence(X_np, self.seq_len)
        Xs = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        return Xs

    def predict_proba(self, X):
        self.model.eval()
        Xs = self._prepare_sequences(X)
        with torch.no_grad():
            probs = self.model(Xs).cpu().numpy().flatten()

        # numeric safety
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        proba_2d = np.vstack([1.0 - probs, probs]).T   # shape (n, 2)
        return proba_2d

def prepare_loaded_model(model):
    """
    Ensures DL models load safely:
    - moved to CPU
    - switched to eval mode
    """
    try:
        # If it's DL wrapper, it has .model inside
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.to("cpu")
            model.model.eval()
    except:
        pass

    return model

def run_optuna_tuning(X_train, y_train, X_valid, y_valid, is_classifier=True, n_trials=20):
    """
    Improved Optuna tuner with DL support.
    Handles:
    ✔ RF / XGB / LGB / CAT / DL
    ✔ pruning
    ✔ safety recall
    ✔ returns best model + params
    """
    best_score = -999
    best_params = None
    best_model = None
    best_model_obj = None   # <<< DL wrapper stored here

    trial_history = []
    no_gain_count = 0
    patience = max(5, n_trials // 4)

    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    model_options = ["rf", "xgb", "lgb", "cat"] if is_classifier else ["rf", "xgb", "lgb", "cat"]
    print(f"[OPTUNA] Search candidates = {model_options}")
    print(f"[OPTUNA] DL enabled: {TORCH_AVAILABLE}")

    # ---------------------------------------------------
    def objective(trial):
        nonlocal best_score, best_params, best_model, best_model_obj, no_gain_count

        model_name = trial.suggest_categorical("model", model_options)

        # ===================== DL BRANCH =====================
        if model_name == "dl":
            if not TORCH_AVAILABLE:
                return 1e-6

            params = {
                "model_type": trial.suggest_categorical("dl_type", ["lstm", "gru", "transformer"]),
                "hidden_dim": trial.suggest_int("hidden_dim", 32, 128),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 5e-3),
                "seq_len": trial.suggest_int("seq_len", 7, 30),
                "gamma": trial.suggest_float("gamma", 1.0, 5.0),
                "label_smooth": trial.suggest_float("label_smooth", 0.02, 0.10),
            }

            # ★ HARD BLOCK: skip Transformer when dataset too large
            if params["model_type"] == "transformer" and X_train.shape[0] > 100000:
                print("⚠ Transformer skipped due to large dataset size")
                return 1e-6

            # ----- build sequences -----
            Xs_train = torch.tensor(make_sequence(X_train.values, params["seq_len"]), dtype=torch.float32).to(device)
            Xs_valid = torch.tensor(make_sequence(X_valid.values, params["seq_len"]), dtype=torch.float32).to(device)
            ys_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
            ys_valid = torch.tensor(y_valid.values.reshape(-1, 1), dtype=torch.float32).to(device)

            smooth = params["label_smooth"]
            ys_train = ys_train * (1 - smooth) + smooth / 2
            ys_valid = ys_valid * (1 - smooth) + smooth / 2

            dl = FireDL(params["model_type"], X_train.shape[1], params["hidden_dim"], params["dropout"]).to(device)
            optimizer = torch.optim.Adam(dl.parameters(), lr=params["lr"])
            criterion = FocalLoss(gamma=params["gamma"])

            best_loss = 1e9
            ep_patience = 3

            for epoch in range(15):
                dl.train()
                preds = dl(Xs_train)
                loss = criterion(preds, ys_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # validation scoring
                dl.eval()
                with torch.no_grad():
                    val_preds = dl(Xs_valid)
                    val_loss = criterion(val_preds, ys_valid).item()

                trial.report(-val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if val_loss < best_loss:
                    best_loss = val_loss
                else:
                    ep_patience -= 1
                    if ep_patience == 0:
                        break

            dl.eval()
            with torch.no_grad():
                val_probs = dl(Xs_valid).cpu().numpy().flatten()

            # ====== FAIRNESS CHECKS ======
            pr_auc = average_precision_score(y_valid, val_probs)

            best_local_rec = max(
                recall_score(y_valid, (val_probs >= t).astype(int))
                for t in np.linspace(0.1, 0.9, 7)
            )
            best_f2 = max(
                fbeta_score(y_valid, (val_probs >= t).astype(int), beta=2)
                for t in np.linspace(0.1, 0.9, 7)
            )
            score = pr_auc

        # ================= TREE MODELS =================
        else:
            if model_name == "lgb":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 2, 12),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                    "random_state": CFG.random_state,
                }
                model = (LGBMClassifier if is_classifier else LGBMRegressor)(**params)

            elif model_name == "xgb":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 2, 12),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
                    "random_state": CFG.random_state,
                }
                model = (xgb.XGBClassifier if is_classifier else xgb.XGBRegressor)(**params)

            elif model_name == "cat":
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 400),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.4),
                    "random_seed": CFG.random_state
                }
                model = (CatBoostClassifier if is_classifier else CatBoostRegressor)(**params, verbose=0)

            else:  # RF
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "random_state": CFG.random_state,
                }
                model = (RandomForestClassifier if is_classifier else RandomForestRegressor)(**params)

            model.fit(X_train, y_train)

            if is_classifier:
                probs = model.predict_proba(X_valid)[:, 1]
                best_local_rec = max(
                    recall_score(y_valid, (probs >= t).astype(int))
                    for t in np.linspace(0.1, 0.9, 7)
                )

                score = average_precision_score(y_valid, probs)
            else:
                score = r2_score(y_valid, model.predict(X_valid))

        # ================= TRACK BEST =================
        trial_history.append((trial.number, model_name, score))

        if score > best_score:
            print(f"  [OPTUNA] Trial {trial.number}: NEW BEST {model_name} {score:.4f}")
            best_score = score
            best_model = model_name
            best_params = params.copy()
            no_gain_count = 0

            if model_name == "dl":
                dl_cpu = copy.deepcopy(dl).to("cpu")
                best_model_obj = DLClassifierWrapper(
                    dl_model=dl_cpu,
                    seq_len=params["seq_len"],
                    device="cpu"
                )
            else:
                best_model_obj = None
        else:
            no_gain_count += 1

        if no_gain_count >= patience:
            raise optuna.TrialPruned()

        return score

    print(f"[OPTUNA] running tuning... patience={patience}")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    if best_model is None:
        return None, None, None, None

    return best_model, best_params, best_score, best_model_obj

def evaluate_cv(model, X, y, is_classifier=True, metric="pr_auc"):
    """
    Cross-validation evaluation aligned with new wildfire metrics.

    metric options:
        "pr_auc"  -> average precision scoring
        "f2"      -> recall-weighted scoring
        "r2"      -> regression quality
    """

    try:
        cv = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=2, random_state=CFG.random_state
        )

        # Select scorer correctly
        if is_classifier:

            if metric == "pr_auc":
                scorer = "average_precision"

            elif metric == "f2":
                scorer = make_scorer(fbeta_score, beta=2)

            else:
                raise ValueError("Invalid classifier metric for CV")

        else:  # regression
            if metric == "r2":
                scorer = "r2"
            else:
                raise ValueError("Invalid regression metric for CV")

        scores = cross_val_score(model, X, y, scoring=scorer, cv=cv)

        return float(scores.mean()), float(scores.std())

    except Exception as e:
        print("[!] CV evaluation failed:", e)
        return None, None

# def get_candidate_classifiers():
#     models = {}

#     models['lgb'] = LGBMClassifier(n_estimators=200, random_state=CFG.random_state)
#     models['xgb'] = xgb.XGBClassifier(
#         n_estimators=200, random_state=CFG.random_state,
#         verbosity=0, use_label_encoder=False
#     )
#     models['cat'] = CatBoostClassifier(iterations=100, random_seed=CFG.random_state, verbose=0)

#     models['rf'] = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=CFG.random_state)
#     # models['et'] = ExtraTreesClassifier(n_estimators=120, n_jobs=-1, random_state=CFG.random_state)

#     # models['svc'] = SVC(probability=True, random_state=CFG.random_state)
#     # models['mlp'] = MLPClassifier(hidden_layer_sizes=(60,), early_stopping=True, random_state=CFG.random_state)

#     print("[i] Classifier candidates:", list(models.keys()))
#     return models
def get_candidate_classifiers():
    print("\n[✔] Base classifier = LightGBM only. Other models will be tried while Optimising using Optuna\n")
    return {
        "lgb": LGBMClassifier(
            n_estimators=200,
            random_state=CFG.random_state
        )
    }

# def get_candidate_regressors():
#     models = {}

#     models['lgb'] = LGBMRegressor(n_estimators=200, random_state=CFG.random_state)
#     models['xgb'] = xgb.XGBRegressor(n_estimators=180, random_state=CFG.random_state)
#     models['cat'] = CatBoostRegressor(iterations=100, random_seed=CFG.random_state, verbose=0)

#     # models['rf']  = RandomForestRegressor(n_estimators=120, random_state=CFG.random_state)
#     # models['et']  = ExtraTreesRegressor(n_estimators=120, random_state=CFG.random_state)
#     # models['svr'] = SVR()
#     # models['mlp'] = MLPRegressor(hidden_layer_sizes=(60,), early_stopping=True, random_state=CFG.random_state)

#     print("[i] Regressor candidates:", list(models.keys()))
#     return models

def get_candidate_regressors():
    print("\n[✔] Base regressor = LightGBM only\n")
    return {
        "lgb": LGBMRegressor(
            n_estimators=200,
            random_state=CFG.random_state
        )
    }

def save_gis_overlay(df_zone, save_path, value_column="fire_detected"):
    """
    Robust GIS overlay. 
    1. Tries GeoPandas (Heatmap).
    2. Falls back to Matplotlib Scatter if GeoPandas fails.
    """
    # Safety Check: Data existence
    if "latitude" not in df_zone.columns or "longitude" not in df_zone.columns:
        print("[!] GIS Aborted: Missing 'latitude' or 'longitude' columns.")
        return
    if value_column not in df_zone.columns:
        print(f"[!] GIS Aborted: Missing value column '{value_column}'.")
        return

    # 1. Try GeoPandas
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_zone,
            geometry=[Point(xy) for xy in zip(df_zone["longitude"], df_zone["latitude"])],
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        gdf.plot(
            ax=ax,
            column=value_column,
            markersize=20,
            cmap="inferno",
            legend=True,
            alpha=0.7,
            edgecolor='none'
        )
        ax.set_title(f"GIS Overlay: {value_column}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Save
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return # Success

    except ImportError:
        print("[!] GeoPandas not installed. Falling back to Scatter Plot.")
    except Exception as e:
        print(f"[!] GeoPandas overlay failed: {e}. Falling back to Scatter Plot.")

    # 2. Fallback: Standard Scatter Plot
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            df_zone["longitude"], 
            df_zone["latitude"], 
            c=df_zone[value_column], 
            cmap="inferno", 
            s=20, 
            alpha=0.7
        )
        plt.colorbar(sc, label=value_column)
        ax.set_title(f"Spatial Map: {value_column}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[i] Saved GIS fallback map: {save_path}")

    except Exception as e:
        print(f"[!] GIS Fallback failed: {e}")

def apply_smogn_regression(X_train, y_train):
    """ 
    Safe SMOGN rebalancing:
    ✔ works only for continuous imbalance
    ✔ keeps pipeline stable
    ✔ always returns (DataFrame, Series)
    """

    def _to_df_series(X, y):
        # X → DataFrame
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_np = np.asarray(X)
            X_df = pd.DataFrame(
                X_np,
                columns=[f"f_{i}" for i in range(X_np.shape[1])]
            )

        # y → Series
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_ser = pd.Series(y).copy()
        else:
            y_ser = pd.Series(np.asarray(y), name=CFG.target_reg)

        y_ser.name = CFG.target_reg
        return X_df, y_ser

    try:
        X_df, y_ser = _to_df_series(X_train, y_train)

        df_temp = X_df.copy()
        df_temp[CFG.target_reg] = pd.to_numeric(y_ser, errors="coerce")

        if df_temp[CFG.target_reg].isna().all():
            print("[!] SMOGN skipped — target is all NaN after coercion.")
            return X_df, y_ser

        df_temp[CFG.target_reg] = df_temp[CFG.target_reg].fillna(
            df_temp[CFG.target_reg].median()
        )

        # Run SMOGN on well-formed DataFrame
        df_smogn = smogn.smoter(
            data=df_temp,
            y=CFG.target_reg
        )

        print(f"[i] SMOGN applied → new shape {df_smogn.shape}")

        X_new = df_smogn.drop(columns=[CFG.target_reg])
        y_new = df_smogn[CFG.target_reg]

        return X_new, y_new

    except Exception as e:
        print("[!] SMOGN failed — fallback to original:", e)
        # Always fallback as pandas structures
        return _to_df_series(X_train, y_train)

def save_curve(data, title, xlabel, ylabel, save_path):
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(data) + 1), data, marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        save_fig(fig, save_path)
    except Exception as e:
        print("[!] Curve save failed:", e)

def _check_safe_zone(df_zone, outdir_zone, min_samples=50):
    """
    Smart safety screening function.
    Detects empty / micro zones AND performs auto-merge advisory.
    """

    y_cls = df_zone[CFG.target_class].astype(int)

    total_samples = len(df_zone)
    fire_count = int((y_cls == 1).sum())
    if total_samples == 0:
      metrics = {"zone_status": "empty_zone_no_samples"}
      json.dump(metrics, open(os.path.join(outdir_zone, "classifier_meta.json"), "w"), indent=2)
      generate_zone_report(outdir_zone, os.path.join(outdir_zone, "report.html"))
      return True, metrics

    fire_rate = fire_count / total_samples

    drought_mean = float(df_zone.get("drought_index", pd.Series([None])).mean())
    vegetation_dryness = float(df_zone.get("vegetation_dryness", pd.Series([None])).mean())
    population_density = float(df_zone.get("population_density", pd.Series([None])).mean())

    # ================================
    # CASE A: Mono-class
    # ================================
    if y_cls.nunique() < 2:

        tn = total_samples if y_cls.iloc[0] == 0 else 0
        tp = total_samples if y_cls.iloc[0] == 1 else 0

        metrics = {
            "zone_status": "no_fire_variation_detected",
            "historical_fire_probability": fire_rate,
            "drought_index_mean": drought_mean,
            "vegetation_dryness_mean": vegetation_dryness,
            "population_density_mean": population_density,
            "recommendation": (
                "Zone lacks variability. Merge into ecologically nearest region."
            ),
            "tn": tn, "fp": 0, "fn": 0, "tp": tp,
            "pr_auc": 1.0, "f2": 0.0, "accuracy": 1.0,
            "precision": 0.0, "recall": 0.0,
            "cv_pr_auc_mean": None, "cv_pr_auc_std": None
        }

        json.dump(metrics, open(os.path.join(outdir_zone, "classifier_meta.json"), "w"), indent=2)
        json.dump({}, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)
        generate_zone_report(outdir_zone, os.path.join(outdir_zone, "report.html"))
        return True, metrics

    # ================================
    # CASE B: Too few samples
    # ================================
    if total_samples < min_samples:

        metrics = {
            "zone_status": "micro_zone_insufficient_samples",
            "total_samples": total_samples,
            "historical_fire_probability": fire_rate,
            "drought_index_mean": drought_mean,
            "vegetation_dryness_mean": vegetation_dryness,
            "population_density_mean": population_density,
            "recommendation": (
                "Merge this zone with nearest ecological match "
                "(similar dryness + population)."
            ),
        }

        json.dump(metrics, open(os.path.join(outdir_zone, "classifier_meta.json"), "w"), indent=2)
        json.dump({}, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)
        generate_zone_report(outdir_zone, os.path.join(outdir_zone, "report.html"))
        return True, metrics

    return False, None

def _prepare_data(df_zone):
    """
    Final wildfire-optimized preprocessing:
    ✓ Feature cleanup
    ✓ Missing imputation
    ✓ Scaling/encoding
    ✓ Temporal-safe split
    ✓ Smart balancing (SMOTEENN fallback ADASYN)
    ✓ Saves transformer for inference
    """

    # ======================================================
    # 1) Extract targets
    # ======================================================
    y_cls = df_zone[CFG.target_class].astype(int)
    y_reg = np.log1p(df_zone[CFG.target_reg])

    X = df_zone.drop(columns=[CFG.target_class, CFG.target_reg], errors="ignore")

    # ======================================================
    # 2) Drop known useless / leakage features
    # ======================================================

    drop_candidates = [
        "zone_id",
        "pixel_id",
        "timestamp",       # date/time should be encoded separately
        "acquisition_time",
        "detection_date",  # future leakage risk
        "fire_size_after", # outcome leakage
    ]

    X = X.drop(columns=[c for c in drop_candidates if c in X.columns], errors="ignore")

    # ======================================================
    # 3) Feature screening
    # ======================================================

    # Drop columns with >80% missing
    missing_ratio = X.isna().mean()
    high_missing = missing_ratio[missing_ratio > 0.8].index.tolist()
    X = X.drop(columns=high_missing, errors="ignore")

    # Drop zero-variance columns
    selector = VarianceThreshold(threshold=0.0)
    try:
        selector.fit(X.select_dtypes(include=["int64", "float64"]))
        zero_var = [col for col, keep in zip(
            X.select_dtypes(include=["int64", "float64"]).columns, selector.get_support()
        ) if not keep]
        X = X.drop(columns=zero_var, errors="ignore")
    except:
        pass

    # ======================================================
    # 4) Identify numeric & categorical
    # ======================================================
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # ======================================================
    # 5) Preprocessing Pipelines
    # ======================================================

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop"
    )

    # ======================================================
    # 6) Train-test split (wildfire-aware)
    # ======================================================

    # If temporal split exists—use it
    if "year" in df_zone.columns:
        df_zone = df_zone.sort_values("year")
        X = X.loc[df_zone.index]
        y_cls = y_cls.loc[df_zone.index]
        y_reg = y_reg.loc[df_zone.index]

    X_train, X_test, y_tr_cls, y_te_cls, y_tr_reg, y_te_reg = train_test_split(
        X, y_cls, y_reg,
        test_size=CFG.test_size,
        stratify=y_cls,
        random_state=CFG.random_state
    )

    # ======================================================
    # 7) Fit preprocessing on training only (no leakage!)
    # ======================================================

    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre  = preprocessor.transform(X_test)

    # ======================================================
    # 8) Smart Resampling
    # ======================================================

    try:
        sm = SMOTEENN(random_state=CFG.random_state)
        X_train_bal, y_tr_bal = sm.fit_resample(X_train_pre, y_tr_cls)
        balance_method = "SMOTEENN"
    except:
        try:
            ada = ADASYN(random_state=CFG.random_state)
            X_train_bal, y_tr_bal = ada.fit_resample(X_train_pre, y_tr_cls)
            balance_method = "ADASYN"
        except:
            X_train_bal, y_tr_bal = X_train_pre, y_tr_cls
            balance_method = "none"

    print(f"[✓] Balancing applied using: {balance_method}")

    # ======================================================
    # 9) Return enriched data & preprocessor for inference
    # ======================================================

    return {
    "X": X,
    "X_train": X_train_pre,
    "X_test": X_test_pre,
    "y_tr_cls": y_tr_cls,
    "y_te_cls": y_te_cls,
    "y_tr_reg": y_tr_reg,
    "y_te_reg": y_te_reg,
    "X_train_sm": X_train_bal,
    "y_tr_cls_sm": y_tr_bal,
    "preprocessor": preprocessor,
  }

def _select_best_classifier(data, args, outdir_zone=None):
    import time
    from sklearn.model_selection import cross_val_predict

    X_train = data["X_train_sm"]
    y_train = data["y_tr_cls_sm"]
    X_test  = data["X_test"]
    y_test  = data["y_te_cls"]

    best_clf, best_name = None, None
    best_selection_score = -1e9
    best_full_pr = -1
    best_params, best_cv_mean, best_cv_std = None, None, None

    warmup_n      = min(4000, len(X_train))
    speed_budget  = 2.0   # seconds

    for name, model in get_candidate_classifiers().items():
        try:
            # ==== Warm training =====
            t0 = time.time()
            model.fit(X_train[:warmup_n], y_train[:warmup_n])
            warm_time = time.time() - t0

            warm_pr = average_precision_score(
                y_test, model.predict_proba(X_test)[:,1]
            )
            if warm_pr < 0.30:
                continue

            # ==== full training =====
            t0 = time.time()
            model.fit(X_train, y_train)
            full_time = time.time() - t0

            full_pr = average_precision_score(
                y_test, model.predict_proba(X_test)[:,1]
            )

            cv_mean, cv_std = evaluate_cv(
                model, X_train, y_train, is_classifier=True, metric="pr_auc"
            )

            # improve stability weighting
            stability_penalty = (cv_std / (cv_mean + 1e-6)) if cv_mean else 1

            # Selection heuristic
            selection = (
                0.55 * full_pr +
                0.30 * (cv_mean or 0) +
                0.10 * (1 - stability_penalty) -
                0.05 * (full_time / speed_budget)
            )

            if selection > best_selection_score:
                best_selection_score = selection
                best_full_pr = full_pr
                best_clf = clone(model).set_params(random_state=CFG.random_state)
                best_name = name
                best_cv_mean = cv_mean
                best_cv_std  = cv_std

        except Exception as e:
            print(f"[!] Candidate {name} failed:", e)

    # ===== fallback =====
    if best_clf is None:
        best_clf = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, random_state=CFG.random_state
        )
        best_clf.fit(X_train, y_train)
        best_full_pr = average_precision_score(
            y_test, best_clf.predict_proba(X_test)[:,1]
        )
        best_name = "fallback_lgb"

    # ===== Optuna refinement =====
    if args and getattr(args, "run_optuna", True):
        tuned = run_optuna_tuning(
            X_train, y_train, X_test, y_test,
            is_classifier=True, n_trials=args.trials
        )
        if tuned:
            tuned_name, tuned_params, tuned_pr = tuned[:3]
            if tuned_pr > best_full_pr:
                best_full_pr = tuned_pr
                best_params = tuned_params

                if tuned_name == "lgb":
                    best_clf = LGBMClassifier(**tuned_params)
                elif tuned_name == "xgb":
                    best_clf = xgb.XGBClassifier(**tuned_params)
                elif tuned_name == "cat":
                    tuned_params.pop("verbose", None)
                    best_clf = CatBoostClassifier(verbose=0, **tuned_params)
                elif tuned_name == "rf":
                    best_clf = RandomForestClassifier(**tuned_params)
                best_clf.fit(X_train, y_train)

    # ===== store out-of-fold preds for stacking / calibration =====
    try:
        oof = cross_val_predict(best_clf, X_train, y_train, cv=5,
                                method="predict_proba")[:,1]
        joblib.dump(oof, os.path.join(outdir_zone, "oof_predictions.pkl"))
    except Exception as e:
        print("[!] Could not save oof:", e)

    return best_clf, best_params, best_full_pr, best_cv_mean, best_cv_std


def _evaluate_classifier(best_clf, data, outdir_zone, zone_id, meta_info):
    """
    Full wildfire-aware classifier evaluation:
    ✔ ROC curve
    ✔ custom threshold optimisation
    ✔ confusion matrix + curve diagnostics
    ✔ JSON metrics export
    """

    X_test  = data["X_test"]
    y_true  = data["y_te_cls"]

    probs = best_clf.predict_proba(X_test)[:, 1]

    clf_metrics = {}

    # ---------------- ROC Curve ----------------
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_score = np.trapz(tpr, fpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve – Zone {zone_id}")

        save_fig(fig, os.path.join(outdir_zone, "roc_curve.png"))
        clf_metrics["auc"] = float(auc_score)

    except Exception as e:
        print("[!] ROC curve failed:", e)

    # ---------------- Custom threshold tuning ----------------
    best_thr = 0.5
    best_score = -1
    scores_curve = []

    pr_auc_test = average_precision_score(y_true, probs)

    for t in np.linspace(0.05, 0.95, 19):

        preds = (probs >= t).astype(int)

        f2 = fbeta_score(y_true, preds, beta=2)
        f1 = fbeta_score(y_true, preds, beta=1)

        weighted = 0.6 * f2 + 0.3 * f1 + 0.1 * pr_auc_test
        scores_curve.append(weighted)

        if weighted > best_score:
            best_score = weighted
            best_thr = t

    save_curve(
        scores_curve,
        f"Classifier Threshold Curve Zone {zone_id}",
        "Threshold Step",
        "Weighted Score (F2/F1/AUC)",
        os.path.join(outdir_zone, "classifier_curve.png")
    )

    # Extract best individual metrics
    best_f2 = max(
        fbeta_score(y_true, (probs >= t).astype(int), beta=2)
        for t in np.linspace(0.05, 0.95, 19)
    )

    best_f1 = max(
        fbeta_score(y_true, (probs >= t).astype(int), beta=1)
        for t in np.linspace(0.05, 0.95, 19)
    )

    # ---------------- Confusion matrix ----------------
    y_pred_bin = (probs >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)

    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix Zone {zone_id}")
        save_fig(fig, os.path.join(outdir_zone, "confusion_matrix.png"))
    except Exception as e:
        print("[!] Confusion matrix save failed:", e)

    # ---------------- Store final metrics ----------------
    clf_metrics = {
        "pr_auc": float(meta_info["pr_auc"]),
        "auc": clf_metrics.get("auc", None),
        "f2": float(best_f2),
        "f1": float(best_f1),
        "accuracy": float(accuracy_score(y_true, y_pred_bin)),
        "precision": float(precision_score(y_true, y_pred_bin)),
        "recall": float(recall_score(y_true, y_pred_bin)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "optuna_params": meta_info["params"],
        "cv_pr_auc_mean": meta_info["cv_mean"],
        "cv_pr_auc_std": meta_info["cv_std"],
        "best_threshold": float(best_thr),
    }

    joblib.dump(best_clf, os.path.join(outdir_zone, "classifier.pkl"))
    json.dump(clf_metrics, open(os.path.join(outdir_zone, "classifier_meta.json"), "w"), indent=2)

    return best_thr, clf_metrics, y_true.tolist(), probs.tolist()


def _train_and_evaluate_regressor(data, args, outdir_zone, zone_id):
    """
    Fire-only regression with risk-aware tuning:
    - minimises catastrophic underprediction
    - leak-safe validation split for tuning
    """

    fire_tr = data["y_tr_cls"] == 1
    fire_te = data["y_te_cls"] == 1

    X_fire = data["X_train"][fire_tr].copy()
    y_fire = data["y_tr_reg"][fire_tr].copy()

    X_test_fire  = data["X_test"][fire_te].copy()
    y_test_fire  = data["y_te_reg"][fire_te].copy()

    if len(X_fire) < 10 or fire_te.sum() < 5:
        json.dump({}, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)
        return None, None

    # rebalancing
    try:
        X_fire, y_fire = apply_smogn_regression(X_fire, y_fire)
        X_fire = X_fire.reset_index(drop=True)
        y_fire = y_fire.reset_index(drop=True)
    except:
        pass

    # ≡≡≡ risk-aware loss ≡≡≡
    def wildfire_loss(true_raw, pred_raw):
        under = np.maximum(true_raw - pred_raw, 0)
        over  = np.maximum(pred_raw - true_raw, 0)

        return np.mean(
            under**2 + 0.1 * over**2
        )

    best_reg = None
    best_loss = 1e30

    # leak-safe split for tuning
    if args and getattr(args, "run_optuna", False):
        try:
            X_tn, X_val, y_tn, y_val = train_test_split(
                X_fire, y_fire,
                test_size=0.25,
                random_state=CFG.random_state
            )

            def objective(trial):
                model_type = trial.suggest_categorical("model", ["lgb", "xgb", "cat", "rf"])

                # param spaces (unchanged)
                if model_type == "lgb":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                        "max_depth": trial.suggest_int("max_depth", -1, 14),
                        "random_state": CFG.random_state,
                    }
                    model = LGBMRegressor(**params)

                elif model_type == "xgb":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "random_state": CFG.random_state,
                    }
                    model = xgb.XGBRegressor(**params)

                elif model_type == "cat":
                    params = {
                        "depth": trial.suggest_int("depth", 4, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                        "loss_function": "RMSE",
                        "random_seed": CFG.random_state,
                    }
                    model = CatBoostRegressor(verbose=0, **params)

                else:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                        "max_depth": trial.suggest_int("max_depth", 6, 14),
                        "max_features": "sqrt",
                        "random_state": CFG.random_state,
                    }
                    model = RandomForestRegressor(**params)

                model.fit(X_tn, y_tn)

                pred_val_log = model.predict(X_val)
                true_val_raw = np.expm1(y_val)
                pred_val_raw = np.expm1(pred_val_log)

                # catastrophic penalty term
                cat_val = ((true_val_raw > 500) & ((true_val_raw - pred_val_raw) > 0.3 * true_val_raw)).sum()

                loss = wildfire_loss(true_val_raw, pred_val_raw) + 10.0 * (cat_val / max(1, len(true_val_raw)))

                return loss

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.trials)

            best_trial = study.best_trial
            tuned_params = {k: v for k, v in best_trial.params.items() if k != "model"}

            model_type = best_trial.params["model"]
            if model_type == "lgb":
                best_reg = LGBMRegressor(**tuned_params)
            elif model_type == "xgb":
                best_reg = xgb.XGBRegressor(**tuned_params)
            elif model_type == "cat":
                tuned_params.pop("verbose", None)
                best_reg = CatBoostRegressor(verbose=0, **tuned_params)
            else:
                best_reg = RandomForestRegressor(**tuned_params)

            best_reg.fit(X_fire, y_fire)

        except Exception as e:
            print("[!] risk-aware Optuna failed:", e)

    # fallback risk search
    if best_reg is None:
        for _, model in get_candidate_regressors().items():
            try:
                model.fit(X_fire, y_fire)
                pred_log = model.predict(X_test_fire)
                loss = wildfire_loss(np.expm1(y_test_fire), np.expm1(pred_log))
                if loss < best_loss:
                    best_loss = loss
                    best_reg = model
            except:
                pass

    if best_reg is None:
        json.dump({}, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)
        return None, None

    joblib.dump(best_reg, os.path.join(outdir_zone, "regressor.pkl"))

    # evaluate
    pred_log = best_reg.predict(X_test_fire)
    true_raw = np.expm1(y_test_fire)
    pred_raw = np.expm1(pred_log)

    r2_log = r2_score(y_test_fire, pred_log)
    mse = mean_squared_error(true_raw, pred_raw)

    under = np.maximum(true_raw - pred_raw, 0)
    cat = ((true_raw > 500) & ((true_raw - pred_raw) > (0.3 * true_raw))).sum()

    metrics = {
        "best_model": best_reg.__class__.__name__,
        "r2_log_space": float(r2_log),
        "mse_raw": float(mse),
        "rmse_raw": float(np.sqrt(mse)),
        "mae_raw": float(mean_absolute_error(true_raw, pred_raw)),
        "asymmetric_underestimation_cost": float(np.mean(under**2)),
        "catastrophic_miss_count": int(cat),
        "catastrophic_rate": cat / max(1, len(true_raw)),
    }

    json.dump(metrics, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)

    return best_reg, metrics

def _generate_explainability_artifacts(df_zone, best_clf, best_reg, data, outdir_zone, zone_id):
    """
    Corrected Explainability Module:
    - Fixes length mismatch between transformed features and feature name list.
    - Uses post-preprocessor feature names for importance/SHAP.
    - Uses raw feature names only when slicing original df_zone.
    """
    df_zone = df_zone.apply(
      lambda col: pd.to_numeric(
          col.astype(str).str.replace(r"[\[\]]", "", regex=True),
          errors="ignore"
      ) if col.dtype == "object" else col
    )

    try:
        if best_clf is None:
            print("[i] Explainability skipped — no classifier.")
            return

        preprocessor = data.get("preprocessor")
        raw_X = data["X"]
        raw_features = list(raw_X.columns)

        # --------- Transformed feature names (after ColumnTransformer) ---------
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            transformed_features = list(preprocessor.get_feature_names_out())
        else:
            n_feats = data["X_train"].shape[1]
            transformed_features = [f"f_{i}" for i in range(n_feats)]

        # ======================================================================
        # 1) Global Feature Importance (Classifier + Regressor)
        # ======================================================================
        try:
            save_feature_importance(
                best_clf,
                transformed_features,
                os.path.join(outdir_zone, "feature_importance_classifier.png"),
                csv_path=os.path.join(outdir_zone, "feature_importance_classifier.csv"),
            )
        except Exception as e:
            print("[!] Classifier feature-importance failed:", e)

        if best_reg is not None:
            try:
                save_feature_importance(
                    best_reg,
                    transformed_features,
                    os.path.join(outdir_zone, "feature_importance_regressor.png"),
                    csv_path=os.path.join(outdir_zone, "feature_importance_regressor.csv"),
                )
            except Exception as e:
                print("[!] Regressor feature-importance failed:", e)

        # ======================================================================
        # 2) SHAP Summary (on processed test set)
        # ======================================================================
        X_sample_df = None  # for optional narrative later
        try:
            X_test_pre = data["X_test"]

            if X_test_pre is not None and len(X_test_pre) > 0:

                sample_size = min(100, X_test_pre.shape[0])
                idx = np.random.choice(X_test_pre.shape[0], sample_size, replace=False)

                X_sample_np = X_test_pre[idx].astype(str)

                # CLEAN string artifacts like '[5.074E-1]'
                X_sample_np = np.vectorize(
                    lambda v: str(v).replace("[", "").replace("]", "")
                )(X_sample_np)

                # Convert to float safely
                X_sample_np = pd.DataFrame(X_sample_np, dtype="float64")

                # Proper column assignment
                X_sample_df = pd.DataFrame(X_sample_np.values, columns=transformed_features)

                save_shap_summary(
                    best_clf,
                    X_sample_df,
                    os.path.join(outdir_zone, "shap_summary_classifier.png"),
                )

        except Exception as e:
            print("[!] SHAP summary generation failed:", e)

        # ======================================================================
        # 3) GIS SHAP Risk Map  + Pretty Base Geo Scatter
        # ======================================================================
        try:
            if preprocessor is not None and "latitude" in df_zone.columns and "longitude" in df_zone.columns:

                df_geo_sample = df_zone.sample(min(50, len(df_zone)), random_state=42).copy()
                df_geo_sample["latitude"] = pd.to_numeric(df_geo_sample["latitude"], errors="coerce")
                df_geo_sample["longitude"] = pd.to_numeric(df_geo_sample["longitude"], errors="coerce")
                df_geo_sample = df_geo_sample.dropna(subset=["latitude", "longitude"])

                if len(df_geo_sample) > 0:

                    # ------------------------------------------------------------
                    # 1) PRETTY BASIC MAP (visual reference only)
                    # ------------------------------------------------------------
                    try:
                      base_ok = save_gis_overlay(
                        df_geo_sample,
                        os.path.join(outdir_zone, "gis_overlay_base.png"),
                        value_column=None
                      )

                      if base_ok:
                          print(f"[✓] Pretty geo overlay saved → {outdir_zone}/gis_overlay_base.png")
                      else:
                          print(f"[!] Pretty geo overlay failed → {outdir_zone}/gis_overlay_base.png")
                    except Exception as e:
                          print("[!] Pretty geospatial overlay failed:", e)

                    # ------------------------------------------------------------
                    # 2) SHAP-Weighted Risk Map (your intelligence layer)
                    # ------------------------------------------------------------

                    X_geo_raw = df_geo_sample[raw_features].copy()
                    X_geo_processed = preprocessor.transform(X_geo_raw)

                    # CLEAN artifacts: vector conversion + numeric casting
                    X_geo_processed = X_geo_processed.astype(str)
                    X_geo_processed = np.vectorize(
                        lambda v: str(v).replace("[", "").replace("]", "")
                    )(X_geo_processed)

                    X_geo_processed = pd.DataFrame(X_geo_processed, dtype="float64")

                    explainer = shap.TreeExplainer(best_clf)
                    shap_vals = explainer.shap_values(X_geo_processed)

                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                    elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                        shap_vals = shap_vals[:, :, 1]

                    geo_risk = np.abs(shap_vals).sum(axis=1)
                    df_geo_sample["shap_risk"] = geo_risk

                    save_gis_overlay(
                        df_geo_sample,
                        os.path.join(outdir_zone, "gis_risk_heatmap.png"),
                        value_column="shap_risk",
                    )
                    print(f"[✓] SHAP risk heatmap saved → {outdir_zone}/gis_risk_heatmap.png")

        except Exception as e:
            print(f"[!] GIS SHAP mapping failed: {e}")



        # ======================================================================
        # 4) Zone Map
        # ======================================================================
        try:
            save_zone_map(df_zone, os.path.join(outdir_zone, "zone_map.png"))
        except Exception as e:
            print("[!] Zone map failed:", e)

        # ======================================================================
        # 5) Narrative Insights (top SHAP drivers)
        # ======================================================================
        try:
            if X_sample_df is not None:
                
                explainer = shap.TreeExplainer(best_clf)
                shap_vals = explainer.shap_values(X_sample_df)

                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1]

                shap_df = pd.DataFrame(shap_vals, columns=X_sample_df.columns)
                dom_features = shap_df.abs().mean().sort_values(ascending=False).head(10)
                narrative = "\n".join(
                    f"* {feat}: major driver of risk" for feat in dom_features.index
                )

                with open(
                    os.path.join(outdir_zone, "feature_narrative.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(f"Top wildfire drivers in zone {zone_id}:\n{narrative}\n")
        except Exception as e:
            print("[!] Narrative insight generation failed:", e)

    except Exception as e:
        print(f"[!] Explainability module failed: {e}")


def train_zone(df_zone, outdir_zone, zone_id, args=None, min_samples=50):

    print(f"\n=== 🔥 TRAINING ECOZONE {zone_id} | {len(df_zone)} samples ===")
    
    # Clean only if NOT resuming (handled in main, but good safety here)
    if not (args and getattr(args, "resume", False)):
        clean_output_directory(outdir_zone, delete_subfolders=True)
    else:
        if not os.path.exists(outdir_zone):
            os.makedirs(outdir_zone)


    # --- Safety Screening ---
    is_unsafe, safety_metrics = _check_safe_zone(df_zone, outdir_zone, min_samples=min_samples)
    if is_unsafe:
        return {
            "clf": None, "thr": 0.5, "reg": None,
            "metrics_clf": safety_metrics, "metrics_reg": None
        }

    # --- Preprocessing ---
    # This dictionary 'data' contains the preprocessor needed for the GIS fix
    data = _prepare_data(df_zone)
    print("[i] Train/Test shapes:", data["X_train"].shape, data["X_test"].shape)

    # --- Classifier Selection ---
    best_clf, best_params, best_pr_auc, cv_mean, cv_std = _select_best_classifier(data, args, outdir_zone)

    clf_meta = {
        "pr_auc": best_pr_auc,
        "params": best_params,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }

    # --- Classifier Evaluation ---
    best_thr, clf_metrics, y_true, y_prob = _evaluate_classifier(
        best_clf, data, outdir_zone, zone_id, clf_meta
    )

    # --- Regression ---
    best_reg, reg_metrics = _train_and_evaluate_regressor(
        data, args, outdir_zone, zone_id
    )

    # --- Explainability (The Fixed Function) ---
    _generate_explainability_artifacts(
        df_zone, best_clf, best_reg, data, outdir_zone, zone_id
    )

    # --- Generate Report ---
    generate_zone_report(outdir_zone, os.path.join(outdir_zone, "report.html"))

    return {
        "clf": best_clf,
        "thr": best_thr,
        "reg": best_reg,
        "metrics_clf": clf_metrics,
        "metrics_reg": reg_metrics,
        "y_true": y_true,
        "y_prob": y_prob
    }

def assign_zones(df, router):
    print("[i] Assigning ecozones via router…")
    df["ecozone"] = router.predict(df[["latitude","longitude"]])
    return df

def predict_zone(df_zone, zone_id, registry):
    entry = registry.get(zone_id, {})

    clf = entry.get("clf")
    reg = entry.get("reg")
    thr = entry.get("thr", 0.5)

    # Feature matrix like training
    X = df_zone.drop(columns=[CFG.target_class, CFG.target_reg], errors="ignore")

    n = len(df_zone)

    # --- Classification ---
    if clf is None:
        fire_prob = np.zeros(n, dtype=float)
    else:
        fire_prob = clf.predict_proba(X)[:, 1]

    # --- Regression ---
    if reg is not None:
        intensity_log = reg.predict(X)
        pred_intensity = np.expm1(intensity_log)
    else:
        pred_intensity = np.zeros(n, dtype=float)

    # --- Derived ---
    soft_severity = fire_prob * pred_intensity
    fire_label = (fire_prob >= thr).astype(int)

    return pd.DataFrame({
        "fire_prob": fire_prob,
        "pred_intensity": pred_intensity,
        "soft_severity": soft_severity,
        "fire_label": fire_label,
        "ecozone": zone_id,
    }, index=df_zone.index)

def train_ecozone_knn(df, outdir):
    print("[i] Training ecozone KNN router …")
    
    k = max(20, int(np.sqrt(len(df)) / 3))
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(df[['latitude','longitude']], df['ecozone'])
    
    joblib.dump(knn, os.path.join(outdir, "ecozone_knn.pkl"))
    return knn
  
def plot_india_risk(df, risk_col="fire_prob", save_path="india_risk_map.png"):
    """
    India ke map par risk_col ke basis pe heatmap (scatter) banata hai.
    Example: risk_col = 'fire_prob' ya 'shap_risk'
    """
    required = {"latitude", "longitude", risk_col}
    if not required <= set(df.columns):
        print(f"[!] plot_india_risk: df missing columns: {required - set(df.columns)}")
        return

    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df[risk_col] = pd.to_numeric(df[risk_col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", risk_col])

    india = _get_india_shape()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=(7, 9))

    # base India
    india.plot(ax=ax, color="white", edgecolor="black", linewidth=0.8)

    # risk overlay
    gdf.plot(
        ax=ax,
        column=risk_col,
        cmap="inferno",
        markersize=5,
        alpha=0.8,
        legend=True,
    )

    ax.set_title(f"High-Risk Areas over India ({risk_col})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[i] Saved India risk map → {save_path}")
 
  
def ecofire_predict(input_csv, model_dir="ecozone_results"):
    print("[i] Running ECOFIRE prediction…")

    registry, router = load_registry(model_dir)

    df = pd.read_csv(input_csv)
    df = engineer_features(df)

    df = assign_zones(df, router)

    results = []
    for z in sorted(df["ecozone"].unique()):
        df_z = df[df["ecozone"] == z]
        print(f"[i] Zone {z} → {len(df_z)} samples")

        preds = predict_zone(df_z, z, registry)
        preds["ecozone"] = z     # preserve zone
        results.append(preds)

    output = pd.concat(results).sort_index()
    
    final = pd.concat([df, output], axis=1)

    out_file = "prediction_output.csv"
    final.to_csv(out_file, index=False)

    print(f"[i] Prediction complete → saved {out_file}")
    try:
        plot_india_risk(final, risk_col="fire_prob", 
                        save_path=os.path.join(model_dir, "india_fire_risk.png"))
    except Exception as e:
        print("[!] Could not generate India risk map:", e)
    return final

def train_zone_wrapper(params):
    df_zone, out_zone, zone_id, args = params
    result = train_zone(df_zone, out_zone, zone_id, args)
    return zone_id, result

def generate_zone_report(zone_dir, out_html):
    print("[i] Generating zone report:", zone_dir)

    try:
        meta = json.load(open(os.path.join(zone_dir,"classifier_meta.json")))
    except:
        meta = {}

    try:
        meta_r = json.load(open(os.path.join(zone_dir,"regressor_meta.json")))
    except:
        meta_r = {}

    imgs = [f for f in os.listdir(zone_dir) if f.endswith(".png")]

    img_html = ""
    for f in imgs:
        label = f.replace(".png","").replace("_"," ").title()
        img_html += f"<h3>{label}</h3><img src='{f}' width='600'/><br>"

    html = f"""
    <html>
    <body>
    <h1>Zone Report: {os.path.basename(zone_dir)}</h1>

    <h2>Classifier Summary</h2>
    <pre>{json.dumps(meta, indent=2)}</pre>

    <h2>Regressor Summary</h2>
    <pre>{json.dumps(meta_r, indent=2)}</pre>

    <h2>Diagnostics & Training Curves</h2>
    {img_html}
    </body>
    </html>
    """

    with open(out_html,"w",encoding="utf-8") as f:
        f.write(html)

    print("[i] Report written:", out_html)

def smogn_like_rebalance(X, y):
    """Safer SMOGN augmentation — perturb only numeric features without destabilizing scaling."""
    if len(y) < 50:
        return X, y

    q_hi, q_lo = y.quantile(0.9), y.quantile(0.1)
    rare = (y >= q_hi) | (y <= q_lo)

    X_r, y_r = X[rare], y[rare]
    if len(X_r) < 5:
        return X, y

    # Separate numeric & non-numeric
    num_cols = X_r.select_dtypes(include=np.number).columns
    non_num_cols = X_r.select_dtypes(exclude=np.number).columns

    # Numeric perturbations
    noise = X_r[num_cols].sample(len(X_r), replace=True)
    noise = noise * (1 + 0.05*np.random.randn(*noise.shape))
    noise = pd.DataFrame(noise, columns=num_cols).reset_index(drop=True)

    # Non-numeric copied as repetition
    non_noise = X_r[non_num_cols].sample(len(X_r), replace=True)
    non_noise = non_noise.reset_index(drop=True)


    X_aug = pd.concat([X, pd.concat([noise, non_noise], axis=1)], axis=0)
    y_aug = pd.concat([y, y_r.sample(len(y_r), replace=True)], axis=0)

    return X_aug, y_aug

def save_feature_importance(model, feature_names, save_path, csv_path=None):
    try:
        # Get importances
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            importances = np.asarray(model.coef_).ravel()
        else:
            print("[!] No feature importance available for model:", model.__class__.__name__)
            return

        feature_names = list(feature_names)
        n_imp = importances.shape[0]

        # ---- Length safety ----
        if len(feature_names) != n_imp:
            print(f"[!] Feature importance length mismatch: "
                  f"{len(feature_names)} names vs {n_imp} importances. Auto-aligning.")
            if len(feature_names) > n_imp:
                feature_names = feature_names[:n_imp]
            else:
                feature_names = feature_names + [
                    f"f_{i}" for i in range(len(feature_names), n_imp)
                ]

        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        # ===== Save ranking table =====
        if csv_path:
            fi.to_csv(csv_path, index=False)
            print("[i] Feature importance ranking written:", csv_path)

        # ===== Plot sorted importance (top 40) =====
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="importance",
            y="feature",
            data=fi.head(40),
            ax=ax
        )

        ax.set_title("Feature Importance (Sorted)")
        ax.set_xlabel("Importance weight")
        ax.set_ylabel("Feature")

        save_fig(fig, save_path)

    except Exception as e:
        print("[!] Error saving feature importance:", e)

def save_shap_summary(model, X_sample, save_path):
    try:

        # ========= SAFE PREPROCESSING BLOCK =========
        X_sample = X_sample.copy()

        # Convert string numerics like "5E-1" → 0.5
        for col in X_sample.columns:
            if X_sample[col].dtype == 'object':
                try:
                    X_sample[col] = pd.to_numeric(X_sample[col], errors='coerce')
                except Exception:
                    pass

        # Drop columns entirely NaN
        X_sample = X_sample.dropna(axis=1, how="all")

        # Keep only numeric columns
        X_sample = X_sample.select_dtypes(include=[np.number])

        # Replace remaining NaNs column-wise using median
        if not X_sample.empty:
            X_sample = X_sample.fillna(X_sample.median())

        # ============================================

        if X_sample.empty:
            print("[!] SHAP skipped — no numeric columns after cleaning.")
            return

        # ===== Compute SHAP values =====
        try:
            X_numeric = X_sample.apply(pd.to_numeric, errors='coerce').fillna(0)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_numeric)

            if isinstance(shap_values, list):
                # Typically [class0, class1]; take positive class
                shap_values = shap_values[1]
            shap_values = np.asarray(shap_values)

        except Exception as e:
            print("[!] TreeExplainer failed, falling back to approximate SHAP:", e)

            X_numeric = X_sample.select_dtypes(include=[np.number])
            n_rows, n_feats = X_numeric.shape

            if hasattr(model, "feature_importances_"):
                base = np.abs(np.asarray(model.feature_importances_))
            elif hasattr(model, "coef_"):
                base = np.abs(np.asarray(model.coef_).ravel())
            else:
                base = np.ones(n_feats)

            # Normalise to sum to 1 to avoid huge scale issues
            base = base[:n_feats]
            base = base / (base.sum() + 1e-9)
            shap_values = np.tile(base, (n_rows, 1))

        # ===== Dimension alignment =====
        X_numeric = X_sample.select_dtypes(include=[np.number])
        n_rows, n_feats = X_numeric.shape
        n_imp = shap_values.shape[1]

        if n_imp != n_feats:
            print(f"[!] SHAP value dimension mismatch: {n_imp} vs {n_feats}. Auto-aligning.")
            k = min(n_imp, n_feats)
            shap_values = shap_values[:, :k]
            X_numeric = X_numeric.iloc[:, :k]

        # ===== Plot =====
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_numeric, show=False)
        save_fig(fig, save_path)

    except Exception as e:
        print("[!] SHAP failed completely:", e)

def load_registry(model_dir="ecozone_results"):
    print("[i] Loading registry + router…")
    registry = joblib.load(f"{model_dir}/models_registry.pkl")

    # apply DL safety
    for z, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        clf = entry.get("clf")
        if clf is not None:
            entry["clf"] = prepare_loaded_model(clf)

    router = joblib.load(f"{model_dir}/ecozone_knn.pkl")
    return registry, router

def save_zone_map(df_zone, save_path):
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.kdeplot(
            x=df_zone["longitude"],
            y=df_zone["latitude"],
            fill=True,
            cmap="Reds",
            thresh=0.05,
            ax=ax
        )
        ax.set_title("Spatial Heatmap of Samples")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print("[i] Saved zone map:", save_path)

    except Exception as e:
        print("[!] Failed zone map:", e)

def save_gis_overlay(df_zone, save_path, value_column="fire_detected"):
    """
    Creates geospatial scatter/heat overlay maps.

    value_column:
        - None          → plain geospatial map (no heatmap)
        - fire_detected → default classifier signal
        - shap_risk     → explainability overlay
    """
    try:
        print(f"[i] GIS overlay requested → {save_path} using '{value_column}'")

        # Validate coordinates exist
        if "latitude" not in df_zone or "longitude" not in df_zone:
            print("[!] GIS overlay aborted: missing latitude/longitude columns")
            return False

        df_zone = df_zone.copy()
        df_zone["latitude"]  = pd.to_numeric(df_zone["latitude"],  errors="coerce")
        df_zone["longitude"] = pd.to_numeric(df_zone["longitude"], errors="coerce")
        df_zone = df_zone.dropna(subset=["latitude", "longitude"])

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_zone,
            geometry=[Point(xy) for xy in zip(df_zone["longitude"], df_zone["latitude"])],
            crs="EPSG:4326"
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        # Case 1: plain scatter if value_column=None
        if value_column is None:
            gdf.plot(
                ax=ax,
                markersize=6,
                color="red",
                alpha=0.6
            )
        else:
            # If column missing, fail gracefully
            if value_column not in gdf.columns:
                print(f"[!] GIS overlay aborted: missing value column '{value_column}'")
                return False

            df_zone[value_column] = pd.to_numeric(df_zone[value_column], errors="coerce")

            gdf.plot(
                ax=ax,
                column=value_column,
                cmap="inferno",
                legend=True,
                markersize=6,
                alpha=0.7
            )

        title = f"GIS Overlay — {value_column}" if value_column else "GIS Overlay"
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[✓] GIS overlay actually saved → {save_path}")
        return True

    except Exception as e:
        print("[!] GIS overlay failed:", e)
        return False
import geopandas as gpd
from shapely.geometry import Point

def _get_india_shape():
    """
    Natural Earth dataset se India polygon le aata hai.
    CRS = EPSG:4326 (lat/lon)
    """
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    india = world[world["name"] == "India"].to_crs(epsg=4326)
    return india

def plot_india_zones(df, save_path="india_ecozones.png"):
    """
    India ke map par har sample ka ecozone dikhata hai.
    df: columns required -> ['latitude', 'longitude', 'ecozone']
    """
    # safety
    if not {"latitude", "longitude", "ecozone"} <= set(df.columns):
        print("[!] plot_india_zones: df missing latitude/longitude/ecozone")
        return

    # clean coords
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    india = _get_india_shape()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=(7, 9))

    # base India boundary
    india.plot(ax=ax, color="white", edgecolor="black", linewidth=0.8)

    # ecozone points
    gdf.plot(
        ax=ax,
        column="ecozone",
        markersize=5,
        alpha=0.7,
        legend=True,
        cmap="tab20",
    )

    ax.set_title("Ecozones over India")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[i] Saved India ecozone map → {save_path}")


def generate_global_reports(registry, outdir):
    print("[i] Building ULTIMATE global performance reports...")

    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    global_dir = os.path.join(outdir, "global")
    os.makedirs(global_dir, exist_ok=True)

    summary_rows = []
    
    all_y_true = []
    all_y_prob = []
    spatial_data = []

    # Extract stored centroid structure safely
    centroids = registry.get("centroids", {})

    for zone in sorted([z for z in registry.keys() if isinstance(z, (int, np.integer))]):
        entry = registry[zone] or {}

        mclf = entry.get("metrics_clf", {}) or {}
        mreg = entry.get("metrics_reg", {}) or {}

        tn = mclf.get("tn", 0)
        fp = mclf.get("fp", 0)
        fn = mclf.get("fn", 0)
        tp = mclf.get("tp", 0)
        total_samples = tn + fp + fn + tp
        fire_rate = (tp + fn) / max(1, total_samples)

        # Collect prediction distributions
        if "y_true" in entry and "y_prob" in entry:
            yt, yp = entry["y_true"], entry["y_prob"]
            if yt is not None and yp is not None:
                all_y_true.extend(list(yt))
                all_y_prob.extend(list(yp))

        # Safe centroid lookup
        if isinstance(centroids, dict) and zone in centroids:
            lat, lon = centroids[zone]
        else:
            lat, lon = (np.nan, np.nan)

        if not np.isnan(lat):
            spatial_data.append({
                "zone": zone, "lat": lat, "lon": lon,
                "pr_auc": mclf.get("pr_auc", np.nan)
            })

        summary_rows.append({
            "zone": zone,
            "total_samples": total_samples,
            "fire_rate": fire_rate,
            "latitude": lat,

            # CLASSIFICATION
            "pr_auc": mclf.get("pr_auc", np.nan),
            "f2": mclf.get("f2", np.nan),
            "accuracy": mclf.get("accuracy", np.nan),
            "precision": mclf.get("precision", np.nan),
            "recall": mclf.get("recall", np.nan),

            # REGRESSION
            "rmse": mreg.get("rmse", np.nan),
            "r2": mreg.get("r2", np.nan),

            # VALIDATION STABILITY
            "cv_mean": mclf.get("cv_pr_auc_mean", np.nan),
            "cv_std": mclf.get("cv_pr_auc_std", np.nan),

            # STORE counts so global CM works!
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(global_dir, "performance_summary.csv"), index=False)

    # ============================================================================
    # 1. METRIC RANGE BOXPLOTS
    # ============================================================================
    print("[i] Generating metric range boxplots...")
    try:
        melted = summary_df.melt(
            id_vars=["zone"],
            value_vars=["pr_auc", "f2", "accuracy", "recall", "precision"],
            var_name="Metric", value_name="Score"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=melted, x="Metric", y="Score", ax=ax, palette="Set2")
        sns.stripplot(data=melted, x="Metric", y="Score",
                      color="black", jitter=True, alpha=0.3, ax=ax)

        ax.set_title("Global Performance Distribution (Range Summary)")
        ax.set_ylim(0, 1.05)
        save_fig(fig, os.path.join(global_dir, "global_metric_ranges.png"))
    except Exception as e:
        print("[!] Boxplots failed:", e)

    # ============================================================================
    # 2. META CORR HEATMAP
    # ============================================================================
    print("[i] Generating meta-correlation heatmap...")
    try:
        meta_cols = ["total_samples", "fire_rate", "latitude",
                     "pr_auc", "f2", "cv_std"]
        corr_df = summary_df[meta_cols].dropna().corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f",
                    center=0, ax=ax)
        ax.set_title("Meta-Analysis: Drivers of Zone Performance")
        save_fig(fig, os.path.join(global_dir, "meta_correlation_matrix.png"))
    except Exception as e:
        print("[!] Heatmap failed:", e)

    # ============================================================================
    # 3. SAMPLE SIZE VS PERFORMANCE
    # ============================================================================
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=summary_df,
                        x="total_samples", y="pr_auc",
                        alpha=0.6, s=120, ax=ax)

        ax.set_xscale("log")
        ax.set_title("Data Efficiency: Sample Size vs PR-AUC")
        ax.set_xlabel("Log(Samples)")
        save_fig(fig, os.path.join(global_dir, "samples_vs_performance.png"))
    except Exception as e:
        print("[!] Scatter failed:", e)

    # ============================================================================
    # 4. GLOBAL PROBABILITY HISTOGRAM
    # ============================================================================
    try:
        if len(all_y_prob) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(all_y_prob, bins=50, kde=True, ax=ax)
            ax.set_title("Global Fire Probability Distribution")
            ax.set_xlabel("Predicted Fire Probability")
            save_fig(fig, os.path.join(global_dir, "global_prediction_dist.png"))
    except Exception as e:
        print("[!] Global histogram failed:", e)

    # ============================================================================
    # 5. SPATIAL MAP
    # ============================================================================
    if len(spatial_data) > 0:
        try:
            sdf = pd.DataFrame(spatial_data)
            fig, ax = plt.subplots(figsize=(8, 6))
            pts = ax.scatter(sdf["lon"], sdf["lat"], c=sdf["pr_auc"],
                             cmap="RdYlGn", s=80, edgecolor="k")
            plt.colorbar(pts, label="PR-AUC")
            ax.set_title("Geographic Performance Map")
            save_fig(fig, os.path.join(global_dir, "spatial_performance_map.png"))
        except Exception as e:
            print("[!] Spatial plot failed:", e)

    # ============================================================================
    # 6. GLOBAL CONFUSION MATRIX
    # ============================================================================
    all_tn = summary_df["tn"].sum()
    all_fp = summary_df["fp"].sum()
    all_fn = summary_df["fn"].sum()
    all_tp = summary_df["tp"].sum()

    global_cm = np.array([[all_tn, all_fp],
                          [all_fn, all_tp]])

    pd.DataFrame(global_cm,
                 index=["Actual 0", "Actual 1"],
                 columns=["Pred 0", "Pred 1"]).to_csv(
        os.path.join(global_dir, "global_confusion_matrix.csv")
    )

    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(global_cm, annot=True, fmt=".0f",
                    cmap="Blues", ax=ax)
        ax.set_title("Global Confusion Matrix")
        save_fig(fig, os.path.join(global_dir, "global_confusion_matrix.png"))
    except Exception:
        pass

    # ============================================================================
    # 7. GLOBAL HTML DASHBOARD
    # ============================================================================
    print("[i] Writing global index page...")

    index_html = """
    <html><head><title>AutoML-ECOFIRE Global Analysis</title></head><body>
    <h1>Global Performance Summary</h1>
    <ul>
    """

    for z in sorted([z for z in registry.keys() if isinstance(z, (int, np.integer))]):
        index_html += f"<li><a href='../zone_{z}/report.html'>Zone {z} Report</a></li>"

    index_html += """
    </ul><hr>
    <p><a href='performance_summary.csv'>Download CSV Summary</a></p>
    </body></html>
    """

    with open(os.path.join(global_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print("[i] DONE Global reporting.\n")


def compute_centroids(df):
    centroids = {}
    for z in sorted(df["ecozone"].unique()):
        if z == -1: 
            continue
        part = df[df["ecozone"] == z]
        centroids[int(z)] = (
            float(part["latitude"].mean()),
            float(part["longitude"].mean())
        )
    return centroids

def clean_output_directory(path, delete_subfolders=True):
    """
    Cleans old training results for reproducibility.
    
    - Removes files inside directory
    - Optionally removes subdirectories
    - Creates directory if missing
    """
    import shutil

    # If not exist — create directory
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[i] Created output directory: {path}")
        return

    print(f"[i] Cleaning previous outputs in: {path}")

    for root, dirs, files in os.walk(path):
        # Delete files
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except:
                pass

        # Delete folder contents if allowed
        if delete_subfolders:
            for d in dirs:
                try:
                    shutil.rmtree(os.path.join(root, d))
                except:
                    pass

def build_zone_tasks(df, args, registry, outdir):
    tasks = []
    zones = sorted([z for z in df["ecozone"].unique() if z != -1])

    for zone in zones:
        df_zone = df[df["ecozone"] == zone]

        if len(df_zone) < args.min_zone_rows:
            print(f"[i] Skipping zone {zone} — too small")
            continue

        if df_zone[CFG.target_class].nunique() < 2:
            print(f"[!] Zone {zone} is SAFE — no variation")

            only_val = df_zone[CFG.target_class].iloc[0]
            tn = len(df_zone) if only_val == 0 else 0
            tp = len(df_zone) if only_val == 1 else 0

            registry[zone] = {
                "clf": None,
                "reg": None,
                "thr": 0.5,
                "metrics_clf": {
                    "pr_auc": 1.0,
                    "f2": 0.0,
                    "accuracy": 1.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "tn": tn,
                    "fp": 0,
                    "fn": 0,
                    "tp": tp,
                },
                "metrics_reg": None,
            }

            out_zone = os.path.join(outdir, f"zone_{zone}")
            os.makedirs(out_zone, exist_ok=True)
            safe_report = {
              "status": "SAFE_ZONE",
              "samples": int(len(df_zone)),
              "message": "no fires observed here — model training skipped"
            }

            json.dump(safe_report, open(os.path.join(out_zone, "classifier_meta.json"), "w"), indent=2)
            json.dump(safe_report, open(os.path.join(out_zone, "regressor_meta.json"), "w"), indent=2)
            generate_zone_report(out_zone, os.path.join(out_zone, "report.html"))
            continue

        out_zone = os.path.join(outdir, f"zone_{zone}")
        tasks.append((df_zone, out_zone, zone, args))

    return tasks

class DualLogger:
    def __init__(self, path):
        # utf-8 so emojis and unicode don't crash
        self.file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def write(self, message):
        # print to original console
        try:
            sys.__stdout__.write(message)
        except UnicodeEncodeError:
            # if console can't show emoji, at least don’t crash
            sys.__stdout__.write(message.encode("utf-8", "replace").decode("utf-8"))

        # write to log file
        self.file.write(message)
        self.file.flush()

    def flush(self):
        try:
            sys.__stdout__.flush()
        except:
            pass
        self.file.flush()

def train_all_zones(tasks, registry, parallel=False):
    """
    Train models for all ecozones.
    
    Args:
        tasks       : list of (df_zone, out_zone, zone_id, args)
        registry    : existing registry dict to update
        parallel    : if True -> use parallel processing
    """

    task_count = len(tasks)
    
    if task_count == 0:
        print("[!] No eligible ecozones to train — exiting training stage.")
        return registry

    # ================================================
    # ---- SEQUENTIAL EXECUTION (default) ----
    # ================================================
    if not parallel:
        print("\n[i] Running SEQUENTIAL ecozone training...")
        for params in tasks:
            zone_id = params[2]
            print(f"[i] ---> Running zone {zone_id} sequentially")
            _, result = train_zone_wrapper(params)
            registry[zone_id] = result

        return registry

    # ================================================
    # ---- PARALLEL EXECUTION ----
    # ================================================
    print(f"\n[i] Launching PARALLEL ecozone training "
          f"on {min(task_count, os.cpu_count())} workers...")

    worker_count = max(1, min(task_count, os.cpu_count()))

    with ProcessPoolExecutor(max_workers=worker_count) as exe:

        futures = {exe.submit(train_zone_wrapper, t): t for t in tasks}

        for future in as_completed(futures):
            try:
                zone_id, result = future.result()
                registry[zone_id] = result
            except Exception as e:
                params = futures[future]
                z = params[2]
                print(f"[!] Zone {z} failed during parallel execution:", e)

    return registry

# -----------------------------
# Train only one leftover zone
# -----------------------------
def train_single_zone(zone_id, args):
    print(f"[▶] Resuming: Training only zone {zone_id}")

    # 1. Load & preprocess
    df = load_data(args.data)
    df = engineer_features(df)
    df, _, knn_router = extract_ecozones(df, eps=args.eps, min_samples=args.min_samples)

    # 2. Create output folder if needed (DON'T DELETE OLD DIRS)
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    zone_dir = os.path.join(outdir, f"zone_{zone_id}")
    os.makedirs(zone_dir, exist_ok=True)

    # 3. Filter only selected zone
    df_zone = df[df["ecozone"] == zone_id]
    if df_zone.empty:
        print(f"❗ Zone {zone_id} not found in dataset.")
        return

    # 4. Train zone
    registry = {}
    model = train_zone(df_zone, zone_dir, zone_id, args)
    registry[zone_id] = model

    # 5. Update registry file
    reg_path = os.path.join(outdir, "models_registry.pkl")
    if os.path.exists(reg_path):                         # load existing registry
        old_reg = joblib.load(reg_path)
        old_reg[zone_id] = model
        registry = old_reg

    joblib.dump(registry, reg_path)

    print(f"[✓] Zone {zone_id} training saved at {zone_dir}")


# -----------------------------
# Resume training for missing zones
# -----------------------------
def resume_leftover_zones(args):

    print("\n[▶] Resume mode: training leftover zones")

    df = load_data(args.data)
    df = engineer_features(df)
    df, _, _ = extract_ecozones(df, eps=args.eps, min_samples=args.min_samples)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    reg_path = os.path.join(outdir, "models_registry.pkl")

    # Load registry safely
    registry = joblib.load(reg_path) if os.path.exists(reg_path) else {}
    
    # Detect trained zones by report existence + registry entry
    trained = set()
    for z in registry.keys():
        if isinstance(z, int):
            zone_report = os.path.join(outdir, f"zone_{z}", "report.html")
            if os.path.exists(zone_report):      # <-- KEY FIX
                trained.add(z)

    print(f"[i] Already trained zones (verified): {trained}")

    available_zones = sorted(df["ecozone"].unique())
    print(f"[i] Available zones: {available_zones}")

    missing = [z for z in available_zones if z not in trained]
    print(f"[i] Leftover zones to train: {missing}")

    if not missing:
        print("✔ Nothing to resume — all zones completed")
        return

    for zone_id in missing:
        print(f"\n[🔸 Training Zone {zone_id}]")

        zone_dir = os.path.join(outdir, f"zone_{zone_id}")
        os.makedirs(zone_dir, exist_ok=True)

        df_zone = df[df["ecozone"] == zone_id]

        model = train_zone(df_zone, zone_dir, zone_id, args)

        registry[zone_id] = model
        joblib.dump(registry, reg_path)

    print("\n[✓] Resume training finished.")
def main(args):
    print("[i] Starting pipeline…")
    
    # 1. Load & Preprocess Data (Must happen in both modes to get zone definitions)
    df = load_data(args.data)
    df = engineer_features(df)
    
    # 2. Extract Ecozones
    df, _, knn_router = extract_ecozones(df, eps=args.eps, min_samples=args.min_samples)

    outdir = args.outdir
    reg_path = os.path.join(outdir, "models_registry.pkl")
    
    # --- RESUME LOGIC  ---
    completed_zones = set()
    registry = {}

    if args.resume:
        print(f"\n[i] 🟡 RESUME MODE DETECTED: Preserving contents of {outdir}")
        
        # Try to load existing registry
        if os.path.exists(reg_path):
            try:
                registry = joblib.load(reg_path)
                print(f"[i] Loaded existing registry with {len(registry)} entries.")
                
                # Validation: A zone is only 'done' if it is in registry AND report exists
                for z in list(registry.keys()):
                    # Skip non-integer keys like 'centroids'
                    if not isinstance(z, (int, np.integer)):
                        continue
                        
                    report_path = os.path.join(outdir, f"zone_{z}", "report.html")
                    if os.path.exists(report_path):
                        completed_zones.add(z)
                
                print(f"[i] Verified {len(completed_zones)} zones already completed.")
                
            except Exception as e:
                print(f"[!] Error loading registry: {e}. Starting fresh registry.")
                registry = {"centroids": compute_centroids(df)}
        else:
            print("[!] No registry found. Creating new one.")
            registry = {"centroids": compute_centroids(df)}
            
    else:
        # Default: Fresh Start -> Wipe everything
        print(f"\n[i] 🟢 FRESH START: Cleaning {outdir}...")
        clean_output_directory(outdir, delete_subfolders=True)
        registry = {"centroids": compute_centroids(df)}
    # --- RESUME LOGIC END ---

    # 3. Build Tasks
    # Note: build_zone_tasks calculates all potential tasks
    all_tasks = build_zone_tasks(df, args, registry, outdir)

    # 4. Filter Tasks (The Critical Step)
    # Task tuple structure: (df_zone, out_zone, zone_id, args)
    tasks_to_run = []
    for t in all_tasks:
        zone_id = t[2]
        if zone_id in completed_zones:
            # Skip if resuming and already done
            continue
        tasks_to_run.append(t)

    print(f"[i] Total Zones: {len(all_tasks)}")
    print(f"[i] Completed  : {len(completed_zones)}")
    print(f"[i] Remaining  : {len(tasks_to_run)}")

    if not tasks_to_run and len(all_tasks) > 0:
        print("[✓] All zones are already trained. Nothing to do.")
    else:
        # 5. Train Remaining Zones
        registry = train_all_zones(tasks_to_run, registry, parallel=False)

    # 6. Finalize (Run this every time to ensure global reports encompass old + new data)
    print("[i] Updating Global Artifacts (Router, Registry, Reports)...")
    
    # Retrain router (fast) to ensure it maps to all zones correctly
    train_ecozone_knn(df, outdir)

    # Save Registry
    joblib.dump(registry, reg_path)

    # Generate Reports
    generate_global_reports(registry, outdir)
    plot_india_zones(df, os.path.join(outdir, "india_ecozones.png"))


    print("[✓] Pipeline completed successfully.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ecozone-aware AutoML Wildfire Detection")

    # Training mode arguments
    parser.add_argument("--data", help="Path to training CSV file")
    parser.add_argument("--outdir", default="ecozone_results", help="Output directory")
    
    # Logic flags
    parser.add_argument("--resume", action="store_true", help="If set, skip directory clean and train only missing zones.")
    parser.add_argument("--predict", help="Path to CSV for prediction (inference mode)")
    parser.add_argument("--models", default="ecozone_results", help="Model directory for prediction")

    # Hyperparameters
    parser.add_argument("--eps", type=float, default=CFG.eps)
    parser.add_argument("--min_samples", type=int, default=CFG.min_samples)
    parser.add_argument("--min_zone_rows", type=int, default=CFG.min_zone_rows)
    parser.add_argument("--trials", type=int, default=CFG.optuna_trials)
    parser.add_argument("--ensemble_n", type=int, default=CFG.ensemble_n)
    parser.add_argument("--run_optuna", action="store_true")

    # Parse
    args = parser.parse_args()

    # Logging setup
    log_path = "log.txt"
    # Only clear log if NOT resuming
    if not args.resume:
        open(log_path, 'w').close() 

    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    # --- EXECUTION FLOW ---
    if args.predict:
        # Inference Mode
        ecofire_predict(args.predict, args.models)

    elif args.data:
        # Training Mode (Fresh or Resume handled inside main)
        main(args)

    else:
        parser.print_help()

        print("\n[!] Error: You must provide --data (for training) or --predict (for inference).")
