#!/usr/bin/env python3
"""
Adapted AutoML Two-Stage Fire Prediction
Works with the 'final dataset.csv' format
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)

# Try to import optional libraries
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
    print("✓ LightGBM available")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not available, using HistGradientBoosting")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✓ Optuna available")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠ Optuna not available, using default parameters")

# Configuration
class Config:
    data_path = "final dataset.csv"
    target_col = "fire"
    test_size = 0.2
    random_state = 42
    n_trials = 50  # Optuna trials
    
    # Model parameters
    n_estimators = 200
    learning_rate = 0.05
    max_depth = 7

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='AutoML Two-Stage Fire Prediction')
    parser.add_argument('--data', type=str, default="final dataset.csv",
                        help='Path to input CSV file')
    return parser.parse_args()

CFG = Config()

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    df = pd.read_csv(CFG.data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        print("\n✓ No missing values found")
    
    # Display target distribution
    print(f"\nTarget variable '{CFG.target_col}' distribution:")
    print(df[CFG.target_col].value_counts().sort_index())
    print(f"\nTarget statistics:")
    print(df[CFG.target_col].describe())
    
    # Check if target is binary or count
    unique_vals = df[CFG.target_col].nunique()
    is_binary = unique_vals == 2
    print(f"\nTarget type: {'Binary' if is_binary else 'Count/Continuous'} ({unique_vals} unique values)")
    
    return df

def engineer_features(df):
    """Create additional features"""
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    df_feat = df.copy()
    
    # Temperature-based features
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df_feat['temp_range'] = df_feat['tmax'] - df_feat['tmin']
        df_feat['temp_avg'] = (df_feat['tmax'] + df_feat['tmin']) / 2
        print("✓ Temperature features created")
    
    # Fire danger index (simplified FWI-like)
    if all(col in df.columns for col in ['tmax', 'humidity', 'windspeed']):
        df_feat['temp_component'] = np.maximum(0, df_feat['tmax'] - 273.15)  # Convert to Celsius if in Kelvin
        df_feat['humidity_component'] = np.maximum(0, 100 - df_feat['humidity'])
        df_feat['wind_component'] = df_feat['windspeed'] ** 1.5
        df_feat['fire_danger_index'] = (
            df_feat['temp_component'] * 
            df_feat['humidity_component'] * 
            df_feat['wind_component']
        ) / 10000
        print("✓ Fire danger index created")
    
    # Drought index
    if 'soil_moisture' in df.columns and 'rain' in df.columns:
        df_feat['drought_index'] = np.maximum(0, 1 - df_feat['soil_moisture'] - df_feat['rain'])
        print("✓ Drought index created")
    
    # NDVI-based vegetation features
    if 'ndvi' in df.columns:
        df_feat['ndvi_normalized'] = df_feat['ndvi'] / 10000  # Normalize if needed
        df_feat['vegetation_stress'] = 1 / (df_feat['ndvi'] + 1)
        print("✓ Vegetation features created")
    
    # Topographic interaction
    if all(col in df.columns for col in ['elevation', 'slope']):
        df_feat['topo_risk'] = df_feat['elevation'] * df_feat['slope'] / 1000
        print("✓ Topographic features created")
    
    # Aspect-based features (cyclical encoding)
    if 'aspect' in df.columns:
        df_feat['aspect_sin'] = np.sin(2 * np.pi * df_feat['aspect'] / 360)
        df_feat['aspect_cos'] = np.cos(2 * np.pi * df_feat['aspect'] / 360)
        print("✓ Aspect features created")
    
    print(f"\nTotal features: {len(df_feat.columns) - 1}")  # Exclude target
    
    return df_feat

def prepare_train_test(df, target_col):
    """Prepare train/test split"""
    print(f"\n{'='*60}")
    print("PREPARING TRAIN/TEST SPLIT")
    print(f"{'='*60}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.test_size, random_state=CFG.random_state, stratify=y if y.nunique() < 10 else None
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Positive rate in train: {(y_train > 0).mean():.3f}")
    print(f"Positive rate in test: {(y_test > 0).mean():.3f}")
    
    return X_train, X_test, y_train, y_test

def train_stage1_classifier(X_train, X_test, y_train, y_test):
    """Stage 1: Binary Classification"""
    print(f"\n{'='*60}")
    print("STAGE 1: BINARY CLASSIFICATION")
    print(f"{'='*60}")
    
    # Determine threshold for binary classification
    # If data is continuous (many unique values), use median as threshold
    # If data is binary (2 unique values), use 0 as threshold
    unique_vals = y_train.nunique()
    
    if unique_vals > 10:
        # Continuous data: use threshold of 5 (middle of 0-10 scale)
        threshold = 5.0
        print(f"Continuous fire values detected. Using threshold = {threshold}")
        print(f"Low fire (< {threshold}): No fire class")
        print(f"High fire (>= {threshold}): Fire class")
    else:
        # Binary or count data: use 0 as threshold
        threshold = 0
        print(f"Binary/count fire values detected. Using threshold = {threshold}")
    
    # Create binary target
    y_train_cls = (y_train >= threshold).astype(int)
    y_test_cls = (y_test >= threshold).astype(int)
    
    print(f"Positive class rate: {y_train_cls.mean():.3f}")
    print(f"Negative class rate: {(1 - y_train_cls).mean():.3f}")
    
    # Train classifier
    if LIGHTGBM_AVAILABLE:
        clf = LGBMClassifier(
            n_estimators=CFG.n_estimators,
            learning_rate=CFG.learning_rate,
            max_depth=CFG.max_depth,
            class_weight='balanced',
            random_state=CFG.random_state,
            verbose=-1
        )
        print("Using LightGBM Classifier")
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=CFG.n_estimators,
            learning_rate=CFG.learning_rate,
            max_depth=CFG.max_depth,
            random_state=CFG.random_state
        )
        print("Using HistGradientBoosting Classifier")
    
    clf.fit(X_train, y_train_cls)
    
    # Predictions
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_cls = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test_cls, y_pred_proba)
    ap = average_precision_score(y_test_cls, y_pred_proba)
    f1 = f1_score(y_test_cls, y_pred_cls)
    
    print(f"\n📊 CLASSIFICATION RESULTS:")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test_cls, y_pred_cls))
    
    print(f"\nClassification Report:")
    print(classification_report(y_test_cls, y_pred_cls, digits=3))
    
    # Feature importance
    if hasattr(clf, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'feature': X_train.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Important Features:")
        print(feature_imp.head(10).to_string(index=False))
    
    return clf, y_pred_proba

def train_stage2_regressor(X_train, X_test, y_train, y_test, y_pred_proba):
    """Stage 2: Regression on fire days"""
    print(f"\n{'='*60}")
    print("STAGE 2: REGRESSION (Fire Intensity)")
    print(f"{'='*60}")
    
    # Determine threshold based on data type
    unique_vals = y_train.nunique()
    if unique_vals > 10:
        # Continuous data: use all samples for regression
        threshold = 0
        print(f"Continuous fire values: Training on all samples")
    else:
        # Binary/count data: filter fire days only
        threshold = 0
        print(f"Binary/count fire values: Training on fire days only")
    
    # Filter fire days only
    fire_mask_train = y_train > threshold
    fire_mask_test = y_test > threshold
    
    if fire_mask_train.sum() < 10:
        print("⚠ Not enough fire days for regression training")
        return None, None
    
    X_train_fire = X_train[fire_mask_train]
    y_train_fire = np.log1p(y_train[fire_mask_train])
    
    X_test_fire = X_test[fire_mask_test]
    y_test_fire = np.log1p(y_test[fire_mask_test])
    
    print(f"Training on {len(X_train_fire)} fire days")
    print(f"Testing on {len(X_test_fire)} fire days")
    
    # Train regressor
    if LIGHTGBM_AVAILABLE:
        reg = LGBMRegressor(
            n_estimators=CFG.n_estimators,
            learning_rate=CFG.learning_rate,
            max_depth=CFG.max_depth,
            random_state=CFG.random_state,
            verbose=-1
        )
        print("Using LightGBM Regressor")
    else:
        reg = HistGradientBoostingRegressor(
            max_iter=CFG.n_estimators,
            learning_rate=CFG.learning_rate,
            max_depth=CFG.max_depth,
            random_state=CFG.random_state
        )
        print("Using HistGradientBoosting Regressor")
    
    reg.fit(X_train_fire, y_train_fire)
    
    # Predictions
    y_pred_reg_fire = reg.predict(X_test_fire)
    
    # Metrics on fire days
    mae = mean_absolute_error(y_test_fire, y_pred_reg_fire)
    rmse = np.sqrt(mean_squared_error(y_test_fire, y_pred_reg_fire))
    r2 = r2_score(y_test_fire, y_pred_reg_fire)
    
    print(f"\n📊 REGRESSION RESULTS (Log Scale):")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Transform back to original scale
    y_test_fire_orig = np.expm1(y_test_fire)
    y_pred_reg_fire_orig = np.expm1(y_pred_reg_fire)
    
    mae_orig = mean_absolute_error(y_test_fire_orig, y_pred_reg_fire_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_test_fire_orig, y_pred_reg_fire_orig))
    
    print(f"\n📊 REGRESSION RESULTS (Original Scale):")
    print(f"MAE: {mae_orig:.4f}")
    print(f"RMSE: {rmse_orig:.4f}")
    
    return reg, y_pred_reg_fire

def evaluate_combined_model(clf, reg, X_test, y_test):
    """Evaluate the combined two-stage model"""
    print(f"\n{'='*60}")
    print("COMBINED TWO-STAGE MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Stage 1: Predict fire probability
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Stage 2: Predict fire intensity
    y_pred_reg = reg.predict(X_test)
    
    # Combined prediction: p(fire) * intensity
    y_pred_combined = y_pred_proba * np.expm1(y_pred_reg)
    
    # Evaluate
    mae_combined = mean_absolute_error(y_test, y_pred_combined)
    rmse_combined = np.sqrt(mean_squared_error(y_test, y_pred_combined))
    
    print(f"\n📊 COMBINED MODEL RESULTS:")
    print(f"MAE: {mae_combined:.4f}")
    print(f"RMSE: {rmse_combined:.4f}")
    
    # Additional statistics
    print(f"\nPrediction Statistics:")
    print(f"Mean predicted fire count: {y_pred_combined.mean():.4f}")
    print(f"Mean actual fire count: {y_test.mean():.4f}")
    print(f"Max predicted: {y_pred_combined.max():.4f}")
    print(f"Max actual: {y_test.max():.4f}")
    
    return y_pred_combined

def optimize_with_optuna(X_train, y_train):
    """Optimize hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
        print("⚠ Optuna or LightGBM not available, skipping optimization")
        return None
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print(f"{'='*60}")
    
    y_train_cls = (y_train > 0).astype(int)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'class_weight': 'balanced',
            'random_state': CFG.random_state,
            'verbose': -1
        }
        
        clf = LGBMClassifier(**params)
        scores = cross_val_score(clf, X_train, y_train_cls, cv=3, scoring='roc_auc')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=CFG.n_trials, show_progress_bar=True)
    
    print(f"\n✓ Best AUC: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params

def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_args()
    CFG.data_path = args.data
    
    print("\n" + "="*60)
    print("FOREST FIRE PREDICTION - AUTOML TWO-STAGE APPROACH")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Feature engineering
    df_feat = engineer_features(df)
    
    # Prepare train/test
    X_train, X_test, y_train, y_test = prepare_train_test(df_feat, CFG.target_col)
    
    # Optional: Optimize hyperparameters
    if OPTUNA_AVAILABLE and LIGHTGBM_AVAILABLE:
        print("\nStarting hyperparameter optimization...")
        best_params = optimize_with_optuna(X_train, y_train)
        if best_params:
            CFG.n_estimators = best_params.get('n_estimators', CFG.n_estimators)
            CFG.learning_rate = best_params.get('learning_rate', CFG.learning_rate)
            CFG.max_depth = best_params.get('max_depth', CFG.max_depth)
    
    # Stage 1: Classification
    clf, y_pred_proba = train_stage1_classifier(X_train, X_test, y_train, y_test)
    
    # Stage 2: Regression
    reg, y_pred_reg = train_stage2_regressor(X_train, X_test, y_train, y_test, y_pred_proba)
    
    # Combined evaluation
    if reg is not None:
        y_pred_combined = evaluate_combined_model(clf, reg, X_test, y_test)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
