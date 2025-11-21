#!/usr/bin/env python3
"""
Adapted Deep Learning Fire Prediction
Works with the 'final dataset.csv' format using a feedforward neural network
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Configuration
class Config:
    data_path = "final dataset.csv"
    target_col = "fire"
    test_size = 0.2
    random_state = 42
    
    # Training parameters
    batch_size = 64
    learning_rate = 0.001
    n_epochs = 200
    patience = 20
    
    # Model architecture
    hidden_dims = [256, 128, 64, 32]
    dropout_rate = 0.3
    
CFG = Config()

# Set random seeds
torch.manual_seed(CFG.random_state)
np.random.seed(CFG.random_state)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
        # Fill missing values
        df = df.fillna(df.median())
        print("✓ Missing values filled with median")
    else:
        print("\n✓ No missing values found")
    
    # Display target distribution
    print(f"\nTarget variable '{CFG.target_col}' distribution:")
    print(df[CFG.target_col].value_counts().sort_index())
    print(f"\nTarget statistics:")
    print(df[CFG.target_col].describe())
    
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
    
    # Fire danger index
    if all(col in df.columns for col in ['tmax', 'humidity', 'windspeed']):
        df_feat['temp_component'] = np.maximum(0, df_feat['tmax'] - 273.15)
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
    
    # NDVI-based features
    if 'ndvi' in df.columns:
        df_feat['ndvi_normalized'] = df_feat['ndvi'] / 10000
        df_feat['vegetation_stress'] = 1 / (df_feat['ndvi'] + 1)
        print("✓ Vegetation features created")
    
    # Topographic interaction
    if all(col in df.columns for col in ['elevation', 'slope']):
        df_feat['topo_risk'] = df_feat['elevation'] * df_feat['slope'] / 1000
        print("✓ Topographic features created")
    
    # Aspect features (cyclical encoding)
    if 'aspect' in df.columns:
        df_feat['aspect_sin'] = np.sin(2 * np.pi * df_feat['aspect'] / 360)
        df_feat['aspect_cos'] = np.cos(2 * np.pi * df_feat['aspect'] / 360)
        print("✓ Aspect features created")
    
    print(f"\nTotal features: {len(df_feat.columns) - 1}")
    
    return df_feat

class FirePredictionNN(nn.Module):
    """Neural Network for Fire Prediction"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3, task='binary'):
        super(FirePredictionNN, self).__init__()
        
        self.task = task
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer
        if task == 'binary':
            self.output = nn.Linear(prev_dim, 1)
            self.activation = nn.Sigmoid()
        else:  # regression
            self.output = nn.Linear(prev_dim, 1)
            self.activation = nn.Softplus()  # Ensure positive output
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output(x)
        x = self.activation(x)
        return x.squeeze()

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_classifier(X_train, X_test, y_train, y_test, scaler):
    """Train binary classifier"""
    print(f"\n{'='*60}")
    print("TRAINING BINARY CLASSIFIER (Neural Network)")
    print(f"{'='*60}")
    
    # Create binary target
    y_train_cls = (y_train > 0).astype(np.float32)
    y_test_cls = (y_test > 0).astype(np.float32)
    
    print(f"Positive class rate: {y_train_cls.mean():.3f}")
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_cls)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test_cls)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
    
    # Create model
    model = FirePredictionNN(
        input_dim=X_train.shape[1],
        hidden_dims=CFG.hidden_dims,
        dropout_rate=CFG.dropout_rate,
        task='binary'
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("\nTraining...")
    for epoch in range(CFG.n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 20 == 0 or epoch == CFG.n_epochs - 1:
            print(f"Epoch {epoch:3d}/{CFG.n_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= CFG.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred_proba = model(X_test_tensor).cpu().numpy()
    
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
    
    return model, train_losses, val_losses, y_pred_proba

def train_regressor(X_train, X_test, y_train, y_test, scaler):
    """Train regression model on fire days"""
    print(f"\n{'='*60}")
    print("TRAINING REGRESSOR (Neural Network)")
    print(f"{'='*60}")
    
    # Filter fire days
    fire_mask_train = y_train > 0
    fire_mask_test = y_test > 0
    
    if fire_mask_train.sum() < 10:
        print("⚠ Not enough fire days for regression training")
        return None, None, None
    
    X_train_fire = X_train[fire_mask_train]
    y_train_fire = np.log1p(y_train[fire_mask_train]).astype(np.float32)
    
    X_test_fire = X_test[fire_mask_test]
    y_test_fire = np.log1p(y_test[fire_mask_test]).astype(np.float32)
    
    print(f"Training on {len(X_train_fire)} fire days")
    print(f"Testing on {len(X_test_fire)} fire days")
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train_fire)
    X_test_scaled = scaler.transform(X_test_fire)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_fire)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test_fire)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
    
    # Create model
    model = FirePredictionNN(
        input_dim=X_train_fire.shape[1],
        hidden_dims=CFG.hidden_dims,
        dropout_rate=CFG.dropout_rate,
        task='regression'
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("\nTraining...")
    for epoch in range(CFG.n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 20 == 0 or epoch == CFG.n_epochs - 1:
            print(f"Epoch {epoch:3d}/{CFG.n_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= CFG.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred_reg = model(X_test_tensor).cpu().numpy()
    
    # Metrics
    mae = mean_absolute_error(y_test_fire, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_fire, y_pred_reg))
    r2 = r2_score(y_test_fire, y_pred_reg)
    
    print(f"\n📊 REGRESSION RESULTS (Log Scale):")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Original scale
    y_test_orig = np.expm1(y_test_fire)
    y_pred_orig = np.expm1(y_pred_reg)
    
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    
    print(f"\n📊 REGRESSION RESULTS (Original Scale):")
    print(f"MAE: {mae_orig:.4f}")
    print(f"RMSE: {rmse_orig:.4f}")
    
    return model, train_losses, val_losses

def plot_training_curves(clf_train_losses, clf_val_losses, reg_train_losses, reg_val_losses):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Classification
    axes[0].plot(clf_train_losses, label='Train Loss', alpha=0.8)
    axes[0].plot(clf_val_losses, label='Val Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Classification Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Regression
    if reg_train_losses:
        axes[1].plot(reg_train_losses, label='Train Loss', alpha=0.8)
        axes[1].plot(reg_val_losses, label='Val Loss', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Regression Training Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training curves saved to 'training_curves.png'")
    plt.close()

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("FOREST FIRE PREDICTION - DEEP LEARNING APPROACH")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Feature engineering
    df_feat = engineer_features(df)
    
    # Prepare train/test
    X = df_feat.drop(columns=[CFG.target_col])
    y = df_feat[CFG.target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.test_size, random_state=CFG.random_state,
        stratify=(y > 0) if (y > 0).sum() > 10 else None
    )
    
    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT")
    print(f"{'='*60}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Positive rate in train: {(y_train > 0).mean():.3f}")
    print(f"Positive rate in test: {(y_test > 0).mean():.3f}")
    
    # Train classifier
    scaler_cls = StandardScaler()
    clf_model, clf_train_losses, clf_val_losses, y_pred_proba = train_classifier(
        X_train.values, X_test.values, y_train, y_test, scaler_cls
    )
    
    # Train regressor
    scaler_reg = StandardScaler()
    reg_model, reg_train_losses, reg_val_losses = train_regressor(
        X_train.values, X_test.values, y_train, y_test, scaler_reg
    )
    
    # Plot training curves
    plot_training_curves(clf_train_losses, clf_val_losses, reg_train_losses, reg_val_losses)
    
    # Combined evaluation
    if reg_model is not None:
        print(f"\n{'='*60}")
        print("COMBINED MODEL EVALUATION")
        print(f"{'='*60}")
        
        # Scale test data
        X_test_scaled = scaler_reg.transform(X_test.values)
        
        # Predictions
        reg_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_reg = reg_model(X_test_tensor).cpu().numpy()
        
        # Combined: p(fire) * intensity
        y_pred_combined = y_pred_proba * np.expm1(y_pred_reg)
        
        mae_combined = mean_absolute_error(y_test, y_pred_combined)
        rmse_combined = np.sqrt(mean_squared_error(y_test, y_pred_combined))
        
        print(f"\n📊 COMBINED MODEL RESULTS:")
        print(f"MAE: {mae_combined:.4f}")
        print(f"RMSE: {rmse_combined:.4f}")
        
        print(f"\nPrediction Statistics:")
        print(f"Mean predicted: {y_pred_combined.mean():.4f}")
        print(f"Mean actual: {y_test.mean():.4f}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
