# Forest Fire Prediction - Training Scripts

This project contains adapted training scripts for forest fire prediction using both AutoML and Deep Learning approaches.

## 📋 Overview

Two training approaches are available:

1. **AutoML Two-Stage Approach** (`train_automl_adapted.py`)
   - Stage 1: Binary classification to predict fire occurrence
   - Stage 2: Regression to predict fire intensity on fire days
   - Uses LightGBM (or HistGradientBoosting as fallback)
   - Includes Optuna for hyperparameter optimization

2. **Deep Learning Approach** (`train_dl_adapted.py`)
   - Neural network with customizable architecture
   - Two-stage approach (classification + regression)
   - PyTorch-based implementation
   - GPU support available

## 🔧 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install with conda

```bash
# Create a new environment
conda create -n forestfire python=3.9
conda activate forestfire

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

## 📊 Data Requirements

The scripts expect a CSV file named `final dataset.csv` with the following columns:

**Features:**
- `tmax`: Maximum temperature
- `tmin`: Minimum temperature
- `humidity`: Relative humidity
- `windspeed`: Wind speed
- `soil_moisture`: Soil moisture content
- `ndvi`: Normalized Difference Vegetation Index
- `rain`: Precipitation amount
- `cloudcover`: Cloud cover percentage
- `elevation`: Elevation (meters)
- `slope`: Terrain slope
- `aspect`: Terrain aspect (degrees)
- `landcover`: Land cover type

**Target:**
- `fire`: Fire occurrence (0 for no fire, >0 for fire count/intensity)

### Data Sample

```csv
tmax,tmin,humidity,windspeed,soil_moisture,ndvi,rain,cloudcover,elevation,slope,aspect,landcover,fire
304.73,296.13,66.66,0.71,0.29,7067.28,0.0043,14.26,477,1,282,4,0
```

## 🚀 Usage

### 1. AutoML Two-Stage Approach

```bash
python train_automl_adapted.py
```

**Features:**
- Automatic feature engineering
- Fire danger indices (FWI-like)
- Drought indices
- Vegetation stress indicators
- Topographic risk features
- Hyperparameter optimization with Optuna (optional)

**Output:**
- Classification metrics (AUC-ROC, Average Precision, F1)
- Regression metrics (MAE, RMSE, R²)
- Combined two-stage model evaluation
- Feature importance rankings

### 2. Deep Learning Approach

```bash
python train_dl_adapted.py
```

**Features:**
- Feedforward neural network
- Batch normalization and dropout
- Early stopping
- Learning rate scheduling
- GPU acceleration (if available)
- Training curve visualization

**Output:**
- Classification metrics (AUC-ROC, Average Precision, F1)
- Regression metrics (MAE, RMSE, R²)
- Combined model evaluation
- Training curves saved as PNG

## 📈 Expected Performance

### Classification Metrics
- **AUC-ROC**: 0.75-0.95 (higher is better)
- **Average Precision**: 0.70-0.90 (higher is better)
- **F1 Score**: 0.60-0.85 (higher is better)

### Regression Metrics (on fire days)
- **MAE**: Depends on fire intensity scale
- **RMSE**: Depends on fire intensity scale
- **R² Score**: 0.40-0.80 (higher is better)

## ⚙️ Configuration

### AutoML Configuration

Edit the `Config` class in `train_automl_adapted.py`:

```python
class Config:
    data_path = "final dataset.csv"  # Path to your data
    target_col = "fire"              # Target column name
    test_size = 0.2                  # Test set proportion
    random_state = 42                # Random seed
    n_trials = 50                    # Optuna trials
    n_estimators = 200               # Number of trees
    learning_rate = 0.05             # Learning rate
    max_depth = 7                    # Maximum tree depth
```

### Deep Learning Configuration

Edit the `Config` class in `train_dl_adapted.py`:

```python
class Config:
    data_path = "final dataset.csv"  # Path to your data
    target_col = "fire"              # Target column name
    test_size = 0.2                  # Test set proportion
    batch_size = 64                  # Batch size
    learning_rate = 0.001            # Learning rate
    n_epochs = 200                   # Maximum epochs
    patience = 20                    # Early stopping patience
    hidden_dims = [256, 128, 64, 32] # Network architecture
    dropout_rate = 0.3               # Dropout rate
```

## 📁 Output Files

After training, you'll get:

- **AutoML**: Console output with detailed metrics and feature importance
- **Deep Learning**: 
  - Console output with metrics
  - `training_curves.png` - Visualization of training progress

## 🔬 Model Approaches

### Two-Stage Strategy

Both approaches use a two-stage strategy:

1. **Stage 1 (Classification)**: Predict whether a fire will occur (binary: 0 or 1)
2. **Stage 2 (Regression)**: Predict fire intensity/count only on predicted fire days

**Combined Prediction**: `P(fire) × Intensity = Expected Fire Count`

### Issue: "FileNotFoundError: final dataset.csv"
**Solution**: Make sure `final dataset.csv` is in the same directory as the scripts

### Issue: "CUDA out of memory" (Deep Learning)
**Solution**: Reduce `batch_size` in Config or use CPU

### Issue: "LightGBM not available"
**Solution**: Install with `pip install lightgbm` or scripts will use scikit-learn fallback

### Issue: Poor performance
**Solutions**:
- Check data quality and feature distributions
- Enable hyperparameter optimization
- Try both approaches and compare
- Increase training epochs (Deep Learning)
- Increase n_trials for Optuna (AutoML)

## 📚 References

- **AutoML**: Based on two-stage fire prediction methodology
- **Deep Learning**: PyTorch neural network with domain-specific features
- **Fire Indices**: Simplified Fire Weather Index (FWI) components

## 🤝 Contributing

Feel free to improve the scripts by:
- Adding more sophisticated feature engineering
- Implementing additional model architectures
- Adding visualization capabilities
- Improving documentation

## 📝 License

MIT License - Feel free to use and modify for your research/projects.

---

**Note**: Make sure to place your `final dataset.csv` file in the project directory before running the training scripts!
