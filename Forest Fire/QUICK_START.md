# Quick Start Guide - Forest Fire Prediction

## ⚡ Fast Setup (3 Steps)

### Step 1: Install Dependencies

```bash
# Option A: Using the setup script (recommended)
bash setup.sh

# Option B: Manual installation
pip install -r requirements.txt
```

### Step 2: Add Your Data

Place your `final dataset.csv` file in this directory with these columns:
- Features: tmax, tmin, humidity, windspeed, soil_moisture, ndvi, rain, cloudcover, elevation, slope, aspect, landcover
- Target: fire

### Step 3: Train Models

```bash
# Train AutoML model (faster, typically 2-5 minutes)
python train_automl_adapted.py

# Train Deep Learning model (slower, typically 5-15 minutes)
python train_dl_adapted.py
```

## 📊 Understanding the Output

### AutoML Output (train_automl_adapted.py)

You'll see three sets of results:

1. **Stage 1 - Classification Results:**
   ```
   AUC-ROC: 0.8500
   Average Precision: 0.7800
   F1 Score: 0.7200
   ```

2. **Stage 2 - Regression Results (fire days only):**
   ```
   MAE: 0.4500
   RMSE: 0.6200
   R² Score: 0.6500
   ```

3. **Combined Model Results:**
   ```
   MAE: 0.3200
   RMSE: 0.5100
   ```

### Deep Learning Output (train_dl_adapted.py)

Similar structure plus:
- Training progress every 20 epochs
- `training_curves.png` saved to disk
- Model automatically stops early if not improving

## 🎯 What Are Good Scores?

### Classification (Stage 1)
- **AUC-ROC > 0.80**: Excellent fire/no-fire discrimination
- **AUC-ROC 0.70-0.80**: Good
- **AUC-ROC 0.60-0.70**: Fair
- **AUC-ROC < 0.60**: Poor (investigate data quality)

### Regression (Stage 2)
- **R² > 0.60**: Excellent fire intensity prediction
- **R² 0.40-0.60**: Good
- **R² 0.20-0.40**: Fair
- **R² < 0.20**: Poor

### Combined Model
- Lower MAE/RMSE is better
- Compare against baseline (mean prediction)

## 🔧 Improving Scores

### If Classification Scores Are Low:

1. **Enable Hyperparameter Optimization** (AutoML only):
   - The script may ask if you want to run Optuna
   - This will take longer but usually improves scores

2. **Check Data Quality**:
   ```python
   import pandas as pd
   df = pd.read_csv('final dataset.csv')
   
   # Check target distribution
   print(df['fire'].value_counts())
   
   # Check for missing values
   print(df.isnull().sum())
   
   # Check feature correlations
   print(df.corr()['fire'].sort_values(ascending=False))
   ```

3. **Increase Model Complexity** (Deep Learning):
   - Edit `train_dl_adapted.py`
   - Change `hidden_dims = [256, 128, 64, 32]` to `[512, 256, 128, 64]`
   - Increase `n_epochs = 300`

### If Regression Scores Are Low:

1. **More Fire Days Needed**:
   - Regression only trains on fire days
   - Need at least 50-100 fire days for good results

2. **Feature Engineering**:
   - Both scripts auto-create fire danger indices
   - Check if they're being created (watch console output)

3. **Check Target Distribution**:
   ```python
   # Look at fire intensity distribution
   df_fire = df[df['fire'] > 0]
   print(df_fire['fire'].describe())
   ```

## 🚀 Advanced Usage

### Running Both Models and Comparing

```bash
# Train both models
python train_automl_adapted.py > automl_results.txt
python train_dl_adapted.py > dl_results.txt

# Compare results
echo "=== AutoML Results ==="
grep "AUC-ROC\|MAE\|RMSE" automl_results.txt

echo "=== Deep Learning Results ==="
grep "AUC-ROC\|MAE\|RMSE" dl_results.txt
```

### Custom Configuration

Edit the `Config` class at the top of each script:

```python
# Example: Use 30% for testing instead of 20%
class Config:
    test_size = 0.3  # Changed from 0.2
    
# Example: More aggressive hyperparameter search
class Config:
    n_trials = 100  # Changed from 50 (AutoML only)
```

### GPU Acceleration (Deep Learning)

If you have CUDA-capable GPU:

```bash
# Check if PyTorch detects GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Training will automatically use GPU if available
python train_dl_adapted.py
```

## 📈 Typical Training Times

- **AutoML (without Optuna)**: 2-5 minutes
- **AutoML (with Optuna, 50 trials)**: 10-20 minutes
- **Deep Learning (CPU)**: 5-15 minutes
- **Deep Learning (GPU)**: 2-5 minutes

## ❓ Common Questions

**Q: Which model should I use?**
- Start with AutoML - it's faster and often performs well
- Try Deep Learning if you have GPU or want to experiment
- Use both and ensemble predictions for best results

**Q: Can I use my own feature names?**
- Yes! Edit the `Config` class and update column names
- Make sure to update feature engineering code accordingly

**Q: How do I save the trained models?**
- AutoML: Models are in memory, add pickle/joblib save code
- Deep Learning: Add `torch.save(model.state_dict(), 'model.pth')`

**Q: My data has temporal information, should I use time-based CV?**
- Yes! The original notebooks use `TimeSeriesSplit`
- Current scripts use simple split for simplicity
- Modify to use time-aware splitting if data is temporal

## 🐛 Troubleshooting

```bash
# If import errors:
pip install --upgrade -r requirements.txt

# If "CUDA out of memory":
# Edit Config in train_dl_adapted.py
# Set: batch_size = 32  # or even 16

# If "Not enough fire days":
# Your dataset needs more positive samples
# Try class balancing techniques or get more data
```

## 📞 Need Help?

1. Check the detailed README.md
2. Review error messages carefully
3. Ensure data format matches exactly
4. Try with a smaller subset first

---

**Remember**: Machine learning is iterative. Run experiments, analyze results, adjust parameters, and repeat!
