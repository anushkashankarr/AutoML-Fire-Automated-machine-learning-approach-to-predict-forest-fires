# 🔥 Forest Fire Prediction — AutoML Two-Stage Model

### *(LightGBM + Optuna + Feature Engineering)*

This repository provides an **AutoML-style two-stage machine learning pipeline** for predicting forest fire occurrence and intensity using environmental, vegetation, and topographic features.

The approach includes:

* **Stage 1:** Binary classification *(fire vs. no fire)*
* **Stage 2:** Regression *(predicting fire intensity on fire days)*
* **Feature Engineering:** Temperature indices, drought metrics, vegetation stress, topographic risk, fire danger index
* **Optional Hyperparameter Optimization:** Optuna
* **Training Curve Visualization**

---

## 📌 Key Features

### ✔ Two-Stage Prediction Strategy

* **Classification:** Probability of fire (0/1)
* **Regression:** Fire intensity *only on fire days*
* **Combined Output:**

[
\text{Predicted Fire Count} = 1(\text{fire predicted}) \times \text{Intensity}
]

---

### ✔ Automatic Feature Engineering

The script creates multiple engineered features:

* **Temperature features:** `temp_range`, `temp_avg`
* **Fire danger index (FWI-like)**
* **Drought index**
* **Vegetation stress:** `ndvi_normalized`, `vegetation_stress`
* **Topographic risk:** *slope × elevation*
* **Cyclic aspect encoding:** `aspect_sin`, `aspect_cos`

---

### ✔ Hyperparameter Optimization (Optional)

If Optuna is installed:

* **Stage 1 (classification):** optimized for *AUC*
* **Stage 2 (regression):** optimized for *MSE*

---

### ✔ LightGBM First, HistGradientBoosting Fallback

Automatic fallback depending on availability.

---

### ✔ Visual Outputs

Classification training curves saved to:

```
automl_training_curves.png
```

---

## 📦 Installation

### 1. Install required libraries

```bash
pip install numpy pandas scikit-learn matplotlib
```

### 2. (Optional, recommended)

```bash
pip install lightgbm optuna
```

### 3. Conda setup (optional)

```bash
conda create -n fire python=3.9
conda activate fire
pip install -r requirements.txt
```

---

## 📁 Data Requirements

The script expects a CSV named:

```
final_dataset_automl.csv
```

#### Required columns:

```
tmax, tmin, humidity, windspeed
soil_moisture, rain, ndvi
cloudcover
elevation, slope, aspect
landcover
```

**Target:**

```
fire intensity index
```

#### Example entry:

```
system:index,aspect,cloudcover,elevation,fire,fireClass,fire_intensity,humidity,landcover,ndvi,rain,slope,soil_moisture,solar_radiation,tmax,tmin,u_wind,v_wind,windspeed,.geo
0,282,14.255...,477,0,0,0,66.65,4,7067.27,0.0043,1,0.293,1.776E7,304.72,296.13,0.043,0.708,0.710,{...}
```

---

## 🚀 Usage

### Run normally:

```bash
python3 train_automl_adapted.py
```

### Custom input file:

```bash
python train_automl_adapted.py --data my_data.csv
```

---

## ⚙ Configuration

The `Config` class controls defaults:

```python
class Config:
    data_path = "final_dataset_automl.csv"
    target_col = "fire"
    test_size = 0.2
    random_state = 42
    n_trials = 100

    n_estimators = 300
    learning_rate = 0.05
    max_depth = 7
```

---

## 📊 Outputs

### **Stage 1 — Classification**

* AUC-ROC
* Average Precision
* F1 Score
* Confusion matrix
* Classification report
* Feature importance

### **Stage 2 — Regression (fire days ONLY)**

* MAE
* RMSE
* R²
* Feature importance

### **Combined Two-Stage Metrics**

* MAE
* RMSE
* R²
* Predicted vs. actual fire-day counts

### **Generated Files**

```
automl_training_curves.png
```

---

## 🧠 How the Model Works

### 🔹 Stage 1 — Fire vs. No Fire

A LightGBM (fallback: HistGradientBoosting) classifier predicts:

[
P(\text{fire} > 0)
]

Stratification ensures balanced splits.

### 🔹 Stage 2 — Intensity on Fire Days

Regression trained only where:

[
\text{fire} > 0
]

### 🔹 Combined Output

Final predicted fire count:

[
\text{pred_fire} = (\text{pred_proba} > 0.5) \times \text{predicted_intensity}
]

---

## 🧪 Hyperparameter Optimization

If **Optuna** is installed:

* Classification → optimized using **ROC-AUC**
* Regression → optimized using **negative MSE**
* Uses **3-fold cross-validation**

Disable Optuna simply by **not installing it**.

---
