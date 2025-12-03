# 🔥 AGNITRA : Ecozone-Aware AutoML Wildfire System for pan-India

### A Spatially Adaptive, Dual-Task Machine Learning Pipeline

**EcoFire AutoML** is a production-ready Python framework designed to predict wildfire occurrence and intensity. Unlike traditional models that treat an entire continent as a single dataset, this system automatically discovers micro-climatic regions ("Ecozones") using density-based clustering and trains specialized models for each zone.

-----

## 🚀 Key Features

  * **🌍 Dynamic Ecozone Discovery:** Uses a hybrid **K-Means + DBSCAN** approach to identify distinct geographical clusters based on data density, not arbitrary grid lines.
  * **🤖 Dual-Task Learning:** Simultaneously trains:
      * **Classifier:** Probability of fire occurrence ($P_{fire}$).
      * **Regressor:** Predicted fire intensity (Log-transformed MW).
  * **🧠 Hybrid Model Search:** Automatically selects the best architecture among **LightGBM, XGBoost, CatBoost, RandomForest,** and custom **PyTorch Transformers/LSTMs**.
  * **🎛️ Optuna Integration:** Automated hyperparameter tuning with pruning for maximum efficiency.
  * **🔍 Deep Explainability:** Generates SHAP summary plots, Feature Importance charts, and GIS Spatial Overlay maps for every zone.
  * **📄 Automated Reporting:** Compiles HTML reports for every zone and a global index dashboard.

-----

## 🛠️ Pipeline Architecture

The system follows a strict linear pipeline with parallel execution capabilities:

1.  **Ingestion & Engineering:** Loads raw CSV data and computes derived features (e.g., `vpd_wind_idx`, scaled NDVI).
2.  **Spatial Clustering:**
    \*
      * **Step A:** Coarse clustering via K-Means.
      * **Step B:** Fine-grained density refinement via DBSCAN.
      * **Step C:** A KNN Router is trained to assign future/test data to these zones.
3.  **Zone-wise Training (Parallelized):**
      * Data is split by zone.
      * **AutoML Loop:** Candidate models are warm-started. The best performer is selected based on `PR-AUC` and `CV-Stability`.
      * **Tuning:** If enabled, Optuna refines the best model.
4.  **Evaluation & Export:**
      * Generates Confusion Matrices, ROC Curves, and Calibration plots.
      * Saves models (`.pkl`) and metadata (`.json`).
5.  **Global Reporting:** Aggregates metrics across all zones into a master HTML dashboard.

-----

## 📦 Installation & Requirements

Ensure you have Python 3.8+ and the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn \
            xgboost lightgbm catboost optuna shap geopandas torch
```

-----

## 💻 Command Line Usage

### 1\. Training Mode

Run the full pipeline on your dataset.

```bash
# Basic run with default settings
python ecofire.py --data /path/to/training_data.csv --outdir ./my_results

# Advanced run with Optuna tuning and Parallel processing
python ecofire.py --data train.csv \
    --run_optuna \
    --trials 50 \
    --eps 20 \
    --min_samples 100
```

**Arguments:**

  * `--data`: Path to input CSV.
  * `--outdir`: Directory to save models and reports.
  * `--eps`: DBSCAN epsilon (radius in km) for zone discovery.
  * `--run_optuna`: Enable hyperparameter tuning (slower but better results).
  * `--quantile`: (Optional) Use quantile regression techniques.

### 2\. Prediction Mode

Use the trained "Ecozone Router" to predict on new data.

```bash
python ecofire.py --predict /path/to/test_data.csv --models ./my_results
```

  * The script will automatically route new data points to the correct Ecozone model and generate a `prediction_output.csv`.

-----

## 📂 Code Structure & Function Definitions

The code is organized into modular blocks. Here is a breakdown of the critical functions:

### 1\. Data & Spatial Engineering

  * `load_data(path)`: Loads CSV and drops excluded columns defined in `CFG`.
  * `engineer_features(df)`: Calculates meteorological indices (e.g., Temperature in Celsius, VPD-Wind Index).
  * `extract_ecozones(df)`: **The Core Logic.** Uses K-Means for macro-separation, then applies DBSCAN within those clusters to find dense spatial pockets. Returns the dataframe with an `ecozone` ID.
  * `train_ecozone_knn(...)`: Trains a K-Nearest Neighbors classifier to route *new* coordinates to the clusters discovered during training.

### 2\. Deep Learning (`FireDL`)

  * `FireDL(nn.Module)`: A custom PyTorch module supporting LSTM, GRU, or Transformer Encoder architectures.
  * 
[Image of Transformer neural network architecture]

  * `DLClassifierWrapper`: Wraps the PyTorch model to behave like a Scikit-Learn estimator (methods like `.fit()` and `.predict_proba()`), making it compatible with the AutoML loop.
  * `FocalLoss`: A custom loss function to handle extreme class imbalance (fire events are rare).

### 3\. AutoML & Tuning

  * `run_optuna_tuning(...)`: The optimization engine. It defines search spaces for XGBoost, LightGBM, CatBoost, and PyTorch. It uses `TrialPruned` to stop bad trials early.
  * `evaluate_cv(...)`: Performs Repeated Stratified K-Fold cross-validation to ensure model stability.
  * `get_candidate_classifiers()`: Returns a dictionary of "warm-start" models to test against the data.

### 4\. Training Loop

  * `train_zone(...)`: **The Worker Function.**
    1.  Receives data for a single Zone ID.
    2.  Handles "Safe Zones" (zones with 0 fires) deterministically.
    3.  Applies SMOTE (oversampling) if needed.
    4.  Runs the Model Selection -\> Optuna Tuning -\> Evaluation pipeline.
    5.  Generates SHAP plots and local HTML reports.
  * `train_all_zones(...)`: Orchestrator that manages parallel execution (using `ProcessPoolExecutor`) of `train_zone`.

### 5\. Visualization & Reporting

  * `save_gis_overlay(...)`: Creates a geospatial scatter plot showing fire detections or SHAP drivers.
  * `save_shap_summary(...)`: Generates beeswarm plots to show feature impact.
  * `generate_global_reports(...)`: Aggregates stats from all zones into `index.html` and `performance_summary.csv`.

-----

## 📊 Output Directory Structure

After training,`--outdir` will look like:

```text
ecozone_results/
├── index.html                  # <--- START HERE: Global Dashboard
├── performance_summary.csv     # CSV table of all zone metrics
├── global_confusion_matrix.png # Aggregate confusion matrix
├── models_registry.pkl         # Master dictionary of all zone models
├── ecozone_knn.pkl             # Router for new data
├── log.txt                     # Execution log
├── zone_0/
│   ├── classifier.pkl          # Trained Model
│   ├── report.html             # Zone-specific visual report
│   ├── roc_curve.png           # ROC Plot
│   ├── shap_summary.png        # Explainability plot
│   └── gis_overlay.png         # Map of the zone
├── zone_1/
│   └── ...
└── ...
```

