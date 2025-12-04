# AutoML-ECOFIRE

### RESULTS

> Note:
> The current public release consolidates all core logic (training, modelling, reporting, GIS overlays, explainability, registry management, and prediction) into a **single integrated script** for reproducibility.
> A modular structure (`utils/`, `models/`, `pipelines/`) will be introduced in **v2**.
--

## 📌 Full Results, Trained Models, and Analysis Dashboards

Due to GitHub’s **100MB per-file limit**, trained models, geospatial overlays, reports, and results are hosted externally:

### 🔗 **Download Zone-wise Model Artefacts & Reports**
[https://drive.google.com/drive/folders/1Q8wu_sYGScowKygSP9utcJbm9B4AaXBd?usp=sharing](https://drive.google.com/drive/folders/1Q8wu_sYGScowKygSP9utcJbm9B4AaXBd?usp=sharing)

This archive contains:
* Trained models (.pkl) for every ecozone
* HTML zone assessment reports
* GIS wildfire risk heatmaps
* ROC curves and confusion matrices
* SHAP interpretability insights
* Feature ranking visualisations
* Global performance dashboards
* Registry metadata for inference

--

# 📦 Results & Drive Folder Architecture

The model outputs and analysis reports are organised in the following structure (seen in Drive):

```
ecozone_results/
│
├── zone_0/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── classifier_meta.json
│   ├── regressor_meta.json
│   ├── shap_summary_classifier.png
│   ├── confusion_matrix.png
│   ├── feature_importance_classifier.csv
│   ├── roc_curve.png
│   ├── zone_map.png
│   └── report.html
│
├── zone_1/
│   └── (same structure)
│
├── ...
│
├── zone_11/               # final ecozone
│
├── global/
│   ├── performance_summary.csv
│   ├── global_confusion_matrix.png
│   ├── spatial_performance_map.png
│   ├── meta_correlation_matrix.png
│   └── index.html         # global dashboard
│
├── ecozone_knn.pkl        # ecozone routing model
└── models_registry.pkl    # registry linking zones to estimators
```

This layout reflects a **geographically partitioned ML pipeline**, where models are optimised independently for each ecological zone.

---

---



## 📁 Repository Structure (v1 Release)

```
project/
│
├── automl_adapted.py      # Single consolidated script: training, inference, GIS, reporting & explainability
└── README.md
└── Results
└── Logs
└── GEE Script to get dataset
```

> All pipeline stages and utilities are intentionally packaged together in v1 to simplify replication and review.

---

---

## 🔍 System Objective

The project builds **operational wildfire intelligence** by:

* Detecting fire occurrence
* Estimating intensity
* Learning zone-specific behaviour
* Generating interpretable artefacts for planners
* Visualising risk surfaces spatially

Traditional one-model-fits-all approaches underperform in heterogeneous ecozones; this framework addresses that gap.

---

---

## ⚙️ Pipeline Overview

1. Dataset ingestion & ecozone assignment
2. Zone-wise preprocessing
3. Classifier and regressor selection via AutoML
4. Optuna tuning and metric optimisation
5. SHAP explainability and ranking
6. Zone-level reporting and HTML generation
7. GIS risk heatmaps and overlays
8. Global insight aggregation

---

---

## 🧪 Execution (current version)

Training, explainability, dashboards, reporting, and GIS layers are executed entirely through:

```
automl_adapted.py
```

> Structured modules (`utils/`, `models/`, `geo/`, `reporting/`) are planned for **v2**.

---

## 📌 Why are trained models stored externally?

GitHub enforces a **100MB per-file upper limit**.
Model artefacts, high-resolution maps, and results exceed this limit.

To maintain research accessibility:

✔ Artefacts are hosted on Drive
✔ Repository remains lightweight
✔ Model versions can update without history rewrite

---

---

## 📬 Future Release (v2 Roadmap)

* Separate modules for preprocessing, training, inference, GIS, and reporting
* Updated CLI interface (`main.py`)
* Deployment-ready predictor API wrapper
* Unit-tested reusable utilities
Agar chaho toh main **badge version**, **PDF one-pager**, ya **LinkedIn project summary** bhi bana deta 👍
