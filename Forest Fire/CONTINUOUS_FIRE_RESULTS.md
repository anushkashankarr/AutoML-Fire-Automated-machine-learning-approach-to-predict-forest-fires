# Forest Fire Prediction - Two-Stage Model Results

## Dataset Information

- **Total Samples**: 4,194
- **Features**: 24 (12 original + 12 engineered)
- **Target Variable**: `Fire intensity index` 

## The dataset contains meteorological, vegetation, and topographic attributes that influence fire behaviour, along with an intensity index representing relative fire severity.

## Model Architecture: Two-Stage Approach
Stage 1 – Classification Component
Predicts whether a given sample belongs to a higher-risk category based on the intensity index.

Stage 2 – Regression Component
Predicts the magnitude of fire intensity for intensity-relevant samples.

Combined Output
The final expected intensity is obtained by combining the classifier’s probability with the regressor's intensity estimation.

This design allows the system to separately learn (a) the likelihood of a high-risk event, and (b) the expected intensity when such an event occurs.

### Stage 1: Binary Classification 
**Model**: LightGBM Classifier (or equivalent gradient-boosted tree)
**Purpose**: Identify which samples fall into a higher-risk category using a threshold-based strategy on the intensity index.


#### Performance Metrics
- **AUC-ROC**: 0.8760
- **Average Precision**: 0.8379
- **F1 Score**: 0.7414
- **Accuracy**: 80.2%

**Confusion Matrix**:
```
              Predicted
              Low   High
Actual Low    435    77
Actual High    89   238
```
## Interpretation 
Strong ability to differentiate higher-risk conditions.

High AUC-ROC indicates reliable ranking of risk.

Balanced behavior across both classes, making it suitable for early warning use cases.


### Stage 2: Regression (Fire Intensity)
**Purpose**: Predict fire intensity for fire events

#### Training Configuration
Regression is performed on intensity-relevant samples, helping the model focus on meaningful fire conditions.

A log-transformation (log1p) stabilizes the target distribution and improves learning for higher intensities.

#### Performance Metrics

**Log Scale (log1p transformed)**:
- **MAE**: 0.4587
- **RMSE**: 0.5800
- **R² Score**: 0.2780

**Original Scale**:
- **MAE**: 2.0406
- **RMSE**: 2.6470

## Interpretation 
The regressor captures moderate variation in fire intensity.

Errors increase for extreme events, which is expected given the rarity of high-intensity conditions.

### Combined Two-Stage Model

The final prediction uses both components:

**Combined Prediction** = P(high fire) × Predicted Intensity

#### Combined Performance Metrics
- **MAE**: 2.6284
- **RMSE**: 3.4403

## Interpretation 
The combined model provides conservative estimates, which is generally preferable in risk-sensitive applications.

The two-stage structure offers clearer separation of “risk detection” vs “intensity estimation.”

## Engineered Features

The model uses 24 features including:

1. **Temperature Features**:
   - temp_range (tmax - tmin)
   - temp_avg (average temperature)

2. **Fire Danger Index**:
   - Combines temperature, humidity, and wind speed
   - Mimics Fire Weather Index (FWI) concept

3. **Drought Index**:
   - Based on soil moisture and rainfall

4. **Vegetation Features**:
   - NDVI normalization
   - Vegetation stress indicator

5. **Topographic Features**:
   - Elevation × slope interaction
   - Aspect (cyclical encoding with sin/cos)

## Recommendations

### For Better Performance

1. **Threshold Optimization**:
   Selecting thresholds based on domain-specific risk categories may refine Stage-1 classification.

2. **Hyperparameter Tuning**:
   Using Optuna or grid search for both stages could yield performance gains.

3. **Additional and Spatial Features**:
   Incorporating temporal signals (e.g., seasonality) or neighbouring-cell information may improve accuracy.

4. **Alternative Models**:
   Testing spatial deep-learning architectures (e.g., U-Net) could be explored for gridded data.

## Conclusion

The two-stage model effectively separates fire-risk detection (Stage 1) from intensity estimation (Stage 2), producing a flexible and modular prediction system. This approach demonstrates:

Strong classification performance (AUC-ROC ≈ 0.88)

Reasonable intensity regression accuracy (R² ≈ 0.28)

Useful combined predictions for decision-support applications

The framework is fully reproducible, extendable, and suitable for integration into fire-risk assessment workflows.

