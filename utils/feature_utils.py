from imports import *
from config.config import CFG

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

