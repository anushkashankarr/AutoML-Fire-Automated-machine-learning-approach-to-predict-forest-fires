from imports import *
from config.config import CFG

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

