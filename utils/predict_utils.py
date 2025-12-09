from imports import *
from config.config import CFG

def ecofire_predict(input_csv, model_dir="ecozone_results"):
    print("[i] Running ECOFIRE prediction…")

    registry, router = load_registry(model_dir)

    df = pd.read_csv(input_csv)
    df = engineer_features(df)

    df = assign_zones(df, router)

    results = []
    for z in sorted(df["ecozone"].unique()):
        df_z = df[df["ecozone"] == z]
        print(f"[i] Zone {z} → {len(df_z)} samples")

        preds = predict_zone(df_z, z, registry)
        preds["ecozone"] = z     # preserve zone
        results.append(preds)

    output = pd.concat(results).sort_index()
    
    final = pd.concat([df, output], axis=1)

    out_file = "prediction_output.csv"
    final.to_csv(out_file, index=False)

    print(f"[i] Prediction complete → saved {out_file}")
    try:
        plot_india_risk(final, risk_col="fire_prob", 
                        save_path=os.path.join(model_dir, "india_fire_risk.png"))
    except Exception as e:
        print("[!] Could not generate India risk map:", e)
    return final

def predict_zone(df_zone, zone_id, registry):
    entry = registry.get(zone_id, {})

    clf = entry.get("clf")
    reg = entry.get("reg")
    thr = entry.get("thr", 0.5)

    # Feature matrix like training
    X = df_zone.drop(columns=[CFG.target_class, CFG.target_reg], errors="ignore")

    n = len(df_zone)

    # --- Classification ---
    if clf is None:
        fire_prob = np.zeros(n, dtype=float)
    else:
        fire_prob = clf.predict_proba(X)[:, 1]

    # --- Regression ---
    if reg is not None:
        intensity_log = reg.predict(X)
        pred_intensity = np.expm1(intensity_log)
    else:
        pred_intensity = np.zeros(n, dtype=float)

    # --- Derived ---
    soft_severity = fire_prob * pred_intensity
    fire_label = (fire_prob >= thr).astype(int)

    return pd.DataFrame({
        "fire_prob": fire_prob,
        "pred_intensity": pred_intensity,
        "soft_severity": soft_severity,
        "fire_label": fire_label,
        "ecozone": zone_id,
    }, index=df_zone.index)

def load_registry(model_dir="ecozone_results"):
    print("[i] Loading registry + router…")
    registry = joblib.load(f"{model_dir}/models_registry.pkl")

    # apply DL safety
    for z, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        clf = entry.get("clf")
        if clf is not None:
            entry["clf"] = prepare_loaded_model(clf)

    router = joblib.load(f"{model_dir}/ecozone_knn.pkl")
    return registry, router
