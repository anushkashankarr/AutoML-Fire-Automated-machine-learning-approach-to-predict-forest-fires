from imports import *
from config.config import CFG

def build_zone_tasks(df, args, registry, outdir):
    tasks = []
    zones = sorted([z for z in df["ecozone"].unique() if z != -1])

    for zone in zones:
        df_zone = df[df["ecozone"] == zone]

        if len(df_zone) < args.min_zone_rows:
            print(f"[i] Skipping zone {zone} — too small")
            continue

        if df_zone[CFG.target_class].nunique() < 2:
            print(f"[!] Zone {zone} is SAFE — no variation")

            only_val = df_zone[CFG.target_class].iloc[0]
            tn = len(df_zone) if only_val == 0 else 0
            tp = len(df_zone) if only_val == 1 else 0

            registry[zone] = {
                "clf": None,
                "reg": None,
                "thr": 0.5,
                "metrics_clf": {
                    "pr_auc": 1.0,
                    "f2": 0.0,
                    "accuracy": 1.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "tn": tn,
                    "fp": 0,
                    "fn": 0,
                    "tp": tp,
                },
                "metrics_reg": None,
            }

            out_zone = os.path.join(outdir, f"zone_{zone}")
            os.makedirs(out_zone, exist_ok=True)
            safe_report = {
              "status": "SAFE_ZONE",
              "samples": int(len(df_zone)),
              "message": "no fires observed here — model training skipped"
            }

            json.dump(safe_report, open(os.path.join(out_zone, "classifier_meta.json"), "w"), indent=2)
            json.dump(safe_report, open(os.path.join(out_zone, "regressor_meta.json"), "w"), indent=2)
            generate_zone_report(out_zone, os.path.join(out_zone, "report.html"))
            continue

        out_zone = os.path.join(outdir, f"zone_{zone}")
        tasks.append((df_zone, out_zone, zone, args))

    return tasks

def train_all_zones(tasks, registry, parallel=False):
    """
    Train models for all ecozones.
    
    Args:
        tasks       : list of (df_zone, out_zone, zone_id, args)
        registry    : existing registry dict to update
        parallel    : if True -> use parallel processing
    """

    task_count = len(tasks)
    
    if task_count == 0:
        print("[!] No eligible ecozones to train — exiting training stage.")
        return registry

    # ================================================
    # ---- SEQUENTIAL EXECUTION (default) ----
    # ================================================
    if not parallel:
        print("\n[i] Running SEQUENTIAL ecozone training...")
        for params in tasks:
            zone_id = params[2]
            print(f"[i] ---> Running zone {zone_id} sequentially")
            _, result = train_zone_wrapper(params)
            registry[zone_id] = result

        return registry

    # ================================================
    # ---- PARALLEL EXECUTION ----
    # ================================================
    print(f"\n[i] Launching PARALLEL ecozone training "
          f"on {min(task_count, os.cpu_count())} workers...")

    worker_count = max(1, min(task_count, os.cpu_count()))

    with ProcessPoolExecutor(max_workers=worker_count) as exe:

        futures = {exe.submit(train_zone_wrapper, t): t for t in tasks}

        for future in as_completed(futures):
            try:
                zone_id, result = future.result()
                registry[zone_id] = result
            except Exception as e:
                params = futures[future]
                z = params[2]
                print(f"[!] Zone {z} failed during parallel execution:", e)

    return registry
# -----------------------------
# Train only one leftover zone
# -----------------------------
def train_single_zone(zone_id, args):
    print(f"[▶] Resuming: Training only zone {zone_id}")

    # 1. Load & preprocess
    df = load_data(args.data)
    df = engineer_features(df)
    df, _, knn_router = extract_ecozones(df, eps=args.eps, min_samples=args.min_samples)

    # 2. Create output folder if needed (DON'T DELETE OLD DIRS)
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    zone_dir = os.path.join(outdir, f"zone_{zone_id}")
    os.makedirs(zone_dir, exist_ok=True)

    # 3. Filter only selected zone
    df_zone = df[df["ecozone"] == zone_id]
    if df_zone.empty:
        print(f"❗ Zone {zone_id} not found in dataset.")
        return

    # 4. Train zone
    registry = {}
    model = train_zone(df_zone, zone_dir, zone_id, args)
    registry[zone_id] = model

    # 5. Update registry file
    reg_path = os.path.join(outdir, "models_registry.pkl")
    if os.path.exists(reg_path):                         # load existing registry
        old_reg = joblib.load(reg_path)
        old_reg[zone_id] = model
        registry = old_reg

    joblib.dump(registry, reg_path)

    print(f"[✓] Zone {zone_id} training saved at {zone_dir}")
# -----------------------------
# Resume training for missing zones
# -----------------------------
def resume_leftover_zones(args):

    print("\n[▶] Resume mode: training leftover zones")

    df = load_data(args.data)
    df = engineer_features(df)
    df, _, _ = extract_ecozones(df, eps=args.eps, min_samples=args.min_samples)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    reg_path = os.path.join(outdir, "models_registry.pkl")

    # Load registry safely
    registry = joblib.load(reg_path) if os.path.exists(reg_path) else {}
    
    # Detect trained zones by report existence + registry entry
    trained = set()
    for z in registry.keys():
        if isinstance(z, int):
            zone_report = os.path.join(outdir, f"zone_{z}", "report.html")
            if os.path.exists(zone_report):      # <-- KEY FIX
                trained.add(z)

    print(f"[i] Already trained zones (verified): {trained}")

    available_zones = sorted(df["ecozone"].unique())
    print(f"[i] Available zones: {available_zones}")

    missing = [z for z in available_zones if z not in trained]
    print(f"[i] Leftover zones to train: {missing}")

    if not missing:
        print("✔ Nothing to resume — all zones completed")
        return

    for zone_id in missing:
        print(f"\n[🔸 Training Zone {zone_id}]")

        zone_dir = os.path.join(outdir, f"zone_{zone_id}")
        os.makedirs(zone_dir, exist_ok=True)

        df_zone = df[df["ecozone"] == zone_id]

        model = train_zone(df_zone, zone_dir, zone_id, args)

        registry[zone_id] = model
        joblib.dump(registry, reg_path)

    print("\n[✓] Resume training finished.")

def train_zone_wrapper(params):
    df_zone, out_zone, zone_id, args = params
    result = train_zone(df_zone, out_zone, zone_id, args)
    return zone_id, result

def train_zone(df_zone, outdir_zone, zone_id, args=None, min_samples=50):

    print(f"\n=== 🔥 TRAINING ECOZONE {zone_id} | {len(df_zone)} samples ===")
    
    # Clean only if NOT resuming (handled in main, but good safety here)
    if not (args and getattr(args, "resume", False)):
        clean_output_directory(outdir_zone, delete_subfolders=True)
    else:
        if not os.path.exists(outdir_zone):
            os.makedirs(outdir_zone)


    # --- Safety Screening ---
    is_unsafe, safety_metrics = _check_safe_zone(df_zone, outdir_zone, min_samples=min_samples)
    if is_unsafe:
        return {
            "clf": None, "thr": 0.5, "reg": None,
            "metrics_clf": safety_metrics, "metrics_reg": None
        }

    # --- Preprocessing ---
    # This dictionary 'data' contains the preprocessor needed for the GIS fix
    data = _prepare_data(df_zone)
    print("[i] Train/Test shapes:", data["X_train"].shape, data["X_test"].shape)

    # --- Classifier Selection ---
    best_clf, best_params, best_pr_auc, cv_mean, cv_std = _select_best_classifier(data, args, outdir_zone)

    clf_meta = {
        "pr_auc": best_pr_auc,
        "params": best_params,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }

    # --- Classifier Evaluation ---
    best_thr, clf_metrics, y_true, y_prob = _evaluate_classifier(
        best_clf, data, outdir_zone, zone_id, clf_meta
    )

    # --- Regression ---
    best_reg, reg_metrics = _train_and_evaluate_regressor(
        data, args, outdir_zone, zone_id
    )

    # --- Explainability (The Fixed Function) ---
    _generate_explainability_artifacts(
        df_zone, best_clf, best_reg, data, outdir_zone, zone_id
    )

    # --- Generate Report ---
    generate_zone_report(outdir_zone, os.path.join(outdir_zone, "report.html"))

    return {
        "clf": best_clf,
        "thr": best_thr,
        "reg": best_reg,
        "metrics_clf": clf_metrics,
        "metrics_reg": reg_metrics,
        "y_true": y_true,
        "y_prob": y_prob
    }

def _check_safe_zone(df_zone, outdir_zone, min_samples=50):
    """
    Smart safety screening function.
    Detects empty / micro zones AND performs auto-merge advisory.
    """

    y_cls = df_zone[CFG.target_class].astype(int)

    total_samples = len(df_zone)
    fire_count = int((y_cls == 1).sum())
    if total_samples == 0:
      metrics = {"zone_status": "empty_zone_no_samples"}
      json.dump(metrics, open(os.path.join(outdir_zone, "classifier_meta.json"), "w"), indent=2)
      generate_zone_report(outdir_zone, os.path.join(outdir_zone, "report.html"))
      return True, metrics

    fire_rate = fire_count / total_samples

    drought_mean = float(df_zone.get("drought_index", pd.Series([None])).mean())
    vegetation_dryness = float(df_zone.get("vegetation_dryness", pd.Series([None])).mean())
    population_density = float(df_zone.get("population_density", pd.Series([None])).mean())

    # ================================
    # CASE A: Mono-class
    # ================================
    if y_cls.nunique() < 2:

        tn = total_samples if y_cls.iloc[0] == 0 else 0
        tp = total_samples if y_cls.iloc[0] == 1 else 0

        metrics = {
            "zone_status": "no_fire_variation_detected",
            "historical_fire_probability": fire_rate,
            "drought_index_mean": drought_mean,
            "vegetation_dryness_mean": vegetation_dryness,
            "population_density_mean": population_density,
            "recommendation": (
                "Zone lacks variability. Merge into ecologically nearest region."
            ),
            "tn": tn, "fp": 0, "fn": 0, "tp": tp,
            "pr_auc": 1.0, "f2": 0.0, "accuracy": 1.0,
            "precision": 0.0, "recall": 0.0,
            "cv_pr_auc_mean": None, "cv_pr_auc_std": None
        }

        json.dump(metrics, open(os.path.join(outdir_zone, "classifier_meta.json"), "w"), indent=2)
        json.dump({}, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)
        generate_zone_report(outdir_zone, os.path.join(outdir_zone, "report.html"))
        return True, metrics

    # ================================
    # CASE B: Too few samples
    # ================================
    if total_samples < min_samples:

        metrics = {
            "zone_status": "micro_zone_insufficient_samples",
            "total_samples": total_samples,
            "historical_fire_probability": fire_rate,
            "drought_index_mean": drought_mean,
            "vegetation_dryness_mean": vegetation_dryness,
            "population_density_mean": population_density,
            "recommendation": (
                "Merge this zone with nearest ecological match "
                "(similar dryness + population)."
            ),
        }

        json.dump(metrics, open(os.path.join(outdir_zone, "classifier_meta.json"), "w"), indent=2)
        json.dump({}, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)
        generate_zone_report(outdir_zone, os.path.join(outdir_zone, "report.html"))
        return True, metrics

    return False, None



