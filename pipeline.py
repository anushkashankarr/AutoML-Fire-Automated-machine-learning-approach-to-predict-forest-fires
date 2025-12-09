from imports import *
from config.config import CFG
from utils import *
# from utils.cluster_utils import extract_ecozones
# from utils.feature_utils import engineer_features
# from utils.io_utils import clean_output_directory, load_data
# from utils.training_scheduler import build_zone_tasks
# from utils.cluster_utils import train_ecozone_knn

def main(args):
    print("[i] Starting pipeline…")

    # 1. Load dataset
    df = load_data(args.data)
    df = engineer_features(df)

    # 2. Ecozone clustering
    df, _, knn_router = extract_ecozones(df, eps=args.eps, min_samples=args.min_samples)

    outdir   = args.outdir
    reg_path = os.path.join(outdir, "models_registry.pkl")

    # === RESUME / FRESH MODE SIMPLIFIED ===
    if args.resume:
        print("\n[i] 🟡 RESUME MODE — Skipping zones with existing report.html")

        # Load existing registry or make new one
        if os.path.exists(reg_path):
            registry = joblib.load(reg_path)
        else:
            registry = {}

        completed_zones = set()

        # Detect completed zones *only* using report existence
        for name in os.listdir(outdir):
            if name.startswith("zone_"):
                try:
                    z = int(name.split("_")[1])
                except:
                    continue

                report = os.path.join(outdir, name, "report.html")
                if os.path.exists(report):
                    completed_zones.add(z)

        print(f"[i] Detected already completed zones → {sorted(completed_zones)}")

    else:
        print("\n[i] 🟢 FRESH START — wiping output directory")
        clean_output_directory(outdir, delete_subfolders=True)

        registry = {}
        completed_zones = set()

    # 3. Build task list
    all_tasks = build_zone_tasks(df, args, registry, outdir)

    tasks_to_run = [t for t in all_tasks if t[2] not in completed_zones]

    print(f"[i] Total zones: {len(all_tasks)}")
    print(f"[i] Completed via report.html: {len(completed_zones)}")
    print(f"[i] Remaining to train: {len(tasks_to_run)}")

    # 4. Train remaining ones
    if tasks_to_run:
        registry = train_all_zones(tasks_to_run, registry, parallel=False)
    else:
        print("[✓] Resume mode: nothing to retrain")

    # 5. Router + Save + Reports
    print("[i] Updating router + registry + global reports")

    train_ecozone_knn(df, outdir)
    joblib.dump(registry, reg_path)

    generate_global_reports(registry, outdir)
    plot_india_zones(df, os.path.join(outdir, "india_ecozones.png"))

    print("[✓] Pipeline completed successfully.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ecozone-aware AutoML Wildfire Detection")

    # Training mode arguments
    parser.add_argument("--data", help="Path to training CSV file")
    parser.add_argument("--outdir", default="ecozone_results", help="Output directory")
    
    # Logic flags
    parser.add_argument("--resume", action="store_true", help="If set, skip directory clean and train only missing zones.")
    parser.add_argument("--predict", help="Path to CSV for prediction (inference mode)")
    parser.add_argument("--models", default="ecozone_results", help="Model directory for prediction")

    # Hyperparameters
    parser.add_argument("--eps", type=float, default=CFG.eps)
    parser.add_argument("--min_samples", type=int, default=CFG.min_samples)
    parser.add_argument("--min_zone_rows", type=int, default=CFG.min_zone_rows)
    parser.add_argument("--trials", type=int, default=CFG.optuna_trials)
    parser.add_argument("--ensemble_n", type=int, default=CFG.ensemble_n)
    parser.add_argument("--run_optuna", action="store_true")

    # Parse
    args = parser.parse_args()

    # Logging setup
    log_path = "log.txt"
    # Only clear log if NOT resuming
    if not args.resume:
        open(log_path, 'w').close() 

    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    # --- EXECUTION FLOW ---
    if args.predict:
        # Inference Mode
        ecofire_predict(args.predict, args.models)

    elif args.data:
        # Training Mode (Fresh or Resume handled inside main)
        main(args)

    else:
        parser.print_help()
        print("\n[!] Error: You must provide --data (for training) or --predict (for inference).")