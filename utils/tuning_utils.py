from imports import *
from config.config import CFG

# def get_candidate_classifiers():
#     models = {}

#     models['lgb'] = LGBMClassifier(n_estimators=200, random_state=CFG.random_state)
#     models['xgb'] = xgb.XGBClassifier(
#         n_estimators=200, random_state=CFG.random_state,
#         verbosity=0, use_label_encoder=False
#     )
#     models['cat'] = CatBoostClassifier(iterations=100, random_seed=CFG.random_state, verbose=0)

#     models['rf'] = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=CFG.random_state)
#     # models['et'] = ExtraTreesClassifier(n_estimators=120, n_jobs=-1, random_state=CFG.random_state)

#     # models['svc'] = SVC(probability=True, random_state=CFG.random_state)
#     # models['mlp'] = MLPClassifier(hidden_layer_sizes=(60,), early_stopping=True, random_state=CFG.random_state)

#     print("[i] Classifier candidates:", list(models.keys()))
#     return models
def get_candidate_classifiers():
    print("\n[✔] Base classifier = LightGBM only. Other models will be tried while Optimising using Optuna\n")
    return {
        "lgb": LGBMClassifier(
            n_estimators=200,
            random_state=CFG.random_state
        )
    }

# def get_candidate_regressors():
#     models = {}

#     models['lgb'] = LGBMRegressor(n_estimators=200, random_state=CFG.random_state)
#     models['xgb'] = xgb.XGBRegressor(n_estimators=180, random_state=CFG.random_state)
#     models['cat'] = CatBoostRegressor(iterations=100, random_seed=CFG.random_state, verbose=0)

#     # models['rf']  = RandomForestRegressor(n_estimators=120, random_state=CFG.random_state)
#     # models['et']  = ExtraTreesRegressor(n_estimators=120, random_state=CFG.random_state)
#     # models['svr'] = SVR()
#     # models['mlp'] = MLPRegressor(hidden_layer_sizes=(60,), early_stopping=True, random_state=CFG.random_state)

#     print("[i] Regressor candidates:", list(models.keys()))
#     return models

def get_candidate_regressors():
    print("\n[✔] Base regressor = LightGBM only\n")
    return {
        "lgb": LGBMRegressor(
            n_estimators=200,
            random_state=CFG.random_state
        )
    }


def run_optuna_tuning(X_train, y_train, X_valid, y_valid, is_classifier=True, n_trials=20):
    """
    Improved Optuna tuner with DL support.
    Handles:
    ✔ RF / XGB / LGB / CAT / DL
    ✔ pruning
    ✔ safety recall
    ✔ returns best model + params
    """
    best_score = -999
    best_params = None
    best_model = None
    best_model_obj = None   # <<< DL wrapper stored here

    trial_history = []
    no_gain_count = 0
    patience = max(5, n_trials // 4)

    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    model_options = ["rf", "xgb", "lgb", "cat", "dl"] if is_classifier else ["rf", "xgb", "lgb", "cat"]
    model_options = ["rf", "xgb", "lgb", "cat", "dl"] if is_classifier else ["rf", "xgb", "lgb", "cat"]
    print(f"[OPTUNA] Search candidates = {model_options}")
    print(f"[OPTUNA] DL enabled: {TORCH_AVAILABLE}")

    # ---------------------------------------------------
    def objective(trial):
        nonlocal best_score, best_params, best_model, best_model_obj, no_gain_count

        model_name = trial.suggest_categorical("model", model_options)

        # ===================== DL BRANCH =====================
        if model_name == "dl":
            if not TORCH_AVAILABLE:
                return 1e-6

            params = {
                "model_type": trial.suggest_categorical("dl_type", ["lstm", "gru", "transformer"]),
                "hidden_dim": trial.suggest_int("hidden_dim", 32, 128),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 5e-3),
                "seq_len": trial.suggest_int("seq_len", 7, 30),
                "gamma": trial.suggest_float("gamma", 1.0, 5.0),
                "label_smooth": trial.suggest_float("label_smooth", 0.02, 0.10),
            }

            # ★ HARD BLOCK: skip Transformer when dataset too large
            if params["model_type"] == "transformer" and X_train.shape[0] > 100000:
                print("⚠ Transformer skipped due to large dataset size")
                return 1e-6

            # ----- build sequences -----
            # convert safely regardless of pandas/numpy format
            X_train_np = X_train if isinstance(X_train, np.ndarray) else X_train.values
            X_valid_np = X_valid if isinstance(X_valid, np.ndarray) else X_valid.values
            y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.values
            y_valid_np = y_valid if isinstance(y_valid, np.ndarray) else y_valid.values

            Xs_train = torch.tensor(make_sequence(X_train_np, params["seq_len"]), dtype=torch.float32).to(device)
            Xs_valid = torch.tensor(make_sequence(X_valid_np, params["seq_len"]), dtype=torch.float32).to(device)
            ys_train = torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32).to(device)
            ys_valid  = torch.tensor(y_valid_np.reshape(-1, 1), dtype=torch.float32).to(device)

            smooth = params["label_smooth"]
            ys_train = ys_train * (1 - smooth) + smooth / 2
            ys_valid = ys_valid * (1 - smooth) + smooth / 2

            dl = FireDL(params["model_type"], X_train.shape[1], params["hidden_dim"], params["dropout"]).to(device)
            optimizer = torch.optim.Adam(dl.parameters(), lr=params["lr"])
            criterion = FocalLoss(gamma=params["gamma"])

            best_loss = 1e9
            ep_patience = 3

            for epoch in range(15):
                dl.train()
                preds = dl(Xs_train)
                loss = criterion(preds, ys_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # validation scoring
                dl.eval()
                with torch.no_grad():
                    val_preds = dl(Xs_valid)
                    val_loss = criterion(val_preds, ys_valid).item()

                trial.report(-val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if val_loss < best_loss:
                    best_loss = val_loss
                else:
                    ep_patience -= 1
                    if ep_patience == 0:
                        break

            dl.eval()
            with torch.no_grad():
                val_probs = dl(Xs_valid).cpu().numpy().flatten()

            # ====== FAIRNESS CHECKS ======
            pr_auc = average_precision_score(y_valid, val_probs)

            best_local_rec = max(
                recall_score(y_valid, (val_probs >= t).astype(int))
                for t in np.linspace(0.1, 0.9, 7)
            )
            best_f2 = max(
                fbeta_score(y_valid, (val_probs >= t).astype(int), beta=2)
                for t in np.linspace(0.1, 0.9, 7)
            )
            score = pr_auc

        # ================= TREE MODELS =================
        else:
            if model_name == "lgb":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 2, 12),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                    "random_state": CFG.random_state,
                }
                model = (LGBMClassifier if is_classifier else LGBMRegressor)(**params)

            elif model_name == "xgb":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 2, 12),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
                    "random_state": CFG.random_state,
                }
                model = (xgb.XGBClassifier if is_classifier else xgb.XGBRegressor)(**params)

            elif model_name == "cat":
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 400),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.4),
                    "random_seed": CFG.random_state
                }
                model = (CatBoostClassifier if is_classifier else CatBoostRegressor)(**params, verbose=0)

            else:  # RF
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "random_state": CFG.random_state,
                }
                model = (RandomForestClassifier if is_classifier else RandomForestRegressor)(**params)

            model.fit(X_train, y_train)

            if is_classifier:
                probs = model.predict_proba(X_valid)[:, 1]
                best_local_rec = max(
                    recall_score(y_valid, (probs >= t).astype(int))
                    for t in np.linspace(0.1, 0.9, 7)
                )

                score = average_precision_score(y_valid, probs)
            else:
                score = r2_score(y_valid, model.predict(X_valid))

        # ================= TRACK BEST =================
        trial_history.append((trial.number, model_name, score))

        if score > best_score:
            print(f"  [OPTUNA] Trial {trial.number}: NEW BEST {model_name} {score:.4f}")
            best_score = score
            best_model = model_name
            best_params = params.copy()
            no_gain_count = 0

            if model_name == "dl":
                dl_cpu = copy.deepcopy(dl).to("cpu")
                best_model_obj = DLClassifierWrapper(
                    dl_model=dl_cpu,
                    seq_len=params["seq_len"],
                    device="cpu"
                )
            else:
                best_model_obj = None
        else:
            no_gain_count += 1

        if no_gain_count >= patience:
            raise optuna.TrialPruned()

        return score

    print(f"[OPTUNA] running tuning... patience={patience}")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    if best_model is None:
        return None, None, None, None

    return best_model, best_params, best_score, best_model_obj


