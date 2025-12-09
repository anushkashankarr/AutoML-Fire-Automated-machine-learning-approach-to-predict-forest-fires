from imports import *
from config.config import CFG

def _select_best_classifier(data, args, outdir_zone=None):
    import time
    from sklearn.model_selection import cross_val_predict

    X_train = data["X_train_sm"]
    y_train = data["y_tr_cls_sm"]
    X_test  = data["X_test"]
    y_test  = data["y_te_cls"]

    best_clf, best_name = None, None
    best_selection_score = -1e9
    best_full_pr = -1
    best_params, best_cv_mean, best_cv_std = None, None, None

    warmup_n      = min(4000, len(X_train))
    speed_budget  = 2.0   # seconds

    for name, model in get_candidate_classifiers().items():
        try:
            # ==== Warm training =====
            t0 = time.time()
            model.fit(X_train[:warmup_n], y_train[:warmup_n])
            warm_time = time.time() - t0

            warm_pr = average_precision_score(
                y_test, model.predict_proba(X_test)[:,1]
            )
            if warm_pr < 0.30:
                continue

            # ==== full training =====
            t0 = time.time()
            model.fit(X_train, y_train)
            full_time = time.time() - t0

            full_pr = average_precision_score(
                y_test, model.predict_proba(X_test)[:,1]
            )

            cv_mean, cv_std = evaluate_cv(
                model, X_train, y_train, is_classifier=True, metric="pr_auc"
            )

            # improve stability weighting
            stability_penalty = (cv_std / (cv_mean + 1e-6)) if cv_mean else 1

            # Selection heuristic
            selection = (
                0.55 * full_pr +
                0.30 * (cv_mean or 0) +
                0.10 * (1 - stability_penalty) -
                0.05 * (full_time / speed_budget)
            )

            if selection > best_selection_score:
                best_selection_score = selection
                best_full_pr = full_pr
                best_clf = clone(model).set_params(random_state=CFG.random_state)
                best_name = name
                best_cv_mean = cv_mean
                best_cv_std  = cv_std

        except Exception as e:
            print(f"[!] Candidate {name} failed:", e)

    # ===== fallback =====
    if best_clf is None:
        best_clf = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, random_state=CFG.random_state
        )
        best_clf.fit(X_train, y_train)
        best_full_pr = average_precision_score(
            y_test, best_clf.predict_proba(X_test)[:,1]
        )
        best_name = "fallback_lgb"

    # ===== Optuna refinement =====
    if args and getattr(args, "run_optuna", True):
        tuned = run_optuna_tuning(
            X_train, y_train, X_test, y_test,
            is_classifier=True, n_trials=args.trials
        )
        if tuned:
            tuned_name, tuned_params, tuned_pr = tuned[:3]
            if tuned_pr > best_full_pr:
                best_full_pr = tuned_pr
                best_params = tuned_params

                if tuned_name == "lgb":
                    best_clf = LGBMClassifier(**tuned_params)
                elif tuned_name == "xgb":
                    best_clf = xgb.XGBClassifier(**tuned_params)
                elif tuned_name == "cat":
                    tuned_params.pop("verbose", None)
                    best_clf = CatBoostClassifier(verbose=0, **tuned_params)
                elif tuned_name == "rf":
                    best_clf = RandomForestClassifier(**tuned_params)
                best_clf.fit(X_train, y_train)

    # ===== store out-of-fold preds for stacking / calibration =====
    try:
        oof = cross_val_predict(best_clf, X_train, y_train, cv=5,
                                method="predict_proba")[:,1]
        joblib.dump(oof, os.path.join(outdir_zone, "oof_predictions.pkl"))
    except Exception as e:
        print("[!] Could not save oof:", e)

    return best_clf, best_params, best_full_pr, best_cv_mean, best_cv_std

def _evaluate_classifier(best_clf, data, outdir_zone, zone_id, meta_info):
    """
    Full wildfire-aware classifier evaluation:
    ✔ ROC curve
    ✔ custom threshold optimisation
    ✔ confusion matrix + curve diagnostics
    ✔ JSON metrics export
    """

    X_test  = data["X_test"]
    y_true  = data["y_te_cls"]

    probs = best_clf.predict_proba(X_test)[:, 1]

    clf_metrics = {}

    # ---------------- ROC Curve ----------------
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_score = np.trapz(tpr, fpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve – Zone {zone_id}")

        save_fig(fig, os.path.join(outdir_zone, "roc_curve.png"))
        clf_metrics["auc"] = float(auc_score)

    except Exception as e:
        print("[!] ROC curve failed:", e)

    # ---------------- Custom threshold tuning ----------------
    best_thr = 0.5
    best_score = -1
    scores_curve = []

    pr_auc_test = average_precision_score(y_true, probs)

    for t in np.linspace(0.05, 0.95, 19):

        preds = (probs >= t).astype(int)

        f2 = fbeta_score(y_true, preds, beta=2)
        f1 = fbeta_score(y_true, preds, beta=1)

        weighted = 0.6 * f2 + 0.3 * f1 + 0.1 * pr_auc_test
        scores_curve.append(weighted)

        if weighted > best_score:
            best_score = weighted
            best_thr = t

    save_curve(
        scores_curve,
        f"Classifier Threshold Curve Zone {zone_id}",
        "Threshold Step",
        "Weighted Score (F2/F1/AUC)",
        os.path.join(outdir_zone, "classifier_curve.png")
    )

    # Extract best individual metrics
    best_f2 = max(
        fbeta_score(y_true, (probs >= t).astype(int), beta=2)
        for t in np.linspace(0.05, 0.95, 19)
    )

    best_f1 = max(
        fbeta_score(y_true, (probs >= t).astype(int), beta=1)
        for t in np.linspace(0.05, 0.95, 19)
    )

    # ---------------- Confusion matrix ----------------
    y_pred_bin = (probs >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)

    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix Zone {zone_id}")
        save_fig(fig, os.path.join(outdir_zone, "confusion_matrix.png"))
    except Exception as e:
        print("[!] Confusion matrix save failed:", e)

    # ---------------- Store final metrics ----------------
    clf_metrics = {
        "pr_auc": float(meta_info["pr_auc"]),
        "auc": clf_metrics.get("auc", None),
        "f2": float(best_f2),
        "f1": float(best_f1),
        "accuracy": float(accuracy_score(y_true, y_pred_bin)),
        "precision": float(precision_score(y_true, y_pred_bin)),
        "recall": float(recall_score(y_true, y_pred_bin)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "optuna_params": meta_info["params"],
        "cv_pr_auc_mean": meta_info["cv_mean"],
        "cv_pr_auc_std": meta_info["cv_std"],
        "best_threshold": float(best_thr),
    }

    joblib.dump(best_clf, os.path.join(outdir_zone, "classifier.pkl"))
    json.dump(clf_metrics, open(os.path.join(outdir_zone, "classifier_meta.json"), "w"), indent=2)

    return best_thr, clf_metrics, y_true.tolist(), probs.tolist()

def _train_and_evaluate_regressor(data, args, outdir_zone, zone_id):
    """
    Fire-only regression with risk-aware tuning:
    - minimises catastrophic underprediction
    - leak-safe validation split for tuning
    """

    fire_tr = data["y_tr_cls"] == 1
    fire_te = data["y_te_cls"] == 1

    X_fire = data["X_train"][fire_tr].copy()
    y_fire = data["y_tr_reg"][fire_tr].copy()

    X_test_fire  = data["X_test"][fire_te].copy()
    y_test_fire  = data["y_te_reg"][fire_te].copy()

    if len(X_fire) < 10 or fire_te.sum() < 5:
        json.dump({}, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)
        return None, None

    # rebalancing
    try:
        X_fire, y_fire = apply_smogn_regression(X_fire, y_fire)
        X_fire = X_fire.reset_index(drop=True)
        y_fire = y_fire.reset_index(drop=True)
    except:
        pass

    # ≡≡≡ risk-aware loss ≡≡≡
    def wildfire_loss(true_raw, pred_raw):
        under = np.maximum(true_raw - pred_raw, 0)
        over  = np.maximum(pred_raw - true_raw, 0)

        return np.mean(
            under**2 + 0.1 * over**2
        )

    best_reg = None
    best_loss = 1e30

    # leak-safe split for tuning
    if args and getattr(args, "run_optuna", False):
        try:
            X_tn, X_val, y_tn, y_val = train_test_split(
                X_fire, y_fire,
                test_size=0.25,
                random_state=CFG.random_state
            )

            def objective(trial):
                model_type = trial.suggest_categorical("model", ["lgb", "xgb", "cat", "rf"])

                # param spaces (unchanged)
                if model_type == "lgb":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                        "max_depth": trial.suggest_int("max_depth", -1, 14),
                        "random_state": CFG.random_state,
                    }
                    model = LGBMRegressor(**params)

                elif model_type == "xgb":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "random_state": CFG.random_state,
                    }
                    model = xgb.XGBRegressor(**params)

                elif model_type == "cat":
                    params = {
                        "depth": trial.suggest_int("depth", 4, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                        "loss_function": "RMSE",
                        "random_seed": CFG.random_state,
                    }
                    model = CatBoostRegressor(verbose=0, **params)

                else:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                        "max_depth": trial.suggest_int("max_depth", 6, 14),
                        "max_features": "sqrt",
                        "random_state": CFG.random_state,
                    }
                    model = RandomForestRegressor(**params)

                model.fit(X_tn, y_tn)

                pred_val_log = model.predict(X_val)
                true_val_raw = np.expm1(y_val)
                pred_val_raw = np.expm1(pred_val_log)

                # catastrophic penalty term
                cat_val = ((true_val_raw > 500) & ((true_val_raw - pred_val_raw) > 0.3 * true_val_raw)).sum()

                loss = wildfire_loss(true_val_raw, pred_val_raw) + 10.0 * (cat_val / max(1, len(true_val_raw)))

                return loss

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.trials)

            best_trial = study.best_trial
            tuned_params = {k: v for k, v in best_trial.params.items() if k != "model"}

            model_type = best_trial.params["model"]
            if model_type == "lgb":
                best_reg = LGBMRegressor(**tuned_params)
            elif model_type == "xgb":
                best_reg = xgb.XGBRegressor(**tuned_params)
            elif model_type == "cat":
                tuned_params.pop("verbose", None)
                best_reg = CatBoostRegressor(verbose=0, **tuned_params)
            else:
                best_reg = RandomForestRegressor(**tuned_params)

            best_reg.fit(X_fire, y_fire)

        except Exception as e:
            print("[!] risk-aware Optuna failed:", e)

    # fallback risk search
    if best_reg is None:
        for _, model in get_candidate_regressors().items():
            try:
                model.fit(X_fire, y_fire)
                pred_log = model.predict(X_test_fire)
                loss = wildfire_loss(np.expm1(y_test_fire), np.expm1(pred_log))
                if loss < best_loss:
                    best_loss = loss
                    best_reg = model
            except:
                pass

    if best_reg is None:
        json.dump({}, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)
        return None, None

    joblib.dump(best_reg, os.path.join(outdir_zone, "regressor.pkl"))

    # evaluate
    pred_log = best_reg.predict(X_test_fire)
    true_raw = np.expm1(y_test_fire)
    pred_raw = np.expm1(pred_log)

    r2_log = r2_score(y_test_fire, pred_log)
    mse = mean_squared_error(true_raw, pred_raw)

    under = np.maximum(true_raw - pred_raw, 0)
    cat = ((true_raw > 500) & ((true_raw - pred_raw) > (0.3 * true_raw))).sum()

    metrics = {
        "best_model": best_reg.__class__.__name__,
        "r2_log_space": float(r2_log),
        "mse_raw": float(mse),
        "rmse_raw": float(np.sqrt(mse)),
        "mae_raw": float(mean_absolute_error(true_raw, pred_raw)),
        "asymmetric_underestimation_cost": float(np.mean(under**2)),
        "catastrophic_miss_count": int(cat),
        "catastrophic_rate": cat / max(1, len(true_raw)),
    }

    json.dump(metrics, open(os.path.join(outdir_zone, "regressor_meta.json"), "w"), indent=2)

    return best_reg, metrics

def evaluate_cv(model, X, y, is_classifier=True, metric="pr_auc"):
    """
    Cross-validation evaluation aligned with new wildfire metrics.

    metric options:
        "pr_auc"  -> average precision scoring
        "f2"      -> recall-weighted scoring
        "r2"      -> regression quality
    """

    try:
        cv = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=2, random_state=CFG.random_state
        )

        # Select scorer correctly
        if is_classifier:

            if metric == "pr_auc":
                scorer = "average_precision"

            elif metric == "f2":
                scorer = make_scorer(fbeta_score, beta=2)

            else:
                raise ValueError("Invalid classifier metric for CV")

        else:  # regression
            if metric == "r2":
                scorer = "r2"
            else:
                raise ValueError("Invalid regression metric for CV")

        scores = cross_val_score(model, X, y, scoring=scorer, cv=cv)

        return float(scores.mean()), float(scores.std())

    except Exception as e:
        print("[!] CV evaluation failed:", e)
        return None, None
