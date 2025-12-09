class CFG:
    target_class = 'fire_detected'
    target_reg = 'fire_intensity'
    drop_cols = ['system:index', 'geometry_json', '.geo', 'date']

    eps = 25
    min_samples = 120
    test_size = 0.2
    random_state = 42

    optuna_trials = 30
    ensemble_n = 3
    min_zone_rows = 200
