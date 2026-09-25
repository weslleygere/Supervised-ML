"""Versioned conditional Optuna search over configuration AND hyperparameters."""
import optuna

SEARCH_SPACE_VERSION = "hfi_full_training_v1"


def suggest_parameters(trial: optuna.Trial, model, max_components: int = 16) -> dict:
    name = getattr(model, "value", model)
    if name == "RIDGE_REGRESSION":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e5, log=True)}
    if name == "ELASTIC_NET":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 2.0, log=True), "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1.0)}
    if name == "HUBER":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True), "epsilon": trial.suggest_float("epsilon", 1.05, 2.5)}
    if name == "PLS":
        return {"n_components": trial.suggest_int("n_components", 1, min(16, max_components))}
    if name in {"SVR", "KERNEL_RIDGE"}:
        p = {"gamma_multiplier": trial.suggest_float("gamma_multiplier", 1e-2, 1e2, log=True)}
        if name == "SVR":
            p.update(C=trial.suggest_float("C", 0.1, 100.0, log=True), epsilon=trial.suggest_float("epsilon", 1e-4, 0.3, log=True))
        else:
            p["alpha"] = trial.suggest_float("alpha", 1e-5, 100.0, log=True)
        return p
    if name in {"RANDOM_FOREST", "EXTRA_TREES"}:
        return {"n_estimators": 300, "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 10, 20]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12), "max_features": trial.suggest_float("max_features", 0.2, 1.0)}
    if name == "XGBOOST":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 1, 6), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True), "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True)}
    if name == "CATBOOST":
        return {"iterations": trial.suggest_int("iterations", 100, 800, step=100), "depth": trial.suggest_int("depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True), "random_strength": trial.suggest_float("random_strength", 0.0, 2.0)}
    if name == "GAUSSIAN_PROCESS":
        return {"kernel": trial.suggest_categorical("kernel", ["rbf", "matern15", "matern25"]),
                "length_scale": trial.suggest_float("length_scale", 0.1, 100.0, log=True),
                "amplitude": trial.suggest_float("amplitude", 0.01, 10.0, log=True), "noise": trial.suggest_float("noise", 1e-5, 1.0, log=True)}
    raise ValueError(f"No search space for {name}")


def suggest_configuration(trial, model, settings, dimensions, min_train_size):
    feature = trial.suggest_categorical("feature_set", list(settings.feature_sets))
    aggregation = trial.suggest_categorical("aggregation", list(settings.aggregation_strategies))
    scaling = trial.suggest_categorical("scaling", list(settings.feature_scalings))
    # PLS is itself supervised reduction; PCA is deliberately not composed with it.
    reduction = "none" if model == "PLS" else trial.suggest_categorical("reduction", list(settings.reduction_methods))
    config = {"model": model, "feature_set": feature, "aggregation": aggregation, "scaling": scaling, "reduction": reduction}
    total_dim = dimensions[aggregation]["availability"]
    for block in ("indices", "embeddings"):
        config[f"pca_{block}"] = None
        if feature not in (block, "both"):
            continue
        dim = dimensions[aggregation][block]
        if dim == 0:
            raise ValueError(f"No features available for {block}")
        total_dim += dim
        if reduction == "pca":
            feasible = [v for v in settings.pca_component_candidates if v <= min(dim, min_train_size - 1)]
            if not feasible:
                raise optuna.TrialPruned(f"No feasible PCA count for {block}/{aggregation}.")
            # Different block/aggregation dimensions must not redefine one categorical distribution.
            config[f"pca_{block}"] = trial.suggest_categorical(f"pca_{block}_{aggregation}", feasible)
        elif reduction == "pca_variance":
            config[f"pca_{block}"] = trial.suggest_categorical(f"variance_{block}", list(settings.pca_variance_candidates))
    config["model_params"] = suggest_parameters(trial, model, min(total_dim, min_train_size - 1))
    return config
