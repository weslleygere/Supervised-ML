import optuna

from .factory import RegressionModels


# =============================================================================
# HYPERPARAMETER SEARCH
# =============================================================================


def suggest_parameters(
    trial: optuna.Trial,
    model: RegressionModels,
) -> dict:
    """
    Suggest hyperparameters for one Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.

    model : RegressionModels
        Regression algorithm being optimized.

    Returns
    -------
    dict
        Hyperparameters proposed for the model.
    """

    match model:

        # =====================================================================
        # RIDGE
        # =====================================================================

        case RegressionModels.RIDGE_REGRESSION:
            return {
                "alpha": trial.suggest_float(
                    "alpha",
                    1e-4,
                    1e3,
                    log=True,
                ),
            }

        # =====================================================================
        # ELASTIC NET
        # =====================================================================

        case RegressionModels.ELASTIC_NET:
            return {
                "alpha": trial.suggest_float(
                    "alpha",
                    1e-4,
                    1e1,
                    log=True,
                ),
                "l1_ratio": trial.suggest_float(
                    "l1_ratio",
                    0.05,
                    0.95,
                ),
                "max_iter": 10000,
            }

        # =====================================================================
        # SVR
        # =====================================================================

        case RegressionModels.SVR:
            return {
                "C": trial.suggest_float(
                    "C",
                    1e-2,
                    1e2,
                    log=True,
                ),
                "epsilon": trial.suggest_float(
                    "epsilon",
                    1e-2,
                    0.5,
                    log=True,
                ),
                "kernel": "rbf",
                "gamma": "scale",
                "cache_size": 2000,
                "max_iter": 50000,
            }

        # =====================================================================
        # RANDOM FOREST
        # =====================================================================

        case RegressionModels.RANDOM_FOREST:
            return {
                "n_estimators": 300,
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    [
                        None,
                        10,
                        20,
                        30,
                    ],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    10,
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    0.3,
                    1.0,
                ),
            }

        # =====================================================================
        # EXTRA TREES
        # =====================================================================

        case RegressionModels.EXTRA_TREES:
            return {
                "n_estimators": 300,
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    [
                        None,
                        10,
                        20,
                        30,
                    ],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    10,
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    0.3,
                    1.0,
                ),
            }

        # =====================================================================
        # XGBOOST
        # =====================================================================

        case RegressionModels.XGBOOST:
            return {
                "n_estimators": 500,
                "max_depth": trial.suggest_int(
                    "max_depth",
                    2,
                    6,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.15,
                    log=True,
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight",
                    1.0,
                    10.0,
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    0.6,
                    1.0,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.6,
                    1.0,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    1e2,
                    log=True,
                ),
                "objective": "reg:squarederror",
                "tree_method": "hist",
            }

        # =====================================================================
        # UNSUPPORTED MODEL
        # =====================================================================

        case _:
            raise ValueError(
                f"No Optuna search space defined for model: {model}"
            )
