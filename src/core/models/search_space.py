import optuna

from .factory import RegressionModels


# =============================================================================
# COMPLETE PIPELINE SEARCH
# =============================================================================


def suggest_pipeline_configuration(
    trial: optuna.Trial,
    model: RegressionModels,
    feature_sets: tuple[str, ...],
    aggregations: tuple[str, ...],
    reductions: tuple[str, ...],
    pca_indices_candidates: tuple[int, ...],
    pca_embeddings_candidates: tuple[int, ...],
    feature_dimensions: dict[str, dict[str, int]],
    min_inner_train_size: int,
) -> dict:
    """
    Suggest one complete pipeline configuration for a fixed model family.

    The search includes:

        feature representation
        aggregation strategy
        dimensionality reduction
        PCA dimensionality
        model hyperparameters

    Model family itself is not an Optuna parameter. Each model family is
    optimized in a separate study using the same trial budget.
    """

    feature_set = trial.suggest_categorical(
        "feature_set",
        list(feature_sets),
    )

    aggregation = trial.suggest_categorical(
        "aggregation",
        list(aggregations),
    )

    reduction = trial.suggest_categorical(
        "reduction",
        list(reductions),
    )

    pca_indices_components = None
    pca_embeddings_components = None

    # =========================================================================
    # PCA
    # =========================================================================

    if reduction == "pca":

        dimensions = feature_dimensions[
            aggregation
        ]

        if feature_set in {
            "indices",
            "both",
        }:

            pca_indices_components = (
                _suggest_pca_components(
                    trial=trial,
                    name=(
                        f"pca_indices_components"
                        f"__{aggregation}"
                    ),
                    candidates=pca_indices_candidates,
                    feature_dimension=dimensions[
                        "indices"
                    ],
                    min_inner_train_size=(
                        min_inner_train_size
                    ),
                )
            )

        if feature_set in {
            "embeddings",
            "both",
        }:

            pca_embeddings_components = (
                _suggest_pca_components(
                    trial=trial,
                    name=(
                        f"pca_embeddings_components"
                        f"__{aggregation}"
                    ),
                    candidates=pca_embeddings_candidates,
                    feature_dimension=dimensions[
                        "embeddings"
                    ],
                    min_inner_train_size=(
                        min_inner_train_size
                    ),
                )
            )

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================

    model_params = suggest_parameters(
        trial=trial,
        model=model,
    )

    return {
        "model": model,
        "feature_set": feature_set,
        "aggregation": aggregation,
        "reduction": reduction,
        "pca_indices_components":
            pca_indices_components,
        "pca_embeddings_components":
            pca_embeddings_components,
        "model_params": model_params,
    }


# =============================================================================
# PCA SEARCH
# =============================================================================


def _suggest_pca_components(
    trial: optuna.Trial,
    name: str,
    candidates: tuple[int, ...],
    feature_dimension: int,
    min_inner_train_size: int,
) -> int:
    """
    Suggest a PCA dimensionality feasible in every inner-training fold.

    PCA dimensionality is bounded by:

        number of original features
        smallest inner-training sample size - 1
    """

    max_components = min(
        feature_dimension,
        min_inner_train_size - 1,
    )

    feasible = [
        value
        for value in candidates
        if value <= max_components
    ]

    if not feasible:
        raise optuna.TrialPruned(
            f"No feasible PCA components for {name}."
        )

    return trial.suggest_categorical(
        name,
        feasible,
    )


# =============================================================================
# MODEL HYPERPARAMETER SEARCH
# =============================================================================


def suggest_parameters(
    trial: optuna.Trial,
    model: RegressionModels,
) -> dict:
    """
    Suggest hyperparameters for one regression model family.
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
                    1e4,
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
                    10.0,
                    log=True,
                ),
                "l1_ratio": trial.suggest_float(
                    "l1_ratio",
                    0.05,
                    1.0,
                ),
                "max_iter": 100000,
            }

        # =====================================================================
        # SVR
        # =====================================================================

        case RegressionModels.SVR:

            return {
                "C": trial.suggest_float(
                    "C",
                    1e-2,
                    1e3,
                    log=True,
                ),
                "epsilon": trial.suggest_float(
                    "epsilon",
                    1e-3,
                    0.5,
                    log=True,
                ),
                "gamma": trial.suggest_float(
                    "gamma",
                    1e-4,
                    1.0,
                    log=True,
                ),
                "kernel": "rbf",
                "cache_size": 2000,
                "max_iter": 100000,
            }

        # =====================================================================
        # RANDOM FOREST
        # =====================================================================

        case RegressionModels.RANDOM_FOREST:

            return {
                "n_estimators": 500,
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    [
                        None,
                        3,
                        5,
                        8,
                        12,
                        20,
                    ],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    12,
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    0.2,
                    1.0,
                ),
            }

        # =====================================================================
        # EXTRA TREES
        # =====================================================================

        case RegressionModels.EXTRA_TREES:

            return {
                "n_estimators": 500,
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    [
                        None,
                        3,
                        5,
                        8,
                        12,
                        20,
                    ],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    12,
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    0.2,
                    1.0,
                ),
            }

        # =====================================================================
        # GRADIENT BOOSTING
        # =====================================================================

        case RegressionModels.GRADIENT_BOOSTING:

            loss = trial.suggest_categorical(
                "loss",
                [
                    "squared_error",
                    "huber",
                    "absolute_error",
                ],
            )

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    800,
                    step=50,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.005,
                    0.2,
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    1,
                    5,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    15,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    0.5,
                    1.0,
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    0.3,
                    1.0,
                ),
                "loss": loss,
            }

            if loss == "huber":
                params["alpha"] = (
                    trial.suggest_float(
                        "huber_alpha",
                        0.80,
                        0.95,
                    )
                )

            return params

        # =====================================================================
        # XGBOOST
        # =====================================================================

        case RegressionModels.XGBOOST:

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    200,
                    1200,
                    step=100,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    1,
                    6,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.005,
                    0.2,
                    log=True,
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight",
                    0.5,
                    20.0,
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    0.5,
                    1.0,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.4,
                    1.0,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    1e3,
                    log=True,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    1e-4,
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
