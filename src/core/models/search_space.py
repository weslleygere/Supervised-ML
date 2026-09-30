import optuna

from .factory import RegressionModels


# =============================================================================
# COMPLETE PIPELINE SEARCH
# =============================================================================


def suggest_pipeline_configuration(
    trial: optuna.Trial,
    models: list[RegressionModels],
    feature_sets: tuple[str, ...],
    aggregations: tuple[str, ...],
    reductions: tuple[str, ...],
    pca_indices_candidates: tuple[int, ...],
    pca_embeddings_candidates: tuple[int, ...],
    feature_dimensions: dict[str, dict[str, int]],
    min_cv_train_size: int,
) -> dict:
    """
    Suggest one complete machine-learning pipeline.

    A single Optuna trial jointly selects:

        model family
        feature representation
        aggregation strategy
        dimensionality reduction
        PCA dimensionality, when applicable
        model-specific hyperparameters

    The resulting configuration remains fixed while the candidate is
    evaluated across all cross-validation folds and repetitions.
    """

    if not models:
        raise ValueError(
            "At least one regression model must be available."
        )

    # =========================================================================
    # GLOBAL PIPELINE CONFIGURATION
    # =========================================================================

    feature_set = trial.suggest_categorical(
        "feature_set",
        list(
            feature_sets
        ),
    )

    aggregation = trial.suggest_categorical(
        "aggregation",
        list(
            aggregations
        ),
    )

    reduction = trial.suggest_categorical(
        "reduction",
        list(
            reductions
        ),
    )

    # =========================================================================
    # MODEL FAMILY
    # =========================================================================

    model_name = trial.suggest_categorical(
        "model",
        [
            model.name
            for model in models
        ],
    )

    model = RegressionModels[
        model_name
    ]

    # =========================================================================
    # PCA
    # =========================================================================

    pca_indices_components = None
    pca_embeddings_components = None

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
                        "pca_indices_components"
                        f"__{aggregation}"
                    ),
                    candidates=(
                        pca_indices_candidates
                    ),
                    feature_dimension=(
                        dimensions[
                            "indices"
                        ]
                    ),
                    min_cv_train_size=(
                        min_cv_train_size
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
                        "pca_embeddings_components"
                        f"__{aggregation}"
                    ),
                    candidates=(
                        pca_embeddings_candidates
                    ),
                    feature_dimension=(
                        dimensions[
                            "embeddings"
                        ]
                    ),
                    min_cv_train_size=(
                        min_cv_train_size
                    ),
                )
            )

    # =========================================================================
    # MODEL-SPECIFIC HYPERPARAMETERS
    # =========================================================================

    model_params = suggest_parameters(
        trial=trial,
        model=model,
    )

    return {
        "model":
            model,

        "feature_set":
            feature_set,

        "aggregation":
            aggregation,

        "reduction":
            reduction,

        "pca_indices_components":
            pca_indices_components,

        "pca_embeddings_components":
            pca_embeddings_components,

        "model_params":
            model_params,
    }


# =============================================================================
# PCA SEARCH
# =============================================================================


def _suggest_pca_components(
    trial: optuna.Trial,
    name: str,
    candidates: tuple[int, ...],
    feature_dimension: int,
    min_cv_train_size: int,
) -> int:
    """
    Suggest a PCA dimensionality feasible in every CV training fold.

    The maximum admissible dimensionality is bounded by:

        original feature dimensionality
        smallest CV training-set size minus one

    The minus one reflects the maximum non-zero rank after centering.
    """

    if feature_dimension < 1:
        raise optuna.TrialPruned(
            f"No features available for PCA parameter '{name}'."
        )

    if min_cv_train_size < 2:
        raise optuna.TrialPruned(
            "At least two training observations are required for PCA."
        )

    max_components = min(
        feature_dimension,
        min_cv_train_size - 1,
    )

    feasible = [
        value
        for value in candidates
        if value <= max_components
    ]

    if not feasible:
        raise optuna.TrialPruned(
            f"No feasible PCA components for '{name}'. "
            f"Maximum allowed: {max_components}."
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
    Suggest model-specific hyperparameters.

    Optuna parameter names are prefixed by model family so that different
    conditional branches of the same study never reuse one parameter name
    with incompatible distributions.

    The returned dictionary uses the parameter names expected by the
    underlying scikit-learn or XGBoost estimator.
    """

    match model:

        # =====================================================================
        # RIDGE REGRESSION
        # =====================================================================

        case RegressionModels.RIDGE_REGRESSION:

            return {
                "alpha":
                    trial.suggest_float(
                        "ridge_alpha",
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
                "alpha":
                    trial.suggest_float(
                        "elastic_alpha",
                        1e-4,
                        10.0,
                        log=True,
                    ),

                "l1_ratio":
                    trial.suggest_float(
                        "elastic_l1_ratio",
                        0.05,
                        1.0,
                    ),

                "max_iter":
                    100000,
            }

        # =====================================================================
        # SUPPORT VECTOR REGRESSION
        # =====================================================================

        case RegressionModels.SVR:

            return {
                "C":
                    trial.suggest_float(
                        "svr_c",
                        1e-2,
                        1e3,
                        log=True,
                    ),

                "epsilon":
                    trial.suggest_float(
                        "svr_epsilon",
                        1e-3,
                        0.5,
                        log=True,
                    ),

                "gamma":
                    trial.suggest_float(
                        "svr_gamma",
                        1e-4,
                        1.0,
                        log=True,
                    ),

                "kernel":
                    "rbf",

                "cache_size":
                    2000,

                "max_iter":
                    100000,
            }

        # =====================================================================
        # RANDOM FOREST
        # =====================================================================

        case RegressionModels.RANDOM_FOREST:

            return {
                "n_estimators":
                    500,

                "max_depth":
                    trial.suggest_categorical(
                        "rf_max_depth",
                        [
                            None,
                            3,
                            5,
                            8,
                            12,
                            20,
                        ],
                    ),

                "min_samples_leaf":
                    trial.suggest_int(
                        "rf_min_samples_leaf",
                        1,
                        12,
                    ),

                "max_features":
                    trial.suggest_float(
                        "rf_max_features",
                        0.2,
                        1.0,
                    ),
            }

        # =====================================================================
        # EXTRA TREES
        # =====================================================================

        case RegressionModels.EXTRA_TREES:

            return {
                "n_estimators":
                    500,

                "max_depth":
                    trial.suggest_categorical(
                        "et_max_depth",
                        [
                            None,
                            3,
                            5,
                            8,
                            12,
                            20,
                        ],
                    ),

                "min_samples_leaf":
                    trial.suggest_int(
                        "et_min_samples_leaf",
                        1,
                        12,
                    ),

                "max_features":
                    trial.suggest_float(
                        "et_max_features",
                        0.2,
                        1.0,
                    ),
            }

        # =====================================================================
        # GRADIENT BOOSTING
        # =====================================================================

        case RegressionModels.GRADIENT_BOOSTING:

            loss = trial.suggest_categorical(
                "gb_loss",
                [
                    "squared_error",
                    "huber",
                    "absolute_error",
                ],
            )

            params = {
                "n_estimators":
                    trial.suggest_int(
                        "gb_n_estimators",
                        100,
                        800,
                        step=50,
                    ),

                "learning_rate":
                    trial.suggest_float(
                        "gb_learning_rate",
                        0.005,
                        0.2,
                        log=True,
                    ),

                "max_depth":
                    trial.suggest_int(
                        "gb_max_depth",
                        1,
                        5,
                    ),

                "min_samples_leaf":
                    trial.suggest_int(
                        "gb_min_samples_leaf",
                        1,
                        15,
                    ),

                "subsample":
                    trial.suggest_float(
                        "gb_subsample",
                        0.5,
                        1.0,
                    ),

                "max_features":
                    trial.suggest_float(
                        "gb_max_features",
                        0.3,
                        1.0,
                    ),

                "loss":
                    loss,
            }

            if loss == "huber":

                params[
                    "alpha"
                ] = trial.suggest_float(
                    "gb_huber_alpha",
                    0.80,
                    0.95,
                )

            return params

        # =====================================================================
        # XGBOOST
        # =====================================================================

        case RegressionModels.XGBOOST:

            return {
                "n_estimators":
                    trial.suggest_int(
                        "xgb_n_estimators",
                        200,
                        1200,
                        step=100,
                    ),

                "max_depth":
                    trial.suggest_int(
                        "xgb_max_depth",
                        1,
                        6,
                    ),

                "learning_rate":
                    trial.suggest_float(
                        "xgb_learning_rate",
                        0.005,
                        0.2,
                        log=True,
                    ),

                "min_child_weight":
                    trial.suggest_float(
                        "xgb_min_child_weight",
                        0.5,
                        20.0,
                        log=True,
                    ),

                "subsample":
                    trial.suggest_float(
                        "xgb_subsample",
                        0.5,
                        1.0,
                    ),

                "colsample_bytree":
                    trial.suggest_float(
                        "xgb_colsample_bytree",
                        0.4,
                        1.0,
                    ),

                "reg_lambda":
                    trial.suggest_float(
                        "xgb_reg_lambda",
                        1e-3,
                        1e3,
                        log=True,
                    ),

                "reg_alpha":
                    trial.suggest_float(
                        "xgb_reg_alpha",
                        1e-4,
                        1e2,
                        log=True,
                    ),

                "objective":
                    "reg:squarederror",

                "tree_method":
                    "hist",
            }

        # =====================================================================
        # UNSUPPORTED MODEL
        # =====================================================================

        case _:

            raise ValueError(
                "No Optuna search space defined for "
                f"model: {model}"
            )
