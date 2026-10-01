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
    selection_indices_candidates: tuple[int, ...] | None = None,
    selection_embeddings_candidates: tuple[int, ...] | None = None,
) -> dict:
    """
    Suggest one complete machine-learning pipeline.

    A single Optuna trial jointly selects:

        feature representation
        acoustic aggregation
        dimensionality reduction
        reduction dimensionality, when applicable
        model family
        model-specific hyperparameters

    The resulting configuration remains fixed across all CV folds and
    repetitions used to evaluate the candidate.
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

    dimensions = feature_dimensions[
        aggregation
    ]

    # =========================================================================
    # PCA
    # =========================================================================

    pca_indices_components = None
    pca_embeddings_components = None

    if reduction == "pca":

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
    # SUPERVISED FEATURE SELECTION
    # =========================================================================

    selection_indices_features = None
    selection_embeddings_features = None

    if reduction == "supervised_selection":

        indices_candidates = (
            selection_indices_candidates
            if selection_indices_candidates is not None
            else pca_indices_candidates
        )

        embeddings_candidates = (
            selection_embeddings_candidates
            if selection_embeddings_candidates is not None
            else pca_embeddings_candidates
        )

        if feature_set in {
            "indices",
            "both",
        }:

            selection_indices_features = (
                _suggest_selected_features(
                    trial=trial,
                    name=(
                        "selection_indices_features"
                        f"__{aggregation}"
                    ),
                    candidates=(
                        indices_candidates
                    ),
                    feature_dimension=(
                        dimensions[
                            "indices"
                        ]
                    ),
                )
            )

        if feature_set in {
            "embeddings",
            "both",
        }:

            selection_embeddings_features = (
                _suggest_selected_features(
                    trial=trial,
                    name=(
                        "selection_embeddings_features"
                        f"__{aggregation}"
                    ),
                    candidates=(
                        embeddings_candidates
                    ),
                    feature_dimension=(
                        dimensions[
                            "embeddings"
                        ]
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

    configuration = {
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

    if reduction == "supervised_selection":

        configuration[
            "selection_indices_features"
        ] = (
            selection_indices_features
        )

        configuration[
            "selection_embeddings_features"
        ] = (
            selection_embeddings_features
        )

    return configuration


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
# SUPERVISED FEATURE-SELECTION SEARCH
# =============================================================================


def _suggest_selected_features(
    trial: optuna.Trial,
    name: str,
    candidates: tuple[int, ...],
    feature_dimension: int,
) -> int:
    """
    Suggest the number of features retained by supervised selection.

    Correlation-based selection is not subject to the PCA rank constraint.
    The selected dimension only needs to be no larger than the original
    feature block.
    """

    if feature_dimension < 1:
        raise optuna.TrialPruned(
            "No features available for supervised-selection "
            f"parameter '{name}'."
        )

    feasible = [
        value
        for value in candidates
        if value <= feature_dimension
    ]

    if not feasible:
        raise optuna.TrialPruned(
            "No feasible supervised-selection dimension for "
            f"'{name}'. Maximum allowed: {feature_dimension}."
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

    Optuna parameter names are namespaced by model family so conditional
    branches never reuse the same parameter name with incompatible
    distributions.
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
        # BAYESIAN RIDGE
        # =====================================================================

        case RegressionModels.BAYESIAN_RIDGE:

            alpha_prior = (
                trial.suggest_float(
                    "bayesian_alpha_prior",
                    1e-8,
                    1e-2,
                    log=True,
                )
            )

            lambda_prior = (
                trial.suggest_float(
                    "bayesian_lambda_prior",
                    1e-8,
                    1e-2,
                    log=True,
                )
            )

            return {
                "alpha_1":
                    alpha_prior,

                "alpha_2":
                    alpha_prior,

                "lambda_1":
                    lambda_prior,

                "lambda_2":
                    lambda_prior,

                "max_iter":
                    1000,
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
        # KERNEL RIDGE
        # =====================================================================

        case RegressionModels.KERNEL_RIDGE:

            return {
                "alpha":
                    trial.suggest_float(
                        "kernel_ridge_alpha",
                        1e-4,
                        1e3,
                        log=True,
                    ),

                "gamma":
                    trial.suggest_float(
                        "kernel_ridge_gamma",
                        1e-4,
                        1.0,
                        log=True,
                    ),

                "kernel":
                    "rbf",
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

                "min_samples_split":
                    trial.suggest_int(
                        "rf_min_samples_split",
                        2,
                        12,
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

                "min_samples_split":
                    trial.suggest_int(
                        "et_min_samples_split",
                        2,
                        12,
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

                "min_samples_split":
                    trial.suggest_int(
                        "gb_min_samples_split",
                        2,
                        12,
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

                "gamma":
                    trial.suggest_float(
                        "xgb_gamma",
                        0.0,
                        5.0,
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
        # LIGHTGBM
        # =====================================================================

        case RegressionModels.LIGHTGBM:

            return {
                "n_estimators":
                    trial.suggest_int(
                        "lgbm_n_estimators",
                        100,
                        800,
                        step=50,
                    ),

                "learning_rate":
                    trial.suggest_float(
                        "lgbm_learning_rate",
                        0.005,
                        0.2,
                        log=True,
                    ),

                "num_leaves":
                    trial.suggest_categorical(
                        "lgbm_num_leaves",
                        [
                            4,
                            8,
                            16,
                            32,
                        ],
                    ),

                "max_depth":
                    trial.suggest_categorical(
                        "lgbm_max_depth",
                        [
                            -1,
                            3,
                            5,
                            7,
                        ],
                    ),

                "min_child_samples":
                    trial.suggest_int(
                        "lgbm_min_child_samples",
                        2,
                        20,
                    ),

                "subsample":
                    trial.suggest_float(
                        "lgbm_subsample",
                        0.5,
                        1.0,
                    ),

                "subsample_freq":
                    1,

                "colsample_bytree":
                    trial.suggest_float(
                        "lgbm_colsample_bytree",
                        0.4,
                        1.0,
                    ),

                "reg_lambda":
                    trial.suggest_float(
                        "lgbm_reg_lambda",
                        1e-3,
                        1e3,
                        log=True,
                    ),

                "reg_alpha":
                    trial.suggest_float(
                        "lgbm_reg_alpha",
                        1e-4,
                        1e2,
                        log=True,
                    ),

                "objective":
                    "regression",
            }

        # =====================================================================
        # CATBOOST
        # =====================================================================

        case RegressionModels.CATBOOST:

            return {
                "iterations":
                    trial.suggest_int(
                        "catboost_iterations",
                        200,
                        1000,
                        step=100,
                    ),

                "learning_rate":
                    trial.suggest_float(
                        "catboost_learning_rate",
                        0.005,
                        0.2,
                        log=True,
                    ),

                "depth":
                    trial.suggest_int(
                        "catboost_depth",
                        2,
                        8,
                    ),

                "l2_leaf_reg":
                    trial.suggest_float(
                        "catboost_l2_leaf_reg",
                        1e-2,
                        1e2,
                        log=True,
                    ),

                "random_strength":
                    trial.suggest_float(
                        "catboost_random_strength",
                        1e-3,
                        10.0,
                        log=True,
                    ),

                "rsm":
                    trial.suggest_float(
                        "catboost_rsm",
                        0.5,
                        1.0,
                    ),

                "loss_function":
                    "RMSE",
            }

        # =====================================================================
        # UNSUPPORTED MODEL
        # =====================================================================

        case _:

            raise ValueError(
                "No Optuna search space defined for "
                f"model: {model}"
            )