import optuna
import pytest

from src.core.models.factory import RegressionModels
from src.core.models.search_space import (
    suggest_parameters,
    suggest_pipeline_configuration,
)


# =============================================================================
# TEST TRIAL
# =============================================================================


class DummyTrial:
    """
    Minimal Optuna-like trial used to inspect the conditional search space
    without running an optimization study.
    """

    def __init__(
        self,
        values: dict | None = None,
    ) -> None:

        self.values = values or {}

        self.categorical_choices = {}
        self.float_parameters = {}
        self.int_parameters = {}

        self.suggested_names = set()

    def suggest_categorical(
        self,
        name,
        choices,
    ):

        self.suggested_names.add(
            name
        )

        self.categorical_choices[
            name
        ] = tuple(
            choices
        )

        if name in self.values:
            return self.values[
                name
            ]

        return choices[0]

    def suggest_float(
        self,
        name,
        low,
        high,
        **kwargs,
    ):

        self.suggested_names.add(
            name
        )

        self.float_parameters[
            name
        ] = {
            "low":
                low,
            "high":
                high,
            **kwargs,
        }

        if name in self.values:
            return self.values[
                name
            ]

        return low

    def suggest_int(
        self,
        name,
        low,
        high,
        **kwargs,
    ):

        self.suggested_names.add(
            name
        )

        self.int_parameters[
            name
        ] = {
            "low":
                low,
            "high":
                high,
            **kwargs,
        }

        if name in self.values:
            return self.values[
                name
            ]

        return low


# =============================================================================
# COMMON SEARCH SPACE
# =============================================================================


MODELS = list(
    RegressionModels
)

FEATURE_SETS = (
    "indices",
    "embeddings",
    "both",
)

AGGREGATIONS = (
    "mean",
    "mean_std",
    "hierarchical",
    "robust_daily",
    "dawn_profile",
    "dawn_trend",
)

REDUCTIONS = (
    "none",
    "pca",
    "supervised_selection",
)

PCA_INDICES = (
    2,
    4,
    6,
    8,
    10,
)

PCA_EMBEDDINGS = (
    2,
    4,
    6,
    8,
    10,
)

SELECTION_INDICES = (
    2,
    4,
    6,
    8,
    10,
)

SELECTION_EMBEDDINGS = (
    2,
    4,
    8,
    16,
    32,
)

FEATURE_DIMENSIONS = {
    "mean": {
        "indices":
            10,
        "embeddings":
            20,
    },

    "mean_std": {
        "indices":
            20,
        "embeddings":
            40,
    },

    "hierarchical": {
        "indices":
            30,
        "embeddings":
            60,
    },

    "robust_daily": {
        "indices":
            30,
        "embeddings":
            60,
    },

    "dawn_profile": {
        "indices":
            40,
        "embeddings":
            80,
    },

    "dawn_trend": {
        "indices":
            30,
        "embeddings":
            60,
    },
}


# =============================================================================
# COMPLETE PIPELINE MODEL SELECTION
# =============================================================================


def test_model_family_is_part_of_pipeline_search() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "mean",

            "reduction":
                "none",

            "model":
                "SVR",

            "svr_c":
                10.0,

            "svr_epsilon":
                0.1,

            "svr_gamma":
                0.01,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "model"
    ] == RegressionModels.SVR

    assert trial.categorical_choices[
        "model"
    ] == tuple(
        model.name
        for model in MODELS
    )


def test_complete_pipeline_returns_all_required_components() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "embeddings",

            "aggregation":
                "hierarchical",

            "reduction":
                "pca",

            "model":
                "RIDGE_REGRESSION",

            "pca_embeddings_components__hierarchical":
                4,

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert set(
        configuration
    ) == {
        "model",
        "feature_set",
        "aggregation",
        "reduction",
        "pca_indices_components",
        "pca_embeddings_components",
        "model_params",
    }

    assert configuration[
        "feature_set"
    ] == "embeddings"

    assert configuration[
        "aggregation"
    ] == "hierarchical"

    assert configuration[
        "reduction"
    ] == "pca"

    assert configuration[
        "pca_indices_components"
    ] is None

    assert configuration[
        "pca_embeddings_components"
    ] == 4

    assert configuration[
        "model_params"
    ][
        "alpha"
    ] == pytest.approx(
        1.0
    )


# =============================================================================
# NO REDUCTION
# =============================================================================


def test_no_reduction_does_not_propose_reduction_dimensions() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "both",

            "aggregation":
                "hierarchical",

            "reduction":
                "none",

            "model":
                "RIDGE_REGRESSION",

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            selection_indices_candidates=(
                SELECTION_INDICES
            ),
            selection_embeddings_candidates=(
                SELECTION_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "pca_indices_components"
    ] is None

    assert configuration[
        "pca_embeddings_components"
    ] is None

    assert not any(
        name.startswith(
            "pca_"
        )
        or name.startswith(
            "selection_"
        )
        for name in trial.suggested_names
    )


# =============================================================================
# PCA ACTIVE FEATURE BLOCK
# =============================================================================


def test_indices_only_proposes_indices_pca() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "mean",

            "reduction":
                "pca",

            "model":
                "RIDGE_REGRESSION",

            "pca_indices_components__mean":
                4,

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "pca_indices_components"
    ] == 4

    assert configuration[
        "pca_embeddings_components"
    ] is None

    assert (
        "pca_indices_components__mean"
        in trial.categorical_choices
    )

    assert (
        "pca_embeddings_components__mean"
        not in trial.categorical_choices
    )


def test_embeddings_only_proposes_embeddings_pca() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "embeddings",

            "aggregation":
                "mean_std",

            "reduction":
                "pca",

            "model":
                "RIDGE_REGRESSION",

            "pca_embeddings_components__mean_std":
                6,

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "pca_indices_components"
    ] is None

    assert configuration[
        "pca_embeddings_components"
    ] == 6


# =============================================================================
# PCA FEASIBILITY
# =============================================================================


def test_pca_candidates_are_limited_by_cv_training_size() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "mean",

            "reduction":
                "pca",

            "model":
                "RIDGE_REGRESSION",

            "pca_indices_components__mean":
                4,

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                2,
                4,
                6,
                8,
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=5,
        )
    )

    assert trial.categorical_choices[
        "pca_indices_components__mean"
    ] == (
        2,
        4,
    )

    assert configuration[
        "pca_indices_components"
    ] == 4


def test_pca_candidates_are_limited_by_feature_dimension() -> None:

    dimensions = {
        "mean": {
            "indices":
                4,
            "embeddings":
                20,
        }
    }

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "mean",

            "reduction":
                "pca",

            "model":
                "RIDGE_REGRESSION",

            "pca_indices_components__mean":
                4,

            "ridge_alpha":
                1.0,
        }
    )

    suggest_pipeline_configuration(
        trial=trial,
        models=MODELS,
        feature_sets=(
            "indices",
        ),
        aggregations=(
            "mean",
        ),
        reductions=(
            "pca",
        ),
        pca_indices_candidates=(
            2,
            4,
            6,
            8,
        ),
        pca_embeddings_candidates=(
            PCA_EMBEDDINGS
        ),
        feature_dimensions=dimensions,
        min_cv_train_size=20,
    )

    assert trial.categorical_choices[
        "pca_indices_components__mean"
    ] == (
        2,
        4,
    )


def test_no_feasible_pca_dimension_prunes_trial() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "mean",

            "reduction":
                "pca",

            "model":
                "RIDGE_REGRESSION",

            "ridge_alpha":
                1.0,
        }
    )

    dimensions = {
        "mean": {
            "indices":
                3,
            "embeddings":
                10,
        }
    }

    with pytest.raises(
        optuna.TrialPruned
    ):

        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=(
                "indices",
            ),
            aggregations=(
                "mean",
            ),
            reductions=(
                "pca",
            ),
            pca_indices_candidates=(
                6,
                8,
                10,
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=dimensions,
            min_cv_train_size=4,
        )


# =============================================================================
# BOTH PCA FEATURE BLOCKS
# =============================================================================


def test_both_feature_blocks_receive_independent_pca_dimensions() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "both",

            "aggregation":
                "mean_std",

            "reduction":
                "pca",

            "model":
                "RIDGE_REGRESSION",

            "pca_indices_components__mean_std":
                4,

            "pca_embeddings_components__mean_std":
                6,

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "pca_indices_components"
    ] == 4

    assert configuration[
        "pca_embeddings_components"
    ] == 6


# =============================================================================
# SUPERVISED FEATURE SELECTION
# =============================================================================


def test_indices_only_proposes_indices_supervised_selection() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "hierarchical",

            "reduction":
                "supervised_selection",

            "model":
                "RIDGE_REGRESSION",

            "selection_indices_features__hierarchical":
                6,

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            selection_indices_candidates=(
                SELECTION_INDICES
            ),
            selection_embeddings_candidates=(
                SELECTION_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "selection_indices_features"
    ] == 6

    assert configuration[
        "selection_embeddings_features"
    ] is None

    assert configuration[
        "pca_indices_components"
    ] is None

    assert configuration[
        "pca_embeddings_components"
    ] is None

    assert (
        "selection_indices_features__hierarchical"
        in trial.categorical_choices
    )

    assert (
        "selection_embeddings_features__hierarchical"
        not in trial.categorical_choices
    )


def test_both_feature_blocks_receive_independent_selection_dimensions() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "both",

            "aggregation":
                "dawn_profile",

            "reduction":
                "supervised_selection",

            "model":
                "RIDGE_REGRESSION",

            "selection_indices_features__dawn_profile":
                8,

            "selection_embeddings_features__dawn_profile":
                16,

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            selection_indices_candidates=(
                SELECTION_INDICES
            ),
            selection_embeddings_candidates=(
                SELECTION_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "selection_indices_features"
    ] == 8

    assert configuration[
        "selection_embeddings_features"
    ] == 16


def test_selection_candidates_are_limited_by_feature_dimension() -> None:

    dimensions = {
        "mean": {
            "indices":
                5,
            "embeddings":
                20,
        }
    }

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "mean",

            "reduction":
                "supervised_selection",

            "model":
                "RIDGE_REGRESSION",

            "selection_indices_features__mean":
                4,

            "ridge_alpha":
                1.0,
        }
    )

    suggest_pipeline_configuration(
        trial=trial,
        models=MODELS,
        feature_sets=(
            "indices",
        ),
        aggregations=(
            "mean",
        ),
        reductions=(
            "supervised_selection",
        ),
        pca_indices_candidates=(
            PCA_INDICES
        ),
        pca_embeddings_candidates=(
            PCA_EMBEDDINGS
        ),
        selection_indices_candidates=(
            2,
            4,
            6,
            8,
        ),
        feature_dimensions=dimensions,
        min_cv_train_size=20,
    )

    assert trial.categorical_choices[
        "selection_indices_features__mean"
    ] == (
        2,
        4,
    )


def test_selection_is_not_limited_by_cv_training_size() -> None:
    """
    Correlation selection has no PCA rank restriction.

    A selected feature count may therefore exceed n_train - 1 as long as
    enough original features exist.
    """

    dimensions = {
        "mean": {
            "indices":
                20,
            "embeddings":
                20,
        }
    }

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "mean",

            "reduction":
                "supervised_selection",

            "model":
                "RIDGE_REGRESSION",

            "selection_indices_features__mean":
                10,

            "ridge_alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=(
                "indices",
            ),
            aggregations=(
                "mean",
            ),
            reductions=(
                "supervised_selection",
            ),
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            selection_indices_candidates=(
                4,
                8,
                10,
            ),
            feature_dimensions=dimensions,
            min_cv_train_size=5,
        )
    )

    assert trial.categorical_choices[
        "selection_indices_features__mean"
    ] == (
        4,
        8,
        10,
    )

    assert configuration[
        "selection_indices_features"
    ] == 10


def test_no_feasible_selection_dimension_prunes_trial() -> None:

    dimensions = {
        "mean": {
            "indices":
                3,
            "embeddings":
                10,
        }
    }

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "mean",

            "reduction":
                "supervised_selection",

            "model":
                "RIDGE_REGRESSION",

            "ridge_alpha":
                1.0,
        }
    )

    with pytest.raises(
        optuna.TrialPruned
    ):

        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=(
                "indices",
            ),
            aggregations=(
                "mean",
            ),
            reductions=(
                "supervised_selection",
            ),
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            selection_indices_candidates=(
                4,
                6,
                8,
            ),
            feature_dimensions=dimensions,
            min_cv_train_size=5,
        )


# =============================================================================
# AGGREGATION-SPECIFIC REDUCTION PARAMETERS
# =============================================================================


def test_pca_parameter_name_depends_on_aggregation() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "hierarchical",

            "reduction":
                "pca",

            "model":
                "RIDGE_REGRESSION",

            "pca_indices_components__hierarchical":
                4,

            "ridge_alpha":
                1.0,
        }
    )

    suggest_pipeline_configuration(
        trial=trial,
        models=MODELS,
        feature_sets=FEATURE_SETS,
        aggregations=AGGREGATIONS,
        reductions=REDUCTIONS,
        pca_indices_candidates=(
            PCA_INDICES
        ),
        pca_embeddings_candidates=(
            PCA_EMBEDDINGS
        ),
        feature_dimensions=(
            FEATURE_DIMENSIONS
        ),
        min_cv_train_size=8,
    )

    assert (
        "pca_indices_components__hierarchical"
        in trial.categorical_choices
    )

    assert (
        "pca_indices_components__mean"
        not in trial.categorical_choices
    )


def test_selection_parameter_name_depends_on_aggregation() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "indices",

            "aggregation":
                "dawn_trend",

            "reduction":
                "supervised_selection",

            "model":
                "RIDGE_REGRESSION",

            "selection_indices_features__dawn_trend":
                4,

            "ridge_alpha":
                1.0,
        }
    )

    suggest_pipeline_configuration(
        trial=trial,
        models=MODELS,
        feature_sets=FEATURE_SETS,
        aggregations=AGGREGATIONS,
        reductions=REDUCTIONS,
        pca_indices_candidates=(
            PCA_INDICES
        ),
        pca_embeddings_candidates=(
            PCA_EMBEDDINGS
        ),
        selection_indices_candidates=(
            SELECTION_INDICES
        ),
        selection_embeddings_candidates=(
            SELECTION_EMBEDDINGS
        ),
        feature_dimensions=(
            FEATURE_DIMENSIONS
        ),
        min_cv_train_size=8,
    )

    assert (
        "selection_indices_features__dawn_trend"
        in trial.categorical_choices
    )

    assert (
        "selection_indices_features__mean"
        not in trial.categorical_choices
    )


# =============================================================================
# MODEL SEARCH SPACES
# =============================================================================


@pytest.mark.parametrize(
    "model",
    list(
        RegressionModels
    ),
)
def test_every_model_family_has_a_search_space(
    model: RegressionModels,
) -> None:

    trial = DummyTrial()

    params = suggest_parameters(
        trial=trial,
        model=model,
    )

    assert isinstance(
        params,
        dict,
    )

    assert params


# =============================================================================
# EXISTING MODEL PARAMETER NAMESPACING
# =============================================================================


def test_ridge_uses_namespaced_optuna_parameter() -> None:

    trial = DummyTrial(
        {
            "ridge_alpha":
                2.5,
        }
    )

    params = suggest_parameters(
        trial=trial,
        model=(
            RegressionModels
            .RIDGE_REGRESSION
        ),
    )

    assert (
        "ridge_alpha"
        in trial.suggested_names
    )

    assert params[
        "alpha"
    ] == pytest.approx(
        2.5
    )


def test_svr_uses_only_svr_hyperparameters() -> None:

    trial = DummyTrial(
        {
            "svr_c":
                10.0,

            "svr_epsilon":
                0.05,

            "svr_gamma":
                0.01,
        }
    )

    params = suggest_parameters(
        trial=trial,
        model=RegressionModels.SVR,
    )

    assert {
        "svr_c",
        "svr_epsilon",
        "svr_gamma",
    } <= trial.suggested_names

    assert params[
        "C"
    ] == pytest.approx(
        10.0
    )

    assert params[
        "epsilon"
    ] == pytest.approx(
        0.05
    )

    assert params[
        "gamma"
    ] == pytest.approx(
        0.01
    )


def test_gradient_boosting_searches_supported_losses() -> None:

    trial = DummyTrial(
        {
            "gb_loss":
                "huber",

            "gb_huber_alpha":
                0.90,
        }
    )

    params = suggest_parameters(
        trial=trial,
        model=(
            RegressionModels
            .GRADIENT_BOOSTING
        ),
    )

    assert trial.categorical_choices[
        "gb_loss"
    ] == (
        "squared_error",
        "huber",
        "absolute_error",
    )

    assert params[
        "loss"
    ] == "huber"

    assert params[
        "alpha"
    ] == pytest.approx(
        0.90
    )


def test_gradient_boosting_non_huber_does_not_propose_alpha() -> None:

    trial = DummyTrial(
        {
            "gb_loss":
                "squared_error",
        }
    )

    params = suggest_parameters(
        trial=trial,
        model=(
            RegressionModels
            .GRADIENT_BOOSTING
        ),
    )

    assert (
        "gb_huber_alpha"
        not in trial.suggested_names
    )

    assert (
        "alpha"
        not in params
    )


def test_xgboost_uses_xgb_namespaced_parameters() -> None:

    trial = DummyTrial()

    params = suggest_parameters(
        trial=trial,
        model=(
            RegressionModels
            .XGBOOST
        ),
    )

    assert (
        "xgb_max_depth"
        in trial.suggested_names
    )

    assert (
        "xgb_learning_rate"
        in trial.suggested_names
    )

    assert (
        "xgb_gamma"
        in trial.suggested_names
    )

    assert (
        "max_depth"
        in params
    )

    assert (
        "learning_rate"
        in params
    )

    assert (
        "gamma"
        in params
    )


# =============================================================================
# NEW MODEL SEARCH SPACES
# =============================================================================


def test_bayesian_ridge_uses_compact_prior_search() -> None:

    trial = DummyTrial(
        {
            "bayesian_alpha_prior":
                1e-5,

            "bayesian_lambda_prior":
                1e-4,
        }
    )

    params = suggest_parameters(
        trial=trial,
        model=(
            RegressionModels
            .BAYESIAN_RIDGE
        ),
    )

    assert {
        "bayesian_alpha_prior",
        "bayesian_lambda_prior",
    } <= trial.suggested_names

    assert params[
        "alpha_1"
    ] == pytest.approx(
        1e-5
    )

    assert params[
        "alpha_2"
    ] == pytest.approx(
        1e-5
    )

    assert params[
        "lambda_1"
    ] == pytest.approx(
        1e-4
    )

    assert params[
        "lambda_2"
    ] == pytest.approx(
        1e-4
    )


def test_kernel_ridge_uses_namespaced_parameters() -> None:

    trial = DummyTrial(
        {
            "kernel_ridge_alpha":
                0.5,

            "kernel_ridge_gamma":
                0.01,
        }
    )

    params = suggest_parameters(
        trial=trial,
        model=(
            RegressionModels
            .KERNEL_RIDGE
        ),
    )

    assert {
        "kernel_ridge_alpha",
        "kernel_ridge_gamma",
    } <= trial.suggested_names

    assert params[
        "alpha"
    ] == pytest.approx(
        0.5
    )

    assert params[
        "gamma"
    ] == pytest.approx(
        0.01
    )

    assert params[
        "kernel"
    ] == "rbf"


def test_lightgbm_uses_lgbm_namespaced_parameters() -> None:

    trial = DummyTrial()

    params = suggest_parameters(
        trial=trial,
        model=(
            RegressionModels
            .LIGHTGBM
        ),
    )

    assert {
        "lgbm_n_estimators",
        "lgbm_learning_rate",
        "lgbm_num_leaves",
        "lgbm_max_depth",
        "lgbm_min_child_samples",
        "lgbm_subsample",
        "lgbm_colsample_bytree",
        "lgbm_reg_lambda",
        "lgbm_reg_alpha",
    } <= trial.suggested_names

    assert params[
        "objective"
    ] == "regression"

    assert params[
        "subsample_freq"
    ] == 1


def test_catboost_uses_catboost_namespaced_parameters() -> None:

    trial = DummyTrial()

    params = suggest_parameters(
        trial=trial,
        model=(
            RegressionModels
            .CATBOOST
        ),
    )

    assert {
        "catboost_iterations",
        "catboost_learning_rate",
        "catboost_depth",
        "catboost_l2_leaf_reg",
        "catboost_random_strength",
        "catboost_rsm",
    } <= trial.suggested_names

    assert params[
        "loss_function"
    ] == "RMSE"


# =============================================================================
# CONDITIONAL PIPELINE BRANCH
# =============================================================================


def test_pipeline_activates_only_selected_model_branch() -> None:

    trial = DummyTrial(
        {
            "feature_set":
                "embeddings",

            "aggregation":
                "mean",

            "reduction":
                "none",

            "model":
                "SVR",

            "svr_c":
                5.0,

            "svr_epsilon":
                0.05,

            "svr_gamma":
                0.01,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            models=MODELS,
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "model"
    ] == RegressionModels.SVR

    assert {
        "svr_c",
        "svr_epsilon",
        "svr_gamma",
    } <= trial.suggested_names

    forbidden_prefixes = (
        "ridge_",
        "elastic_",
        "bayesian_",
        "kernel_ridge_",
        "rf_",
        "et_",
        "gb_",
        "xgb_",
        "lgbm_",
        "catboost_",
    )

    assert not any(
        name.startswith(
            forbidden_prefixes
        )
        for name in trial.suggested_names
    )


# =============================================================================
# EMPTY MODEL SPACE
# =============================================================================


def test_empty_model_space_is_rejected() -> None:

    trial = DummyTrial()

    with pytest.raises(
        ValueError,
        match="At least one regression model",
    ):

        suggest_pipeline_configuration(
            trial=trial,
            models=[],
            feature_sets=FEATURE_SETS,
            aggregations=AGGREGATIONS,
            reductions=REDUCTIONS,
            pca_indices_candidates=(
                PCA_INDICES
            ),
            pca_embeddings_candidates=(
                PCA_EMBEDDINGS
            ),
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )