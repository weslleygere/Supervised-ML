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
)

REDUCTIONS = (
    "none",
    "pca",
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
}


# =============================================================================
# COMPLETE PIPELINE MODEL SELECTION
# =============================================================================


def test_model_family_is_part_of_pipeline_search() -> None:
    """
    Model family itself must be selected inside the Optuna trial.
    """

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
    """
    Every Optuna trial must describe one complete pipeline candidate.
    """

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
# NO PCA
# =============================================================================


def test_no_reduction_does_not_propose_pca_components() -> None:
    """
    reduction='none' must leave both PCA dimensions unset.
    """

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
            feature_dimensions=(
                FEATURE_DIMENSIONS
            ),
            min_cv_train_size=8,
        )
    )

    assert configuration[
        "reduction"
    ] == "none"

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
        for name in (
            trial.suggested_names
        )
    )


# =============================================================================
# ACTIVE FEATURE BLOCK
# =============================================================================


def test_indices_only_proposes_indices_pca() -> None:
    """
    PCA must only be proposed for feature blocks that are actually used.
    """

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
    """
    Embedding-only pipelines must not propose an index PCA dimension.
    """

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
    """
    PCA candidates must be feasible in every CV training fold.

    With five training observations:

        max components = 5 - 1 = 4
    """

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
    """
    PCA cannot exceed the dimensionality of the active feature block.
    """

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
    """
    An impossible PCA configuration must be pruned rather than silently
    modified.
    """

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
# BOTH FEATURE BLOCKS
# =============================================================================


def test_both_feature_blocks_receive_independent_pca_dimensions() -> None:
    """
    Indices and embeddings must receive independent PCA choices.
    """

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

    assert (
        "pca_indices_components__mean_std"
        in trial.categorical_choices
    )

    assert (
        "pca_embeddings_components__mean_std"
        in trial.categorical_choices
    )


# =============================================================================
# AGGREGATION-SPECIFIC PCA PARAMETERS
# =============================================================================


def test_pca_parameter_name_depends_on_aggregation() -> None:
    """
    Aggregation strategies use different PCA parameter names because their
    original feature dimensionalities differ.
    """

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

    assert (
        "pca_indices_components__mean_std"
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
    """
    Every model exposed by RegressionModels must have a corresponding
    hyperparameter search space.
    """

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
# NAMESPACED MODEL PARAMETERS
# =============================================================================


def test_ridge_uses_namespaced_optuna_parameter() -> None:
    """
    Ridge must use a model-specific Optuna parameter name while returning
    the estimator-compatible parameter name.
    """

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

    assert (
        "alpha"
        not in trial.suggested_names
    )

    assert params[
        "alpha"
    ] == pytest.approx(
        2.5
    )


def test_svr_uses_only_svr_hyperparameters() -> None:
    """
    Selecting SVR must activate the SVR branch only.
    """

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

    assert not any(
        name.startswith(
            "ridge_"
        )
        for name in trial.suggested_names
    )

    assert not any(
        name.startswith(
            "gb_"
        )
        for name in trial.suggested_names
    )

    assert not any(
        name.startswith(
            "xgb_"
        )
        for name in trial.suggested_names
    )

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
    """
    Gradient Boosting must expose the intended loss functions and conditionally
    activate Huber alpha.
    """

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

    assert (
        "gb_huber_alpha"
        in trial.suggested_names
    )


def test_gradient_boosting_non_huber_does_not_propose_alpha() -> None:
    """
    Huber alpha must not exist when another Gradient Boosting loss is used.
    """

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
    """
    XGBoost hyperparameters must not share Optuna names with other tree
    families.
    """

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
        "gb_max_depth"
        not in trial.suggested_names
    )

    assert (
        "rf_max_depth"
        not in trial.suggested_names
    )

    assert (
        "et_max_depth"
        not in trial.suggested_names
    )

    assert (
        "max_depth"
        in params
    )

    assert (
        "learning_rate"
        in params
    )


# =============================================================================
# CONDITIONAL PIPELINE BRANCH
# =============================================================================


def test_pipeline_activates_only_selected_model_branch() -> None:
    """
    A complete pipeline trial must propose hyperparameters only for the
    selected model family.
    """

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
        "rf_",
        "et_",
        "gb_",
        "xgb_",
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
    """
    A pipeline search without candidate model families is invalid.
    """

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
