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
    Minimal Optuna-like trial used to inspect the proposed search space
    without running an optimization study.
    """

    def __init__(
        self,
        values: dict | None = None,
    ) -> None:

        self.values = values or {}
        self.categorical_choices = {}

    def suggest_categorical(
        self,
        name,
        choices,
    ):

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

        if name in self.values:
            return self.values[
                name
            ]

        return low


# =============================================================================
# COMMON SEARCH SPACE
# =============================================================================


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
        "indices": 10,
        "embeddings": 20,
    },
    "mean_std": {
        "indices": 20,
        "embeddings": 40,
    },
    "hierarchical": {
        "indices": 30,
        "embeddings": 60,
    },
}


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
            "alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            model=(
                RegressionModels
                .RIDGE_REGRESSION
            ),
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
            min_inner_train_size=8,
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
            trial.categorical_choices
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
            "pca_indices_components__mean":
                4,
            "alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            model=(
                RegressionModels
                .RIDGE_REGRESSION
            ),
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
            min_inner_train_size=8,
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


# =============================================================================
# PCA FEASIBILITY
# =============================================================================


def test_pca_candidates_are_limited_by_inner_training_size() -> None:
    """
    PCA candidates must be feasible in every inner-training fold.

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
            "pca_indices_components__mean":
                4,
            "alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            model=(
                RegressionModels
                .RIDGE_REGRESSION
            ),
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
            min_inner_train_size=5,
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
            "indices": 4,
            "embeddings": 20,
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
            "pca_indices_components__mean":
                4,
            "alpha":
                1.0,
        }
    )

    suggest_pipeline_configuration(
        trial=trial,
        model=(
            RegressionModels
            .RIDGE_REGRESSION
        ),
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
        min_inner_train_size=20,
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
        }
    )

    dimensions = {
        "mean": {
            "indices": 3,
            "embeddings": 10,
        }
    }

    with pytest.raises(
        optuna.TrialPruned
    ):
        suggest_pipeline_configuration(
            trial=trial,
            model=(
                RegressionModels
                .RIDGE_REGRESSION
            ),
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
            min_inner_train_size=4,
        )


# =============================================================================
# BOTH FEATURE BLOCKS
# =============================================================================


def test_both_feature_blocks_receive_independent_pca_dimensions() -> None:
    """
    Indices and embeddings must have independent PCA choices.
    """

    trial = DummyTrial(
        {
            "feature_set":
                "both",
            "aggregation":
                "mean_std",
            "reduction":
                "pca",
            "pca_indices_components__mean_std":
                4,
            "pca_embeddings_components__mean_std":
                6,
            "alpha":
                1.0,
        }
    )

    configuration = (
        suggest_pipeline_configuration(
            trial=trial,
            model=(
                RegressionModels
                .RIDGE_REGRESSION
            ),
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
            min_inner_train_size=8,
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
    Different aggregation strategies must use different Optuna parameter
    names because their feasible PCA spaces may differ.
    """

    trial = DummyTrial(
        {
            "feature_set":
                "indices",
            "aggregation":
                "hierarchical",
            "reduction":
                "pca",
            "pca_indices_components__hierarchical":
                4,
            "alpha":
                1.0,
        }
    )

    suggest_pipeline_configuration(
        trial=trial,
        model=(
            RegressionModels
            .RIDGE_REGRESSION
        ),
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
        min_inner_train_size=8,
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
    Optuna search space.
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


def test_gradient_boosting_searches_supported_losses() -> None:

    trial = DummyTrial(
        {
            "loss":
                "huber",
            "huber_alpha":
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
        "loss"
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
