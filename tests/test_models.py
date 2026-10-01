import numpy as np
import pandas as pd
import pytest

from src.core.models.definitions import (
    AbstractModel,
    BayesianRidgeModel,
    CatBoostModel,
    ElasticNetModel,
    ExtraTreesModel,
    GradientBoostingModel,
    KernelRidgeModel,
    LightGBMModel,
    RandomForestModel,
    RidgeRegressionModel,
    SVRModel,
    XGBoostModel,
)
from src.core.models.factory import (
    ModelFactory,
    RegressionModels,
)


# =============================================================================
# TEST DATA
# =============================================================================


@pytest.fixture
def regression_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    """
    Small deterministic regression dataset used to verify the common model
    interface.
    """

    x = pd.DataFrame(
        {
            "feature_1": [
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],

            "feature_2": [
                1.0,
                0.0,
                2.0,
                1.0,
                3.0,
                2.0,
                4.0,
                3.0,
            ],

            "feature_3": [
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
            ],
        },
        index=[
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
        ],
    )

    y = pd.DataFrame(
        {
            "meanHFI": [
                10.0,
                14.0,
                18.0,
                23.0,
                29.0,
                34.0,
                41.0,
                47.0,
            ],
        },
        index=x.index,
    )

    sample_weight = np.array(
        [
            0.5,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.5,
            1.5,
        ],
        dtype=float,
    )

    return (
        x,
        y,
        sample_weight,
    )


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================


MODEL_CLASSES = {
    RegressionModels.RIDGE_REGRESSION:
        RidgeRegressionModel,

    RegressionModels.ELASTIC_NET:
        ElasticNetModel,

    RegressionModels.BAYESIAN_RIDGE:
        BayesianRidgeModel,

    RegressionModels.SVR:
        SVRModel,

    RegressionModels.KERNEL_RIDGE:
        KernelRidgeModel,

    RegressionModels.RANDOM_FOREST:
        RandomForestModel,

    RegressionModels.EXTRA_TREES:
        ExtraTreesModel,

    RegressionModels.GRADIENT_BOOSTING:
        GradientBoostingModel,

    RegressionModels.XGBOOST:
        XGBoostModel,

    RegressionModels.LIGHTGBM:
        LightGBMModel,

    RegressionModels.CATBOOST:
        CatBoostModel,
}


MODEL_PARAMS = {
    RegressionModels.RIDGE_REGRESSION: {
        "alpha":
            1.0,
    },

    RegressionModels.ELASTIC_NET: {
        "alpha":
            0.01,

        "l1_ratio":
            0.5,

        "max_iter":
            10000,
    },

    RegressionModels.BAYESIAN_RIDGE: {
        "max_iter":
            100,
    },

    RegressionModels.SVR: {
        "C":
            1.0,

        "epsilon":
            0.1,

        "gamma":
            0.1,

        "max_iter":
            10000,
    },

    RegressionModels.KERNEL_RIDGE: {
        "alpha":
            1.0,

        "gamma":
            0.1,
    },

    RegressionModels.RANDOM_FOREST: {
        "n_estimators":
            5,

        "max_depth":
            3,
    },

    RegressionModels.EXTRA_TREES: {
        "n_estimators":
            5,

        "max_depth":
            3,
    },

    RegressionModels.GRADIENT_BOOSTING: {
        "n_estimators":
            5,

        "max_depth":
            2,
    },

    RegressionModels.XGBOOST: {
        "n_estimators":
            5,

        "max_depth":
            2,

        "learning_rate":
            0.1,

        "objective":
            "reg:squarederror",

        "tree_method":
            "hist",
    },

    RegressionModels.LIGHTGBM: {
        "n_estimators":
            5,

        "learning_rate":
            0.1,

        "num_leaves":
            4,

        "min_child_samples":
            2,
    },

    RegressionModels.CATBOOST: {
        "iterations":
            5,

        "depth":
            2,

        "learning_rate":
            0.1,

        "loss_function":
            "RMSE",
    },
}


# =============================================================================
# FACTORY
# =============================================================================


@pytest.mark.parametrize(
    "model_family",
    list(
        RegressionModels
    ),
)
def test_factory_creates_correct_model_class(
    model_family: RegressionModels,
) -> None:
    """
    Every enum member must map to the intended model wrapper.
    """

    model = ModelFactory.create_model(
        model=model_family,
        params=(
            MODEL_PARAMS[
                model_family
            ]
        ),
    )

    assert isinstance(
        model,
        MODEL_CLASSES[
            model_family
        ],
    )

    assert isinstance(
        model,
        AbstractModel,
    )


# =============================================================================
# COMMON FIT / PREDICT INTERFACE
# =============================================================================


@pytest.mark.parametrize(
    "model_family",
    list(
        RegressionModels
    ),
)
def test_every_model_supports_weighted_fit_and_prediction(
    model_family: RegressionModels,
    regression_data,
) -> None:
    """
    Every candidate model must support the common experiment interface:

        fit(X, y, sample_weight)
        predict(X)

    This verifies that Point-balanced sample weights can be passed through
    every model wrapper used by the CASH search.
    """

    (
        x,
        y,
        sample_weight,
    ) = regression_data

    model = ModelFactory.create_model(
        model=model_family,
        params=(
            MODEL_PARAMS[
                model_family
            ]
        ),
    )

    fit_time = model.fit(
        x,
        y,
        sample_weight=sample_weight,
    )

    (
        predictions,
        prediction_time,
    ) = model.predict(
        x
    )

    assert fit_time >= 0.0

    assert prediction_time >= 0.0

    assert isinstance(
        predictions,
        pd.DataFrame,
    )

    assert predictions.shape == (
        len(
            x
        ),
        1,
    )

    assert predictions.columns.tolist() == [
        "meanHFI",
    ]

    assert predictions.index.equals(
        x.index
    )

    assert np.isfinite(
        predictions.to_numpy()
    ).all()


# =============================================================================
# ENUM / FACTORY COVERAGE
# =============================================================================


def test_factory_mapping_covers_every_regression_model() -> None:
    """
    The explicit factory test mapping must stay synchronized with the enum.
    """

    assert set(
        MODEL_CLASSES
    ) == set(
        RegressionModels
    )

    assert set(
        MODEL_PARAMS
    ) == set(
        RegressionModels
    )