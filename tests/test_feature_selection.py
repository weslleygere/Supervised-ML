import numpy as np

from src.core.processors.feature_selection import (
    WeightedCorrelationSelector,
)


# =============================================================================
# FEATURE RANKING
# =============================================================================


def test_selector_keeps_features_most_correlated_with_target() -> None:

    x = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 1.0],
        ]
    )

    y = np.array(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    weights = np.ones(
        len(y)
    )

    selector = WeightedCorrelationSelector(
        n_features=1
    )

    transformed = selector.fit_transform(
        x=x,
        y=y,
        sample_weight=weights,
    )

    assert selector.selected_indices_.tolist() == [
        0
    ]

    np.testing.assert_allclose(
        transformed[:, 0],
        x[:, 0],
    )


# =============================================================================
# SAMPLE WEIGHTS
# =============================================================================


def test_sample_weights_can_change_feature_ranking() -> None:

    x = np.array(
        [
            [-2.0, -2.0],
            [-1.0, 2.0],
            [3.0, -2.0],
            [-1.0, -1.0],
        ]
    )

    y = np.array(
        [
            0.0,
            1.0,
            2.0,
            3.0,
        ]
    )

    # -------------------------------------------------------------------------
    # Equal weights
    # -------------------------------------------------------------------------

    equal_selector = WeightedCorrelationSelector(
        n_features=1
    )

    equal_selector.fit(
        x=x,
        y=y,
        sample_weight=np.ones(
            len(y)
        ),
    )

    assert equal_selector.selected_indices_.tolist() == [
        0
    ]

    # -------------------------------------------------------------------------
    # Weighted observations
    # -------------------------------------------------------------------------

    weighted_selector = WeightedCorrelationSelector(
        n_features=1
    )

    weighted_selector.fit(
        x=x,
        y=y,
        sample_weight=np.array(
            [
                1.0,
                1.0,
                1.0,
                20.0,
            ]
        ),
    )

    assert weighted_selector.selected_indices_.tolist() == [
        1
    ]


# =============================================================================
# MULTIPLE FEATURES
# =============================================================================


def test_selector_returns_requested_number_of_features() -> None:

    x = np.array(
        [
            [0.0, 4.0, 1.0],
            [1.0, 3.0, 1.0],
            [2.0, 2.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 0.0, 1.0],
        ]
    )

    y = np.array(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    selector = WeightedCorrelationSelector(
        n_features=2
    )

    transformed = selector.fit_transform(
        x=x,
        y=y,
        sample_weight=np.ones(
            len(y)
        ),
    )

    assert transformed.shape == (
        5,
        2,
    )

    assert selector.selected_indices_.tolist() == [
        0,
        1,
    ]


# =============================================================================
# TRANSFORM
# =============================================================================


def test_transform_uses_features_selected_during_fit() -> None:

    x_train = np.array(
        [
            [0.0, 4.0, 10.0],
            [1.0, 3.0, 10.0],
            [2.0, 2.0, 10.0],
            [3.0, 1.0, 10.0],
            [4.0, 0.0, 10.0],
        ]
    )

    y_train = np.array(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    selector = WeightedCorrelationSelector(
        n_features=1
    )

    selector.fit(
        x=x_train,
        y=y_train,
        sample_weight=np.ones(
            len(y_train)
        ),
    )

    x_test = np.array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
        ]
    )

    transformed = selector.transform(
        x_test
    )

    np.testing.assert_allclose(
        transformed,
        np.array(
            [
                [10.0],
                [11.0],
            ]
        ),
    )