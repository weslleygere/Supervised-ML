import numpy as np


# =============================================================================
# WEIGHTED SUPERVISED FEATURE SELECTION
# =============================================================================


class WeightedCorrelationSelector:
    """
    Select features using absolute weighted correlation with the target.

    The selector is fitted only on training data. Sample weights allow
    physical Points to contribute equally to feature selection.
    """

    def __init__(
        self,
        n_features: int,
    ) -> None:

        self.n_features = n_features

        self.scores_: np.ndarray | None = None
        self.selected_indices_: np.ndarray | None = None

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "WeightedCorrelationSelector":

        x = np.asarray(
            x,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        ).reshape(-1)

        weights = np.asarray(
            sample_weight,
            dtype=float,
        )

        if self.n_features > x.shape[1]:
            raise ValueError(
                "Requested number of selected features exceeds "
                "the available feature dimension."
            )

        # ---------------------------------------------------------------------
        # Weighted centering
        # ---------------------------------------------------------------------

        x_mean = np.average(
            x,
            axis=0,
            weights=weights,
        )

        y_mean = np.average(
            y,
            weights=weights,
        )

        x_centered = (
            x
            - x_mean
        )

        y_centered = (
            y
            - y_mean
        )

        # ---------------------------------------------------------------------
        # Weighted correlation
        # ---------------------------------------------------------------------

        covariance = np.average(
            x_centered
            * y_centered[:, None],
            axis=0,
            weights=weights,
        )

        x_variance = np.average(
            x_centered**2,
            axis=0,
            weights=weights,
        )

        y_variance = np.average(
            y_centered**2,
            weights=weights,
        )

        denominator = np.sqrt(
            x_variance
            * y_variance
        )

        scores = np.divide(
            np.abs(
                covariance
            ),
            denominator,
            out=np.zeros_like(
                covariance
            ),
            where=(
                denominator
                > 0
            ),
        )

        self.scores_ = scores

        self.selected_indices_ = np.argsort(
            -scores,
            kind="stable",
        )[
            :self.n_features
        ]

        return self

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        if self.selected_indices_ is None:
            raise RuntimeError(
                "Selector must be fitted before transform()."
            )

        return np.asarray(
            x,
            dtype=float,
        )[
            :,
            self.selected_indices_
        ]

    def fit_transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:

        self.fit(
            x=x,
            y=y,
            sample_weight=sample_weight,
        )

        return self.transform(
            x
        )