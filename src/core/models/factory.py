from enum import IntEnum, auto

from .definitions import (
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


# =============================================================================
# REGRESSION MODELS
# =============================================================================


class RegressionModels(IntEnum):
    """
    Regression model families evaluated in the HFI prediction experiment.
    """

    RIDGE_REGRESSION = auto()
    ELASTIC_NET = auto()
    BAYESIAN_RIDGE = auto()

    SVR = auto()
    KERNEL_RIDGE = auto()

    RANDOM_FOREST = auto()
    EXTRA_TREES = auto()

    GRADIENT_BOOSTING = auto()
    XGBOOST = auto()
    LIGHTGBM = auto()
    CATBOOST = auto()


# =============================================================================
# MODEL FACTORY
# =============================================================================


class ModelFactory:
    """
    Factory responsible for creating regression model instances.
    """

    @staticmethod
    def create_model(
        model: RegressionModels,
        params: dict | None = None,
    ) -> AbstractModel:
        """
        Create a regression model using the supplied hyperparameters.
        """

        params = params or {}

        match model:

            # =================================================================
            # REGULARIZED LINEAR MODELS
            # =================================================================

            case RegressionModels.RIDGE_REGRESSION:

                return RidgeRegressionModel(
                    **params
                )

            case RegressionModels.ELASTIC_NET:

                return ElasticNetModel(
                    **params
                )

            case RegressionModels.BAYESIAN_RIDGE:

                return BayesianRidgeModel(
                    **params
                )

            # =================================================================
            # KERNEL MODELS
            # =================================================================

            case RegressionModels.SVR:

                return SVRModel(
                    **params
                )

            case RegressionModels.KERNEL_RIDGE:

                return KernelRidgeModel(
                    **params
                )

            # =================================================================
            # BAGGED TREE ENSEMBLES
            # =================================================================

            case RegressionModels.RANDOM_FOREST:

                return RandomForestModel(
                    **params
                )

            case RegressionModels.EXTRA_TREES:

                return ExtraTreesModel(
                    **params
                )

            # =================================================================
            # BOOSTED TREE ENSEMBLES
            # =================================================================

            case RegressionModels.GRADIENT_BOOSTING:

                return GradientBoostingModel(
                    **params
                )

            case RegressionModels.XGBOOST:

                return XGBoostModel(
                    **params
                )

            case RegressionModels.LIGHTGBM:

                return LightGBMModel(
                    **params
                )

            case RegressionModels.CATBOOST:

                return CatBoostModel(
                    **params
                )

            # =================================================================
            # UNSUPPORTED MODEL
            # =================================================================

            case _:

                raise ValueError(
                    f"Unsupported model: {model}"
                )