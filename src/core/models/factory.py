from enum import Enum
from .definitions import make_estimator


class RegressionModels(str, Enum):
    RIDGE_REGRESSION = "RIDGE_REGRESSION"
    ELASTIC_NET = "ELASTIC_NET"
    HUBER = "HUBER"
    PLS = "PLS"
    SVR = "SVR"
    KERNEL_RIDGE = "KERNEL_RIDGE"
    GAUSSIAN_PROCESS = "GAUSSIAN_PROCESS"
    RANDOM_FOREST = "RANDOM_FOREST"
    EXTRA_TREES = "EXTRA_TREES"
    XGBOOST = "XGBOOST"
    CATBOOST = "CATBOOST"


class ModelFactory:
    @staticmethod
    def create_model(model, params=None, random_state=42, n_jobs=1):
        name = model.value if isinstance(model, RegressionModels) else str(model)
        return make_estimator(name, params or {}, random_state, n_jobs)
