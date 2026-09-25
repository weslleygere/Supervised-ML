"""Estimator construction. All families use equal-installation fitting weights."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.svm import SVR


class RelativeKernelRegressor(RegressorMixin, BaseEstimator):
    """Compute bandwidth relative to training-only gamma='scale'."""
    def __init__(self, kind="SVR", gamma_multiplier=1.0, C=1.0, epsilon=0.1, alpha=1.0):
        self.kind = kind
        self.gamma_multiplier = gamma_multiplier
        self.C = C
        self.epsilon = epsilon
        self.alpha = alpha

    def fit(self, x, y):
        variance = float(np.var(x))
        self.gamma_ = self.gamma_multiplier / (x.shape[1] * variance) if variance > 0 else self.gamma_multiplier
        self.estimator_ = (
            SVR(C=self.C, epsilon=self.epsilon, gamma=self.gamma_, max_iter=200000, cache_size=512)
            if self.kind == "SVR" else KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma_)
        )
        self.estimator_.fit(x, y)
        self.fit_status_ = getattr(self.estimator_, "fit_status_", 0)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        return self.estimator_.predict(x)


def make_estimator(name: str, params: dict, seed: int, jobs: int):
    p = dict(params)
    if name == "RIDGE_REGRESSION":
        return Ridge(**p)
    if name == "ELASTIC_NET":
        return ElasticNet(**p, max_iter=100000, random_state=seed)
    if name == "HUBER":
        return HuberRegressor(**p, max_iter=3000)
    if name == "PLS":
        return PLSRegression(**p, scale=False, max_iter=2000)
    if name in {"SVR", "KERNEL_RIDGE"}:
        return RelativeKernelRegressor(kind=name, **p)
    if name in {"RANDOM_FOREST", "EXTRA_TREES"}:
        klass = RandomForestRegressor if name == "RANDOM_FOREST" else ExtraTreesRegressor
        return klass(**p, n_jobs=jobs, random_state=seed)
    if name == "XGBOOST":
        from xgboost import XGBRegressor
        return XGBRegressor(**p, objective="reg:squarederror", tree_method="hist", n_jobs=jobs, random_state=seed)
    if name == "CATBOOST":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**p, loss_function="RMSE", verbose=False, allow_writing_files=False, thread_count=jobs, random_seed=seed)
    if name == "GAUSSIAN_PROCESS":
        kernel_name = p.pop("kernel")
        base = RBF(p["length_scale"]) if kernel_name == "rbf" else Matern(p["length_scale"], nu=1.5 if kernel_name == "matern15" else 2.5)
        kernel = ConstantKernel(p["amplitude"]) * base + WhiteKernel(p["noise"])
        # Optuna optimizes the kernel parameters; do not hide an extra optimizer.
        return GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-8, random_state=seed)
    if name in {"DUMMY_MEAN", "DUMMY_MEDIAN"}:
        return DummyRegressor(strategy="mean" if name == "DUMMY_MEAN" else "median")
    raise ValueError(f"Unknown model {name}")
