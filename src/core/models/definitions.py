import time
import torch
import pandas as pd
from typing import Optional
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class AbstractModel:
    """
    Abstract base class for all model implementations.

    Attributes
    ----------
    seed : int
        Global random seed for reproducibility, set externally via configuration.
    device : str
        Indicates 'cuda' if a compatible GPU is detected by PyTorch, otherwise 'cpu'.
    """
    seed: Optional[int] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self) -> None:
        pass

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Fit the model to the training data.

        Parameters
        ----------
            x : pd.DataFrame
                The input features for training.
            y : pd.DataFrame
                The target variable for training.

        Returns
        -------
            float
                The time taken to fit the model.
        """
        start = time.perf_counter()
        self._fit(x, y)
        return time.perf_counter() - start

    def predict(self, x: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """
        Generate predictions using the fitted model.
        
        Parameters
        ----------
            x : pd.DataFrame
                The input features for prediction.
        Returns
        -------
        tuple[pd.DataFrame, float]
            The predicted values and the time taken to generate the predictions.
        """
        start = time.perf_counter()
        preds = self._predict(x)
        return preds, time.perf_counter() - start

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the training data.

        Parameters
        ----------
            x : pd.DataFrame
                The input features for training.
            y : pd.DataFrame
                The target variable for training.
        """
        raise NotImplementedError("Subclasses must implement the _fit method.")

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted model.

        Parameters
        ----------
            x : pd.DataFrame
                The input features for prediction.

        Returns
        -------
            pd.DataFrame
                The predicted values.
        """
        raise NotImplementedError("Subclasses must implement the _predict method.")

    @property
    def name(self):
        return self.__class__.__name__.replace("Model", "").replace("_", " ").title()
    
# ======================================================
#                    Linear Models
# ======================================================

class LinearRegressionModel(AbstractModel):
    """Linear Regression Model."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(LinearRegression())

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x))  # type: ignore


class RidgeRegressionModel(AbstractModel):
    """Ridge Regression Model."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(Ridge(random_state=AbstractModel.seed))

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore


class LassoRegressionModel(AbstractModel):
    """Lasso Regression Model."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(Lasso(random_state=AbstractModel.seed))

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore


class ElasticNetModel(AbstractModel):
    """ElasticNet Regression Model."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(ElasticNet(random_state=AbstractModel.seed))

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore
    

class PLSRegressorModel(AbstractModel):
    """Partial Least Squares Regressor model."""
    def __init__(self) -> None:
        super().__init__()
        self.model = PLSRegression(n_components=2)

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x))
    
# ======================================================
#                Non-Parametric Models
# ======================================================

class KNeighborsModel(AbstractModel):
    """K-Nearest Neighbors Regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(KNeighborsRegressor())

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore
    
# ======================================================
#                Ensemble Models
# ======================================================
    
class DecisionTreeModel(AbstractModel):
    """Decision Tree Regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(DecisionTreeRegressor(random_state=AbstractModel.seed))

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore
    

class RandomForestModel(AbstractModel):
    """Random Forest Regressor model."""
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestRegressor(random_state=AbstractModel.seed)

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x))
    

class ExtraTreesModel(AbstractModel):
    """Extra Trees Regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(ExtraTreesRegressor(random_state=AbstractModel.seed))

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore


class AdaBoostModel(AbstractModel):
    """AdaBoost Regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(AdaBoostRegressor(random_state=AbstractModel.seed))

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore


class XGBoostModel(AbstractModel):
    """XGBoost Regressor model."""
    def __init__(self) -> None:
        super().__init__()
        self.model = XGBRegressor(random_state=AbstractModel.seed)

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x))
    

class LightGBMModel(AbstractModel):
    """LightGBM Regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(LGBMRegressor(random_state=AbstractModel.seed, verbosity=-1)) # type: ignore

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore


class CatBoostModel(AbstractModel):
    """CatBoost Regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(CatBoostRegressor(verbose=0, random_state=AbstractModel.seed)) # type: ignore

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore

# ======================================================
#                Support Vector Models
# ======================================================

class SVRModel(AbstractModel):
    """Support Vector Regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(SVR())

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore
    
# ======================================================
#                Neural Network Models
# ======================================================

class MLPModel(AbstractModel):
    """Multilayer Perceptron Regressor."""
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(MLPRegressor(random_state=AbstractModel.seed, 
                                                       hidden_layer_sizes=(8, 6),
                                                       activation='tanh',
                                                       solver='adam',
                                                       max_iter=1000))

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(x)) # type: ignore