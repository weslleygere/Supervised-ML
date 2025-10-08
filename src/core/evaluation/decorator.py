import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections.abc import Iterator
from typing import Callable, TypeVar, Any, TYPE_CHECKING

from functools import wraps

if TYPE_CHECKING:
    from src.core.models.factory import RegressionModels

F = TypeVar("F", bound=Callable[..., Any])

def progress_bar(func: F) -> F:
    """
    Decorator that wraps a model evaluation function to display a progress bar
    with timing information for each fold during cross-validation.

    Parameters
    ----------
    func : Callable
        The function to be wrapped. It must accept the following arguments:
        - self: an object with a `models` attribute (list of RegressionModels enums).
        - regressor: a RegressionModels enum with a `.name` attribute.
        - features: pandas DataFrame with input features.
        - target: pandas DataFrame with target values.
        - fold_iterator: an iterator that yields (train_idx, test_idx) tuples.

    Returns
    -------
    Callable
        A wrapped version of the input function, enhanced with a `tqdm` progress bar.
    """
    @wraps(func)
    def wrapper(
        self,
        regressor: "RegressionModels",
        features: pd.DataFrame,
        target: pd.DataFrame,
        fold_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        desc_width = max(len(m.name) for m in self.models)
        desc = f"{regressor.name:<{desc_width}}"

        fold_list = list(fold_iterator)
        total = len(fold_list)
        start_time = time.time()

        iterator = tqdm(
            fold_list,
            desc=desc,
            total=total,
            unit="fold",
            ncols=80,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
            mininterval=0.01,
        )

        def timed_iterator() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for i, (train_idx, test_idx) in enumerate(iterator, start=1):
                elapsed = time.time() - start_time
                iterator.set_postfix_str(f"{elapsed / i:.2f}s/it")
                yield train_idx, test_idx

        try:
            result = func(self, regressor, features, target, timed_iterator(), *args, **kwargs)
            return result
        except Exception as e:
            iterator.close()
            raise
    return wrapper  # type: ignore