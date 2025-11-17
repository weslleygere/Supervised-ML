import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from collections.abc import Iterator
from sklearn.model_selection import KFold
from src.core.processors.postsplit import PostSplitProcessor
from src.core.models.factory import ModelFactory, RegressionModels
from src.core.data.schema import Schema, FeatureSchema, TargetSchema

from dataclasses import dataclass
from .decorator import progress_bar
from .plots import plot_scores, plot_times
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

logger = logging.getLogger(__name__)

@dataclass
class StatsMetrics:
    """
    Container for statistical metrics of a single target variable.
    
    Attributes
    ----------
    r2 : float
        R² (coefficient of determination) score.
    rmse : float
        Root Mean Squared Error.
    mae : float
        Mean Absolute Error.
    """
    r2: float
    rmse: float
    mae: float
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert metrics to dictionary format.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with metric names as keys and values as floats.
        """
        return {
            'R2': self.r2,
            'RMSE': self.rmse,
            'MAE': self.mae
        }


@dataclass
class FoldMetrics:
    """
    Container for metrics from a single cross-validation fold.
    
    Attributes
    ----------
    stats_by_target : Dict[str, StatsMetrics]
        Statistical metrics for each target. Format: {target_name: StatsMetrics}
    fit_time : float
        Time taken to fit the model on this fold.
    pred_time : float
        Time taken to generate predictions on this fold.
    """
    stats_by_target: Dict[str, StatsMetrics]
    fit_time: float
    pred_time: float
    
    @classmethod
    def create(cls, 
               target_names: List[str],
               r2_scores: List[float],
               rmse_scores: List[float], 
               mae_scores: List[float],
               fit_time: float,
               pred_time: float) -> 'FoldMetrics':
        """
        Create a FoldMetrics instance from individual metric arrays.
        
        Parameters
        ----------
        target_names : List[str]
            Names of the target variables.
        r2_scores : List[float]
            R² scores for each target.
        rmse_scores : List[float]
            RMSE scores for each target.
        mae_scores : List[float]
            MAE scores for each target.
        fit_time : float
            Time taken to fit the model on this fold.
        pred_time : float
            Time taken to generate predictions on this fold.
            
        Returns
        -------
        FoldMetrics
            New FoldMetrics instance.
        """
        stats_by_target = {}
        for target_name, r2, rmse, mae in zip(target_names, r2_scores, rmse_scores, mae_scores):
            stats_by_target[target_name] = StatsMetrics(r2=r2, rmse=rmse, mae=mae)
        
        return cls(
            stats_by_target=stats_by_target,
            fit_time=fit_time,
            pred_time=pred_time
        )


@dataclass
class ModelMetrics:
    """
    Container for all cross-validation results of a single model.
    
    Parameters
    ----------
    model : RegressionModels
        The model that was evaluated.
        
    Attributes
    ----------
    fold_metrics : List[FoldMetrics]
        List of FoldMetrics objects, one for each fold.
    """
    model: RegressionModels
    
    def __post_init__(self):
        """Initialize internal storage after object creation."""
        self.fold_metrics: List[FoldMetrics] = []
    
    def add_fold_result(self, 
                       target_names: List[str],
                       r2_scores: List[float],
                       rmse_scores: List[float], 
                       mae_scores: List[float],
                       fit_time: float,
                       pred_time: float) -> None:
        """
        Add results from a single fold.
        
        Parameters
        ----------
        target_names : List[str]
            Names of the target variables.
        r2_scores : List[float]
            R² scores for each target.
        rmse_scores : List[float]
            RMSE scores for each target.
        mae_scores : List[float]
            MAE scores for each target.
        fit_time : float
            Time taken to fit the model on this fold.
        pred_time : float
            Time taken to generate predictions on this fold.
        """
        fold_metrics = FoldMetrics.create(
            target_names=target_names,
            r2_scores=r2_scores,
            rmse_scores=rmse_scores,
            mae_scores=mae_scores,
            fit_time=fit_time,
            pred_time=pred_time
        )
        self.fold_metrics.append(fold_metrics)


@dataclass
class ModelResults:
    """
    Container for all evaluation results across multiple models.

    Attributes
    ----------
    model_metrics : List[ModelMetrics]
        List of ModelMetrics objects, one per evaluated model.
    """
    
    def __post_init__(self):
        """Initialize internal storage after object creation."""
        self.model_metrics: List[ModelMetrics] = []
    
    def add_model_result(self, model_metrics: ModelMetrics) -> None:
        """
        Add evaluation results for a model.
        
        Parameters
        ----------
        model_metrics : ModelMetrics
            The ModelMetrics object to add.
        """
        self.model_metrics.append(model_metrics)
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get statistical summary of all model results.
        
        Returns
        -------
        pd.DataFrame
            Summary DataFrame with mean and std of metrics for each model.
        """
        long_df = self._to_long_format().drop(columns=["fold"])
        summary_df = long_df.groupby("model").agg(["mean", "std"]).round(4)
        return summary_df
    
    def _to_long_format(self) -> pd.DataFrame:
        """
        Convert model metrics to long-format DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Long-format DataFrame containing model metrics.
        """
        rows = []
        for model_metrics in self.model_metrics:
            n_folds = len(model_metrics.fold_metrics)
            for fold_idx in range(n_folds):
                fold_metrics = model_metrics.fold_metrics[fold_idx]
                row = {
                    "model": model_metrics.model.name,
                    "fold": fold_idx + 1,
                    "fit_time": fold_metrics.fit_time,
                    "pred_time": fold_metrics.pred_time
                }
                
                for target_name, stats_metrics in fold_metrics.stats_by_target.items():
                    for metric_type, value in stats_metrics.to_dict().items():
                        row[f"{metric_type}_{target_name}"] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)


class ModelEvaluator:
    """
    Perform k-fold cross-validation for multiple models and summarize results.

    Parameters
    ----------
    schema : Schema
        Schema object containing target and feature definitions.
    output_dir : str
        Directory to store evaluation outputs.
    models : list[RegressionModels]
        List of models to evaluate.
    random_state : int
        Random seed for reproducibility.
    n_splits : int
        Number of cross-validation folds.

    Attributes
    ----------
    target : TargetSchema
        Target schema definition.
    features : list[FeatureSchema]
        Feature schema definitions.
    """

    def __init__(self, 
                 schema: Schema, 
                 output_dir: str, 
                 models: List[RegressionModels], 
                 random_state: int, 
                 n_splits: int) -> None:
        self.schema     : Schema                 = schema
        self.output_dir : str                    = output_dir
        self.models     : List[RegressionModels] = models
        self.random_state: int                   = random_state
        self.n_splits   : int                    = n_splits

        self.target     : TargetSchema           = schema.target
        self.features   : List[FeatureSchema]    = schema.features

    def evaluate(self, df_processed: pd.DataFrame) -> None:
        """
        Execute k-fold cross-validation for all models.

        Parameters
        ----------
        df_processed : pd.DataFrame
            Preprocessed dataset including features and targets.
        """
        X = df_processed.drop(columns=self.target.names)
        y = df_processed[self.target.names]

        logger.info(f"Feature variables considered: {list(X.columns)}")
        logger.info(f"Target variables considered: {list(y.columns)}")

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        evaluation_results = ModelResults()

        for i, reg_enum in enumerate(self.models, start=1):
            logger.info("Evaluating model %s of %s", i, len(self.models))
            model_metrics = self._evaluate_model(reg_enum, X, y, kf.split(X, y))
            evaluation_results.add_model_result(model_metrics)

        df_summary = evaluation_results.get_summary()
        df_summary.to_csv(os.path.join(self.output_dir, f"eval_summary.csv"))

        plot_scores(df_summary, os.path.join(self.output_dir, "eval_scores"))
        plot_times(df_summary, os.path.join(self.output_dir, "eval_times"))

        return None

    @progress_bar
    def _evaluate_model(
            self,
            regressor: RegressionModels,
            features: pd.DataFrame,
            target: pd.DataFrame,
            fold_iterator: Iterator[tuple[np.ndarray, np.ndarray]]
        ) -> ModelMetrics:
        """
        Run cross-validation for a single model.

        Parameters
        ----------
        regressor : RegressionModels
            Model to evaluate.
        features : pd.DataFrame
            Feature dataset.
        target : pd.DataFrame
            Target dataset.
        fold_iterator : Iterator[tuple[np.ndarray, np.ndarray]]
            Iterator yielding train-test splits.

        Returns
        -------
        ModelMetrics
            Object containing all evaluation results for the model.
        """
        model_metrics = ModelMetrics(regressor)

        for train_idx, test_idx in fold_iterator:
            X_train = features.iloc[train_idx].copy()
            X_test = features.iloc[test_idx].copy()
            y_train = target.iloc[train_idx].copy()
            y_test = target.iloc[test_idx]

            post_split_processor = PostSplitProcessor(self.schema)
            X_train, y_train = post_split_processor.fit_transform(X_train, y_train)
            X_test = post_split_processor.transform(X_test)

            model_instance = ModelFactory.create_model(regressor)
            fit_time = model_instance.fit(X_train, y_train)
            y_pred, prediction_time = model_instance.predict(X_test)
            y_pred = post_split_processor.inverse_transform_target(y_pred)

            r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
            rmse_scores = root_mean_squared_error(y_test, y_pred, multioutput='raw_values')
            mae_scores = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

            model_metrics.add_fold_result(
                target_names=self.target.names,
                r2_scores=r2_scores.tolist(),
                rmse_scores=rmse_scores.tolist(),
                mae_scores=mae_scores.tolist(),
                fit_time=fit_time,
                pred_time=prediction_time,
            )

        return model_metrics