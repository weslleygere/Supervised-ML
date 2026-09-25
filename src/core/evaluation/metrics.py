"""Point-balanced scoring; sampling repeats never count as independent sites."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sample_weights(df, schema):
    per_installation = df.groupby(schema.bag)[schema.bag].transform("size").to_numpy()
    installations_per_point = df.groupby(schema.group)[schema.bag].transform("nunique").to_numpy()
    return 1.0 / (per_installation * installations_per_point)


def regression_metrics(df, schema):
    y = df[schema.target].to_numpy(float)
    p = df["prediction"].to_numpy(float)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite predictions cannot be scored.")
    w = sample_weights(df, schema)
    ae = np.abs(y - p)
    order = np.argsort(ae)
    quantiles = np.interp([0.5, 0.9], (np.cumsum(w[order]) - 0.5 * w[order]) / w.sum(), ae[order])
    return {
        "MAE": float(mean_absolute_error(y, p, sample_weight=w)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p, sample_weight=w))),
        "R2": float(r2_score(y, p, sample_weight=w, force_finite=False)) if len(y) > 1 and np.average((y - np.average(y, weights=w)) ** 2, weights=w) > 0 else None,
        "bias": float(np.average(p - y, weights=w)),
        "median_absolute_error": float(quantiles[0]), "p90_absolute_error": float(quantiles[1]),
        "n_points": int(df[schema.group].nunique()), "n_installations": int(df[schema.bag].nunique()),
        "n_submissions": len(df),
    }


def macro_mae(predictions, schema, efforts):
    if set(predictions.n_recordings.unique()) != set(efforts):
        raise ValueError("Every selection effort must have validation predictions.")
    return float(np.mean([regression_metrics(predictions[predictions.n_recordings == k], schema)["MAE"] for k in efforts]))


def metric_table(predictions, schema):
    rows = []
    for (evaluation, repeat, effort), group in predictions.groupby(["evaluation", "repeat", "n_recordings"], sort=True):
        rows.append({"evaluation": evaluation, "repeat": repeat, "n_recordings": effort, **regression_metrics(group, schema)})
    return pd.DataFrame(rows)


def point_loss_matrix(predictions, schema):
    df = predictions.assign(absolute_error=abs(predictions.prediction - predictions[schema.target]))
    # Average submissions, installations, then CV repeats. Keep entire Points
    # paired across all efforts when resampling this matrix.
    loss = df.groupby(["repeat", "n_recordings", schema.group, schema.bag]).absolute_error.mean()
    loss = loss.groupby(level=["repeat", "n_recordings", schema.group]).mean()
    loss = loss.groupby(level=["n_recordings", schema.group]).mean()
    matrix = loss.unstack("n_recordings").sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError("Effort curves require the same evaluation Points at every count.")
    return matrix
