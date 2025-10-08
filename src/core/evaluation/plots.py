import logging
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional

logger = logging.getLogger(__name__)

def plot_scores(
    df: pd.DataFrame,
    prefix: Optional[str] = None,
    use_dual_axis: bool = True,
) -> None:
    """
    Plot grouped bars for R2, MAE, and RMSE metrics.

    Creates separate plots for each metric with automatic target detection.
    MAE and RMSE can use dual y-axis when multiple targets are available.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe with MultiIndex columns (metric, agg).
    prefix : str, optional
        If provided, saves to '{prefix}_{metric}.html'. If None, shows the figure.
    use_dual_axis : bool, optional
        If True, uses dual y-axis for MAE and RMSE metrics when multiple targets exist.
        Targets are automatically detected from column names.
    """
    df_flat = _flatten_summary(df)
    metrics = ['R2', 'MAE', 'RMSE']
    
    for metric in metrics:
        bases = _expand_bases(df_flat, [metric])
        if bases:
            should_use_dual_axis = use_dual_axis and metric in ['MAE', 'RMSE']
            _barplot(
                df_flat=df_flat,
                bases=bases,
                title=metric,
                output_path=(f"{prefix}_{metric}.html" if prefix else None),
                use_dual_axis=should_use_dual_axis,
            )


def plot_times(
    df: pd.DataFrame,
    prefix: Optional[str] = None,
) -> None:
    """
    Plot grouped bars for fit_time and pred_time metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe with MultiIndex columns (metric, agg).
    prefix : str, optional
        If provided, saves to '{prefix}_times.html'. If None, shows the figure.
    """
    df_flat = _flatten_summary(df)
    time_metrics = ['fit_time', 'pred_time']
    bases = _expand_bases(df_flat, time_metrics)
    
    if bases:
        _barplot(
            df_flat=df_flat,
            bases=bases,
            title="Training & Prediction Times",
            output_path=f"{prefix}_times.html" if prefix else None,
            use_dual_axis=False,
        )


def _flatten_summary(df_mi: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a MultiIndex summary DataFrame to single-level columns.

    Parameters
    ----------
    df_mi : pd.DataFrame
        Input DataFrame with MultiIndex columns (metric, agg).

    Returns
    -------
    pd.DataFrame
        A copy with single-level columns named as '<metric>_<agg>'.
    """
    if isinstance(df_mi.columns, pd.MultiIndex):
        out = df_mi.copy()
        out.columns = [f"{m}_{a}" for m, a in df_mi.columns]
        return out
    return df_mi.copy()


def _expand_bases(df_flat: pd.DataFrame, metrics: List[str]) -> List[str]:
    """
    Expand base metric names (e.g., ['R2','MAE']) into all '<metric>_<target>' bases
    that have both '<base>_mean' and '<base>_std' columns in `df_flat`.

    Parameters
    ----------
    df_flat : pd.DataFrame
        Flattened summary.
    metrics : List[str]
        Metric base names without target suffix (e.g., ['R2','MAE','RMSE']).

    Returns
    -------
    List[str]
        Bases present in df_flat, ordered by column appearance.
    """
    cols = list(df_flat.columns)
    bases = []
    for m in metrics:
        for c in cols:
            if c.startswith(m + "_") and c.endswith("_mean"):
                base = c[:-5]
                if f"{base}_std" in df_flat.columns and base not in bases:
                    bases.append(base)
    return bases


def _barplot(
    df_flat: pd.DataFrame,
    bases: List[str],
    title: str,
    output_path: Optional[str] = None,
    use_dual_axis: bool = False,
) -> None:
    """
    Draw grouped bar chart with error bars (std) for the given bases.

    Parameters
    ----------
    df_flat : pd.DataFrame
        Flattened summary DataFrame with single-level columns.
    bases : List[str]
        Bases to plot (e.g., ['R2_{target_1}', 'R2_{target_2}']).
    title : str
        Chart title.
    output_path : str, optional
        If provided, saves HTML (and attempts PNG). If None, shows the figure.
    use_dual_axis : bool, optional
        If True, uses dual y-axis for different targets when available.
    """    
    sorted_df = df_flat.sort_values(by=f"{bases[0]}_mean", ascending=True)
    
    targets = [base.split('_', 1)[1] for base in bases if '_' in base]
    unique_targets = list(dict.fromkeys(targets))
    
    if use_dual_axis and len(unique_targets) == 2:
        fig = _create_dual_axis_plot(sorted_df, bases, unique_targets, title)
    else:
        fig = _create_single_axis_plot(sorted_df, bases, title)
    
    _save_or_show_figure(fig, output_path)


def _create_dual_axis_plot(sorted_df: pd.DataFrame, bases: List[str], targets: List[str], title: str) -> go.Figure:
    """
    Create a dual-axis plot with targets on separate y-axes.

    Parameters
    ----------
    sorted_df : pd.DataFrame
        Sorted flattened summary DataFrame.
    bases : List[str]
        List of metric bases to plot.
    targets : List[str]
        List of target names (first two will be used for dual axis).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with dual y-axis configuration.
    """
    target1, target2 = targets[0], targets[1]
    fig = go.Figure()

    for i, target in enumerate(targets):
        target_bases = [base for base in bases if base.endswith(f'_{target}')]
        yaxis = 'y' if i == 0 else 'y2'
        
        for base in target_bases:
            if f"{base}_mean" in sorted_df.columns and f"{base}_std" in sorted_df.columns:
                fig.add_trace(go.Bar(
                    name=target,
                    x=sorted_df.index.astype(str),
                    y=sorted_df[f"{base}_mean"].astype(float),
                    error_y=dict(type="data", array=sorted_df[f"{base}_std"].astype(float), visible=True),
                    yaxis=yaxis,
                    offsetgroup=i+1,
                ))
    
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Model",
        yaxis=dict(title=target1, side='left'),
        yaxis2=dict(title=target2, side='right', overlaying='y'),
        xaxis_tickangle=-45,
        width=1000,
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.05,
            xanchor="left", 
            x=1.05
        ),
        legend_title_text="Targets",
        margin=dict(r=120),
    )
    return fig


def _create_single_axis_plot(sorted_df: pd.DataFrame, bases: List[str], title: str) -> go.Figure:
    """
    Create a single-axis plot.

    Parameters
    ----------
    sorted_df : pd.DataFrame
        Sorted flattened summary DataFrame.
    bases : List[str]
        List of metric bases to plot.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with single y-axis configuration.
    """
    fig = go.Figure([
        go.Bar(
            name=base,
            x=sorted_df.index.astype(str),
            y=sorted_df[f"{base}_mean"].astype(float),
            error_y=dict(type="data", array=sorted_df[f"{base}_std"].astype(float), visible=True),
        )
        for base in bases
        if f"{base}_mean" in sorted_df.columns and f"{base}_std" in sorted_df.columns
    ])
    
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Model",
        yaxis_title=title,
        xaxis_tickangle=-45,
        width=900,
        height=500,
        legend_title_text="Metrics",
    )
    return fig


def _save_or_show_figure(fig: go.Figure, output_path: Optional[str]) -> None:
    """
    Save figure to file or show it.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to save or display.
    output_path : str, optional
        If provided, saves HTML and attempts PNG. If None, shows the figure.
    """
    if output_path:
        fig.write_html(output_path)
        try:
            fig.write_image(output_path.replace(".html", ".png"), width=1000, height=500, scale=2)
        except Exception as e:
            logger.warning(f"Could not save PNG for {output_path}: {e}")
    else:
        fig.show()