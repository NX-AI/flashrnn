from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from pathlib import Path


def create_group_names_from_cols(data_df: pd.DataFrame, colnames: str) -> list[str]:
    group_names = []
    group_cols = data_df[colnames].astype(int)
    for i, row in group_cols.iterrows():
        group_str = ""
        for i, colname in enumerate(colnames):
            group_str += f"{colname}={row[colname]}"
            if i < len(colnames) - 1:
                group_str += "\n"
        # print(group_str)
        group_names.append(group_str)
    return group_names


def create_bar_length_df(
    data_df: pd.DataFrame, colnames: list[str], offset: float = 0.0
) -> pd.DataFrame:
    # subtract the min from selected columns respectively and add an arbitrary offset
    bar_length_df = (
        data_df[colnames]
        .sub(data_df[colnames].min())
        .div(data_df[colnames].max() / offset)
        .add(offset)
    )
    bar_length_data_df = data_df.copy()
    bar_length_data_df[colnames] = bar_length_df
    return bar_length_data_df


def create_runtime_bar_plot(
    data_df: pd.DataFrame,
    group_col_names: list[str],
    title: str = None,
    bar_label_font_size: int = 9,
    bar_length_df: pd.DataFrame = None,
    plot_column_order: list[str] = None,
    style_dict: dict[str, Any] = None,
    fillna_val: float = -0.2,
    fillna_exclude_cols: list[str] = None,
    fillna_str: str = "OOSM",
    legend_args: dict[str, Any] = dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)),
    legend_order: list[str] = None,
    figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54),
    grid_alpha: float = 0.2,
    yticks: list[float] = None,
    ax=None,
):
    group_names = create_group_names_from_cols(
        data_df=data_df, colnames=group_col_names
    )
    raw_data_df = data_df.drop(columns=group_col_names)
    # data df contains only the columns to plot
    if fillna_exclude_cols is not None:
        raw_data_nan_cols_df = raw_data_df[fillna_exclude_cols].round(2)
    else:
        fillna_exclude_cols = []
        raw_data_nan_cols_df = pd.DataFrame()
    raw_data_nonan_cols_df = raw_data_df.drop(columns=fillna_exclude_cols)
    if bar_length_df is None:
        bar_length_df = raw_data_nonan_cols_df.fillna(fillna_val).round(2)
    else:
        bar_length_df = bar_length_df.drop(columns=fillna_exclude_cols)
        bar_length_df = (
            bar_length_df.drop(columns=group_col_names).fillna(fillna_val).round(2)
        )
    raw_data_nonan_cols_df = raw_data_nonan_cols_df.round(2).fillna(fillna_str)

    if fillna_exclude_cols is not None:
        raw_data_df = pd.concat([raw_data_nonan_cols_df, raw_data_nan_cols_df], axis=1)
        bar_length_df = pd.concat([bar_length_df, raw_data_nan_cols_df], axis=1)
    else:
        raw_data_df = raw_data_nonan_cols_df

    # x-axis locations
    x = np.arange(len(raw_data_df))
    width = 1 / (len(raw_data_df.columns) + 1)
    multiplier = 0

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    if plot_column_order is not None:
        columns = plot_column_order
    else:
        columns = bar_length_df.columns

    for col in columns:
        offset = width * multiplier
        if style_dict is None:
            rects = ax.bar(x + offset, bar_length_df[col], width, label=col)
        else:
            rects = ax.bar(x + offset, bar_length_df[col], width, **style_dict[col])
        ax.bar_label(
            rects, labels=raw_data_df[col], padding=2, fontsize=bar_label_font_size
        )
        multiplier += 1

    ax.set_ylabel("Time (ms)")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticks(x + width, group_names)
    if legend_args and legend_order is None:
        ax.legend(**legend_args)
    elif legend_args and legend_order is not None:
        handles, labels = ax.get_legend_handles_labels()
        label_handle_dict = dict(zip(labels, handles))
        handles = [label_handle_dict[label] for label in legend_order]
        ax.legend(handles=handles, **legend_args)
    ax.grid(alpha=grid_alpha, which="both")

    if yticks is not None:
        ax.set_yticks(yticks)
        y_formatter = plt.ScalarFormatter()
        # y_formatter.set_scientific(False)
        # y_formatter.set_useOffset(10.0)
        ax.get_yaxis().set_major_formatter(y_formatter)

    return f


def savefig(f, name: str, savedir: Path = Path(".")) -> None:
    f.savefig(savedir / f"{name}.pdf", bbox_inches="tight")
    f.savefig(savedir / f"{name}.svg", bbox_inches="tight")
    f.savefig(savedir / f"{name}.png", bbox_inches="tight", dpi=200)


def plot_runtime_results(
    data_df: pd.DataFrame,
    plot_column_order: list[str],
    group_cols: list[str],
    slow_cols: list[str],
    slow_cols_offset: float,
    yticks: list[float],
    bar_label_fontsize: int = 9,
    filename: str = None,
    fillna_exclude_cols: list[str] = None,
    legend_args: dict[str, Any] = dict(
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.0, 0.97, 1.0, 0.102),
        frameon=False,
        facecolor="white",
    ),
    ax=None,
):
    from plot_config import (
        FONTSIZE,
        FONTSIZE_SMALL,
        FONTSIZE_TICKS,
        FIGSIZE,
        style_dict,
        save_path,
    )

    with mpl.rc_context(
        rc={
            "text.usetex": True,
            "font.size": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE_SMALL,
            "xtick.labelsize": FONTSIZE_TICKS,
            "ytick.labelsize": FONTSIZE_TICKS,
            "axes.titlesize": FONTSIZE,
            "lines.markersize": 4.0,  # * default: 6.0
        }
    ):
        f = create_runtime_bar_plot(
            data_df=data_df,
            bar_length_df=create_bar_length_df(
                data_df, slow_cols, offset=slow_cols_offset
            ),
            bar_label_font_size=bar_label_fontsize,
            group_col_names=group_cols,
            plot_column_order=plot_column_order,
            style_dict=style_dict,
            legend_args=legend_args,  # {"loc": "upper right", "bbox_to_anchor": (1.1, 1.0)},
            yticks=yticks,
            figsize=FIGSIZE,
            fillna_val=-(slow_cols_offset / 60.0),
            fillna_exclude_cols=fillna_exclude_cols,
            ax=ax,
        )
        if filename is not None:
            savefig(f, filename, save_path)

        return f


def plot_runtime_results_fwbw(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    col_order_left: list[str] = None,
    col_order_right: list[str] = None,
    slow_cols_left: list[str] = [],
    slow_cols_offset_left: float = 0.0,
    yticks_left: list[float] = [],
    slow_cols_right: list[str] = [],
    slow_cols_offset_right: float = 0.0,
    yticks_right: list[float] = [],
    group_cols: list[str] = [],
    filename_wo_ending: str = "",
    legend_args: dict[str, Any] = {
        "loc": "lower center",
        "ncol": 3,
        "bbox_to_anchor": (0.0, 0.97, 1.0, 0.102),
        "frameon": False,
        "facecolor": "white",
    },
    modify_df_func=None,
    fillna_exclude_cols_left: list[str] = None,
    fillna_exclude_cols_right: list[str] = None,
):
    from plot_config import FIGSIZE_2COL, GRIDSPEC_KWARGS, save_path

    f, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=FIGSIZE_2COL, gridspec_kw=GRIDSPEC_KWARGS
    )

    if modify_df_func is not None:
        df_left = modify_df_func(df_left)
        df_right = modify_df_func(df_right)

    f = plot_runtime_results(
        data_df=df_left,
        slow_cols=slow_cols_left,
        slow_cols_offset=slow_cols_offset_left,
        group_cols=group_cols,
        yticks=yticks_left,
        plot_column_order=col_order_left,
        legend_args=legend_args,
        fillna_exclude_cols=fillna_exclude_cols_left,
        ax=ax_left,
    )
    f = plot_runtime_results(
        data_df=df_right,
        slow_cols=slow_cols_right,
        slow_cols_offset=slow_cols_offset_right,
        group_cols=group_cols,
        yticks=yticks_right,
        plot_column_order=col_order_right,
        legend_args=legend_args,
        fillna_exclude_cols=fillna_exclude_cols_right,
        ax=ax_right,
    )
    savefig(f, savedir=save_path, name=filename_wo_ending)
    return f
