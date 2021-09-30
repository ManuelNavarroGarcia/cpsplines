import pandas as pd
from typing import Callable, Optional, Tuple, Union


def boxplot_test(
    df: pd.DataFrame,
    func: Union[Callable, str] = "median",
    decimals: int = 2,
    ylim: Optional[Tuple[Union[int, float]]] = None,
    labels: Optional[Tuple[str]] = None,
    face_colours: Optional[Tuple[str]] = None,
    line_colours: Optional[Tuple[str]] = None,
    legend_boxes_indexes: Optional[Tuple[int]] = None,
    legend_labels: Optional[Tuple[str]] = None,
    legend_location: str = "best",
    **kwargs,
):
    ax, bplot = df.boxplot(
        column=df.columns.tolist(), return_type="both", patch_artist=True, **kwargs
    )
    fig = ax.get_figure()
    xlocs = ax.get_xticks()
    if ylim is None:
        ylim = (0, df.max().max())
    for i, v in enumerate(df.apply(func=func)):
        _ = ax.text(
            xlocs[i] - 0.1,
            1.2 * ylim[1],
            f"{v:2.{decimals}f}",
            weight="bold",
            fontsize=16,
        )
    _ = ax.set_ylim(ylim[0], 1.3 * ylim[1])
    if labels is not None:
        _ = ax.set_xticklabels(labels)
    if line_colours is not None:
        for elem in ["boxes", "fliers"]:
            for patch, color in zip(bplot[elem], line_colours):
                patch.set(color=color)

        for elem in ["caps", "whiskers"]:
            for patch1, patch2, color in zip(
                bplot[elem][::2], bplot[elem][1::2], line_colours
            ):
                patch1.set(color=color)
                patch2.set(color=color)
    if face_colours is not None:
        for patch, color in zip(bplot["boxes"], face_colours):
            patch.set(facecolor=color)
    if legend_boxes_indexes is not None:
        legend_boxes = [bplot["boxes"][i] for i in legend_boxes_indexes]
        _ = ax.legend(
            legend_boxes, legend_labels, loc=legend_location, prop={"size": 16}
        )
    return fig, ax
