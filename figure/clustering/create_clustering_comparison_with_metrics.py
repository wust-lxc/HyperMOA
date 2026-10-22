#!/usr/bin/env python
"""Add compact clustering metric tables to the comparison figures."""

from __future__ import annotations

import json
import os
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from openpyxl import load_workbook
from PIL import Image

import create_clustering_comparison_paper_figures as base


OUTPUT_DIR = base.OUTPUT_DIR
METRICS_PATH = OUTPUT_DIR / "selected_slice_clustering_metrics_with_aggregate.csv"
SELECTION_PATH = OUTPUT_DIR / "slice_selection_scores_ari_ami_moran_pas_chaos.csv"
MANIFEST_PATH = OUTPUT_DIR / "clustering_comparison_with_metrics_manifest.json"
FAIR_METRICS_PATH = base.ROOT / "data/evaluation/fair_benchmark/clustering_slice_metrics_fair.csv"
PAPER_METRICS_PATH = base.ROOT / "data/evaluation/fair_benchmark/clustering_paper_metrics_slice.csv"

CORE_METRICS = ("ARI", "AMI", "Moran's I", "PAS", "CHAOS")
LOWER_IS_BETTER = frozenset(("PAS", "CHAOS"))
AGGREGATE_LABEL = "Aggregate score"
PANEL_TITLE_SPACE = 0.032

FILE_STEMS = {
    "Human_lymph_node": "human_lymph_node_slice2_clustering_comparison_with_metrics",
    "Simulation": "simulation_slice5_clustering_comparison_with_metrics",
    "Mouse_Brain": "mouse_brain_h3k27ac_clustering_comparison_with_metrics",
}

BUBBLE_CMAP = LinearSegmentedColormap.from_list(
    "clustering_metric_bubbles",
    ("#e5f0f5", "#82bad6", "#216a9a"),
)


def format_three_decimals(value: float) -> str:
    return format(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
        "f",
    )


def reserve_panel_title_space(axis) -> None:
    position = axis.get_position()
    axis.set_position(
        [
            position.x0,
            position.y0,
            position.width,
            position.height - PANEL_TITLE_SPACE,
        ]
    )


def normalized_rank_scores(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            ((frame[metric].rank(ascending=metric not in LOWER_IS_BETTER, method="average") - 1.0)
             / (len(frame) - 1.0)).rename(metric)
            for metric in CORE_METRICS
        ],
        axis=1,
    )


def aggregate_scores(frame: pd.DataFrame) -> pd.Series:
    return normalized_rank_scores(frame).mean(axis=1)


def choose_slices():
    workbook = load_workbook(base.WORKBOOK, data_only=True, read_only=True)
    fair = pd.read_csv(FAIR_METRICS_PATH).set_index(["dataset", "slice"]).sort_index()
    paper = pd.read_csv(PAPER_METRICS_PATH).set_index(["dataset", "slice"]).sort_index()
    selected = {}
    selected_metrics = {}
    metric_sources = {}
    selection_records = []
    try:
        for dataset, candidates in base.CANDIDATES.items():
            results = []
            for candidate in candidates:
                frame, cells = base.read_sheet_metrics(workbook, candidate["sheet"])
                slice_key = candidate["slice"]
                if dataset == "Simulation":
                    slice_key = f"Simulation{slice_key.removeprefix('slice')}"
                elif dataset == "Mouse_Brain":
                    slice_key = f"Mouse_Brain_{slice_key}"
                raw = fair.loc[(dataset, slice_key)].set_index("method").reindex(base.METHODS)
                if np.max(np.abs(raw.loc[:, base.METRICS].to_numpy() - frame.loc[:, base.METRICS].to_numpy())) > 0.000501:
                    raise ValueError(f"Rounded workbook metrics disagree with fair CSV: {candidate['sheet']}")
                published = paper.loc[(dataset, slice_key)].set_index("method").reindex(base.METHODS)
                frame = published.loc[:, CORE_METRICS].copy()
                sheet = workbook[candidate["sheet"]]
                if (sheet.cell(1, 6).value, sheet.cell(1, 7).value) != ("PAS", "CHAOS"):
                    raise ValueError(f"Missing PAS/CHAOS workbook columns: {candidate['sheet']}")
                for method in base.METHODS:
                    row = next(
                        row for row in range(2, sheet.max_row + 1)
                        if str(sheet.cell(row, 1).value).strip() == method
                    )
                    for column, metric in ((6, "PAS"), (7, "CHAOS")):
                        value = float(sheet.cell(row, column).value)
                        if abs(value - float(frame.loc[method, metric])) > 0.000501:
                            raise ValueError(f"Rounded workbook {metric} disagrees with fair CSV: {candidate['sheet']}/{method}")
                        cells[method][metric] = sheet.cell(row, column).coordinate
                scores = aggregate_scores(frame)
                competitor_score = float(scores.drop(index="HyperMOA").max())
                score_margin = float(scores.loc["HyperMOA"] - competitor_score)
                selection_records.append(
                    {
                        "dataset": dataset,
                        "slice": candidate["slice"],
                        "sheet": candidate["sheet"],
                        "HyperMOA_aggregate_score": float(scores.loc["HyperMOA"]),
                        "best_competitor_aggregate_score": competitor_score,
                        "aggregate_score_margin": score_margin,
                    }
                )
                results.append((candidate, frame, cells))

            expected = base.EXPECTED_SELECTIONS[dataset]
            matched = next((item for item in results if item[0]["slice"] == expected), None)
            if matched is None:
                raise ValueError(f"{dataset}: fixed slice {expected} is unavailable")
            winner, winner_frame, winner_cells = matched
            selected[dataset] = winner
            selected_metrics[dataset] = winner_frame
            metric_sources[dataset] = {
                "sheet": winner["sheet"],
                "cells": {
                    method: {
                        metric: winner_cells[method][metric]
                        for metric in CORE_METRICS
                    }
                    for method in base.METHODS
                },
            }
    finally:
        workbook.close()

    return selected, selected_metrics, metric_sources, pd.DataFrame(selection_records)


def draw_metric_table(axis, frame: pd.DataFrame):
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    scores = normalized_rank_scores(frame)
    aggregate = scores.mean(axis=1)
    table_values = {
        **{metric: frame[metric] for metric in CORE_METRICS},
        AGGREGATE_LABEL: aggregate,
    }
    table_scores = {
        **{metric: scores[metric] for metric in CORE_METRICS},
        AGGREGATE_LABEL: aggregate,
    }

    content_left = 0.045
    content_right = 0.955
    row_label_width = 0.125
    method_left = content_left + row_label_width
    method_width = (content_right - method_left) / len(base.METHODS)
    method_centers = {
        method: method_left + (index + 0.5) * method_width
        for index, method in enumerate(base.METHODS)
    }

    header_y = 0.925
    top_rule_y = 0.835
    row_centers = {
        "ARI": 0.750,
        "AMI": 0.620,
        "Moran's I": 0.490,
        "PAS": 0.360,
        "CHAOS": 0.230,
        AGGREGATE_LABEL: 0.085,
    }


    axis.text(
        content_left + 0.008,
        header_y,
        "Metric",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=5.7,
        weight="normal",
    )
    for method, x_position in method_centers.items():
        axis.text(
            x_position,
            header_y,
            method,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=5.25,
            weight="normal",
        )

    axis.plot(
        [content_left, content_right],
        [top_rule_y, top_rule_y],
        transform=axis.transAxes,
        color="#50595d",
        lw=0.65,
    )
    axis.plot(
        [content_left, content_right],
        [0.160, 0.160],
        transform=axis.transAxes,
        color="#a7b0b4",
        lw=0.55,
    )

    for row_label, y_position in row_centers.items():
        display_label = (
            "Aggregate score↑"
            if row_label == AGGREGATE_LABEL
            else f"{row_label}{'↓' if row_label in LOWER_IS_BETTER else '↑'}"
        )
        axis.text(
            content_left + 0.008,
            y_position,
            display_label,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            weight="normal",
            linespacing=0.9,
        )

        values = table_values[row_label]
        ranks = table_scores[row_label]
        color_ranks = ranks.copy()
        if row_label == "CHAOS":
            # Use one shade for CHAOS values that coincide at displayed precision.
            displayed_values = values.map(format_three_decimals)
            color_ranks = ranks.groupby(displayed_values).transform("max")
        best_value = float(values.min() if row_label in LOWER_IS_BETTER else values.max())
        for method, x_position in method_centers.items():
            value = float(values.loc[method])
            rank_score = float(ranks.loc[method])
            is_best = np.isclose(value, best_value, atol=5e-7, rtol=0.0)
            face_color = BUBBLE_CMAP(0.08 + 0.90 * float(color_ranks.loc[method]))
            axis.add_patch(
                Ellipse(
                    (x_position, y_position),
                    width=method_width * 0.62,
                    height=0.095,
                    transform=axis.transAxes,
                    facecolor=face_color,
                    edgecolor="#145378" if is_best else "none",
                    linewidth=0.7 if is_best else 0.0,
                    zorder=2,
                    clip_on=False,
                )
            )
            axis.text(
                x_position,
                y_position,
                format_three_decimals(value),
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=5.7,
                color="white" if rank_score >= 0.62 else "#26363e",
                weight="bold" if is_best else "normal",
                zorder=3,
            )

    axis.plot(
        [content_left, content_right],
        [0.0, 0.0],
        transform=axis.transAxes,
        color="#50595d",
        lw=0.65,
        clip_on=False,
    )


def render_dataset(dataset: str, config: dict, frame: pd.DataFrame):
    data = ad.read_h5ad(config["path"])
    spatial = np.asarray(data.obsm["spatial"], dtype=float)
    truth = data.obs["truth"]
    truth_string = base.series_as_strings(truth)
    truth_labels = sorted(truth_string.dropna().unique(), key=base.natural_key)
    truth_palette = {
        label: base.TRUTH_COLORS[index]
        for index, label in enumerate(truth_labels)
    }
    panels = [
        ("Ground truth", None),
        *[(method, base.OBS_COLUMNS[method]) for method in base.METHODS],
    ]

    figure = plt.figure(
        figsize=(base.FIGURE_WIDTH, base.FIGURE_HEIGHT),
        facecolor="white",
    )
    grid = figure.add_gridspec(
        4,
        4,
        height_ratios=(1.0, 1.0, 0.17, 0.74),
        left=0.025,
        right=0.985,
        bottom=0.035,
        top=0.940,
        wspace=0.045,
        hspace=0.070,
    )

    panel_details = {}
    for index, (title, column) in enumerate(panels):
        axis = figure.add_subplot(grid[index // 4, index % 4])
        reserve_panel_title_space(axis)
        prediction = None if column is None else data.obs[column]
        colors, display_palette = base.panel_colors(truth, prediction, truth_palette)
        base.draw_panel(
            axis,
            spatial,
            colors,
            title,
            base.POINT_SIZES[dataset],
            highlight=(title == "HyperMOA"),
        )
        labels = truth if prediction is None else prediction
        panel_details[title] = {
            "cluster_count": int(labels.nunique(dropna=True)),
            "missing_count": int(labels.isna().sum()),
            "display_palette": display_palette,
        }

    legend_axis = figure.add_subplot(grid[2, :])
    base.draw_legend(legend_axis, truth_labels, truth_palette)

    table_axis = figure.add_subplot(grid[3, :])
    draw_metric_table(table_axis, frame)

    stem = FILE_STEMS[dataset]
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    preview_path = OUTPUT_DIR / f"{stem}_preview.png"
    figure.savefig(png_path, dpi=base.FIGURE_DPI, facecolor="white")
    figure.savefig(pdf_path, dpi=base.FIGURE_DPI, facecolor="white")
    plt.close(figure)
    base.save_preview(png_path, preview_path)
    data.file.close()

    with Image.open(png_path) as image:
        png_pixels = list(image.size)
        png_dpi = list(image.info.get("dpi", (None, None)))

    return {
        "input_h5ad": str(config["path"]),
        "input_h5ad_sha256": base.sha256(config["path"]),
        "sheet": config["sheet"],
        "slice": config["slice"],
        "truth_label_count": len(truth_labels),
        "panel_details": panel_details,
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "preview_png": str(preview_path),
        },
        "png_pixels": png_pixels,
        "png_dpi": png_dpi,
    }


def save_metrics(selected, selected_metrics):
    records = []
    for dataset, config in selected.items():
        frame = selected_metrics[dataset]
        aggregate = aggregate_scores(frame)
        for method in base.METHODS:
            records.append(
                {
                    "dataset": dataset,
                    "slice": config["slice"],
                    "sheet": config["sheet"],
                    "method": method,
                    **{
                        metric: float(frame.loc[method, metric])
                        for metric in CORE_METRICS
                    },
                    AGGREGATE_LABEL: float(aggregate.loc[method]),
                }
            )
    pd.DataFrame(records).to_csv(
        METRICS_PATH,
        index=False,
        float_format=format_three_decimals,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected, metrics, metric_sources, selection_scores = choose_slices()
    for config in selected.values():
        base.validate_input(config["path"])

    save_metrics(selected, metrics)
    selection_scores.to_csv(SELECTION_PATH, index=False, float_format="%.6f")

    figures = {
        dataset: render_dataset(dataset, config, metrics[dataset])
        for dataset, config in selected.items()
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "figure_size_cm": [base.FIGURE_WIDTH_CM, base.FIGURE_HEIGHT_CM],
        "figure_dpi": base.FIGURE_DPI,
        "workbook": str(base.WORKBOOK),
        "workbook_sha256": base.sha256(base.WORKBOOK),
        "displayed_metrics": list(CORE_METRICS),
        "rank_input": str(PAPER_METRICS_PATH) + " (full precision; workbook values displayed to 3 decimals)",
        "aggregate_definition": (
            "Equal-weight mean of normalized within-slice ranks for ARI, AMI, Moran's I, PAS, CHAOS. "
            "PAS and CHAOS are lower-is-better; all displayed metrics are included. "
            "For each metric, the worst rank maps to 0 and the best rank maps to 1; "
            "ties use average ranks and full-precision input."
        ),
        "selection_rule": (
            "Fixed previously selected illustrative slices: Human slice2, Simulation slice5, "
            "Mouse Brain H3K27ac. Selection scores are diagnostic only."
        ),
        "selected_slices": {
            dataset: config["slice"]
            for dataset, config in selected.items()
        },
        "method_order": ["Ground truth", *base.METHODS],
        "metric_sources": metric_sources,
        "metrics_csv": str(METRICS_PATH),
        "selection_scores_csv": str(SELECTION_PATH),
        "figures": figures,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for dataset, config in selected.items():
        print(f"{dataset}: {config['slice']} -> {FILE_STEMS[dataset]}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
