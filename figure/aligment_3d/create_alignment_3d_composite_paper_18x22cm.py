#!/usr/bin/env python
"""Create a compact three-dataset 3D alignment figure for a full paper page."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import create_alignment_3d_dataset_paper_figures as paper
import create_alignment_3d_metrics_figure as shared

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manuscript_panel_labels import add_panel_labels_to_figure
from manuscript_typography import apply_manuscript_typography


ROOT = shared.ROOT
WORKBOOK = ROOT / "benchmark .xlsx"
OUTPUT_DIR = ROOT / "figure/aligment_3d"
FILE_STEM = "alignment_3d_three_datasets_paper_18x22cm"
PNG_PATH = OUTPUT_DIR / f"{FILE_STEM}.png"
PDF_PATH = OUTPUT_DIR / f"{FILE_STEM}.pdf"
SVG_PATH = OUTPUT_DIR / f"{FILE_STEM}_grouped.svg"
PREVIEW_PATH = OUTPUT_DIR / f"{FILE_STEM}_preview.png"
METRICS_PATH = OUTPUT_DIR / "alignment_metrics_5_methods.csv"
MANIFEST_PATH = OUTPUT_DIR / "alignment_3d_three_datasets_manifest.json"

METHODS = paper.METHODS
DATASETS = paper.DATASETS
WIDTH_CM = 18.0
HEIGHT_CM = 22.0
WIDTH_IN = WIDTH_CM / 2.54
HEIGHT_IN = HEIGHT_CM / 2.54
DPI = 600
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
DATASET_SVG_IDS = {
    "Human_lymph_node": "human_lymph_node",
    "Simulation": "simulation",
    "Mouse_Brain": "mouse_brain",
}

ET.register_namespace("", SVG_NAMESPACE)
ET.register_namespace("xlink", XLINK_NAMESPACE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_preview(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        preview = image.convert("RGB")
        preview.thumbnail((1600, 1956), Image.Resampling.LANCZOS)
        preview.save(destination, dpi=(150, 150), optimize=True)


def svg_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def method_header_group_id(method: str) -> str:
    return f"method_header_{svg_slug(method)}"


def panel_group_id(dataset: str, method: str) -> str:
    return f"{DATASET_SVG_IDS[dataset]}_panel_{svg_slug(method)}"


def dataset_svg_groups(dataset: str) -> dict[str, object]:
    prefix = DATASET_SVG_IDS[dataset]
    return {
        "results": f"{prefix}_3d_results",
        "title": f"{prefix}_dataset_title",
        "panels": [panel_group_id(dataset, method) for method in METHODS],
        "legend": f"{prefix}_truth_legend",
        "table": f"{prefix}_metrics_table",
    }


def svg_group_hierarchy() -> tuple[dict[str, list[str]], list[str]]:
    hierarchy = {
        "method_headers": [method_header_group_id(method) for method in METHODS],
    }
    table_ids = []
    for dataset in DATASETS:
        groups = dataset_svg_groups(dataset)
        hierarchy[str(groups["results"])] = [
            str(groups["title"]),
            *[str(panel) for panel in groups["panels"]],
            str(groups["legend"]),
        ]
        table_ids.append(str(groups["table"]))
    return hierarchy, table_ids


def svg_axes_in_drawing_order() -> list[str]:
    axes_ids = [method_header_group_id(method) for method in METHODS]
    for dataset in DATASETS:
        groups = dataset_svg_groups(dataset)
        axes_ids.extend(
            [
                str(groups["title"]),
                *[str(panel) for panel in groups["panels"]],
                str(groups["legend"]),
                str(groups["table"]),
            ]
        )
    return axes_ids


def find_svg_element(root: ET.Element, element_id: str) -> ET.Element:
    for element in root.iter():
        if element.get("id") == element_id:
            return element
    raise ValueError(f"SVG element not found: {element_id}")


def flatten_svg_axes(root: ET.Element, axes_ids: list[str]) -> None:
    figure_group = find_svg_element(root, "figure_1")
    parent_map = {
        child: parent
        for parent in root.iter()
        for child in parent
    }
    axes = [find_svg_element(root, axis_id) for axis_id in axes_ids]
    for axis in axes:
        parent = parent_map.get(axis)
        if parent is None:
            raise ValueError(f"SVG axis has no parent: {axis.get('id')}")
        parent.remove(axis)
    for axis in axes:
        figure_group.append(axis)


def wrap_svg_elements(root: ET.Element, group_id: str, child_ids: list[str]) -> None:
    parent_map = {
        child: parent
        for parent in root.iter()
        for child in parent
    }
    children = [find_svg_element(root, child_id) for child_id in child_ids]
    parents = [parent_map.get(child) for child in children]
    if not parents or parents[0] is None or any(parent is not parents[0] for parent in parents):
        raise ValueError(f"SVG children for {group_id} do not share one parent")

    parent = parents[0]
    ordered_children = sorted(children, key=lambda child: list(parent).index(child))
    insert_at = min(list(parent).index(child) for child in ordered_children)
    wrapper = ET.Element(
        f"{{{SVG_NAMESPACE}}}g",
        {"id": group_id, "data-name": group_id},
    )
    for child in ordered_children:
        parent.remove(child)
        child.set("data-name", str(child.get("id")))
        wrapper.append(child)
    parent.insert(insert_at, wrapper)


def group_svg(path: Path) -> dict[str, object]:
    hierarchy, table_ids = svg_group_hierarchy()
    tree = ET.parse(path)
    root = tree.getroot()

    flatten_svg_axes(root, svg_axes_in_drawing_order())
    for group_id, child_ids in hierarchy.items():
        wrap_svg_elements(root, group_id, child_ids)
    for table_id in table_ids:
        find_svg_element(root, table_id).set("data-name", table_id)

    ET.indent(tree, space="  ")
    temporary_path = path.with_suffix(".tmp.svg")
    tree.write(temporary_path, encoding="utf-8", xml_declaration=True)
    temporary_path.replace(path)
    return validate_grouped_svg(path, hierarchy, table_ids)


def validate_grouped_svg(
    path: Path,
    hierarchy: dict[str, list[str]],
    table_ids: list[str],
) -> dict[str, object]:
    tree = ET.parse(path)
    root = tree.getroot()
    figure_group = find_svg_element(root, "figure_1")
    top_level_ids = ["method_headers"] + [
        group_id
        for dataset in DATASETS
        for group_id in (
            str(dataset_svg_groups(dataset)["results"]),
            str(dataset_svg_groups(dataset)["table"]),
        )
    ]
    direct_ids = {child.get("id") for child in figure_group}
    missing_top_level = [group_id for group_id in top_level_ids if group_id not in direct_ids]
    if missing_top_level:
        raise ValueError(f"Missing top-level SVG groups: {missing_top_level}")

    for group_id, child_ids in hierarchy.items():
        group = find_svg_element(root, group_id)
        direct_child_ids = {child.get("id") for child in group}
        missing_children = [child_id for child_id in child_ids if child_id not in direct_child_ids]
        if missing_children:
            raise ValueError(f"Missing children in {group_id}: {missing_children}")

    image_tag = f"{{{SVG_NAMESPACE}}}image"
    text_tag = f"{{{SVG_NAMESPACE}}}text"
    panel_ids = [panel_group_id(dataset, method) for dataset in DATASETS for method in METHODS]
    for panel_id in panel_ids:
        panel = find_svg_element(root, panel_id)
        if not list(panel.iter(image_tag)):
            raise ValueError(f"3D panel has no embedded raster image: {panel_id}")
    for table_id in table_ids:
        table = find_svg_element(root, table_id)
        if not list(table.iter(text_tag)):
            raise ValueError(f"Metric table has no editable text: {table_id}")

    editable_text_count = len(list(root.iter(text_tag)))
    svg_text = path.read_text(encoding="utf-8")
    if editable_text_count == 0 or "font-family: 'Arial'" not in svg_text:
        raise ValueError("SVG does not contain editable Arial text")
    if "DejaVu" in svg_text or re.search(r'id="Arial(?:MT|-BoldMT)?-', svg_text):
        raise ValueError("SVG contains a fallback font or outlined Arial glyphs")

    def dimension_cm(attribute: str) -> float:
        raw_value = root.get(attribute, "")
        if not raw_value.endswith("pt"):
            raise ValueError(f"Unexpected SVG {attribute}: {raw_value}")
        return float(raw_value[:-2]) * 2.54 / 72.0

    size_cm = [dimension_cm("width"), dimension_cm("height")]
    if any(abs(actual - expected) > 1e-5 for actual, expected in zip(size_cm, (WIDTH_CM, HEIGHT_CM))):
        raise ValueError(f"Unexpected SVG size: {size_cm}")

    return {
        "top_level_groups": top_level_ids,
        "top_level_group_count": len(top_level_ids),
        "method_panel_group_count": len(panel_ids),
        "editable_text_element_count": editable_text_count,
        "embedded_panel_image_count": sum(
            len(list(find_svg_element(root, panel_id).iter(image_tag)))
            for panel_id in panel_ids
        ),
        "size_cm": size_cm,
        "font": "Arial/Arial Bold editable text",
    }


def build_metric_export(metric_frames, aggregate_scores) -> pd.DataFrame:
    rows = []
    for dataset, frame in metric_frames.items():
        metrics = [column for column in frame.columns if column != "dataset"]
        for method in METHODS:
            row = {
                "dataset": dataset,
                "method": method,
            }
            for metric in shared.METRIC_DIRECTION:
                row[metric] = frame.loc[method, metric] if metric in metrics else float("nan")
            row["aggregate_score"] = float(shared.score_value_text(aggregate_scores[dataset].loc[method]))
            rows.append(row)
    return pd.DataFrame(rows)


def draw_method_header(figure, subplot_spec) -> None:
    header_grid = subplot_spec.subgridspec(1, len(METHODS), wspace=0.0)
    for column, method in enumerate(METHODS):
        axis = figure.add_subplot(header_grid[0, column])
        axis.set_gid(method_header_group_id(method))
        axis.set_axis_off()
        axis.text(
            0.5,
            0.10,
            method,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.2,
            weight="bold",
        )


def render_dataset_band(
    figure,
    subplot_spec,
    dataset: str,
    config,
    frame: pd.DataFrame,
    cell_scores: pd.DataFrame,
    aggregate: pd.Series,
    show_group_header: bool,
):
    band = subplot_spec.subgridspec(
        4,
        1,
        height_ratios=(0.12, 1.10, 0.24, 0.92) if dataset == "Mouse_Brain" else (0.12, 1.15, 0.19, 0.92),
        hspace=0.0,
    )

    label_axis = figure.add_subplot(band[0])
    groups = dataset_svg_groups(dataset)
    label_axis.set_gid(str(groups["title"]))
    label_axis.set_axis_off()
    label_axis.text(
        0.004,
        0.25,
        config["display"],
        transform=label_axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.4,
        weight="bold",
        color="#263238",
    )

    categories, palette = shared.truth_palette(config["root"], dataset)
    panel_grid = band[1].subgridspec(1, len(METHODS), wspace=0.0)
    panel_manifest = {}
    for column, method in enumerate(METHODS):
        path = config["root"] / method / "alignment_results.h5ad"
        paper.validate_alignment_input(path)
        panel_axis, panel_info = shared.draw_3d_panel(
            figure,
            panel_grid[0, column],
            path,
            dataset,
            method,
            categories,
            palette,
            show_title=False,
            show_coordinate_frame=False,
        )
        panel_axis.set_gid(panel_group_id(dataset, method))
        if dataset == 'Mouse_Brain':
            panel_axis.set_box_aspect((1.0, 1.0, 0.55), zoom=1.10)
        panel_manifest[method] = panel_info

    legend_axis = figure.add_subplot(band[2])
    legend_axis.set_gid(str(groups["legend"]))
    shared.draw_truth_legend(legend_axis, categories, palette, dataset)
    if show_group_header:
        bounds = legend_axis.get_position()
        legend_axis.set_position(
            [bounds.x0, bounds.y0 + 0.008, bounds.width, bounds.height]
        )

    table_axis = figure.add_subplot(band[3])
    table_axis.set_gid(str(groups["table"]))
    metrics = [column for column in frame.columns if column != "dataset"]
    paper.draw_metric_table(
        table_axis,
        frame,
        metrics,
        cell_scores,
        aggregate,
        show_group_header=show_group_header,
        value_fontsize=6.0,
    )

    return {
        "display_name": config["display"],
        "truth_categories": categories,
        "metrics": metrics,
        "panels": panel_manifest,
        "svg_groups": groups,
    }


def main() -> int:
    if not WORKBOOK.exists():
        raise FileNotFoundError(WORKBOOK)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_frames, source_cells = paper.read_metrics(WORKBOOK)
    cell_score_frames = {}
    aggregate_scores = {}
    for dataset, frame in metric_frames.items():
        metrics, cell_scores, aggregate = shared.rank_scores(paper.full_precision_rank_frame(dataset, frame))
        if tuple(metrics) != paper.EXPECTED_METRICS[dataset]:
            raise ValueError(f"Unexpected metric order for {dataset}: {metrics}")
        paper.validate_aggregate(dataset, aggregate)
        cell_score_frames[dataset] = cell_scores
        aggregate_scores[dataset] = aggregate

    metric_export = build_metric_export(metric_frames, aggregate_scores)
    metric_export.to_csv(METRICS_PATH, index=False, float_format="%.3f")
    if len(metric_export) != len(DATASETS) * len(METHODS):
        raise ValueError("Unexpected metric export row count")

    figure = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor="white")
    outer = figure.add_gridspec(
        4,
        1,
        height_ratios=(0.14, 1.0, 1.0, 1.0),
        left=0.025,
        right=0.985,
        bottom=0.018,
        top=0.985,
        hspace=0.035,
    )
    draw_method_header(figure, outer[0])

    figure_details = {}
    for index, (dataset, config) in enumerate(DATASETS.items()):
        figure_details[dataset] = render_dataset_band(
            figure,
            outer[index + 1],
            dataset,
            config,
            metric_frames[dataset],
            cell_score_frames[dataset],
            aggregate_scores[dataset],
            show_group_header=False,
        )

    add_panel_labels_to_figure(figure, "alignment")
    apply_manuscript_typography(figure, "alignment")
    qa_path = os.environ.get('HYPERMOA_FIGURE_QA_PATH')
    if qa_path:
        sys.path.insert(0, qa_path)
        from audit_panel_alignment import require_matplotlib_panel_alignment
        panels = [axis for axis in figure.axes if '_panel_' in (axis.get_gid() or '')]
        require_matplotlib_panel_alignment(
            figure, axes=panels,
            json_out=OUTPUT_DIR / f'{FILE_STEM}.alignment.json',
            overlay_svg=OUTPUT_DIR / f'{FILE_STEM}.alignment.svg',
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    figure.savefig(PNG_PATH, dpi=DPI, facecolor="white")
    figure.savefig(PDF_PATH, dpi=DPI, facecolor="white")
    figure.savefig(SVG_PATH, dpi=DPI, facecolor="white", format="svg")
    plt.close(figure)
    svg_validation = group_svg(SVG_PATH)
    save_preview(PNG_PATH, PREVIEW_PATH)

    with Image.open(PNG_PATH) as image:
        pixel_size = list(image.size)
        png_dpi = list(image.info.get("dpi", (None, None)))
    expected_pixels = [int(WIDTH_IN * DPI), int(HEIGHT_IN * DPI)]
    if pixel_size != expected_pixels:
        raise ValueError(f"Unexpected PNG dimensions: {pixel_size}; expected {expected_pixels}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "figure": {
            "size_cm": [WIDTH_CM, HEIGHT_CM],
            "dpi": DPI,
            "pixel_size": pixel_size,
            "png_dpi": png_dpi,
            "dataset_order": list(DATASETS),
            "method_order": list(METHODS),
            "layout": (
                "Three vertical dataset bands; each contains five 3D panels, "
                "a ground-truth legend, and a compact metric table."
            ),
            "method_headers": "shown once at the top",
            "group_headers": "omitted",
            "xy_display_normalization": "independent x/y min-max normalization per slice",
            "z_display_coordinate": "negative slice_order with fixed spacing; not physical thickness",
            "coloring": "ground-truth labels",
            "coordinate_frame": "hidden; no axes, ticks, panes, or grid",
            "minimum_table_font_pt": 6.0,
            "metric_table_layout": "Compact centered tables: metrics as rows, methods as columns in the same left-to-right order as section panels; aggregate score last",
            "metric_table_method_pitch_pt": paper.TABLE_METHOD_PITCH_PT,
            "metric_table_label_gap_pt": paper.TABLE_LABEL_GAP_PT,
            "section_panel_layout": "Five equal columns across the full width with no leading spacer column",
            "metric_table_direction_arrows": False,
            "metric_table_number_format": "Uniform three decimal places for every metric and aggregate score",
            "metric_table_bubble_size_pt": [paper.TABLE_BUBBLE_WIDTH_PT, paper.TABLE_BUBBLE_HEIGHT_PT],
            "metric_table_bubble_size_reference": "Same physical dimensions as the published 180-mm clustering metric tables",
            "metric_table_rank_encoding": "Blue color encodes rank; bubble dimensions are constant",
            "metric_table_footer": "No regional-spot-count footer in the artwork; counts remain in the manuscript caption and Methods",
            "pdf_font_embedding": "TrueType (PDF font type 42)",
            "svg_text": "editable Arial/Arial Bold text (not outlined)",
        },
        "workbook": {
            "path": str(WORKBOOK),
            "sha256": sha256(WORKBOOK),
            "source_cells": source_cells,
        },
        "aggregate_score": (
            "Equal-weight mean of normalized metric ranks within each dataset. "
            "Mouse metrics are regional ARI, regional AMI, Chamfer and PCC, each with weight 1/4."
        ),
        "metrics_csv": str(METRICS_PATH),
        "rank_input": str(paper.FAIR_METRICS_PATH) + " (full precision; workbook values displayed to 3 decimals)",
        "mouse_main_metrics": list(paper.EXPECTED_METRICS["Mouse_Brain"]),
        "mouse_aggregate_score": "equal-weight mean of normalized ranks of regional ARI, regional AMI, Chamfer and PCC",
        "mouse_region_protocol": str(ROOT / "data/evaluation/fair_benchmark/alignment_mouse_region_manifest.json"),
        "mouse_grid_proxy_caveat": "Grid proxies are supplementary; mouse main metrics use supplied regional partitions.",
        "mouse_two_vs_four_metric_sensitivity": str(ROOT / "data/evaluation/fair_benchmark/alignment_mouse_score_sensitivity.csv"),
        "datasets": figure_details,
        "svg_grouping": svg_validation,
        "outputs": {
            "png": str(PNG_PATH),
            "pdf": str(PDF_PATH),
            "grouped_svg": str(SVG_PATH),
            "grouped_svg_sha256": sha256(SVG_PATH),
            "preview": str(PREVIEW_PATH),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"Grouped SVG: {SVG_PATH}")
    print(f"Preview: {PREVIEW_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
