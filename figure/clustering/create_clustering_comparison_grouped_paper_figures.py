#!/usr/bin/env python
"""Create three grouped, publication-ready clustering comparison figures."""

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

import anndata as ad
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from PIL import Image
from scipy.optimize import linear_sum_assignment

import create_clustering_comparison_paper_figures as base
import create_clustering_comparison_with_metrics as metrics_figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manuscript_panel_labels import add_panel_labels_to_figure
from manuscript_typography import apply_manuscript_typography


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "figure/clustering"
WORKBOOK = base.WORKBOOK
METRICS_PATH = OUTPUT_DIR / "selected_slice_clustering_metrics_with_aggregate.csv"
SELECTION_PATH = OUTPUT_DIR / "slice_selection_scores_ari_ami_moran_pas_chaos.csv"
COLOR_MAPPING_PATH = OUTPUT_DIR / "clustering_display_color_mapping.csv"
MANIFEST_PATH = OUTPUT_DIR / "clustering_grouped_paper_figures_manifest.json"

FILE_STEMS = {
    "Human_lymph_node": "human_lymph_node_slice2_clustering_comparison_with_metrics",
    "Simulation": "simulation_slice5_clustering_comparison_with_metrics",
    "Mouse_Brain": "mouse_brain_h3k27ac_clustering_comparison_with_metrics",
}
DATASET_SVG_IDS = {
    "Human_lymph_node": "human_lymph_node",
    "Simulation": "simulation",
    "Mouse_Brain": "mouse_brain",
}
PANELS = (
    ("Ground truth", None),
    *((method, base.OBS_COLUMNS[method]) for method in base.METHODS),
)
SHADE_VARIANTS = (
    ("base", "base", 0.00),
    ("light_20", "light", 0.20),
    ("dark_15", "dark", 0.15),
    ("light_38", "light", 0.38),
    ("dark_30", "dark", 0.30),
)
SIMULATION_ROTATION_DEGREES = 15.0
SIMULATION_DISPLAY_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NAMESPACE)
ET.register_namespace("xlink", XLINK_NAMESPACE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def svg_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def panel_group_id(dataset: str, title: str) -> str:
    return f"{DATASET_SVG_IDS[dataset]}_panel_{svg_slug(title)}"


def dataset_group_ids(dataset: str) -> dict[str, object]:
    prefix = DATASET_SVG_IDS[dataset]
    return {
        "panels": f"{prefix}_clustering_panels",
        "panel_children": [panel_group_id(dataset, title) for title, _ in PANELS],
        "legend": f"{prefix}_truth_legend",
        "table": f"{prefix}_metrics_table",
    }


def find_svg_element(root: ET.Element, element_id: str) -> ET.Element:
    for element in root.iter():
        if element.get("id") == element_id:
            return element
    raise ValueError(f"SVG element not found: {element_id}")


def flatten_svg_axes(root: ET.Element, axes_ids: list[str]) -> None:
    figure_group = find_svg_element(root, "figure_1")
    parent_map = {child: parent for parent in root.iter() for child in parent}
    axes = [find_svg_element(root, axis_id) for axis_id in axes_ids]
    for axis in axes:
        parent = parent_map.get(axis)
        if parent is None:
            raise ValueError(f"SVG axis has no parent: {axis.get('id')}")
        parent.remove(axis)
    for axis in axes:
        figure_group.append(axis)


def wrap_svg_elements(root: ET.Element, group_id: str, child_ids: list[str]) -> None:
    parent_map = {child: parent for parent in root.iter() for child in parent}
    children = [find_svg_element(root, child_id) for child_id in child_ids]
    parents = [parent_map.get(child) for child in children]
    if not parents or parents[0] is None or any(parent is not parents[0] for parent in parents):
        raise ValueError(f"SVG children for {group_id} do not share one parent")
    parent = parents[0]
    ordered = sorted(children, key=lambda child: list(parent).index(child))
    insert_at = min(list(parent).index(child) for child in ordered)
    wrapper = ET.Element(
        f"{{{SVG_NAMESPACE}}}g",
        {"id": group_id, "data-name": group_id},
    )
    for child in ordered:
        parent.remove(child)
        child.set("data-name", str(child.get("id")))
        wrapper.append(child)
    parent.insert(insert_at, wrapper)


def dimension_cm(root: ET.Element, attribute: str) -> float:
    raw_value = root.get(attribute, "")
    if not raw_value.endswith("pt"):
        raise ValueError(f"Unexpected SVG {attribute}: {raw_value}")
    return float(raw_value[:-2]) * 2.54 / 72.0


def group_and_validate_svg(path: Path, dataset: str) -> dict[str, object]:
    groups = dataset_group_ids(dataset)
    panel_ids = [str(value) for value in groups["panel_children"]]
    legend_id = str(groups["legend"])
    table_id = str(groups["table"])
    axes_ids = [*panel_ids, legend_id, table_id]

    tree = ET.parse(path)
    root = tree.getroot()
    flatten_svg_axes(root, axes_ids)
    wrap_svg_elements(root, str(groups["panels"]), panel_ids)
    find_svg_element(root, legend_id).set("data-name", legend_id)
    find_svg_element(root, table_id).set("data-name", table_id)

    ET.indent(tree, space="  ")
    temporary_path = path.with_suffix(".tmp.svg")
    tree.write(temporary_path, encoding="utf-8", xml_declaration=True)
    temporary_path.replace(path)

    tree = ET.parse(path)
    root = tree.getroot()
    figure_group = find_svg_element(root, "figure_1")
    top_level_ids = [str(groups["panels"]), legend_id, table_id]
    direct_ids = {child.get("id") for child in figure_group}
    if any(group_id not in direct_ids for group_id in top_level_ids):
        raise ValueError(f"{dataset}: missing top-level SVG group")
    panel_parent = find_svg_element(root, str(groups["panels"]))
    direct_panel_ids = {child.get("id") for child in panel_parent}
    if any(panel_id not in direct_panel_ids for panel_id in panel_ids):
        raise ValueError(f"{dataset}: missing SVG panel child")

    image_tag = f"{{{SVG_NAMESPACE}}}image"
    text_tag = f"{{{SVG_NAMESPACE}}}text"
    for panel_id in panel_ids:
        if not list(find_svg_element(root, panel_id).iter(image_tag)):
            raise ValueError(f"{dataset}: panel has no embedded raster image: {panel_id}")
    if not list(find_svg_element(root, table_id).iter(text_tag)):
        raise ValueError(f"{dataset}: metric table text is not editable")

    svg_text = path.read_text(encoding="utf-8")
    editable_text_count = len(list(root.iter(text_tag)))
    if editable_text_count == 0 or "font-family: 'Arial'" not in svg_text:
        raise ValueError(f"{dataset}: SVG does not contain editable Arial text")
    if "DejaVu" in svg_text or re.search(r'id="Arial(?:MT|-BoldMT)?-', svg_text):
        raise ValueError(f"{dataset}: SVG contains fallback or outlined font glyphs")

    size_cm = [dimension_cm(root, "width"), dimension_cm(root, "height")]
    expected_size = [base.FIGURE_WIDTH_CM, base.FIGURE_HEIGHT_CM]
    if any(abs(actual - expected) > 1e-5 for actual, expected in zip(size_cm, expected_size)):
        raise ValueError(f"{dataset}: unexpected SVG size {size_cm}")

    return {
        "top_level_groups": top_level_ids,
        "panel_child_groups": panel_ids,
        "editable_text_element_count": editable_text_count,
        "embedded_image_count": len(list(root.iter(image_tag))),
        "size_cm": size_cm,
        "font": "Arial/Arial Bold editable text",
    }


def rotate_coordinates(spatial: np.ndarray, degrees: float) -> tuple[np.ndarray, list[float]]:
    center = spatial.mean(axis=0)
    radians = np.deg2rad(degrees)
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    rotated = (spatial - center) @ rotation.T + center
    return rotated, center.tolist()


def shade_color(color: str, variant_index: int) -> tuple[str, str]:
    if variant_index >= len(SHADE_VARIANTS):
        raise ValueError(f"Not enough shade variants for index {variant_index}")
    name, operation, amount = SHADE_VARIANTS[variant_index]
    rgb = np.asarray(to_rgb(color), dtype=float)
    if operation == "light":
        rgb = rgb + (1.0 - rgb) * amount
    elif operation == "dark":
        rgb = rgb * (1.0 - amount)
    return to_hex(np.clip(rgb, 0.0, 1.0)), name


def prediction_display_mapping(
    dataset: str,
    slice_name: str,
    method: str,
    truth: pd.Series,
    prediction: pd.Series,
    truth_palette: dict[str, str],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    truth_string = base.series_as_strings(truth)
    prediction_string = base.series_as_strings(prediction)
    truth_labels = sorted(truth_string.dropna().unique(), key=base.natural_key)
    prediction_labels = sorted(prediction_string.dropna().unique(), key=base.natural_key)
    valid = truth_string.notna() & prediction_string.notna()
    contingency = pd.crosstab(
        prediction_string.loc[valid],
        truth_string.loc[valid],
    ).reindex(index=prediction_labels, columns=truth_labels, fill_value=0)

    row_indices, column_indices = linear_sum_assignment(-contingency.to_numpy())
    assignments = {
        prediction_labels[row]: {
            "truth": truth_labels[column],
            "type": "hungarian_anchor",
            "overlap": int(contingency.iat[row, column]),
        }
        for row, column in zip(row_indices, column_indices)
    }
    for prediction_label in prediction_labels:
        if prediction_label in assignments:
            continue
        overlaps = contingency.loc[prediction_label]
        matched_truth = str(overlaps.idxmax())
        assignments[prediction_label] = {
            "truth": matched_truth,
            "type": "overlap_extra",
            "overlap": int(overlaps.loc[matched_truth]),
        }

    if dataset != "Mouse_Brain" and len(prediction_labels) > len(truth_labels):
        raise ValueError(
            f"{dataset}/{method}: expected one-to-one cluster count, got "
            f"{len(prediction_labels)} predictions but only {len(truth_labels)} truth labels"
        )

    grouped_predictions: dict[str, list[str]] = {label: [] for label in truth_labels}
    for prediction_label, assignment in assignments.items():
        grouped_predictions[str(assignment["truth"])].append(prediction_label)

    display_mapping = {}
    records = []
    for truth_label in truth_labels:
        assigned_labels = grouped_predictions[truth_label]
        anchors = [label for label in assigned_labels if assignments[label]["type"] == "hungarian_anchor"]
        extras = sorted(
            [label for label in assigned_labels if assignments[label]["type"] == "overlap_extra"],
            key=lambda label: (-int(assignments[label]["overlap"]), base.natural_key(label)),
        )
        ordered_labels = [*anchors, *extras]
        if len(ordered_labels) > len(SHADE_VARIANTS):
            raise ValueError(
                f"{dataset}/{method}/{truth_label}: {len(ordered_labels)} color variants required"
            )
        for variant_index, prediction_label in enumerate(ordered_labels):
            display_color, variant_name = shade_color(
                truth_palette[truth_label],
                variant_index,
            )
            display_mapping[prediction_label] = display_color
            records.append(
                {
                    "dataset": dataset,
                    "slice": slice_name,
                    "method": method,
                    "original_cluster": prediction_label,
                    "matched_truth_label": truth_label,
                    "assignment_type": assignments[prediction_label]["type"],
                    "overlap_spots": assignments[prediction_label]["overlap"],
                    "variant_index": variant_index,
                    "variant_name": variant_name,
                    "base_color_hex": truth_palette[truth_label],
                    "display_color_hex": display_color,
                }
            )
    if set(display_mapping) != set(prediction_labels):
        raise ValueError(f"{dataset}/{method}: incomplete display color mapping")
    return display_mapping, records


def colors_from_mapping(series: pd.Series, mapping: dict[str, str]) -> np.ndarray:
    values = base.series_as_strings(series)
    colors = np.full(len(values), base.UNLABELLED_COLOR, dtype=object)
    for label, color in mapping.items():
        colors[(values == label).to_numpy()] = color
    return colors


def save_metrics(selected, selected_metrics) -> None:
    records = []
    for dataset, config in selected.items():
        frame = selected_metrics[dataset]
        aggregate = metrics_figure.aggregate_scores(frame)
        for method in base.METHODS:
            records.append(
                {
                    "dataset": dataset,
                    "slice": config["slice"],
                    "sheet": config["sheet"],
                    "method": method,
                    **{
                        metric: float(frame.loc[method, metric])
                        for metric in metrics_figure.CORE_METRICS
                    },
                    metrics_figure.AGGREGATE_LABEL: float(aggregate.loc[method]),
                }
            )
    pd.DataFrame(records).to_csv(
        METRICS_PATH,
        index=False,
        float_format=metrics_figure.format_three_decimals,
    )


def render_dataset(
    dataset: str,
    config: dict,
    frame: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    data = ad.read_h5ad(config["path"])
    spatial_original = np.asarray(data.obsm["spatial"], dtype=float)
    if dataset == "Simulation":
        spatial_display, rotation_center = rotate_coordinates(
            spatial_original,
            SIMULATION_ROTATION_DEGREES,
        )
        rotation_degrees = SIMULATION_ROTATION_DEGREES
    else:
        spatial_display = spatial_original.copy()
        rotation_center = None
        rotation_degrees = 0.0

    truth = data.obs["truth"]
    truth_string = base.series_as_strings(truth)
    truth_labels = sorted(truth_string.dropna().unique(), key=base.natural_key)
    truth_palette = {
        label: (SIMULATION_DISPLAY_COLORS if dataset == "Simulation" else base.TRUTH_COLORS)[index]
        for index, label in enumerate(truth_labels)
    }
    ground_truth_colors = colors_from_mapping(truth, truth_palette)

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
    color_records = []
    for index, (title, column) in enumerate(PANELS):
        axis = figure.add_subplot(grid[index // 4, index % 4])
        metrics_figure.reserve_panel_title_space(axis)
        axis.set_gid(panel_group_id(dataset, title))
        if column is None:
            colors = ground_truth_colors
            labels = truth
            display_mapping = truth_palette
        else:
            prediction = data.obs[column]
            display_mapping, method_records = prediction_display_mapping(
                dataset,
                config["slice"],
                title,
                truth,
                prediction,
                truth_palette,
            )
            color_records.extend(method_records)
            colors = colors_from_mapping(prediction, display_mapping)
            labels = prediction

        base.draw_panel(
            axis,
            spatial_display,
            colors,
            title,
            base.POINT_SIZES[dataset],
            highlight=False,
        )
        panel_details[title] = {
            "cluster_count": int(labels.nunique(dropna=True)),
            "missing_count": int(labels.isna().sum()),
            "display_palette": display_mapping,
        }

    groups = dataset_group_ids(dataset)
    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.set_gid(str(groups["legend"]))
    base.draw_legend(legend_axis, truth_labels, truth_palette)

    table_axis = figure.add_subplot(grid[3, :])
    table_axis.set_gid(str(groups["table"]))
    metrics_figure.draw_metric_table(table_axis, frame)

    stem = FILE_STEMS[dataset]
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    svg_path = OUTPUT_DIR / f"{stem}_grouped.svg"
    preview_path = OUTPUT_DIR / f"{stem}_preview.png"
    add_panel_labels_to_figure(figure, dataset)
    apply_manuscript_typography(figure, dataset)
    figure.savefig(png_path, dpi=base.FIGURE_DPI, facecolor="white")
    figure.savefig(pdf_path, dpi=base.FIGURE_DPI, facecolor="white")
    figure.savefig(svg_path, dpi=base.FIGURE_DPI, facecolor="white", format="svg")
    plt.close(figure)
    svg_validation = group_and_validate_svg(svg_path, dataset)
    base.save_preview(png_path, preview_path)
    data.file.close()

    with Image.open(png_path) as image:
        png_pixels = list(image.size)
        png_dpi = list(image.info.get("dpi", (None, None)))
    expected_pixels = [
        int(base.FIGURE_WIDTH * base.FIGURE_DPI),
        int(base.FIGURE_HEIGHT * base.FIGURE_DPI),
    ]
    if png_pixels != expected_pixels:
        raise ValueError(
            f"{dataset}: unexpected PNG dimensions {png_pixels}; expected {expected_pixels}"
        )

    return (
        {
            "input_h5ad": str(config["path"]),
            "input_h5ad_sha256": sha256(config["path"]),
            "sheet": config["sheet"],
            "slice": config["slice"],
            "truth_label_count": len(truth_labels),
            "truth_missing_count": int(truth.isna().sum()),
            "panel_details": panel_details,
            "display_transform": {
                "rotation_degrees_counterclockwise": rotation_degrees,
                "rotation_center": rotation_center,
                "source_coordinates_modified": False,
            },
            "svg_validation": svg_validation,
            "outputs": {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "grouped_svg": str(svg_path),
                "preview_png": str(preview_path),
            },
            "png_pixels": png_pixels,
            "png_dpi": png_dpi,
        },
        color_records,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected, selected_metrics, metric_sources, selection_scores = metrics_figure.choose_slices()
    for config in selected.values():
        base.validate_input(config["path"])

    save_metrics(selected, selected_metrics)
    selection_scores.to_csv(SELECTION_PATH, index=False, float_format="%.6f")

    figures = {}
    color_records = []
    for dataset, config in selected.items():
        figure_details, dataset_color_records = render_dataset(
            dataset,
            config,
            selected_metrics[dataset],
        )
        figures[dataset] = figure_details
        color_records.extend(dataset_color_records)
    pd.DataFrame(color_records).to_csv(COLOR_MAPPING_PATH, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "figure_size_cm": [base.FIGURE_WIDTH_CM, base.FIGURE_HEIGHT_CM],
        "figure_dpi": base.FIGURE_DPI,
        "font": "Arial/Arial Bold; editable SVG text and PDF Type 42",
        "workbook": str(WORKBOOK),
        "workbook_sha256": sha256(WORKBOOK),
        "displayed_metrics": list(metrics_figure.CORE_METRICS),
        "rank_input": str(metrics_figure.PAPER_METRICS_PATH) + " (full precision; displayed to 3 decimals)",
        "aggregate_definition": (
            "Equal-weight mean of normalized within-slice ranks for ARI, AMI, Moran's I, PAS, CHAOS "
            "across the seven displayed methods; PAS and CHAOS are lower-is-better. "
            "Full-precision values and average-rank ties are used."
        ),
        "selection_rule": "Previously fixed illustrative slices; no reselection after metric changes.",
        "selected_slices": {
            dataset: config["slice"]
            for dataset, config in selected.items()
        },
        "method_order": ["Ground truth", *base.METHODS],
        "color_alignment": {
            "Human_lymph_node": "one-to-one Hungarian matching to truth labels",
            "Simulation": "one-to-one Hungarian matching to truth labels",
            "Mouse_Brain": (
                "Hungarian anchors for 13 truth labels; remaining clusters assigned by "
                "maximum overlap and shown with deterministic shades of the matched truth color"
            ),
            "shade_variants": [variant[0] for variant in SHADE_VARIANTS],
            "labels_modified": False,
        },
        "simulation_display_rotation_degrees_counterclockwise": SIMULATION_ROTATION_DEGREES,
        "metric_sources": metric_sources,
        "metrics_csv": str(METRICS_PATH),
        "selection_scores_csv": str(SELECTION_PATH),
        "color_mapping_csv": str(COLOR_MAPPING_PATH),
        "figures": figures,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Color mapping: {COLOR_MAPPING_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
