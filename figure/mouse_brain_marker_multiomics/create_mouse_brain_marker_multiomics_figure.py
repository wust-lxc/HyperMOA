#!/usr/bin/env python3
"""Create a marker-centered Mouse Brain downstream-analysis paper figure.

Only HyperMOA domains, supplied anatomical labels, RNA counts, and the four raw
epigenomic assays are used. No competing method is loaded. H3K4me3 and H3K27me3
are displayed as independent spatial tracks and are not claimed as modalities in
the existing two-slice HyperMOA fit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manuscript_panel_labels import add_panel_labels_to_figure
from manuscript_typography import apply_manuscript_typography

import anndata as ad
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figure" / "mouse_brain_marker_multiomics"
GTF_PATH = OUT_DIR / "annotation" / "Mus_musculus.GRCm38.102.gtf.gz"
PROCESSED_PATH = (
    ROOT
    / "HyperMOA"
    / "Mouse_brain_result"
    / "H3K27ac_ATAC"
    / "h3k27ac_atac_processed.h5ad"
)
ATAC_RNA_PATH = ROOT / 'data/datasets/Mouse_Brain/ATAC' / "adata_RNA.h5ad"

EPIGENOMIC_ASSAYS = ["ATAC", "H3K27ac", "H3K4me3", "H3K27me3"]
MIN_DOMAIN_SPOTS = 50
MIN_REPRESENTATIVE_DOMAIN_SPOTS = 100
PROMOTER_RADIUS_BP = 2000
MARKER_FDR = 0.05
MIN_MARKER_DETECTION = 0.05
N_REPRESENTATIVE_MARKERS = 3

TRUTH_ORDER = [
    "L1-L3",
    "L4",
    "L5",
    "L6a/b",
    "VL",
    "ccg/aco",
    "CP",
    "ACB",
    "1_others",
]
TRUTH_COLORS = {
    "L1-L3": "#C99182",
    "L4": "#B7A3CC",
    "L5": "#F28E2B",
    "L6a/b": "#A8B84A",
    "VL": "#F2A6C6",
    "ccg/aco": "#8C564B",
    "CP": "#72B7C7",
    "ACB": "#D84A4A",
    "1_others": "#D4D7DB",
}
EXCLUDED_GENE_PATTERN = re.compile(
    r"^(Gm|Rpl|Rps|mt-|Mir|Snor|Hist|Olfr)", flags=re.IGNORECASE
)


@dataclass
class GeneRecord:
    chromosome: str
    start: int
    end: int
    tss: int
    biotype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gtf(path: Path) -> dict[str, GeneRecord]:
    records: dict[str, GeneRecord] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attributes = dict(re.findall(r'(\S+) "([^"]+)"', fields[8]))
            name = attributes.get("gene_name")
            if not name:
                continue
            chromosome = (
                fields[0] if fields[0].startswith("chr") else f"chr{fields[0]}"
            )
            start = int(fields[3]) - 1
            end = int(fields[4])
            tss = start if fields[6] == "+" else end - 1
            biotype = attributes.get(
                "gene_biotype", attributes.get("gene_type", "")
            )
            record = GeneRecord(chromosome, start, end, tss, biotype)
            if name not in records or biotype == "protein_coding":
                records[name] = record
    return records


def log_normalize_counts(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(
        1e4,
        library_size,
        out=np.zeros(library_size.shape, dtype=np.float32),
        where=library_size > 0,
    )
    matrix = matrix.multiply(scale[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    matrix.eliminate_zeros()
    return matrix


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    result = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    values = p_values[finite]
    if values.size == 0:
        return result
    order = np.argsort(values)
    adjusted = values[order] * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(values)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def rank_delta_matrix(
    expression: sparse.csr_matrix,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Represent global ranks as zero-rank plus sparse nonzero rank deltas."""
    ranked = expression.tocsc(copy=True).astype(np.float64)
    ranked.eliminate_zeros()
    n_spots, n_genes = ranked.shape
    zero_rank = np.zeros(n_genes, dtype=float)
    tie_sum = np.zeros(n_genes, dtype=float)
    for gene_index in range(n_genes):
        start, end = ranked.indptr[gene_index : gene_index + 2]
        values = ranked.data[start:end].copy()
        n_nonzero = len(values)
        n_zero = n_spots - n_nonzero
        zero_rank[gene_index] = (n_zero + 1) / 2 if n_zero else 0.0
        if n_nonzero:
            nonzero_ranks = stats.rankdata(values, method="average") + n_zero
            ranked.data[start:end] = nonzero_ranks - zero_rank[gene_index]
            _, counts = np.unique(values, return_counts=True)
            nonzero_ties = np.sum(counts.astype(float) ** 3 - counts)
        else:
            nonzero_ties = 0.0
        zero_ties = float(n_zero**3 - n_zero) if n_zero else 0.0
        tie_sum[gene_index] = zero_ties + nonzero_ties
    return ranked, zero_rank, tie_sum


def domain_marker_statistics(
    expression: sparse.csr_matrix,
    gene_names: np.ndarray,
    domains: np.ndarray,
    truth: np.ndarray,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    n_spots, n_genes = expression.shape
    domain_counts = pd.Series(domains).value_counts().sort_index(
        key=lambda values: values.astype(int)
    )
    stable_domains = sorted(
        domain_counts[domain_counts >= MIN_DOMAIN_SPOTS].index.astype(str), key=int
    )
    total_sum = np.asarray(expression.sum(axis=0)).ravel()
    total_detected = np.asarray((expression > 0).sum(axis=0)).ravel()
    ranked, zero_rank, tie_sum = rank_delta_matrix(expression)
    excluded_names = np.asarray(
        [bool(EXCLUDED_GENE_PATTERN.match(str(gene))) for gene in gene_names]
    )
    records: list[pd.DataFrame] = []

    for domain in stable_domains:
        in_mask = domains == domain
        in_indices = np.flatnonzero(in_mask)
        n_in = len(in_indices)
        n_out = n_spots - n_in
        in_sum = np.asarray(expression[in_indices].sum(axis=0)).ravel()
        in_detected = np.asarray((expression[in_indices] > 0).sum(axis=0)).ravel()
        mean_in = in_sum / n_in
        mean_out = (total_sum - in_sum) / n_out
        pct_in = in_detected / n_in
        pct_out = (total_detected - in_detected) / n_out

        delta_rank_sum = np.asarray(ranked[in_indices].sum(axis=0)).ravel()
        rank_sum = n_in * zero_rank + delta_rank_sum
        u_statistic = rank_sum - n_in * (n_in + 1) / 2
        expected_u = n_in * n_out / 2
        tie_adjustment = tie_sum / (n_spots * (n_spots - 1))
        variance = n_in * n_out / 12 * (n_spots + 1 - tie_adjustment)
        z_score = np.divide(
            u_statistic - expected_u - 0.5,
            np.sqrt(variance),
            out=np.zeros(n_genes, dtype=float),
            where=variance > 0,
        )
        p_value = stats.norm.sf(z_score)
        fdr = benjamini_hochberg(p_value)
        mean_difference = mean_in - mean_out
        detection_difference = pct_in - pct_out
        marker_score = (
            mean_difference
            * np.sqrt(np.maximum(pct_in, 0))
            * np.maximum(detection_difference + 0.05, 0.05)
        )

        domain_truth = pd.Series(truth[in_mask])
        domain_truth = domain_truth[domain_truth != "nan"]
        majority_truth = (
            str(domain_truth.value_counts().index[0]) if len(domain_truth) else "nan"
        )
        purity = (
            float(domain_truth.value_counts().iloc[0] / n_in)
            if len(domain_truth)
            else np.nan
        )
        records.append(
            pd.DataFrame(
                {
                    "domain": domain,
                    "domain_n_spots": n_in,
                    "majority_truth": majority_truth,
                    "truth_purity": purity,
                    "gene": gene_names,
                    "mean_in": mean_in,
                    "mean_out": mean_out,
                    "mean_difference": mean_difference,
                    "pct_in": pct_in,
                    "pct_out": pct_out,
                    "detection_difference": detection_difference,
                    "u_statistic": u_statistic,
                    "z_score": z_score,
                    "p_value": p_value,
                    "fdr": fdr,
                    "marker_score": marker_score,
                    "excluded_name_pattern": excluded_names,
                }
            )
        )
    return pd.concat(records, ignore_index=True), stable_domains, domain_counts


def select_unique_domain_markers(
    marker_statistics: pd.DataFrame, stable_domains: list[str]
) -> pd.DataFrame:
    eligible = marker_statistics[
        (marker_statistics["fdr"] < MARKER_FDR)
        & (marker_statistics["mean_difference"] > 0)
        & (marker_statistics["pct_in"] >= MIN_MARKER_DETECTION)
        & ~marker_statistics["excluded_name_pattern"]
    ].copy()
    eligible = eligible.sort_values("marker_score", ascending=False)
    selected_rows: dict[str, pd.Series] = {}
    used_genes: set[str] = set()
    domain_order = (
        eligible.groupby("domain")["marker_score"]
        .max()
        .sort_values(ascending=False)
        .index.astype(str)
    )
    for domain in domain_order:
        candidates = eligible[eligible["domain"].astype(str) == domain]
        for _, row in candidates.iterrows():
            gene = str(row["gene"])
            if gene not in used_genes:
                selected_rows[domain] = row
                used_genes.add(gene)
                break
    missing = set(stable_domains) - set(selected_rows)
    if missing:
        raise ValueError(f"No unique significant marker for domains: {sorted(missing)}")
    selected = pd.DataFrame([selected_rows[domain] for domain in stable_domains])
    selected["heatmap_order"] = np.arange(len(selected))
    return selected.reset_index(drop=True)


def select_representative_markers(selected_markers: pd.DataFrame) -> pd.DataFrame:
    candidates = selected_markers[
        (selected_markers["domain_n_spots"] >= MIN_REPRESENTATIVE_DOMAIN_SPOTS)
        & (selected_markers["majority_truth"] != "1_others")
        & (selected_markers["majority_truth"] != "nan")
        & (selected_markers["truth_purity"] >= 0.5)
    ].copy()
    candidates["representative_score"] = (
        candidates["marker_score"] * candidates["truth_purity"]
    )
    candidates = candidates.sort_values("representative_score", ascending=False)
    selected: list[pd.Series] = []
    used_regions: set[str] = set()
    for _, row in candidates.iterrows():
        region = str(row["majority_truth"])
        if region in used_regions:
            continue
        selected.append(row)
        used_regions.add(region)
        if len(selected) == N_REPRESENTATIVE_MARKERS:
            break
    if len(selected) != N_REPRESENTATIVE_MARKERS:
        raise ValueError("Unable to select three markers from distinct anatomical regions.")
    return pd.DataFrame(selected).reset_index(drop=True)


def gene_activity_peaks_for_gene(
    gene: str,
    peak_names: np.ndarray,
    annotation: dict[str, GeneRecord],
) -> np.ndarray:
    if gene not in annotation:
        return np.array([], dtype=int)
    record = annotation[gene]
    lower = record.start - PROMOTER_RADIUS_BP
    upper = record.end + PROMOTER_RADIUS_BP
    matches: list[int] = []
    prefix = f"{record.chromosome}-"
    for index, peak in enumerate(peak_names):
        peak = str(peak)
        if not peak.startswith(prefix):
            continue
        try:
            chromosome, start, end = peak.rsplit("-", 2)
        except ValueError:
            continue
        if chromosome == record.chromosome and int(start) <= upper and int(end) >= lower:
            matches.append(index)
    return np.asarray(matches, dtype=int)


def choose_multimodal_gene(
    representatives: pd.DataFrame,
    selected_markers: pd.DataFrame,
    annotation: dict[str, GeneRecord],
) -> tuple[str, dict[str, np.ndarray]]:
    ordered_candidates = list(representatives["gene"].astype(str))
    ordered_candidates.extend(
        gene
        for gene in selected_markers.sort_values("marker_score", ascending=False)[
            "gene"
        ].astype(str)
        if gene not in ordered_candidates
    )
    for gene in ordered_candidates:
        assay_peaks: dict[str, np.ndarray] = {}
        complete = True
        for assay in EPIGENOMIC_ASSAYS:
            path = ROOT / f"data/datasets/Mouse_Brain/{assay}" / "adata_peaks_normalized.h5ad"
            peaks = ad.read_h5ad(path, backed="r")
            indices = gene_activity_peaks_for_gene(
                gene, np.asarray(peaks.var_names.astype(str)), annotation
            )
            assay_peaks[assay] = indices
            if len(indices) == 0:
                complete = False
                break
        if complete:
            return gene, assay_peaks
    raise ValueError("No selected marker has promoter peaks in all four assays.")


def domain_truth_composition(
    domains: np.ndarray, truth: np.ndarray, actual_domains: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = pd.crosstab(
        pd.Categorical(domains, categories=actual_domains, ordered=True),
        pd.Categorical(truth, categories=TRUTH_ORDER, ordered=True),
        dropna=False,
    )
    count.index = actual_domains
    count.columns = TRUTH_ORDER
    proportion = count.div(count.sum(axis=1), axis=0).fillna(0)
    return count, proportion


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standard_deviation = np.nanstd(values)
    if standard_deviation == 0 or not np.isfinite(standard_deviation):
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / standard_deviation


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 6.4,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def domain_palette() -> dict[str, tuple[float, float, float, float]]:
    colors = mpl.colormaps["tab20"](np.linspace(0.02, 0.98, 18))
    return {str(index + 1): colors[index] for index in range(18)}


def style_spatial_axis(axis: plt.Axes, border: bool = True) -> None:
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.margins(0.02)
    if border:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#32363B")
            spine.set_linewidth(0.65)
    else:
        for spine in axis.spines.values():
            spine.set_visible(False)


def plot_truth_overview(
    axis: plt.Axes, coordinates: np.ndarray, truth: np.ndarray
) -> None:
    colors = [TRUTH_COLORS.get(label, "#ECEDEF") for label in truth]
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=colors,
        s=3.5,
        marker="o",
        linewidths=0,
        alpha=0.98,
        rasterized=True,
    )
    for region in TRUTH_ORDER[:-1]:
        mask = truth == region
        if mask.sum() == 0:
            continue
        center = np.median(coordinates[mask], axis=0)
        text = axis.text(
            center[0],
            center[1],
            region,
            ha="center",
            va="center",
            fontsize=5.1,
            weight="semibold",
            color="#20242A",
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=1.0, foreground="white")]
        )
    axis.set_title("Anatomical annotation", weight="semibold", pad=4)
    axis.invert_yaxis()
    style_spatial_axis(axis, border=False)


def plot_truth_mask(
    axis: plt.Axes,
    coordinates: np.ndarray,
    truth: np.ndarray,
    region: str,
) -> None:
    mask = truth == region
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c="#E2E3E5",
        s=1.1,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        coordinates[mask, 0],
        coordinates[mask, 1],
        c=TRUTH_COLORS[region],
        s=1.45,
        linewidths=0,
        rasterized=True,
    )
    axis.set_title(f"{region}  n={mask.sum():,}", fontsize=5.8, pad=1.5)
    axis.invert_yaxis()
    style_spatial_axis(axis)


def plot_truth_legend(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.05, 0.98, "Region colors", weight="bold", fontsize=6.0, va="top")
    for index, region in enumerate(TRUTH_ORDER):
        column = index // 5
        row = index % 5
        x = 0.07 + 0.48 * column
        y = 0.72 - 0.14 * row
        axis.scatter(x, y, s=18, color=TRUTH_COLORS[region], edgecolor="none")
        axis.text(x + 0.08, y, region, va="center", fontsize=5.2)


def plot_hypermoa_domains(
    axis: plt.Axes,
    coordinates: np.ndarray,
    domains: np.ndarray,
    palette: dict[str, object],
) -> None:
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=[palette[domain] for domain in domains],
        s=3.3,
        marker="o",
        linewidths=0,
        alpha=0.98,
        rasterized=True,
    )
    for domain in sorted(np.unique(domains), key=int):
        mask = domains == domain
        center = np.median(coordinates[mask], axis=0)
        text = axis.text(
            center[0],
            center[1],
            domain,
            ha="center",
            va="center",
            fontsize=5.2,
            weight="semibold",
            color="white",
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=0.95, foreground="#252A30")]
        )
    axis.set_title("HyperMOA domains", weight="semibold", pad=4)
    axis.invert_yaxis()
    style_spatial_axis(axis, border=False)


def plot_composition_heatmap(axis: plt.Axes, proportion: pd.DataFrame) -> None:
    values = proportion.to_numpy() * 100
    image = axis.pcolormesh(
        values,
        cmap="Blues",
        vmin=0,
        vmax=100,
        shading="flat",
        edgecolors="white",
        linewidth=0.45,
        antialiased=False,
    )
    axis.set_xlim(0, values.shape[1])
    axis.set_ylim(values.shape[0], 0)
    axis.set_xticks(
        np.arange(len(TRUTH_ORDER)) + 0.5,
        TRUTH_ORDER,
        rotation=42,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(proportion.index)) + 0.5,
        [f"D{domain}" for domain in proportion.index],
    )
    axis.tick_params(length=0, pad=1.5)
    axis.set_title("Anatomical composition of each domain (%)", weight="bold", pad=3)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.026, pad=0.022)
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.tick_params(labelsize=5.2, length=2, pad=1.5)


def expression_vector(
    expression: sparse.csr_matrix, gene_names: np.ndarray, gene: str
) -> np.ndarray:
    matches = np.flatnonzero(gene_names == gene)
    if len(matches) != 1:
        raise KeyError(f"Expected one RNA column for {gene}, found {len(matches)}")
    return np.asarray(expression[:, matches[0]].toarray()).ravel()


def plot_expression_map(
    axis: plt.Axes,
    coordinates: np.ndarray,
    values: np.ndarray,
    gene: str,
    domain: str,
    region: str,
) -> None:
    upper = float(np.quantile(values[np.isfinite(values)], 0.99))
    upper = max(upper, 1e-6)
    clipped = np.clip(values, 0, upper)
    order = np.argsort(clipped)
    scatter = axis.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=clipped[order],
        cmap="viridis",
        vmin=0,
        vmax=upper,
        s=1.35,
        linewidths=0,
        rasterized=True,
    )
    axis.set_title(f"{gene}  |  D{domain}, {region}", fontstyle="italic", pad=2)
    axis.invert_yaxis()
    style_spatial_axis(axis)
    colorbar = axis.figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.018)
    colorbar.ax.tick_params(labelsize=5.1, length=1.5)


def plot_domain_violin(
    axis: plt.Axes,
    values: np.ndarray,
    domains: np.ndarray,
    actual_domains: list[str],
    palette: dict[str, object],
    gene: str,
    show_x_axis: bool,
) -> None:
    distributions = [values[domains == domain] for domain in actual_domains]
    positions = np.arange(1, len(actual_domains) + 1)
    violin = axis.violinplot(
        distributions,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=80,
    )
    for body, domain in zip(violin["bodies"], actual_domains):
        body.set_facecolor(palette[domain])
        body.set_edgecolor("#34383D")
        body.set_linewidth(0.35)
        body.set_alpha(0.92)
    medians = [np.median(values[domains == domain]) for domain in actual_domains]
    axis.scatter(positions, medians, s=5, c="#20242A", linewidths=0, zorder=3)
    axis.set_xticks(positions, actual_domains if show_x_axis else [""] * len(actual_domains))
    axis.set_xlabel("HyperMOA domain" if show_x_axis else "")
    axis.set_ylabel("")
    axis.set_title(f"{gene} expression across domains", fontstyle="italic", pad=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E7E9EC", linewidth=0.45)
    axis.set_axisbelow(True)
    upper = np.quantile(values, 0.997)
    axis.set_ylim(-0.05, max(0.5, upper * 1.05))


def plot_marker_heatmap(
    axis: plt.Axes,
    selected_markers: pd.DataFrame,
    expression: sparse.csr_matrix,
    gene_names: np.ndarray,
    domains: np.ndarray,
    stable_domains: list[str],
) -> pd.DataFrame:
    genes = selected_markers["gene"].astype(str).tolist()
    matrix = np.zeros((len(genes), len(stable_domains)), dtype=float)
    rows: list[dict[str, object]] = []
    for row_index, gene in enumerate(genes):
        values = expression_vector(expression, gene_names, gene)
        means = np.asarray(
            [values[domains == domain].mean() for domain in stable_domains]
        )
        matrix[row_index] = standardized(means)
        for domain, mean_value, z_value in zip(stable_domains, means, matrix[row_index]):
            rows.append(
                {
                    "marker_domain": selected_markers.iloc[row_index]["domain"],
                    "gene": gene,
                    "measured_domain": domain,
                    "mean_log1p_cp10k": mean_value,
                    "scaled_expression": z_value,
                }
            )
    image = axis.pcolormesh(
        matrix,
        cmap="GnBu",
        vmin=-1.5,
        vmax=3.0,
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
        antialiased=False,
    )
    axis.set_xlim(0, matrix.shape[1])
    axis.set_ylim(matrix.shape[0], 0)
    axis.set_xticks(
        np.arange(len(stable_domains)) + 0.5,
        [f"D{domain}" for domain in stable_domains],
        rotation=90,
    )
    axis.set_yticks(
        np.arange(len(genes)) + 0.5,
        [
            f"D{row.domain}  {row.gene}"
            for row in selected_markers[["domain", "gene"]].itertuples(index=False)
        ],
        fontstyle="italic",
        fontsize=5.2,
    )
    axis.tick_params(axis="x", length=0, pad=1.5)
    axis.tick_params(axis="y", length=0, pad=0.5)
    axis.set_title("Domain-associated marker expression", weight="bold", pad=3)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.018)
    colorbar.ax.tick_params(labelsize=5.2, length=2)
    colorbar.ax.set_ylabel("Scaled expression", fontsize=5.2)
    return pd.DataFrame(rows)


def plot_multimodal_map(
    axis: plt.Axes,
    coordinates: np.ndarray,
    values: np.ndarray,
    title: str,
) -> None:
    z_values = np.clip(standardized(values), -2, 3)
    order = np.argsort(z_values)
    scatter = axis.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=z_values[order],
        cmap="viridis",
        vmin=-2,
        vmax=3,
        s=1.15,
        linewidths=0,
        rasterized=True,
    )
    axis.set_title(title, fontsize=6.3, pad=2, weight="bold")
    axis.invert_yaxis()
    style_spatial_axis(axis)
    colorbar = axis.figure.colorbar(scatter, ax=axis, fraction=0.041, pad=0.015)
    colorbar.ax.tick_params(labelsize=5.1, length=1.5)
    colorbar.ax.set_ylabel("z-score", fontsize=5.2, labelpad=1)


def load_multimodal_tracks(
    gene: str,
    assay_peak_indices: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    tracks: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    atac_rna = ad.read_h5ad(ATAC_RNA_PATH)
    atac_rna_normalized = log_normalize_counts(atac_rna.X)
    gene_index = int(atac_rna.var_names.get_loc(gene))
    rna_values = np.asarray(atac_rna_normalized[:, gene_index].toarray()).ravel()
    tracks.append(
        {
            "track": "RNA",
            "title": f"RNA: {gene}",
            "coordinates": np.asarray(atac_rna.obsm["spatial"], dtype=float),
            "values": rna_values,
        }
    )
    summary_rows.append(
        {
            "track": "RNA",
            "gene": gene,
            "source_slice": "Mouse_Brain_ATAC",
                "n_spots": atac_rna.n_obs,
            "n_gene_window_peaks": np.nan,
            "mean_signal": float(np.mean(rna_values)),
            "sd_signal": float(np.std(rna_values)),
        }
    )

    for assay in EPIGENOMIC_ASSAYS:
        path = ROOT / f"data/datasets/Mouse_Brain/{assay}" / "adata_peaks_normalized.h5ad"
        adata = ad.read_h5ad(path)
        indices = assay_peak_indices[assay]
        values = np.asarray(adata.X[:, indices].sum(axis=1)).ravel()
        tracks.append(
            {
                "track": assay,
                "title": f"{assay}: {gene}",
                "coordinates": np.asarray(adata.obsm["spatial"], dtype=float),
                "values": values,
            }
        )
        summary_rows.append(
            {
                "track": assay,
                "gene": gene,
                "source_slice": f"Mouse_Brain_{assay}",
                "n_spots": adata.n_obs,
                "n_gene_window_peaks": len(indices),
                "gene_window_peaks": ";".join(adata.var_names[indices].astype(str)),
                "mean_signal": float(np.mean(values)),
                "sd_signal": float(np.std(values)),
            }
        )
    return tracks, pd.DataFrame(summary_rows)


def force_svg_size(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r'<svg ([^>]*?)width="[^"]+"', r'<svg \1width="18cm"', text, count=1)
    text = re.sub(r'height="[^"]+"', 'height="19cm"', text, count=1)
    svg_path.write_text(text, encoding="utf-8")


def create_figure(
    output_dir: Path,
    coordinates: np.ndarray,
    truth: np.ndarray,
    domains: np.ndarray,
    expression: sparse.csr_matrix,
    gene_names: np.ndarray,
    actual_domains: list[str],
    stable_domains: list[str],
    composition: pd.DataFrame,
    selected_markers: pd.DataFrame,
    representatives: pd.DataFrame,
) -> tuple[dict[str, Path], pd.DataFrame]:
    configure_matplotlib()
    palette = domain_palette()
    figure = plt.figure(figsize=(18 / 2.54, 19 / 2.54))
    outer = figure.add_gridspec(
        3,
        1,
        height_ratios=[1.25, 1.02, 1.72],
        left=0.045,
        right=0.95,
        bottom=0.035,
        top=0.975,
        hspace=0.24,
    )

    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 2.65], wspace=0.15)
    truth_axis = figure.add_subplot(top[0, 0])
    truth_axis.set_gid("anatomical_annotation")
    plot_truth_overview(truth_axis, coordinates, truth)
    masks = top[0, 1].subgridspec(2, 5, wspace=0.14, hspace=0.24)
    for index, region in enumerate(TRUTH_ORDER):
        axis = figure.add_subplot(masks[index // 5, index % 5])
        axis.set_gid(f"anatomical_mask_{region.replace('/', '_').replace('-', '_')}")
        plot_truth_mask(axis, coordinates, truth, region)
    legend_axis = figure.add_subplot(masks[1, 4])
    legend_axis.set_gid("anatomical_region_legend")
    plot_truth_legend(legend_axis)

    middle = outer[1].subgridspec(1, 2, width_ratios=[0.86, 2.15], wspace=0.18)
    domain_axis = figure.add_subplot(middle[0, 0])
    domain_axis.set_gid("hypermoa_domain_map")
    plot_hypermoa_domains(domain_axis, coordinates, domains, palette)
    composition_axis = figure.add_subplot(middle[0, 1])
    composition_axis.set_gid("domain_anatomical_composition")
    plot_composition_heatmap(composition_axis, composition)

    marker_section = outer[2].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.36)
    marker_profiles = marker_section[0, 0].subgridspec(
        3, 2, width_ratios=[0.72, 1.75], hspace=0.34, wspace=0.20
    )
    for row_index, marker in representatives.iterrows():
        gene = str(marker["gene"])
        values = expression_vector(expression, gene_names, gene)
        map_axis = figure.add_subplot(marker_profiles[row_index, 0])
        violin_axis = figure.add_subplot(marker_profiles[row_index, 1])
        violin_position = violin_axis.get_position()
        violin_axis.set_position(
            [
                violin_position.x0 + 0.008,
                violin_position.y0,
                violin_position.width,
                violin_position.height,
            ]
        )
        map_axis.set_gid(f"marker_spatial_{gene}")
        violin_axis.set_gid(f"marker_violin_{gene}")
        plot_expression_map(
            map_axis,
            coordinates,
            values,
            gene,
            str(marker["domain"]),
            str(marker["majority_truth"]),
        )
        plot_domain_violin(
            violin_axis,
            values,
            domains,
            actual_domains,
            palette,
            gene,
            show_x_axis=row_index == len(representatives) - 1,
        )
    marker_heatmap_axis = figure.add_subplot(marker_section[0, 1])
    marker_heatmap_axis.set_gid("domain_marker_expression_heatmap")
    heatmap_values = plot_marker_heatmap(
        marker_heatmap_axis,
        selected_markers,
        expression,
        gene_names,
        domains,
        stable_domains,
    )

    paths = {
        "png": output_dir / "mouse_brain_domain_marker_downstream_18x19cm_600dpi.png",
        "pdf": output_dir / "mouse_brain_domain_marker_downstream_18x19cm.pdf",
        "svg": output_dir / "mouse_brain_domain_marker_downstream_18x19cm_grouped.svg",
        "preview": output_dir / "mouse_brain_domain_marker_downstream_preview.png",
    }
    add_panel_labels_to_figure(figure, "marker")
    apply_manuscript_typography(figure, "marker")
    figure.savefig(paths["png"], dpi=600, facecolor="white")
    figure.savefig(paths["pdf"], dpi=600, facecolor="white")
    figure.savefig(paths["svg"], facecolor="white")
    figure.savefig(paths["preview"], dpi=150, facecolor="white")
    plt.close(figure)
    force_svg_size(paths["svg"])
    return paths, heatmap_values


def write_caption(
    output_dir: Path,
    representatives: pd.DataFrame,
    actual_domains: list[str],
    stable_domains: list[str],
) -> None:
    marker_text = ", ".join(
        f"{row.gene} (D{row.domain}, {row.majority_truth})"
        for row in representatives.itertuples(index=False)
    )
    caption = f"""Marker-centered downstream analysis of the Mouse Brain ATAC slice. Anatomical labels supplied with the dataset are shown as a combined map and individual region masks. HyperMOA produced 18 global domains, of which {len(actual_domains)} occur in this slice. The anatomical-composition matrix reports the percentage of labeled spots in each HyperMOA domain assigned to each supplied region. Domain markers were identified from log1p counts-per-10,000 RNA values using an asymptotic one-sided Mann-Whitney/Wilcoxon rank-sum test with tie and continuity correction, followed by Benjamini-Hochberg correction across all genes within each domain. Domains with fewer than {MIN_DOMAIN_SPOTS} spots were retained in spatial displays but excluded from marker testing; {len(stable_domains)} domains were tested. The automatically selected markers from distinct anatomical regions were {marker_text}. Spatial maps and violin plots show marker expression, and the marker heatmap shows domain-standardized mean expression."""
    (output_dir / "suggested_figure_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [PROCESSED_PATH, ATAC_RNA_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    print("Loading ATAC RNA and HyperMOA domains...")
    rna = ad.read_h5ad(ATAC_RNA_PATH)
    processed = ad.read_h5ad(PROCESSED_PATH, backed="r")
    processed_keys = [f"ATAC_{barcode}" for barcode in rna.obs_names]
    domains = processed.obs.loc[processed_keys, "Domain"].astype(str).to_numpy()
    truth = rna.obs["truth"].astype(str).to_numpy()
    coordinates = np.asarray(rna.obsm["spatial"], dtype=float)
    gene_names = np.asarray(rna.var_names.astype(str))
    expression = log_normalize_counts(rna.X)
    actual_domains = sorted(np.unique(domains), key=int)

    print("Calculating all-gene domain marker statistics...")
    marker_statistics, stable_domains, domain_counts = domain_marker_statistics(
        expression, gene_names, domains, truth
    )
    marker_statistics.to_csv(
        output_dir / "domain_marker_statistics_all_genes.csv.gz",
        index=False,
        compression="gzip",
    )
    selected_markers = select_unique_domain_markers(
        marker_statistics, stable_domains
    )
    selected_markers.to_csv(output_dir / "selected_domain_markers.csv", index=False)
    representatives = select_representative_markers(selected_markers)
    representatives.to_csv(output_dir / "representative_markers.csv", index=False)
    print(
        "  Representatives: "
        + ", ".join(
            f"{row.gene} (D{row.domain}, {row.majority_truth})"
            for row in representatives.itertuples(index=False)
        )
    )

    print("Preparing anatomical composition...")
    count, composition = domain_truth_composition(domains, truth, actual_domains)
    count.to_csv(output_dir / "domain_truth_counts.csv")
    composition.to_csv(output_dir / "domain_truth_proportions.csv")
    domain_counts.rename("n_spots").to_csv(output_dir / "domain_spot_counts.csv")

    print("Rendering 18 x 19 cm paper figure...")
    paths, heatmap_values = create_figure(
        output_dir,
        coordinates,
        truth,
        domains,
        expression,
        gene_names,
        actual_domains,
        stable_domains,
        composition,
        selected_markers,
        representatives,
    )
    heatmap_values.to_csv(output_dir / "marker_heatmap_values.csv", index=False)
    write_caption(
        output_dir,
        representatives,
        actual_domains,
        stable_domains,
    )

    input_paths = [
        ATAC_RNA_PATH,
        PROCESSED_PATH,
    ]
    manifest = {
        "analysis": "Mouse Brain marker-centered multiomics downstream analysis",
        "no_competing_methods_used": True,
        "histology_image_used": False,
        "figure_size_cm": [18, 19],
        "parameters": {
            "rna_normalization": "log1p(counts / library_size * 10000)",
            "marker_test": "one-sided asymptotic Mann-Whitney/Wilcoxon rank-sum",
            "tie_correction": True,
            "continuity_correction": True,
            "multiple_testing": "Benjamini-Hochberg across all RNA genes within each domain",
            "marker_fdr": MARKER_FDR,
            "minimum_domain_spots_for_marker_test": MIN_DOMAIN_SPOTS,
            "minimum_representative_domain_spots": MIN_REPRESENTATIVE_DOMAIN_SPOTS,
            "representative_rule": "top marker_score * truth_purity from three distinct non-other anatomical regions",
        },
        "counts": {
            "global_requested_domains": 18,
            "domains_present_in_atac_slice": len(actual_domains),
            "domains_used_for_marker_testing": len(stable_domains),
            "spots": int(rna.n_obs),
            "rna_genes_tested_per_domain": int(rna.n_vars),
        },
        "representative_markers": representatives[
            ["domain", "majority_truth", "truth_purity", "gene", "fdr", "marker_score"]
        ].to_dict(orient="records"),
        "inputs": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in input_paths
        ],
        "outputs": {key: str(path) for key, path in paths.items()},
        "interpretation_limits": [
            "No histology image was available; supplied truth labels were used.",
            "The anatomical-composition heatmap is row-normalized domain composition, not classification accuracy.",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
