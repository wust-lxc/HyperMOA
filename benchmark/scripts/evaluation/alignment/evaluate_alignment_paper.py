#!/usr/bin/env python3
"""Evaluate directed adjacent-slice alignments with one protocol for every method."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))


import hashlib
import json
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from benchmark.scripts.common import fair_benchmark_common as common
from benchmark.scripts.evaluation.alignment.evaluate_fair_alignment import hypermoa_coordinates, load_raw_slices, pcc_on_common_rna
from benchmark.scripts.evaluation.alignment.evaluate_mouse_region_consistency import compute_region_consistency

ROOT = common.ROOT
OUTPUT = ROOT / "data/evaluation/fair_benchmark"
METHODS = common.METHODS
MAIN_METRICS = {
    "Human_lymph_node": ("Chamfer Distance", "PCC", "CI", "Macro-F1", "Balanced Accuracy"),
    "Simulation": ("PAA", "FOSCTTM", "Chamfer Distance", "CI", "PCC", "SMR"),
    "Mouse_Brain": ("Region ARI", "Region AMI", "Chamfer Distance", "PCC"),
}




def normalized_labels(labels: np.ndarray, dataset: str) -> np.ndarray:
    values = np.array([str(value).strip().lower() if not pd.isna(value) else "" for value in labels])
    if dataset == "Human_lymph_node":
        values[values == "medulla vessels"] = "medulla vessel"
    return values


def pair_result(spec: common.DatasetSpec, method: str, source, target,
                source_xy: np.ndarray, target_xy: np.ndarray, n_genes: int) -> dict:
    predicted = common.nearest_match(source_xy, target_xy)
    spacing = common.median_nn_radius(target_xy)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Invalid target spacing: {spec.name}/{method}/{target.name}")
    raw_chamfer = common.chamfer_distance(source_xy, target_xy)
    pcc, n_pcc = pcc_on_common_rna(source, target, predicted)
    row = {
        "dataset": spec.name, "method": method,
        "pair": f"{source.name}_to_{target.name}",
        "Chamfer Distance": raw_chamfer / spacing,
        "Chamfer_raw": raw_chamfer, "target_median_NN": spacing,
        "PCC": pcc, "n_pcc_pairs": n_pcc,
        "n_source": len(source.obs_names), "n_target": len(target.obs_names),
        "n_common_RNA_genes": n_genes,
    }
    if spec.name != "Mouse_Brain":
        source_labels = normalized_labels(source.labels, spec.name)
        target_labels = normalized_labels(target.labels, spec.name)[predicted]
        valid = common.valid_labels(source_labels) & common.valid_labels(target_labels)
        row["n_valid_label_pairs"] = int(valid.sum())
        row["CI"] = float(np.mean(source_labels[valid] == target_labels[valid])) if valid.any() else np.nan
        if spec.name == "Human_lymph_node":
            row["n_balanced_accuracy_labels"] = len(set(source_labels[valid]))
            row["Balanced Accuracy"] = float(balanced_accuracy_score(
                source_labels[valid], target_labels[valid], adjusted=False)) if valid.any() else np.nan
            labels = sorted(set(source_labels[valid]) | set(target_labels[valid]))
            row["n_macro_f1_labels"] = len(labels)
            row["Macro-F1"] = float(f1_score(source_labels[valid], target_labels[valid],
                                              labels=labels, average="macro", zero_division=0)) if labels else np.nan
    if spec.name in ("Simulation", "Mouse_Brain"):
        true_target = target.obs_names.get_indexer(source.obs_names)
        known = true_target >= 0
        row["n_grid_pairs"] = int(known.sum())
        if not known.any():
            raise ValueError(f"No shared grid IDs: {spec.name}/{source.name}")
        row["PAA"] = float(np.mean(predicted[known] == true_target[known]))
        error = np.linalg.norm(source_xy[known] - target_xy[true_target[known]], axis=1)
        row["SMR"] = float(np.mean(error <= spacing))
        row["FOSCTTM"] = common.foscttm_chunked(source_xy[known], target_xy, true_target[known])
    return row


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    input_files = {}
    gene_records = {}
    for spec in common.SPECS.values():
        slices, genes = load_raw_slices(spec)
        gene_records[spec.name] = {"count": len(genes), "sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest()}
        input_files[spec.name] = {}
        hyper = hypermoa_coordinates(spec, slices)
        for method in METHODS:
            input_files[spec.name][method] = {}
            for source_name, target_name in zip(spec.slices[:-1], spec.slices[1:]):
                source, target = slices[source_name], slices[target_name]
                if method == "HyperMOA":
                    source_xy, target_xy = hyper[source_name], hyper[target_name]
                    path = common.HYPERMOA_SOURCE_FILES[spec.name]
                else:
                    path = spec.result_root / method / "pair_checkpoints" / f"{source_name}_to_{target_name}.npz"
                    with np.load(path) as saved:
                        source_xy = np.asarray(saved["aligned_source"], dtype=np.float64)
                    target_xy = target.xy
                input_files[spec.name][method][f"{source_name}_to_{target_name}"] = str(path)
                row = pair_result(spec, method, source, target,
                                  source_xy, target_xy, len(genes))
                rows.append(row)
                print(f"[paper alignment] {spec.name}/{method}/{source_name}->{target_name}", flush=True)
        del slices
    pairs = pd.DataFrame(rows)
    region, region_protocol = compute_region_consistency()
    for method in METHODS:
        mask = pairs.dataset.eq('Mouse_Brain') & pairs.method.eq(method)
        row = region.loc[region.method.eq(method)].iloc[0]
        for column in ['Region ARI', 'Region AMI', 'n_region_valid_spots']:
            pairs.loc[mask, column] = row[column]
    region.to_csv(OUTPUT / 'alignment_mouse_region_consistency.csv', index=False, float_format='%.9f')
    (OUTPUT / 'alignment_mouse_region_manifest.json').write_text(
        json.dumps(region_protocol, indent=2), encoding='utf-8')
    summary = pairs.groupby(["dataset", "method"], sort=False).mean(numeric_only=True).reset_index()
    main_rows = []
    for dataset, metrics in MAIN_METRICS.items():
        for method in METHODS:
            record = summary.loc[(summary.dataset == dataset) & (summary.method == method)].iloc[0]
            main_rows.append({"dataset": dataset, "method": method, **{name: record[name] for name in metrics}})
    main = pd.DataFrame(main_rows)
    proxy = summary.loc[summary.dataset == "Mouse_Brain", ["dataset", "method", "PAA", "SMR", "FOSCTTM", "n_grid_pairs"]].copy()
    proxy.insert(2, "reference", "shared_grid_coordinate_id_not_verified_biological_spot")
    manifest = {
        "main_metrics_by_dataset": MAIN_METRICS,
        "evaluation_unit": "directed adjacent slice pair; unweighted arithmetic mean across pairs",
        "coordinates": "HyperMOA transform_fine; other methods' saved pairwise aligned_source in their raw target-slice coordinate frame",
        "matches": "same nearest-neighbor rule in paired target coordinates for every method; ignore saved method-native match indices",
        "Chamfer Distance": "symmetric mean nearest-neighbor Euclidean distance divided by target slice median nearest-neighbor distance; dimensionless; direction lower",
        "PCC": "mean per-spot Pearson on dataset-wide common raw RNA genes after CP10K+log1p; nearest aligned target; direction higher",
        "CI": "fraction of nearest aligned pairs with identical valid truth labels; Human lowercase and medulla vessels/vessel alias; unavailable for Mouse due to incompatible ontologies",
        "Macro-F1": "Human only: unweighted mean of exact-label F1 over labels in valid nearest-neighbor pairs; same valid spots and aliases as CI; pair scores averaged without weights; direction higher",
        "PAA": "exact shared-ID nearest-neighbor recovery; Simulation true pairs; Mouse only shared grid-ID proxy",
        "Balanced Accuracy": "Human only: mean recall across true source classes in valid nearest-neighbor pairs; adjusted=False; same valid spots and aliases as CI; unweighted mean of adjacent pair scores; direction higher; anatomical label-transfer consistency, not exact-spot accuracy",
        "SMR": "fraction of shared-ID errors within target median NN radius; Simulation true pairs; Mouse only grid proxy",
        "FOSCTTM": "fraction of other target spots closer than shared-ID partner; Simulation true pairs; Mouse only grid proxy",
        "Region ARI": "ARI of supplied ATAC labels and nearest registered H3K27ac labels on the common valid-label intersection across all methods",
        "Region AMI": "AMI of the same supplied partitions and common spots",
        "mouse_proxy_warning": "Shared grid IDs are not verified biological spot pairs; grid-based proxies are supplemental and are not mouse main metrics",
        "mouse_region_protocol": region_protocol,
        "genes": gene_records, "coordinate_inputs": input_files,
        "algorithm_runs_repeated": False,
    }
    return pairs, main, proxy, manifest


def mouse_score_sensitivity(main_table: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    mouse = main_table.loc[main_table.dataset == "Mouse_Brain"].set_index("method").loc[list(METHODS)]
    old_proxy = proxy.set_index('method').loc[list(METHODS)]
    for name in ['PAA', 'FOSCTTM']:
        mouse[name] = old_proxy[name]

    def score(metrics: tuple[str, ...]) -> pd.Series:
        rank_scores = pd.DataFrame({
            metric: 1.0 - (mouse[metric].rank(
                method="average", ascending=metric in ("Chamfer Distance", "FOSCTTM")
            ) - 1.0) / (len(METHODS) - 1)
            for metric in metrics
        })
        return rank_scores.mean(axis=1)

    two = score(("Chamfer Distance", "PCC"))
    four = score(("PAA", "FOSCTTM", "Chamfer Distance", "PCC"))
    return pd.DataFrame({
        "method": list(METHODS), "two_metric_score": two.to_numpy(),
        "four_metric_score_with_grid_proxies": four.to_numpy(),
    })

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs, main_table, proxy, manifest = evaluate()
    pairs.to_csv(OUTPUT / "alignment_paper_pair_metrics.csv", index=False, float_format="%.9f")
    main_table.to_csv(OUTPUT / "alignment_paper_main_metrics.csv", index=False, float_format="%.9f")
    proxy.to_csv(OUTPUT / "alignment_mouse_grid_proxy_metrics.csv", index=False, float_format="%.9f")
    mouse_score_sensitivity(main_table, proxy).to_csv(
        OUTPUT / "alignment_mouse_score_sensitivity.csv", index=False, float_format="%.9f")
    mouse = main_table.loc[main_table.dataset.eq('Mouse_Brain')].set_index('method').reindex(METHODS)
    normalized = pd.DataFrame(index=mouse.index)
    for metric in MAIN_METRICS['Mouse_Brain']:
        ranks = mouse[metric].rank(method='average', ascending=metric == 'Chamfer Distance')
        normalized[metric + ' rank score'] = 1 - (ranks - 1) / (len(METHODS) - 1)
    normalized['score_with_regions'] = normalized.mean(axis=1)
    normalized['score_geometry_expression'] = normalized[['Chamfer Distance rank score', 'PCC rank score']].mean(axis=1)
    normalized.to_csv(OUTPUT / 'alignment_mouse_region_score_summary.csv', float_format='%.9f')
    (OUTPUT / "alignment_paper_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(main_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
