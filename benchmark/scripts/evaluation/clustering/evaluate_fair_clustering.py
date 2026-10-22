#!/usr/bin/env python3
"""Recompute clustering metrics from saved spot labels on a common spot set."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))


import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from benchmark.scripts.common import fair_benchmark_common as common

ROOT = common.ROOT
OUTPUT = ROOT / "data/evaluation/fair_benchmark"
METHODS = ("HyperMOA", "SpatialGlue", "MISO", "CellCharter", "Harmony", "3d-OT", "SpaMosaic")
METRICS = ("ARI", "NMI", "AMI", "Moran's I")
GLOBAL_METHODS = {"HyperMOA", "Harmony", "SpaMosaic"}


def specs():
    for index in range(1, 4):
        name = f"slice{index}"
        yield "Human_lymph_node", name, ROOT / f"data/results/clustering/Human_lymph_node/{name}"
    for index in range(1, 6):
        name = f"Simulation{index}"
        yield "Simulation", name, ROOT / f"data/results/clustering/Simulation/slice{index}"
    for name in ("Mouse_Brain_ATAC", "Mouse_Brain_H3K27ac"):
        yield "Mouse_Brain", name, ROOT / "data/results/clustering/Mouse_Brain" / name.removeprefix("Mouse_Brain_")


def truth_labels(series: pd.Series, dataset: str) -> pd.Series:
    values = series.astype("string").str.strip()
    if dataset == "Human_lymph_node":
        values = values.str.lower().str.replace("vessels", "vessel", regex=False)
    return values.mask(values.str.lower().isin(("", "na", "n/a", "nan", "none", "null", "unknown")))


def partition_scores(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "ARI": float(adjusted_rand_score(truth, prediction)),
        "NMI": float(normalized_mutual_info_score(truth, prediction)),
        "AMI": float(adjusted_mutual_info_score(truth, prediction)),
    }


def onehot_moran(prediction: np.ndarray, neighbors: np.ndarray) -> float:
    """Prevalence-weighted Moran I of one-hot domain membership; ID invariant."""
    codes = pd.Categorical(prediction).codes
    sizes = np.bincount(codes)
    total = len(codes)
    numerator = 0.0
    weight = 0
    for code, size in enumerate(sizes):
        if size == 0 or size == total:
            continue
        centered = (codes == code).astype(np.float64) - size / total
        denominator = np.dot(centered, centered)
        score = np.sum(centered[:, None] * centered[neighbors]) / neighbors.shape[1] / denominator
        numerator += size * score
        weight += size
    return float(numerator / weight) if weight else float("nan")


def evaluate():
    rows = []
    coverage = []
    pooled = {}
    for dataset, slice_name, directory in specs():
        frame = pd.read_csv(directory / "clustering_labels.csv", index_col=0, dtype="string")
        truth = truth_labels(frame["truth"], dataset)
        columns = {method: f"cluster_{'3dOT' if method == '3d-OT' else method}" for method in METHODS}
        common = truth.notna()
        for column in columns.values():
            common &= frame[column].notna()
        if int(common.sum()) < 3:
            raise ValueError(f"Too few common labelled spots: {dataset}/{slice_name}")

        data = ad.read_h5ad(directory / "clustering_results.h5ad", backed="r")
        try:
            if not np.array_equal(frame.index.astype(str).to_numpy(), data.obs_names.astype(str).to_numpy()):
                raise ValueError(f"Spot order mismatch: {directory}")
            xy = np.asarray(data.obsm["spatial"], dtype=np.float64)[common.to_numpy()]
        finally:
            data.file.close()
        neighbors = NearestNeighbors(n_neighbors=min(7, len(xy))).fit(xy).kneighbors(
            xy, return_distance=False
        )[:, 1:]
        target = truth.loc[common].to_numpy(dtype=str)
        requested = {"Human_lymph_node": 10, "Simulation": 5, "Mouse_Brain": 18}[dataset]
        for method, column in columns.items():
            labels = frame.loc[common, column].to_numpy(dtype=str)
            rows.append({
                "dataset": dataset, "slice": slice_name, "method": method,
                "n_total": len(frame), "n_truth": int(truth.notna().sum()),
                "n_evaluated": int(common.sum()), "truth_classes": len(np.unique(target)),
                "requested_clusters": requested, "actual_clusters": len(np.unique(labels)),
                **partition_scores(target, labels), "Moran's I": onehot_moran(labels, neighbors),
            })
            coverage.append({
                "dataset": dataset, "slice": slice_name, "method": method,
                "n_total": len(frame), "n_truth": int(truth.notna().sum()),
                "n_predicted": int(frame[column].notna().sum()),
                "n_common_evaluated": int(common.sum()),
            })
            if method in GLOBAL_METHODS or (method == "SpatialGlue" and dataset != "Mouse_Brain"):
                pooled.setdefault((dataset, method), []).append((target, labels))
        print(f"{dataset}/{slice_name}: {int(common.sum())} common labelled spots", flush=True)

    slices = pd.DataFrame(rows)
    macro = slices.groupby(["dataset", "method"], sort=False)[list(METRICS)].mean().reset_index()
    macro["n_evaluated_total"] = slices.groupby(["dataset", "method"], sort=False)["n_evaluated"].sum().to_numpy()
    pooled_rows = []
    for (dataset, method), parts in pooled.items():
        target = np.concatenate([item[0] for item in parts])
        labels = np.concatenate([item[1] for item in parts])
        pooled_rows.append({"dataset": dataset, "method": method, "n_evaluated": len(target),
                            **partition_scores(target, labels)})
    return slices, macro, pd.DataFrame(pooled_rows), pd.DataFrame(coverage)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    slices, macro, pooled, coverage = evaluate()
    slices.to_csv(OUTPUT / "clustering_slice_metrics_fair.csv", index=False, float_format="%.9f")
    macro.to_csv(OUTPUT / "clustering_macro_metrics_fair.csv", index=False, float_format="%.9f")
    pooled.to_csv(OUTPUT / "clustering_pooled_global_supplement.csv", index=False, float_format="%.9f")
    coverage.to_csv(OUTPUT / "clustering_coverage.csv", index=False)
    manifest = {
        "metric_input": "Saved clustering labels matched by raw barcode; common labelled spot intersection across all seven methods per slice",
        "Human_truth": "lowercase; replace vessels with vessel",
        "Moran's I": "prevalence-weighted mean of binary one-hot domain Moran I on raw spatial k=6 nearest neighbors",
        "global_workbook_rows": "equal-weight arithmetic mean of per-slice metrics for every method",
        "pooled_global_supplement": "ARI/NMI/AMI only for methods with cross-slice-consistent global cluster IDs",
        "aggregate_score_metrics": ["ARI", "NMI", "AMI"],
        "algorithm_runs_repeated": False,
    }
    (OUTPUT / "clustering_fair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved fair clustering metrics to", OUTPUT)


if __name__ == "__main__":
    main()
