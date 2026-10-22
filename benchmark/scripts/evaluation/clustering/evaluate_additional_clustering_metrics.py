#!/usr/bin/env python3
"""Evaluate complementary clustering metrics on the existing fair spot sets."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))


import hashlib
import json
import os
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from benchmark.scripts.common import fair_benchmark_common as common
from benchmark.scripts.evaluation.clustering.evaluate_fair_clustering import METHODS, specs, truth_labels


ROOT = common.ROOT
OUTPUT = ROOT / "data/evaluation/fair_benchmark/additional_clustering"
SEED = 42
SVD_COMPONENTS = 30
PAS_NEIGHBORS = 10
METRICS = ("HOM", "COM", "FMI", "PAS", "CHAOS", "CHAOS_singleton_penalized", "Silhouette_RNA", "DBI_RNA")
RAW_PATHS = {
    (dataset.name, slice_name): raw_path
    for dataset in common.SPECS.values()
    for slice_name, raw_path in zip(dataset.slices, dataset.raw_paths)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_common_genes() -> dict[str, pd.Index]:
    result = {}
    for dataset in common.SPECS.values():
        genes = None
        for path in dataset.raw_paths:
            data = ad.read_h5ad(path, backed="r")
            try:
                if not data.var_names.is_unique:
                    raise ValueError(f"Duplicate raw gene names: {path}")
                current = data.var_names.copy()
            finally:
                data.file.close()
            genes = current if genes is None else genes.intersection(current, sort=False)
        if len(genes) <= SVD_COMPONENTS:
            raise ValueError(f"Too few shared RNA genes: {dataset.name}")
        result[dataset.name] = genes
    return result


def shared_spots(directory: Path, dataset: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    frame = pd.read_csv(directory / "clustering_labels.csv", index_col=0, dtype="string")
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate curated spot IDs: {directory}")
    columns = {method: f"cluster_{'3dOT' if method == '3d-OT' else method}" for method in METHODS}
    if not set(columns.values()).issubset(frame.columns):
        raise ValueError(f"Missing prediction columns: {directory}")
    truth = truth_labels(frame["truth"], dataset)
    mask = truth.notna()
    for column in columns.values():
        mask &= frame[column].notna()
    selected = frame.loc[mask]
    if len(selected) <= PAS_NEIGHBORS:
        raise ValueError(f"Too few commonly labelled spots: {directory}")
    return selected, truth.loc[mask].to_numpy(dtype=str), columns


def spatial_coordinates(directory: Path, spot_ids: pd.Index) -> np.ndarray:
    data = ad.read_h5ad(directory / "clustering_results.h5ad", backed="r")
    try:
        indexer = data.obs_names.get_indexer(spot_ids.astype(str))
        if np.any(indexer < 0):
            raise ValueError(f"Curated spatial coordinates lack spots: {directory}")
        xy = np.asarray(data.obsm["spatial"], dtype=np.float64)[indexer]
    finally:
        data.file.close()
    if xy.shape != (len(spot_ids), 2) or not np.isfinite(xy).all():
        raise ValueError(f"Invalid spatial coordinates: {directory}")
    return xy


def common_rna_features(path: Path, spot_ids: pd.Index, genes: pd.Index) -> tuple[np.ndarray, dict]:
    data = ad.read_h5ad(path)
    try:
        if not data.obs_names.is_unique:
            raise ValueError(f"Duplicate raw spot IDs: {path}")
        indexer = data.obs_names.get_indexer(spot_ids.astype(str))
        if np.any(indexer < 0):
            raise ValueError(f"Raw RNA lacks {int((indexer < 0).sum())} evaluated spots: {path}")
        gene_indexer = data.var_names.get_indexer(genes)
        if np.any(gene_indexer < 0):
            raise ValueError(f"Raw RNA lacks dataset-common genes: {path}")
        matrix = data.X
        matrix = matrix.tocsr()[indexer][:, gene_indexer].astype(np.float32) if sparse.issparse(matrix) else sparse.csr_matrix(
            np.asarray(matrix[np.ix_(indexer, gene_indexer)], dtype=np.float32)
        )
        n_genes = len(genes)
    finally:
        del data
    if matrix.data.size and (not np.isfinite(matrix.data).all() or np.min(matrix.data) < 0):
        raise ValueError(f"RNA matrix is not finite nonnegative counts: {path}")
    library = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    matrix = (sparse.diags(scale) @ matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    n_components = min(SVD_COMPONENTS, matrix.shape[0] - 1, matrix.shape[1] - 1)
    model = TruncatedSVD(n_components=n_components, n_iter=7, random_state=SEED)
    features = np.asarray(model.fit_transform(matrix), dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite shared RNA features: {path}")
    return features, {
        "raw_path": str(path), "raw_sha256": sha256(path), "n_genes": n_genes,
        "common_gene_names_sha256": hashlib.sha256(
            ("\n".join(map(str, genes)) + "\n").encode("utf-8")
        ).hexdigest(),
        "n_zero_library_spots": int((library == 0).sum()),
        "n_svd_components": n_components,
        "svd_explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
    }


def ten_neighbors(xy: np.ndarray) -> np.ndarray:
    candidates = NearestNeighbors(n_neighbors=PAS_NEIGHBORS + 1, algorithm="kd_tree").fit(xy).kneighbors(
        xy, return_distance=False
    )
    result = np.empty((len(xy), PAS_NEIGHBORS), dtype=np.int64)
    for index, row in enumerate(candidates):
        others = row[row != index]
        if len(others) < PAS_NEIGHBORS:
            raise ValueError("Spatial neighbor search did not return ten other spots")
        result[index] = others[:PAS_NEIGHBORS]
    return result


def spatial_scores(labels: np.ndarray, xy: np.ndarray, neighbors: np.ndarray) -> dict:
    codes = pd.Categorical(labels).codes
    pas = float(np.mean(np.sum(codes[neighbors] != codes[:, None], axis=1) >= 6))
    std = np.std(xy, axis=0, ddof=1)
    std[std == 0] = 1.0
    scaled = (xy - np.mean(xy, axis=0)) / std
    diameter_bound = float(np.linalg.norm(np.ptp(scaled, axis=0)))
    distances = np.zeros(len(xy), dtype=np.float64)
    singleton_count = 0
    for code in np.unique(codes):
        indices = np.flatnonzero(codes == code)
        if len(indices) == 1:
            singleton_count += 1
            continue
        distances[indices] = cKDTree(scaled[indices]).query(scaled[indices], k=2)[0][:, 1]
    standard = float(distances.mean())
    penalized = float((distances.sum() + singleton_count * diameter_bound) / len(xy))
    return {
        "PAS": pas, "CHAOS": standard,
        "CHAOS_singleton_penalized": penalized,
        "n_singleton_spots": singleton_count,
    }


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows = []
    inputs = []
    shared_genes = dataset_common_genes()
    for dataset, slice_name, directory in specs():
        started = time.monotonic()
        selected, target, columns = shared_spots(directory, dataset)
        xy = spatial_coordinates(directory, selected.index)
        neighbors = ten_neighbors(xy)
        raw_path = RAW_PATHS[(dataset, slice_name)]
        features, rna_info = common_rna_features(raw_path, selected.index, shared_genes[dataset])
        expression_distances = pairwise_distances(features, metric="euclidean", n_jobs=1)
        np.fill_diagonal(expression_distances, 0)
        for method, column in columns.items():
            labels = selected[column].to_numpy(dtype=str)
            n_clusters = len(np.unique(labels))
            row = {
                "dataset": dataset, "slice": slice_name, "method": method,
                "n_evaluated": len(selected), "truth_classes": len(np.unique(target)),
                "actual_clusters": n_clusters,
                "HOM": float(homogeneity_score(target, labels)),
                "COM": float(completeness_score(target, labels)),
                "FMI": float(fowlkes_mallows_score(target, labels)),
                **spatial_scores(labels, xy, neighbors),
                "Silhouette_RNA": float(silhouette_score(expression_distances, labels, metric="precomputed"))
                if 1 < n_clusters < len(labels) else float("nan"),
                "DBI_RNA": float(davies_bouldin_score(features, labels))
                if 1 < n_clusters < len(labels) else float("nan"),
            }
            rows.append(row)
        inputs.append({
            "dataset": dataset, "slice": slice_name, "n_common_spots": len(selected),
            "labels_path": str(directory / "clustering_labels.csv"),
            "labels_sha256": sha256(directory / "clustering_labels.csv"),
            "spatial_path": str(directory / "clustering_results.h5ad"),
            **rna_info,
        })
        print(f"{dataset}/{slice_name}: {len(selected)} shared spots, {len(rows)} rows total, "
              f"{time.monotonic() - started:.1f}s", flush=True)
        del expression_distances, features
    slices = pd.DataFrame(rows)
    macro = slices.groupby(["dataset", "method"], sort=False)[list(METRICS)].mean().reset_index()
    macro["n_slices"] = slices.groupby(["dataset", "method"], sort=False).size().to_numpy()
    macro["n_evaluated_total"] = slices.groupby(["dataset", "method"], sort=False)["n_evaluated"].sum().to_numpy()
    macro["n_singleton_spots_total"] = slices.groupby(["dataset", "method"], sort=False)["n_singleton_spots"].sum().to_numpy()
    return slices, macro, inputs


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    slices, macro, inputs = evaluate()
    slices.to_csv(OUTPUT / "additional_clustering_slice_metrics.csv", index=False, float_format="%.9f")
    macro.to_csv(OUTPUT / "additional_clustering_macro_metrics.csv", index=False, float_format="%.9f")
    slices.to_csv(OUTPUT / "additional_clustering_slice_metrics_3dec.csv", index=False, float_format="%.3f")
    macro.to_csv(OUTPUT / "additional_clustering_macro_metrics_3dec.csv", index=False, float_format="%.3f")
    manifest = {
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "method_order": list(METHODS), "evaluation_unit": "one raw slice; equal-weight slice mean for dataset macro",
        "spot_selection": "same intersection of nonmissing truth and all seven saved method labels as evaluate_fair_clustering.py",
        "truth_normalization": "Human lowercase and vessels->vessel; other datasets stripped strings",
        "HOM_COM_FMI": "scikit-learn labels directly, permutation invariant; no Hungarian mapping",
        "PAS": "fraction with at least 6 different labels among 10 raw-spatial nearest neighbors",
        "CHAOS": "SpatialPCA-style mean same-cluster 1NN Euclidean distance after per-axis spatial z-scoring; singleton clusters contribute zero in standard CHAOS",
        "CHAOS_singleton_penalized": "supplementary nonstandard sensitivity: singleton distance replaced with z-scored spatial bounding-box diagonal",
        "Silhouette_RNA_DBI_RNA": "same dataset-wide intersected raw RNA genes, per-slice CP10K+log1p TruncatedSVD features for every method; exact full-spot scores",
        "svd_components": SVD_COMPONENTS, "random_seed": SEED,
        "summary_directions": {"HOM": "higher", "COM": "higher", "FMI": "higher", "PAS": "lower", "CHAOS": "lower", "CHAOS_singleton_penalized": "lower", "Silhouette_RNA": "higher", "DBI_RNA": "lower"},
        "workbooks_modified": False, "algorithm_runs_repeated": False,
        "slices": inputs,
    }
    (OUTPUT / "additional_clustering_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(slices)} slice-method rows and {len(macro)} dataset-method rows to {OUTPUT}")


if __name__ == "__main__":
    main()
