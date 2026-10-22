#!/usr/bin/env python3
"""Recompute alignment metrics from saved matches and continuous coordinates."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))


import hashlib
import json
import os
from dataclasses import dataclass

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmark.scripts.common import fair_benchmark_common as common

ROOT = common.ROOT
OUTPUT = ROOT / "data/evaluation/fair_benchmark"
METHODS = ("HyperMOA", "3d-OT", "SLAT", "PASTE", "SANTO")
SUMMARY_METRICS = ("PAA", "FOSCTTM", "Chamfer Distance", "CI", "PCC", "SMR", "Chamfer_over_target_NN")


@dataclass
class RawSlice:
    name: str
    obs_names: pd.Index
    xy: np.ndarray
    labels: np.ndarray
    expression: sparse.csr_matrix
    sums: np.ndarray
    squared_sums: np.ndarray


def load_raw_slices(spec: common.DatasetSpec) -> tuple[dict[str, RawSlice], pd.Index]:
    names = []
    for path in spec.raw_paths:
        data = ad.read_h5ad(path, backed="r")
        try:
            if not data.var_names.is_unique:
                raise ValueError(f"Duplicate gene names: {path}")
            names.append(data.var_names.copy())
        finally:
            data.file.close()
    genes = names[0]
    for other in names[1:]:
        genes = genes.intersection(other, sort=False)
    if len(genes) == 0:
        raise ValueError(f"No common RNA genes: {spec.name}")

    result = {}
    for name, path in zip(spec.slices, spec.raw_paths):
        data = ad.read_h5ad(path)
        matrix = data[:, genes].X
        matrix = matrix.tocsr().astype(np.float64) if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
        if matrix.data.size and np.min(matrix.data) < 0:
            raise ValueError(f"Negative raw RNA values: {path}")
        library = np.asarray(matrix.sum(axis=1)).ravel()
        scale = np.divide(1e4, library, out=np.zeros_like(library), where=library != 0)
        matrix = (sparse.diags(scale) @ matrix).tocsr()
        matrix.data = np.log1p(matrix.data)
        result[name] = RawSlice(
            name=name, obs_names=data.obs_names.copy(),
            xy=np.asarray(data.obsm["spatial"], dtype=np.float64),
            labels=common.labels(data), expression=matrix,
            sums=np.asarray(matrix.sum(axis=1)).ravel(),
            squared_sums=np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel(),
        )
        del data
    return result, genes


def pcc_on_common_rna(source: RawSlice, target: RawSlice, predicted: np.ndarray) -> tuple[float, int]:
    n_genes = source.expression.shape[1]
    blocks = []
    for start in range(0, len(predicted), 512):
        stop = min(start + 512, len(predicted))
        x = source.expression[start:stop]
        y = target.expression[predicted[start:stop]]
        sum_xy = np.asarray(x.multiply(y).sum(axis=1)).ravel()
        sx = source.sums[start:stop]
        sy = target.sums[predicted[start:stop]]
        variance_x = source.squared_sums[start:stop] - sx * sx / n_genes
        variance_y = target.squared_sums[predicted[start:stop]] - sy * sy / n_genes
        denominator = np.sqrt(np.maximum(variance_x, 0) * np.maximum(variance_y, 0))
        blocks.append(np.divide(sum_xy - sx * sy / n_genes, denominator,
                                out=np.full(stop - start, np.nan), where=denominator > 0))
    values = np.concatenate(blocks)
    finite = np.isfinite(values)
    return (float(values[finite].mean()) if finite.any() else np.nan, int(finite.sum()))


def hypermoa_coordinates(spec: common.DatasetSpec, slices: dict[str, RawSlice]) -> dict[str, np.ndarray]:
    final = ad.read_h5ad(common.HYPERMOA_SOURCE_FILES[spec.name], backed="r")
    try:
        batches = final.obs["batch"].astype(str).to_numpy()
        output = {}
        for name, raw in slices.items():
            mask = batches == common.hypermoa_batch_name(spec.name, name)
            barcodes = (final.obs.loc[mask, "original_barcode"].astype(str).to_numpy()
                        if "original_barcode" in final.obs else final.obs_names[mask].astype(str))
            lookup = pd.Index(barcodes)
            if not lookup.is_unique:
                raise ValueError(f"Duplicate HyperMOA barcodes: {spec.name}/{name}")
            indexer = lookup.get_indexer(raw.obs_names.astype(str))
            if np.any(indexer < 0):
                raise ValueError(f"HyperMOA lacks raw spots: {spec.name}/{name}")
            coords = np.asarray(final.obsm["transform_fine"], dtype=np.float64)[mask]
            output[name] = coords[indexer]
        return output
    finally:
        final.file.close()


def pair_metrics(spec: common.DatasetSpec, method: str, source: RawSlice, target: RawSlice,
                 predicted: np.ndarray, aligned: np.ndarray, target_xy: np.ndarray,
                 n_genes: int) -> dict:
    predicted = np.asarray(predicted, dtype=int)
    aligned = np.asarray(aligned, dtype=np.float64)
    if len(predicted) != len(source.obs_names) or aligned.shape != (len(source.obs_names), 2):
        raise ValueError(f"Shape mismatch: {spec.name}/{method}/{source.name}")
    if np.any(predicted < 0) or np.any(predicted >= len(target.obs_names)):
        raise ValueError(f"Invalid target index: {spec.name}/{method}/{source.name}")

    pcc, n_pcc = pcc_on_common_rna(source, target, predicted)
    chamfer = common.chamfer_distance(aligned, target_xy)
    target_nn = common.median_nn_radius(target_xy)
    result = {
        "dataset": spec.name, "method": method, "pair": f"{source.name}_to_{target.name}",
        "PCC": pcc, "Chamfer Distance": chamfer,
        "Chamfer_over_target_NN": chamfer / target_nn, "target_median_NN": target_nn,
        "n_common_RNA_genes": n_genes, "n_source": len(source.obs_names),
        "n_target": len(target.obs_names), "n_pcc_pairs": n_pcc,
        "n_true_pairs": np.nan, "n_valid_label_pairs": np.nan,
        "coordinate_source": ("HyperMOA transform_fine" if method == "HyperMOA" else
                              "SLAT saved rigid projection" if method == "SLAT" else
                              "saved method continuous coordinates"),
    }
    if spec.has_ci:
        valid = common.valid_labels(source.labels) & common.valid_labels(target.labels[predicted])
        result["n_valid_label_pairs"] = int(valid.sum())
        result["CI"] = float(np.mean(source.labels[valid] == target.labels[predicted[valid]])) if valid.any() else np.nan
    if spec.has_true_pairs:
        true_target = target.obs_names.get_indexer(source.obs_names)
        known = true_target >= 0
        result["n_true_pairs"] = int(known.sum())
        if known.any():
            result["PAA"] = float(np.mean(predicted[known] == true_target[known]))
            errors = np.linalg.norm(aligned[known] - target_xy[true_target[known]], axis=1)
            result["SMR"] = float(np.mean(errors <= target_nn))
            result["FOSCTTM"] = common.foscttm_chunked(aligned[known], target_xy, true_target[known])
    return result


def evaluate():
    rows = []
    genes_record = []
    for spec in common.SPECS.values():
        slices, genes = load_raw_slices(spec)
        gene_path = OUTPUT / f"{spec.name}_common_RNA_genes.txt"
        gene_path.write_text("\n".join(map(str, genes)) + "\n", encoding="utf-8")
        genes_record.append({
            "dataset": spec.name, "n_common_RNA_genes": len(genes),
            "gene_list_sha256": hashlib.sha256(gene_path.read_bytes()).hexdigest(),
            "gene_list_file": str(gene_path),
        })
        hyper = hypermoa_coordinates(spec, slices)
        for method in METHODS:
            for source_name, target_name in zip(spec.slices[:-1], spec.slices[1:]):
                source, target = slices[source_name], slices[target_name]
                if method == "HyperMOA":
                    aligned = hyper[source_name]
                    target_xy = hyper[target_name]
                    predicted = common.nearest_match(aligned, target_xy)
                else:
                    checkpoint = spec.result_root / method / "pair_checkpoints" / f"{source_name}_to_{target_name}.npz"
                    if not checkpoint.exists():
                        print(f"[missing] {checkpoint}", flush=True)
                        continue
                    with np.load(checkpoint) as saved:
                        predicted = np.asarray(saved["predicted_target_index"], dtype=int)
                        aligned = np.asarray(saved["aligned_source"], dtype=np.float64)
                    target_xy = target.xy
                rows.append(pair_metrics(spec, method, source, target, predicted, aligned, target_xy, len(genes)))
                print(f"[alignment] {spec.name}/{method}/{source_name}->{target_name}", flush=True)
        del slices
    pairs = pd.DataFrame(rows)
    columns = [metric for metric in SUMMARY_METRICS if metric in pairs]
    summary = pairs.groupby(["dataset", "method"], sort=False)[columns].mean().reset_index()
    return pairs, summary, genes_record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs, summary, genes = evaluate()
    pairs.to_csv(OUTPUT / "alignment_pair_metrics_fair.csv", index=False, float_format="%.9f")
    summary.to_csv(OUTPUT / "alignment_summary_metrics_fair.csv", index=False, float_format="%.9f")
    manifest = {
        "evaluation_unit": "Directed adjacent slice pair; dataset summary is the unweighted pair mean",
        "PCC": "Same dataset-wide intersected raw RNA genes and CP10K+log1p matrix for every method; rowwise source-predicted-target Pearson mean",
        "geometric_metrics": "Continuous saved aligned source coordinates for every method, including SLAT's saved rigid projection",
        "matches": "Method-native predicted target index, except HyperMOA nearest target in transform_fine",
        "Chamfer Distance": "Symmetric mean Euclidean nearest-neighbor distance in each pair's target coordinate frame",
        "Chamfer_over_target_NN": "Raw Chamfer divided by target median nearest-neighbor spacing; diagnostic only",
        "common_genes": genes,
        "algorithm_runs_repeated": False,
    }
    (OUTPUT / "alignment_fair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved fair alignment metrics to", OUTPUT)


if __name__ == "__main__":
    main()
