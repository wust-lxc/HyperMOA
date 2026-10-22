#!/usr/bin/env python3
"""Reproduce/adopt the fixed refinement from preserved inputs, without archives."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))


import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark.scripts.common import fair_benchmark_common as common
from benchmark.scripts.common.layout_provenance import resolve_recorded_path, verify_protected_file
ROOT = common.ROOT
sys.path.insert(0, str(ROOT / "HyperMOA"))
from hypermoa.alignment_refinement import SliceFrame, refine_slices, refinement_protocol
from hypermoa.final_alignment import finalize_anndata

OUTPUT = ROOT / "data/results/alignment_refinement_final"
SEEDS = (42, 7, 2022)
PROCESSED = {"Human_lymph_node": "human_lymph_node_processed.h5ad",
             "Simulation": "simulation_processed.h5ad", "Mouse_Brain": "h3k27ac_atac_processed.h5ad"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_files():
    from benchmark.scripts.evaluation.clustering.evaluate_fair_clustering import specs
    paths = {ROOT / "benchmark _Clustering.xlsx"}
    for spec in common.SPECS.values():
        for raw in spec.raw_paths:
            paths.update(path for path in raw.parent.glob("*.h5ad") if path.is_file())
        for dataset, _, directory in specs():
            if dataset == spec.name:
                paths.update(path for path in directory.rglob("*") if path.is_file())
        for method in common.METHODS:
            if method != "HyperMOA":
                paths.update(path for path in (spec.result_root / method).rglob("*") if path.is_file())
    for directory in (ROOT / "figure/clustering", ROOT / "benchmark/results"):
        paths.update(path for path in directory.rglob("*") if path.is_file()
                     and path != ROOT / "benchmark/results/RESULTS_INDEX.md")
    for source in common.HYPERMOA_SOURCE_FILES.values():
        paths.update(source.parent.glob("*merged.h5ad"))
    return sorted(paths)


def check_protected(create=False):
    path = OUTPUT / "protected_sha256.json"
    values = {str(p.relative_to(ROOT)): sha256(p) for p in protected_files()}
    if create:
        path.write_text(json.dumps(values, indent=2)+"\n")
    recorded = json.loads(path.read_text())
    expected = {str(resolve_recorded_path(ROOT, relative).relative_to(ROOT)): value
                for relative, value in recorded.items()}
    if len(expected) != len(recorded):
        raise RuntimeError("Duplicate protected paths after layout migration")
    if set(values) != set(expected):
        raise RuntimeError("Raw/clustering/external-method inputs changed")
    for relative, digest in values.items():
        if digest != expected[relative]:
            verify_protected_file(ROOT, relative, expected[relative], expected)
    return len(values)


def fingerprint(data):
    digest = hashlib.sha256()
    obs = data.obs.drop(columns=[key for key in ("x_aligned", "y_aligned") if key in data.obs])
    digest.update(pd.util.hash_pandas_object(obs.astype(str), index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(data.var.astype(str), index=True).values.tobytes())
    x = data.X.tocsr() if sparse.issparse(data.X) else sparse.csr_matrix(data.X)
    for value in (x.data, x.indices, x.indptr):
        digest.update(np.asarray(value).tobytes())
    for key in sorted(data.obsm):
        if key in ("spatial", "HyperMOA", "HyperMOA_bc", "latent", "latent_bc", "ADT"):
            digest.update(key.encode())
            digest.update(np.asarray(data.obsm[key]).tobytes())
    return digest.hexdigest()


def frames_from_data(dataset, data):
    frames = []
    batches = data.obs["batch"].astype(str).to_numpy()
    coordinate_key = "alignment_input" if "alignment_input" in data.obsm else "transform_fine"
    for name in common.SPECS[dataset].slices:
        mask = batches == common.hypermoa_batch_name(dataset, name)
        ids = np.asarray(data.obs.loc[mask, "original_barcode"].astype(str)
                         if "original_barcode" in data.obs else data.obs_names[mask], dtype=str)
        frames.append(SliceFrame(name, ids, np.asarray(data.obsm[coordinate_key])[mask].copy(),
                                np.asarray(data.obsm["HyperMOA_bc"])[mask].copy()))
    return frames


def save_run(dataset, frames, result):
    directory = OUTPUT / dataset / f"seed{result.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    fields = {}
    for frame in frames:
        fields[f"{frame.name}_ids"], fields[f"{frame.name}_xy"] = frame.spot_ids, result.coordinates[frame.name]
    np.savez_compressed(directory / "aligned_coordinates.npz", **fields)
    (directory / "diagnostics.json").write_text(json.dumps({"pairs": result.pair_diagnostics,
        "chain": result.chain_diagnostics, "chain_accepted": result.chain_accepted}, indent=2)+"\n")
    pd.DataFrame(result.pair_diagnostics).to_csv(directory / "pair_decisions.csv", index=False)
    for pair, anchors in result.anchors.items():
        np.savez_compressed(directory / f"{pair}_anchors.npz", training=anchors.training,
                            weights=anchors.weights, heldout=anchors.heldout, heldout_sources=anchors.heldout_sources)


def update_coordinate_fields(data):
    xy = np.asarray(data.obsm["transform_fine"])
    if "x_aligned" in data.obs:
        data.obs["x_aligned"] = xy[:, 0]
    if "y_aligned" in data.obs:
        data.obs["y_aligned"] = xy[:, 1]
    if "z_rec" in data.obs:
        z = data.obs["z_rec"].to_numpy(dtype=float)
        data.obsm["rec_3d"] = np.column_stack([xy, z])
        batches = data.obs["batch"].astype(str).to_numpy()
        normalized = np.zeros_like(xy)
        for name in pd.unique(batches):
            mask = batches == name
            points = xy[mask]
            span = points.max(0)-points.min(0)
            span[span <= 1e-12] = 1.
            normalized[mask] = (points-points.min(0))/span
        if "rec_3d_plot" in data.obsm:
            data.obsm["rec_3d_plot"] = np.column_stack([normalized, -z])
        if "rec_3d_norm" in data.obsm:
            old = np.asarray(data.obsm["rec_3d_norm"]).copy()
            old[:, :2] = normalized
            data.obsm["rec_3d_norm"] = old
    if "alignment_rms_fine" in data.uns:
        data.uns["alignment_rms_fine"] = float(np.sqrt(np.mean(np.sum((xy-data.obsm["spatial"])**2, axis=1))))


def write_checked(path, data, expected_fingerprint):
    staged = path.with_suffix(".new.h5ad")
    data.write_h5ad(staged)
    reread = ad.read_h5ad(staged)
    if fingerprint(reread) != expected_fingerprint:
        raise RuntimeError(f"Labels/embedding/expression changed: {path}")
    np.testing.assert_array_equal(reread.obsm["transform_fine"], data.obsm["transform_fine"])
    staged.replace(path)


def run(adopt=False, parity_root=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    create = not (OUTPUT / "protected_sha256.json").exists()
    count = check_protected(create=create)
    fingerprints, parities, sources = {}, [], {}
    all_frames, all_results = {}, {}
    for dataset, source_path in common.HYPERMOA_SOURCE_FILES.items():
        source = ad.read_h5ad(source_path)
        fingerprints[str(source_path.relative_to(ROOT))] = fingerprint(source)
        sources[dataset] = {"path": str(source_path), "sha256_before_adoption": sha256(source_path)}
        frames = frames_from_data(dataset, source)
        all_frames[dataset], all_results[dataset] = frames, {}
        for seed in SEEDS:
            result = refine_slices(frames, seed=seed)
            save_run(dataset, frames, result)
            all_results[dataset][seed] = result
            if parity_root is not None:
                old_path = parity_root / dataset / "cap140_rot20_r3" / f"seed{seed}" / "aligned_coordinates.npz"
                with np.load(old_path) as old:
                    for frame in frames:
                        np.testing.assert_array_equal(frame.spot_ids, old[f"{frame.name}_ids"])
                        np.testing.assert_array_equal(result.coordinates[frame.name], old[f"{frame.name}_xy"])
                parities.append({"dataset": dataset, "seed": seed, "coordinates_bitwise_identical": True})
            print(f"[final rule] {dataset}/seed{seed}, accepted={sum(r['accepted'] for r in result.pair_diagnostics)}", flush=True)
        del source
    # Adopt only after every independent package run has passed parity checks.
    if adopt:
        for dataset, path in common.HYPERMOA_SOURCE_FILES.items():
            data = ad.read_h5ad(path)
            batches = [common.hypermoa_batch_name(dataset, name) for name in common.SPECS[dataset].slices]
            data, result = finalize_anndata(data, batches, seed=42, copy=False)
            for frame in frames_from_data(dataset, data):
                mask = data.obs["batch"].astype(str).to_numpy() == common.hypermoa_batch_name(dataset, frame.name)
                np.testing.assert_array_equal(data.obsm["transform_fine"][mask], all_results[dataset][42].coordinates[frame.name])
            update_coordinate_fields(data)
            write_checked(path, data, fingerprints[str(path.relative_to(ROOT))])
            processed_path = path.parent / PROCESSED[dataset]
            processed = ad.read_h5ad(processed_path)
            old_fingerprint = fingerprint(processed)
            fingerprints[str(processed_path.relative_to(ROOT))] = old_fingerprint
            processed_frames = frames_from_data(dataset, processed)
            xy, base = np.empty((processed.n_obs, 2)), np.empty((processed.n_obs, 2))
            for frame, primary in zip(processed_frames, all_frames[dataset]):
                order = pd.Index(primary.spot_ids).get_indexer(frame.spot_ids)
                if np.any(order < 0):
                    raise ValueError("Processed-file barcode mismatch")
                np.testing.assert_array_equal(frame.embedding, primary.embedding[order])
                mask = processed.obs["batch"].astype(str).to_numpy() == common.hypermoa_batch_name(dataset, frame.name)
                xy[mask], base[mask] = result.coordinates[common.hypermoa_batch_name(dataset, frame.name)][order], primary.coordinates[order]
            processed.obsm["alignment_input"], processed.obsm["transform_fine"] = base, xy
            processed.obsm["aligned_spatial"] = xy.copy()
            processed.uns["alignment_refinement_protocol"] = data.uns["alignment_refinement_protocol"]
            update_coordinate_fields(processed)
            write_checked(processed_path, processed, old_fingerprint)
            del data, processed
    if parities:
        pd.DataFrame(parities).to_csv(OUTPUT / "package_parity.csv", index=False)
    (OUTPUT / "frozen_content_fingerprints.json").write_text(json.dumps(fingerprints, indent=2)+"\n")
    (OUTPUT / "protocol.json").write_text(json.dumps(refinement_protocol(), indent=2)+"\n")
    (OUTPUT / "run_manifest.json").write_text(json.dumps({"source_inputs": sources, "main_seed": 42,
        "stability_seeds": list(SEEDS), "adopted": adopt, "protected_files_unchanged": check_protected(),
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            (Path(__file__), ROOT/"HyperMOA/hypermoa/alignment_refinement.py", ROOT/"HyperMOA/hypermoa/final_alignment.py")},
        "refinement_input_key": "alignment_input", "final_coordinate_key": "transform_fine",
        "development_data_exposure": True, "full_model_retrained": False}, indent=2)+"\n")


def stability():
    from benchmark.scripts.evaluation.alignment.evaluate_alignment_paper import MAIN_METRICS, pair_result
    from benchmark.scripts.evaluation.alignment.evaluate_fair_alignment import load_raw_slices
    rows = []
    for dataset, spec in common.SPECS.items():
        slices, genes = load_raw_slices(spec)
        for seed in SEEDS:
            with np.load(OUTPUT / dataset / f"seed{seed}/aligned_coordinates.npz") as saved:
                coordinates = {}
                for name, raw in slices.items():
                    ids = pd.Index(saved[f"{name}_ids"].astype(str))
                    order = ids.get_indexer(raw.obs_names.astype(str))
                    if np.any(order < 0):
                        raise ValueError("Final coordinate barcode mismatch")
                    coordinates[name] = saved[f"{name}_xy"][order]
                for source, target in zip(spec.slices[:-1], spec.slices[1:]):
                    row = pair_result(spec, "HyperMOA", slices[source], slices[target], coordinates[source], coordinates[target], len(genes))
                    row["seed"] = seed
                    rows.append(row)
        del slices
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "seed_pair_metrics.csv", index=False, float_format="%.12f")
    overall = frame.groupby(["dataset", "seed"], sort=False).mean(numeric_only=True).reset_index()
    overall.to_csv(OUTPUT / "seed_overall_metrics.csv", index=False, float_format="%.12f")
    records = []
    for dataset, metrics in MAIN_METRICS.items():
        table = overall.loc[overall.dataset.eq(dataset)]
        for metric in metrics:
            records.append({"dataset": dataset, "metric": metric, "mean": table[metric].mean(),
                            "std": table[metric].std(), "n_seeds": len(table)})
    pd.DataFrame(records).to_csv(OUTPUT / "stability.csv", index=False, float_format="%.12f")


def verify():
    fingerprints = json.loads((OUTPUT / "frozen_content_fingerprints.json").read_text())
    for relative, expected in fingerprints.items():
        data = ad.read_h5ad(ROOT / relative)
        if fingerprint(data) != expected:
            raise RuntimeError("Frozen clustering/expression/embedding changed")
        del data
    print(f"[verified] {check_protected()} raw/clustering/external files unchanged; frozen H5AD content unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "stability", "verify"))
    parser.add_argument("--adopt", action="store_true")
    parser.add_argument("--parity-root", type=Path)
    args = parser.parse_args()
    if args.stage == "run":
        run(adopt=args.adopt, parity_root=args.parity_root)
    elif args.stage == "stability":
        stability()
    else:
        verify()


if __name__ == "__main__":
    main()
