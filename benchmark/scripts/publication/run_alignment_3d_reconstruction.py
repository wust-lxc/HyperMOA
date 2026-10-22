#!/usr/bin/env python
"""Run alignment benchmarks and export comparable 3D reconstructions.

This script reuses the exact pairwise preprocessing and recommended parameters
from run_alignment_benchmark.py/run_mouse_brain_alignment.py. It saves only
coordinates, labels and figures; expression matrices are not duplicated.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))


import argparse
import gc
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark.scripts.runners.alignment import run_alignment_benchmark as standard_benchmark
from benchmark.scripts.runners.alignment import run_mouse_brain_alignment as mouse_benchmark


METHODS = ["PASTE", "SLAT", "SANTO", "3d-OT", "HyperMOA"]
EXTERNAL_METHODS = ["PASTE", "SLAT", "SANTO", "3d-OT"]
DATASETS = ["Human_lymph_node", "Simulation", "Mouse_Brain"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    slice_names: tuple[str, ...]
    raw_paths: tuple[Path, ...]
    output_root: Path


SPECS = {
    "Human_lymph_node": DatasetSpec(
        name="Human_lymph_node",
        slice_names=("slice1", "slice2", "slice3"),
        raw_paths=(
            ROOT / "data/datasets/Human_lymph_node/slice1/s1_adata_rna.h5ad",
            ROOT / "data/datasets/Human_lymph_node/slice2/s2_adata_rna.h5ad",
            ROOT / "data/datasets/Human_lymph_node/slice3/s3_adata_rna.h5ad",
        ),
        output_root=ROOT / "data/results/alignment/Human_lymph_node",
    ),
    "Simulation": DatasetSpec(
        name="Simulation",
        slice_names=tuple(f"Simulation{i}" for i in range(1, 6)),
        raw_paths=tuple(ROOT / f"data/datasets/Simulation/slice{i}/adata_RNA.h5ad" for i in range(1, 6)),
        output_root=ROOT / "data/results/alignment/Simulation",
    ),
    "Mouse_Brain": DatasetSpec(
        name="Mouse_Brain",
        slice_names=("Mouse_Brain_ATAC", "Mouse_Brain_H3K27ac"),
        raw_paths=(
            ROOT / "data/datasets/Mouse_Brain/ATAC/adata_RNA.h5ad",
            ROOT / "data/datasets/Mouse_Brain/H3K27ac/adata_RNA.h5ad",
        ),
        output_root=ROOT / "data/results/alignment/Mouse_Brain",
    ),
}


def clustering_sidecar_path(slice_name: str) -> Path:
    if slice_name.startswith("slice"):
        directory = ROOT / "data/results/clustering/Human_lymph_node" / slice_name
    elif slice_name.startswith("Simulation"):
        directory = ROOT / "data/results/clustering/Simulation" / f"slice{slice_name.removeprefix('Simulation')}"
    elif slice_name.startswith("Mouse_Brain_"):
        directory = ROOT / "data/results/clustering/Mouse_Brain" / slice_name.removeprefix("Mouse_Brain_")
    else:
        raise ValueError(f"Unknown clustering slice: {slice_name}")
    return directory / "clustering_results.h5ad"


def method_dir(spec: DatasetSpec, method: str) -> Path:
    return spec.output_root / method


def load_prepared(spec: DatasetSpec):
    if spec.name == "Mouse_Brain":
        source, target = mouse_benchmark.prep_pair(spec.raw_paths[0], spec.raw_paths[1], max_features=3000)
        return [source, target]
    return [standard_benchmark.prep_adata(path, max_genes=2000) for path in spec.raw_paths]


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def nearest_match(aligned_source: np.ndarray, target: np.ndarray):
    distance, index = cKDTree(target).query(aligned_source, k=1)
    return index.astype(int), distance


def barycentric(pi: np.ndarray, target: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi, dtype=np.float64)
    row_sum = pi.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (pi / row_sum) @ np.asarray(target, dtype=np.float64)


def chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    distance_ab, _ = cKDTree(points_b).query(points_a, k=1)
    distance_ba, _ = cKDTree(points_a).query(points_b, k=1)
    return float((distance_ab.mean() + distance_ba.mean()) / 2.0)


def fit_rigid(source: np.ndarray, target: np.ndarray):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def apply_rigid(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def compose_rigid_coordinates(raw_coordinates, pair_transforms):
    output = []
    for slice_index, raw in enumerate(raw_coordinates):
        points = np.asarray(raw, dtype=np.float64)
        for pair_index in range(slice_index, len(pair_transforms)):
            rotation, translation = pair_transforms[pair_index]
            points = apply_rigid(points, rotation, translation)
        output.append(points.astype(np.float32))
    return output


def interpolate_deformation(
    points: np.ndarray,
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    neighbors: int = 5,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    source_landmarks = np.asarray(source_landmarks, dtype=np.float64)
    displacement = np.asarray(target_landmarks, dtype=np.float64) - source_landmarks
    k_use = min(max(1, neighbors), len(source_landmarks))
    distances, indices = cKDTree(source_landmarks).query(points, k=k_use)
    if k_use == 1:
        return points + displacement[np.asarray(indices, dtype=int)]
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=int)
    weights = 1.0 / np.maximum(distances, 1e-8)
    exact = distances[:, 0] <= 1e-12
    weights /= weights.sum(axis=1, keepdims=True)
    interpolated = np.sum(displacement[indices] * weights[:, :, None], axis=1)
    if exact.any():
        interpolated[exact] = displacement[indices[exact, 0]]
    return points + interpolated


def compose_nonrigid_coordinates(raw_coordinates, pair_aligned):
    output = []
    for slice_index, raw in enumerate(raw_coordinates):
        points = np.asarray(raw, dtype=np.float64)
        for pair_index in range(slice_index, len(pair_aligned)):
            if pair_index == slice_index:
                points = np.asarray(pair_aligned[pair_index], dtype=np.float64)
            else:
                points = interpolate_deformation(
                    points,
                    raw_coordinates[pair_index],
                    pair_aligned[pair_index],
                    neighbors=5,
                )
        output.append(points.astype(np.float32))
    return output


def save_pair_checkpoint(output_dir: Path, pair_name: str, pred, aligned, extra=None):
    checkpoint_dir = output_dir / "pair_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "predicted_target_index": np.asarray(pred, dtype=np.int32),
        "aligned_source": np.asarray(aligned, dtype=np.float32),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(checkpoint_dir / f"{pair_name}.npz", **payload)


def run_paste(spec: DatasetSpec, adatas, output_dir: Path):
    import ot
    import paste as pst
    from paste.visualization import generalized_procrustes_analysis

    raw = [np.asarray(item.obsm["spatial"], dtype=np.float64) for item in adatas]
    stacked = []
    rows = []
    for index, (source, target) in enumerate(zip(adatas[:-1], adatas[1:])):
        pair_name = f"{spec.slice_names[index]}_to_{spec.slice_names[index + 1]}"
        print(f"PASTE {spec.name} {pair_name}", flush=True)
        start = time.time()
        try:
            pi = pst.pairwise_align(
                source,
                target,
                use_gpu=True,
                backend=ot.backend.TorchBackend(),
                alpha=0.1,
                norm=True,
                verbose=False,
                gpu_verbose=False,
            )
            pi = to_numpy(pi)
        except Exception as error:
            print(f"PASTE GPU fallback: {type(error).__name__}: {error}", flush=True)
            pi = np.asarray(
                pst.pairwise_align(
                    source,
                    target,
                    alpha=0.1,
                    norm=True,
                    verbose=False,
                    gpu_verbose=False,
                )
            )
        pred = np.argmax(pi, axis=1)
        local_aligned = barycentric(pi, raw[index + 1])
        if index == 0:
            first, second = generalized_procrustes_analysis(raw[0], raw[1], pi)
            stacked.extend([first, second])
        else:
            _, next_coordinates = generalized_procrustes_analysis(stacked[index], raw[index + 1], pi)
            stacked.append(next_coordinates)
        save_pair_checkpoint(output_dir, pair_name, pred, local_aligned)
        rows.append(
            {
                "pair": pair_name,
                "n_source": source.n_obs,
                "n_target": target.n_obs,
                "local_chamfer": chamfer(local_aligned, raw[index + 1]),
                "elapsed_seconds": time.time() - start,
            }
        )
        del pi
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    config = {
        "pairwise": "PASTE pairwise_align",
        "alpha": 0.1,
        "norm": True,
        "stacking": "official generalized_procrustes_analysis equivalent to stack_slices_pairwise",
        "preprocessing": "shared benchmark preprocessing; max_genes=2000 or Mouse_Brain common genes/max_features=3000",
    }
    return [np.asarray(value, dtype=np.float32) for value in stacked], rows, config


def run_slat(spec: DatasetSpec, adatas, output_dir: Path):
    raw = [np.asarray(item.obsm["spatial"], dtype=np.float64) for item in adatas]
    transforms = []
    rows = []
    for index, (source, target) in enumerate(zip(adatas[:-1], adatas[1:])):
        pair_name = f"{spec.slice_names[index]}_to_{spec.slice_names[index + 1]}"
        print(f"SLAT {spec.name} {pair_name}", flush=True)
        start = time.time()
        pred, matched_coordinates = standard_benchmark.run_slat(source, target)
        rotation, translation = fit_rigid(raw[index], matched_coordinates)
        local_aligned = apply_rigid(raw[index], rotation, translation)
        transforms.append((rotation, translation))
        save_pair_checkpoint(
            output_dir,
            pair_name,
            pred,
            local_aligned,
            {"rotation": rotation.astype(np.float64), "translation": translation.astype(np.float64)},
        )
        rows.append(
            {
                "pair": pair_name,
                "n_source": source.n_obs,
                "n_target": target.n_obs,
                "local_chamfer": chamfer(local_aligned, raw[index + 1]),
                "elapsed_seconds": time.time() - start,
            }
        )
    config = {
        "graph": "KNN",
        "k_cutoff": 10,
        "feature": "DPCA",
        "matching": "SLAT spatial_match source-to-target",
        "coordinate_projection": "recommended rigid Kabsch transform fitted to SLAT matches; adjacent transforms composed",
    }
    return compose_rigid_coordinates(raw, transforms), rows, config


def run_santo(spec: DatasetSpec, adatas, output_dir: Path):
    raw = [np.asarray(item.obsm["spatial"], dtype=np.float64) for item in adatas]
    transforms = []
    rows = []
    for index, (source, target) in enumerate(zip(adatas[:-1], adatas[1:])):
        pair_name = f"{spec.slice_names[index]}_to_{spec.slice_names[index + 1]}"
        print(f"SANTO {spec.name} {pair_name}", flush=True)
        start = time.time()
        pred, local_aligned = standard_benchmark.run_santo(source, target)
        rotation, translation = fit_rigid(raw[index], local_aligned)
        transforms.append((rotation, translation))
        save_pair_checkpoint(
            output_dir,
            pair_name,
            pred,
            local_aligned,
            {"rotation": rotation.astype(np.float64), "translation": translation.astype(np.float64)},
        )
        rows.append(
            {
                "pair": pair_name,
                "n_source": source.n_obs,
                "n_target": target.n_obs,
                "local_chamfer": chamfer(local_aligned, raw[index + 1]),
                "elapsed_seconds": time.time() - start,
            }
        )
    config = {
        "epochs": 20,
        "learning_rate": 0.001,
        "k": 10,
        "alpha": 0.9,
        "mode": "align",
        "dimension": 2,
        "diff_omics": False,
        "coordinate_projection": "SANTO returned coarse+fine rigid result; adjacent transforms composed",
    }
    return compose_rigid_coordinates(raw, transforms), rows, config


def run_3dot(spec: DatasetSpec, adatas, output_dir: Path):
    raw = [np.asarray(item.obsm["spatial"], dtype=np.float64) for item in adatas]
    pair_aligned = []
    rows = []
    for index, (source, target) in enumerate(zip(adatas[:-1], adatas[1:])):
        pair_name = f"{spec.slice_names[index]}_to_{spec.slice_names[index + 1]}"
        print(f"3d-OT {spec.name} {pair_name}", flush=True)
        start = time.time()
        pred, local_aligned = standard_benchmark.run_3dot(source, target)
        pair_aligned.append(np.asarray(local_aligned, dtype=np.float32))
        save_pair_checkpoint(output_dir, pair_name, pred, local_aligned)
        rows.append(
            {
                "pair": pair_name,
                "n_source": source.n_obs,
                "n_target": target.n_obs,
                "local_chamfer": chamfer(local_aligned, raw[index + 1]),
                "elapsed_seconds": time.time() - start,
            }
        )
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    config = {
        "training_epochs": 1150,
        "neighbors": 8,
        "reverse_alignment_epochs": 1,
        "simk": 5,
        "otk": 200,
        "coordinate_output": "target_recon unscaled to target spatial range",
        "multi_slice_composition": "adjacent nonrigid displacements composed with 5-NN inverse-distance interpolation",
    }
    return compose_nonrigid_coordinates(raw, pair_aligned), rows, config


RUNNERS = {
    "PASTE": run_paste,
    "SLAT": run_slat,
    "SANTO": run_santo,
    "3d-OT": run_3dot,
}


def minmax_per_slice(coordinates: np.ndarray, slices: np.ndarray):
    output = np.zeros_like(coordinates, dtype=np.float64)
    for slice_name in pd.unique(slices):
        mask = slices == slice_name
        points = coordinates[mask]
        minimum = points.min(axis=0)
        span = points.max(axis=0) - minimum
        span[span <= 1e-12] = 1.0
        output[mask] = (points - minimum) / span
    return output


def minmax_global(coordinates: np.ndarray):
    minimum = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - minimum
    span[span <= 1e-12] = 1.0
    return (coordinates - minimum) / span


def normalize_global_isotropic(coordinates: np.ndarray):
    """Normalize one complete reconstruction without changing slice-to-slice geometry."""
    coordinates = np.asarray(coordinates, dtype=float)
    minimum = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - minimum
    scale = float(np.max(span))
    if scale <= 1e-12:
        scale = 1.0
    return (coordinates - minimum) / scale


def natural_key(value: str):
    import re

    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(value))]


def display_truth(values: np.ndarray, dataset: str) -> np.ndarray:
    output = np.asarray(values, dtype=object).copy()
    if dataset == "Human_lymph_node":
        aliases = {
            "medulla cords": "medulla cords",
            "medulla vessel": "medulla vessel",
            "medulla vessels": "medulla vessel",
        }
        for index, value in enumerate(output):
            if pd.isna(value):
                continue
            normalized = str(value).strip().lower()
            output[index] = aliases.get(normalized, str(value).strip())
    return output


def label_palette(labels: np.ndarray):
    values = sorted({str(value) for value in labels if pd.notna(value) and str(value) != "nan"}, key=natural_key)
    colors = list(plt.get_cmap("tab20").colors)
    if len(values) > len(colors):
        colors = [plt.get_cmap("gist_ncar")(position) for position in np.linspace(0.05, 0.95, len(values))]
    mapping = {label: colors[index % len(colors)] for index, label in enumerate(values)}
    mapping["NA"] = "#bdbdbd"
    return values, mapping


def plot_3d(
    output_path: Path,
    coordinates: np.ndarray,
    slice_names: np.ndarray,
    slice_order: np.ndarray,
    color_values: np.ndarray,
    title: str,
    per_slice_normalize: bool,
    color_mode: str,
):
    xy = (
        minmax_per_slice(coordinates, slice_names)
        if per_slice_normalize
        else normalize_global_isotropic(coordinates)
    )
    z = -np.asarray(slice_order, dtype=float) * 0.48
    fig = plt.figure(figsize=(7.2, 6.1))
    ax = fig.add_subplot(111, projection="3d")
    if color_mode == "slice":
        categories = list(pd.unique(slice_names))
        colors = list(plt.get_cmap("Set2").colors)
        palette = {value: colors[index % len(colors)] for index, value in enumerate(categories)}
        values = slice_names
    else:
        categories, palette = label_palette(color_values)
        if pd.isna(color_values).any():
            categories = categories + ["NA"]
        values = np.asarray(["NA" if pd.isna(value) or str(value) == "nan" else str(value) for value in color_values])
    total = len(coordinates)
    point_size = 8 if total <= 7000 else (3.5 if total <= 13000 else 1.6)
    for category in categories:
        mask = values == category
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            z[mask],
            s=point_size,
            c=[palette[category]],
            label=str(category),
            linewidths=0,
            alpha=0.86,
            depthshade=False,
        )
    ax.set_xlabel("aligned_x")
    ax.set_ylabel("aligned_y")
    ax.set_zlabel("slice_order")
    ax.set_title(title, pad=16)
    ax.view_init(elev=27, azim=-58)
    ax.set_box_aspect((1.0, 1.0, max(0.55, 0.25 * (int(slice_order.max()) + 1))))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, markerscale=2.2)
    # Older outputs may be hard-linked aliases. Unlink first so replacing one
    # rendering cannot silently overwrite another filename that shares its inode.
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def replace_hard_link(source: Path, target: Path):
    if target.exists() or target.is_symlink():
        target.unlink()
    os.link(source, target)


def save_reconstruction_figures(
    spec: DatasetSpec,
    method: str,
    output_dir: Path,
    aligned: np.ndarray,
    slice_values: np.ndarray,
    order_values: np.ndarray,
    truth_display_values: np.ndarray,
):
    title_prefix = f"{spec.name} - {method}"
    rec_path = output_dir / "reconstruction_3d_rec.png"
    plot_3d(
        rec_path,
        aligned,
        slice_values,
        order_values,
        slice_values,
        f"{title_prefix} algorithm-aligned 3D - colored by slice",
        per_slice_normalize=False,
        color_mode="slice",
    )

    truth_path = output_dir / "reconstruction_3d_by_truth.png"
    plot_3d(
        truth_path,
        aligned,
        slice_values,
        order_values,
        truth_display_values,
        f"{title_prefix} algorithm-aligned 3D - colored by ground truth",
        per_slice_normalize=False,
        color_mode="truth",
    )

    # Keep legacy filenames while avoiding duplicate image storage.
    replace_hard_link(truth_path, output_dir / "reconstruction_3d_global_scale.png")
    replace_hard_link(rec_path, output_dir / "reconstruction_3d_by_slice.png")


def compatible_categorical(values):
    series = pd.Series(values, dtype="object")
    series = series.where(pd.notna(series), np.nan)
    return pd.Categorical(series.to_numpy(dtype=object))


def collect_slice_metadata(spec: DatasetSpec, adatas):
    obs_parts = []
    original_parts = []
    for order, (slice_name, data) in enumerate(zip(spec.slice_names, adatas)):
        sidecar_path = clustering_sidecar_path(slice_name)
        sidecar = ad.read_h5ad(sidecar_path)
        source_ids = pd.Index([str(value) for value in data.obs_names])
        if not sidecar.obs_names.equals(source_ids):
            sidecar = sidecar[source_ids].copy()
        obs = sidecar.obs.copy()
        obs["original_barcode"] = source_ids
        obs["slice"] = slice_name
        obs["slice_order"] = order
        obs.index = pd.Index([f"{slice_name}_{value}" for value in source_ids], dtype="object")
        obs_parts.append(obs)
        original_parts.append(np.asarray(data.obsm["spatial"], dtype=np.float32))
    return pd.concat(obs_parts, axis=0), np.vstack(original_parts)


def save_alignment_result(
    spec: DatasetSpec,
    method: str,
    adatas,
    aligned_coordinates,
    pair_rows,
    config,
):
    output_dir = method_dir(spec, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs, original_coordinates = collect_slice_metadata(spec, adatas)
    aligned = np.vstack([np.asarray(value, dtype=np.float32) for value in aligned_coordinates])
    if aligned.shape != original_coordinates.shape or not np.isfinite(aligned).all():
        raise ValueError(f"Invalid aligned coordinates for {method}/{spec.name}: {aligned.shape}")
    slice_values = obs["slice"].astype(str).to_numpy()
    order_values = obs["slice_order"].astype(int).to_numpy()
    truth_values = obs["truth"].astype("string").to_numpy()
    truth_display_values = display_truth(truth_values, spec.name)

    result_obs = obs.copy()
    for column in result_obs.columns:
        if column == "slice_order":
            continue
        if pd.api.types.is_string_dtype(result_obs[column].dtype):
            result_obs[column] = compatible_categorical(result_obs[column])
    result = ad.AnnData(
        X=csr_matrix((len(obs), 0), dtype=np.float32),
        obs=result_obs,
        var=pd.DataFrame(index=pd.Index([], dtype="object", name="feature")),
    )
    result.obsm["spatial_original"] = original_coordinates
    result.obsm["aligned_spatial"] = aligned
    result.obsm["rec_3d"] = np.column_stack([aligned, order_values.astype(np.float32)])
    result.uns["dataset"] = spec.name
    result.uns["method"] = method
    result.uns["slice_order"] = np.asarray(spec.slice_names, dtype=str)
    result.uns["z_interpretation"] = "Equal spacing by slice order; not physical thickness."
    result.uns["run_config_json"] = json.dumps(config, sort_keys=True)
    result.write_h5ad(output_dir / "alignment_results.h5ad", compression="gzip")

    coordinate_frame = pd.DataFrame(
        {
            "barcode": result.obs_names,
            "original_barcode": obs["original_barcode"].astype(str).values,
            "slice": slice_values,
            "slice_order": order_values,
            "truth": truth_values,
            "x_original": original_coordinates[:, 0],
            "y_original": original_coordinates[:, 1],
            "x_aligned": aligned[:, 0],
            "y_aligned": aligned[:, 1],
            "z": order_values,
        }
    )
    coordinate_frame.to_csv(output_dir / "aligned_coordinates.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "pair_alignment_summary.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    save_reconstruction_figures(
        spec,
        method,
        output_dir,
        aligned,
        slice_values,
        order_values,
        truth_display_values,
    )
    validation = {
        "status": "ok",
        "n_obs": int(len(obs)),
        "n_slices": len(spec.slice_names),
        "finite_aligned_coordinates": bool(np.isfinite(aligned).all()),
        "h5ad_shape": [int(result.n_obs), int(result.n_vars)],
        "figure_count": 4,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return result


def raw_shells(spec: DatasetSpec):
    shells = []
    for path in spec.raw_paths:
        raw = ad.read_h5ad(path, backed="r")
        shell = ad.AnnData(
            X=csr_matrix((raw.n_obs, 0), dtype=np.float32),
            obs=pd.DataFrame(index=pd.Index(raw.obs_names.astype(str), dtype="object")),
            var=pd.DataFrame(index=pd.Index([], dtype="object")),
        )
        shell.obsm["spatial"] = np.asarray(raw.obsm["spatial"], dtype=np.float32)
        raw.file.close()
        shells.append(shell)
    return shells


def hypermoa_coordinates(spec: DatasetSpec):
    if spec.name == "Human_lymph_node":
        path = ROOT / "HyperMOA/Human_lymph_node_result/adata.h5ad"
        batch_map = {name: name for name in spec.slice_names}
    elif spec.name == "Simulation":
        path = ROOT / "HyperMOA/Simulation_result/adata.h5ad"
        batch_map = {name: name for name in spec.slice_names}
    else:
        path = ROOT / "HyperMOA/Mouse_brain_result/H3K27ac_ATAC/adata.h5ad"
        batch_map = {"Mouse_Brain_ATAC": "ATAC", "Mouse_Brain_H3K27ac": "H3K27ac"}
    data = ad.read_h5ad(path, backed="r")
    output = []
    for slice_name, raw_path in zip(spec.slice_names, spec.raw_paths):
        raw = ad.read_h5ad(raw_path, backed="r")
        raw_ids = pd.Index(raw.obs_names.astype(str))
        raw.file.close()
        mask = data.obs["batch"].astype(str).to_numpy() == batch_map[slice_name]
        source_ids = (
            pd.Index(data.obs.loc[mask, "original_barcode"].astype(str))
            if "original_barcode" in data.obs
            else pd.Index(data.obs_names[mask].astype(str))
        )
        coordinates = pd.DataFrame(np.asarray(data.obsm["transform_fine"][mask]), index=source_ids)
        output.append(coordinates.loc[raw_ids].to_numpy(dtype=np.float32))
    data.file.close()
    return output, {
        "coordinate_output": "existing HyperMOA obsm[transform_fine]",
        "source": str(path),
        "rerun": False,
    }


def run_external_method(spec: DatasetSpec, method: str, overwrite: bool):
    output_dir = method_dir(spec, method)
    final_path = output_dir / "alignment_results.h5ad"
    if final_path.exists() and not overwrite:
        print(f"SKIP existing {method}/{spec.name}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    status_path.write_text(json.dumps({"status": "running", "started": time.time()}) + "\n")
    adatas = load_prepared(spec)
    started = time.time()
    try:
        aligned, pair_rows, config = RUNNERS[method](spec, adatas, output_dir)
        config["method"] = method
        config["dataset"] = spec.name
        config["slice_order"] = list(spec.slice_names)
        config["elapsed_seconds"] = time.time() - started
        save_alignment_result(spec, method, adatas, aligned, pair_rows, config)
        status_path.write_text(
            json.dumps({"status": "complete", "elapsed_seconds": time.time() - started}, indent=2) + "\n"
        )
    except Exception as error:
        status_path.write_text(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n"
        )
        raise
    finally:
        del adatas
        gc.collect()


def export_hypermoa(spec: DatasetSpec, overwrite: bool):
    output_dir = method_dir(spec, "HyperMOA")
    if (output_dir / "alignment_results.h5ad").exists() and not overwrite:
        print(f"SKIP existing HyperMOA/{spec.name}", flush=True)
        return
    shells = raw_shells(spec)
    coordinates, config = hypermoa_coordinates(spec)
    save_alignment_result(spec, "HyperMOA", shells, coordinates, [], config)


def plot_composite(spec: DatasetSpec):
    available = []
    for method in METHODS:
        path = method_dir(spec, method) / "alignment_results.h5ad"
        if path.exists():
            available.append((method, ad.read_h5ad(path)))
    if not available:
        return
    outputs = {
        "slice": (
            spec.output_root / "all_methods_reconstruction_3d_comparison.png",
            "colored by slice",
        ),
        "truth": (
            spec.output_root / "all_methods_reconstruction_3d_by_truth_comparison.png",
            "colored by ground-truth label",
        ),
    }
    n_columns = len(available)
    for color_mode, (output_path, color_description) in outputs.items():
        fig = plt.figure(figsize=(5.2 * n_columns, 5.0))
        legend_handles = None
        legend_labels = None
        for panel, (method, data) in enumerate(available, start=1):
            ax = fig.add_subplot(1, n_columns, panel, projection="3d")
            coordinates = np.asarray(data.obsm["aligned_spatial"], dtype=float)
            slices = data.obs["slice"].astype(str).to_numpy()
            orders = data.obs["slice_order"].astype(int).to_numpy()
            xy = (
                minmax_per_slice(coordinates, slices)
                if color_mode == "truth"
                else normalize_global_isotropic(coordinates)
            )
            z = -orders.astype(float) * 0.48

            if color_mode == "slice":
                categories = list(pd.unique(slices))
                colors = list(plt.get_cmap("Set2").colors)
                palette = {
                    category: colors[index % len(colors)]
                    for index, category in enumerate(categories)
                }
                values = slices
            else:
                truth = display_truth(data.obs["truth"].astype("string").to_numpy(), spec.name)
                categories, palette = label_palette(truth)
                values = np.asarray(
                    ["NA" if pd.isna(value) or str(value) == "nan" else str(value) for value in truth]
                )
                if np.any(values == "NA"):
                    categories = categories + ["NA"]

            point_size = 5 if len(data) <= 7000 else (2.2 if len(data) <= 13000 else 0.9)
            for category in categories:
                mask = values == category
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    z[mask],
                    s=point_size,
                    c=[palette[category]],
                    linewidths=0,
                    alpha=0.84,
                    depthshade=False,
                    label=str(category) if color_mode == "slice" else None,
                )
            ax.set_title(method)
            ax.set_xlabel("aligned x")
            ax.set_ylabel("aligned y")
            ax.set_zlabel("slice order")
            ax.view_init(elev=27, azim=-58)
            ax.set_box_aspect((1.0, 1.0, max(0.55, 0.25 * len(spec.slice_names))))
            if color_mode == "slice" and panel == 1:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        if color_mode == "slice" and legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.005),
                ncol=min(5, len(legend_labels)),
                frameon=False,
                fontsize=9,
                markerscale=2.2,
            )
            fig.subplots_adjust(bottom=0.14)
        fig.suptitle(
            (
                f"{spec.name} algorithm-aligned 3D comparison - {color_description}"
                + (" - per-slice x/y normalization" if color_mode == "truth" else "")
            ),
            fontsize=17,
            y=0.98,
        )
        spec.output_root.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def rerender_saved_figures(spec: DatasetSpec):
    for method in METHODS:
        output_dir = method_dir(spec, method)
        path = output_dir / "alignment_results.h5ad"
        if not path.exists():
            continue
        print(f"PLOT existing {method}/{spec.name}", flush=True)
        data = ad.read_h5ad(path)
        aligned = np.asarray(data.obsm["aligned_spatial"], dtype=float)
        slice_values = data.obs["slice"].astype(str).to_numpy()
        order_values = data.obs["slice_order"].astype(int).to_numpy()
        truth_values = display_truth(data.obs["truth"].astype("string").to_numpy(), spec.name)
        save_reconstruction_figures(
            spec,
            method,
            output_dir,
            aligned,
            slice_values,
            order_values,
            truth_values,
        )
    plot_composite(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=EXTERNAL_METHODS)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=DATASETS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--composites", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from saved alignment_results.h5ad files without rerunning methods.",
    )
    args = parser.parse_args()

    for dataset_name in args.datasets:
        spec = SPECS[dataset_name]
        spec.output_root.mkdir(parents=True, exist_ok=True)
        if args.plots_only:
            rerender_saved_figures(spec)
            continue
        for method in args.methods:
            print(f"\n=== {method} / {dataset_name} ===", flush=True)
            if method == "HyperMOA":
                export_hypermoa(spec, args.overwrite)
            else:
                run_external_method(spec, method, args.overwrite)
        if args.composites:
            plot_composite(spec)


if __name__ == "__main__":
    main()
