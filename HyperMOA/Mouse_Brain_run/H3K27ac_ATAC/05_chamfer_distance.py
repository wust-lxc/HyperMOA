import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from hypermoa.data_paths import resolve_data_root


TRUTH_SOURCES = {
    "H3K27ac": "Mouse_Brain_H3K27ac",
    "ATAC": "Mouse_Brain_ATAC",
}
OTHER_METHOD_COLUMNS = ("3d-OT", "mclust", "COSMOS", "MISO", "SpatialGlue", "cellcharter")


def _directed_chamfer_sq(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.nan
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(b)
    dist, _ = nn.kneighbors(a, return_distance=True)
    return float((dist * dist).mean())


def _chamfer_row(points: np.ndarray, batches: np.ndarray, slice_a: str, slice_b: str, stage: str) -> dict:
    a = points[batches == slice_a]
    b = points[batches == slice_b]
    d_ab = _directed_chamfer_sq(a, b)
    d_ba = _directed_chamfer_sq(b, a)
    d_sym = np.nan if np.isnan(d_ab) or np.isnan(d_ba) else d_ab + d_ba
    return {
        "stage": stage,
        "slice_a": slice_a,
        "slice_b": slice_b,
        "n_a": int(a.shape[0]),
        "n_b": int(b.shape[0]),
        "cd_sq_a_to_b": d_ab,
        "cd_sq_b_to_a": d_ba,
        "cd_sq_symmetric": float(d_sym),
    }


def _minmax_normalize_by_slice(points: np.ndarray, batches: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    batches = np.asarray(batches).astype(str)
    out = np.zeros_like(points, dtype=np.float64)
    for b in np.unique(batches):
        idx = np.where(batches == b)[0]
        p = points[idx]
        p_min = p.min(axis=0, keepdims=True)
        p_max = p.max(axis=0, keepdims=True)
        span = p_max - p_min
        span[span <= 1e-12] = 1.0
        out[idx] = (p - p_min) / span
    return out


def _moran_i_knn(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    n = values.shape[0]
    if n < 3:
        return np.nan

    k_use = max(1, min(int(k), n - 1))
    neigh = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(coords)
    neigh_idx = neigh.kneighbors(coords, return_distance=False)[:, 1:]

    z = values - values.mean()
    denom = np.sum(z * z)
    if denom <= 0:
        return np.nan

    w = 1.0 / float(k_use)
    num = np.sum(z[:, None] * z[neigh_idx] * w)
    return float(num / denom)


def _add_truth_labels(adata, root_dir: Path) -> None:
    adata.obs["truth"] = pd.NA
    adata.obs["truth_prefixed"] = pd.NA
    data_root = resolve_data_root(root_dir)

    for slice_name, data_dir in TRUTH_SOURCES.items():
        truth_file = data_root / "Mouse_Brain" / data_dir.removeprefix("Mouse_Brain_") / "3d-OT.h5ad"
        if not truth_file.exists():
            continue
        adata_truth = sc.read_h5ad(truth_file, backed="r")
        try:
            if "truth" not in adata_truth.obs:
                continue
            idx = adata.obs["batch"].astype(str) == slice_name
            barcodes = adata.obs.loc[idx, "original_barcode"].astype(str)
            common = barcodes[barcodes.isin(adata_truth.obs_names)]
            if common.empty:
                continue
            truth_map = adata_truth.obs.loc[common.values, "truth"].astype(str).to_dict()
            mapped = barcodes.map(truth_map)
            adata.obs.loc[idx, "truth"] = mapped.values
            adata.obs.loc[idx, "truth_prefixed"] = [
                f"{slice_name}:{x}" if pd.notna(x) else pd.NA for x in mapped.values
            ]
        finally:
            adata_truth.file.close()


def _metric_row_for_labels(slice_name: str, method: str, y_true, y_pred, coords: np.ndarray) -> dict:
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    coords = np.asarray(coords, dtype=np.float64)
    return {
        "slice": slice_name,
        "method": method,
        "n_cells": float(coords.shape[0]),
        "n_valid": float(y_true.shape[0]),
        "n_truth": float(y_true.nunique()),
        "n_pred": float(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "Moran_I": float(_moran_i_knn(pd.Categorical(y_pred).codes.astype(np.float64), coords, k=6)),
    }


def _other_methods_by_slice_metrics(
    adata,
    root_dir: Path,
    truth_key: str = "truth",
    stair_pred_key: str = "Domain",
) -> pd.DataFrame:
    rows = []
    batches = adata.obs["batch"].astype(str)
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64) if "spatial" in adata.obsm else None
    data_root = resolve_data_root(root_dir)

    for slice_name, data_dir in TRUTH_SOURCES.items():
        idx = batches == slice_name
        if int(idx.sum()) == 0:
            continue

        idx_pos = np.where(idx.to_numpy())[0]
        barcodes = adata.obs.loc[idx, "original_barcode"].astype(str)
        truth_file = data_root / "Mouse_Brain" / data_dir.removeprefix("Mouse_Brain_") / "3d-OT.h5ad"
        if not truth_file.exists():
            continue

        adata_truth = sc.read_h5ad(truth_file, backed="r")
        try:
            common_mask = barcodes.isin(adata_truth.obs_names)
            common_barcodes = barcodes[common_mask].values
            if len(common_barcodes) == 0 or "truth" not in adata_truth.obs:
                continue

            coords_common = np.asarray(adata_truth.obsm["spatial"], dtype=np.float64) if "spatial" in adata_truth.obsm else None
            if coords_common is not None:
                truth_indexer = adata_truth.obs_names.get_indexer(common_barcodes)
                coords_common = coords_common[truth_indexer]
            elif spatial is not None:
                coords_common = spatial[idx_pos][common_mask.to_numpy()]
            else:
                continue

            truth_values = adata_truth.obs.loc[common_barcodes, "truth"]
            for method in OTHER_METHOD_COLUMNS:
                if method not in adata_truth.obs:
                    continue
                pred_values = adata_truth.obs.loc[common_barcodes, method]
                valid = ~truth_values.isna() & ~pred_values.isna()
                if int(valid.sum()) == 0:
                    continue
                rows.append(
                    _metric_row_for_labels(
                        slice_name=slice_name,
                        method=method,
                        y_true=truth_values.loc[valid],
                        y_pred=pred_values.loc[valid],
                        coords=coords_common[valid.to_numpy()],
                    )
                )
        finally:
            adata_truth.file.close()

        if truth_key in adata.obs and stair_pred_key in adata.obs and spatial is not None:
            truth_values = adata.obs.loc[idx, truth_key]
            pred_values = adata.obs.loc[idx, stair_pred_key]
            valid = ~truth_values.isna() & ~pred_values.isna()
            if int(valid.sum()) > 0:
                rows.append(
                    _metric_row_for_labels(
                        slice_name=slice_name,
                        method="HyperMOA_current",
                        y_true=truth_values.loc[valid],
                        y_pred=pred_values.loc[valid],
                        coords=spatial[idx_pos][valid.to_numpy()],
                    )
                )

    if not rows:
        return pd.DataFrame(columns=["slice", "method", "n_cells", "n_valid", "n_truth", "n_pred", "ARI", "Moran_I"])
    return pd.DataFrame(rows).sort_values(["slice", "method"]).reset_index(drop=True)


def _add_ari_row(metrics_rows: list[dict], adata, slice_name: str, truth_key: str = "truth", pred_key: str = "Domain") -> None:
    if truth_key not in adata.obs or pred_key not in adata.obs:
        return
    if slice_name == "ALL":
        idx = pd.Series(True, index=adata.obs_names)
        truth_use = "truth_prefixed" if "truth_prefixed" in adata.obs else truth_key
    else:
        idx = adata.obs["batch"].astype(str) == slice_name
        truth_use = truth_key

    valid = idx & ~adata.obs[truth_use].isna() & ~adata.obs[pred_key].isna()
    if int(valid.sum()) == 0:
        return
    y_true = adata.obs.loc[valid, truth_use].astype(str)
    y_pred = adata.obs.loc[valid, pred_key].astype(str)
    metrics_rows.append(
        {
            "metric_group": "clustering",
            "metric_name": "ARI",
            "slice": slice_name,
            "slice_pair": "H3K27ac|ATAC",
            "value": float(adjusted_rand_score(y_true, y_pred)),
            "extra": (
                f"truth_key={truth_use};pred_key={pred_key};n_valid={int(valid.sum())};"
                f"n_truth={int(y_true.nunique())};n_pred={int(y_pred.nunique())}"
            ),
        }
    )


def _add_moran_row(metrics_rows: list[dict], adata, slice_name: str, coord_key: str, pred_key: str = "Domain", k: int = 6) -> None:
    if pred_key not in adata.obs or coord_key not in adata.obsm:
        return
    if slice_name == "ALL":
        idx = np.ones(adata.n_obs, dtype=bool)
    else:
        idx = (adata.obs["batch"].astype(str).values == slice_name)
    if int(idx.sum()) < 3:
        return

    labels = adata.obs.loc[idx, pred_key].astype(str)
    domain_codes = pd.Categorical(labels).codes.astype(np.float64)
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float64)[idx]
    metrics_rows.append(
        {
            "metric_group": "moran",
            "metric_name": "Moran's I",
            "slice": slice_name,
            "slice_pair": "H3K27ac|ATAC",
            "value": float(_moran_i_knn(domain_codes, coords, k=k)),
            "extra": f"source=moran_i_domain_knn;coords={coord_key};k={k};n_cells={int(idx.sum())}",
        }
    )


def _plot_spatial_comparison(adata, result_dir: Path, batch_key: str = "batch", truth_key: str = "truth", pred_key: str = "Domain") -> Path | None:
    if truth_key not in adata.obs or pred_key not in adata.obs or "spatial" not in adata.obsm:
        return None

    adata_plot = adata.copy()
    adata_plot.obs[truth_key] = adata_plot.obs[truth_key].astype("category")
    adata_plot.obs[pred_key] = adata_plot.obs[pred_key].astype("category")

    slices = (
        adata_plot.obs[[batch_key, "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")[batch_key]
        .astype(str)
        .tolist()
        if "slice_order" in adata_plot.obs
        else sorted(adata_plot.obs[batch_key].astype(str).unique())
    )
    if len(slices) == 0:
        return None

    fig, axes = plt.subplots(nrows=len(slices), ncols=2, figsize=(14, 6 * len(slices)))
    axes = np.array(axes).reshape(len(slices), 2)

    spot_size_2d = 100
    for i, slice_id in enumerate(slices):
        adata_sub = adata_plot[adata_plot.obs[batch_key].astype(str) == slice_id].copy()

        sc.pl.embedding(
            adata_sub,
            basis="spatial",
            color=truth_key,
            ax=axes[i, 0],
            show=False,
            title=f"Slice: {slice_id} | Ground Truth",
            frameon=False,
            size=spot_size_2d,
            linewidths=0,
            legend_fontsize=10,
        )

        sc.pl.embedding(
            adata_sub,
            basis="spatial",
            color=pred_key,
            ax=axes[i, 1],
            show=False,
            title=f"Slice: {slice_id} | HyperMOA Domain",
            frameon=False,
            size=spot_size_2d,
            linewidths=0,
            legend_fontsize=10,
        )

    plt.tight_layout()
    output_path = result_dir / "spatial_comparison_2D.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def main():
    root_dir = Path(__file__).resolve().parents[2]
    result_dir = root_dir / "Mouse_brain_result" / "H3K27ac_ATAC"
    processed_file = result_dir / "h3k27ac_atac_processed.h5ad"
    final_file = result_dir / "adata.h5ad"
    chamfer_file = result_dir / "chamfer_distance.csv"
    metric_file = result_dir / "metrics_summary.csv"
    other_methods_file = result_dir / "other_methods_by_slice_metrics.csv"

    if not processed_file.exists():
        raise FileNotFoundError("h3k27ac_atac_processed.h5ad not found. Run previous steps first.")

    adata = sc.read_h5ad(processed_file)
    if "transform_init" not in adata.obsm or "transform_fine" not in adata.obsm:
        raise KeyError("transform_init/transform_fine missing. Run 04_location_alignment.py first.")
    _add_truth_labels(adata, root_dir)

    slice_a, slice_b = "H3K27ac", "ATAC"
    batches = adata.obs["batch"].astype(str).values

    rows = []
    stage_points = {
        "input_raw": np.asarray(adata.obsm["spatial"], dtype=np.float64),
        "init_raw": np.asarray(adata.obsm["transform_init"], dtype=np.float64),
        "fine_raw": np.asarray(adata.obsm["transform_fine"], dtype=np.float64),
    }
    for stage, pts in stage_points.items():
        rows.append(_chamfer_row(pts, batches, slice_a, slice_b, stage))

    for stage, pts in stage_points.items():
        pts_norm = _minmax_normalize_by_slice(pts, batches)
        rows.append(_chamfer_row(pts_norm, batches, slice_a, slice_b, f"{stage}_per_slice_minmax"))

    chamfer_df = pd.DataFrame(rows)
    chamfer_df.to_csv(chamfer_file, index=False)

    adata.obsm["rec_3d"] = np.column_stack(
        [
            np.asarray(adata.obsm["transform_fine"], dtype=np.float64),
            adata.obs["z_rec"].astype(float).values if "z_rec" in adata.obs else np.zeros(adata.n_obs),
        ]
    )
    z_rec = adata.obs["z_rec"].astype(float).values if "z_rec" in adata.obs else np.zeros(adata.n_obs)
    xy_aligned_plot = _minmax_normalize_by_slice(adata.obsm["transform_fine"], batches)
    xy_input_plot = _minmax_normalize_by_slice(adata.obsm["spatial"], batches)
    adata.obsm["rec_3d_plot"] = np.column_stack(
        [
            xy_aligned_plot[:, 0],
            xy_aligned_plot[:, 1],
            -z_rec,
        ]
    )
    adata.obsm["gt_3d_order_plot"] = np.column_stack(
        [
            xy_input_plot[:, 0],
            xy_input_plot[:, 1],
            -z_rec,
        ]
    )

    color_key = "Domain" if "Domain" in adata.obs else "batch"
    spatial_comparison_file = _plot_spatial_comparison(adata, result_dir)
    other_methods_df = _other_methods_by_slice_metrics(adata, root_dir)
    other_methods_df.to_csv(other_methods_file, index=False)

    spot_size_3d = 18
    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="rec_3d_plot",
        projection="3d",
        color=color_key,
        s=spot_size_3d,
        linewidths=0,
        show=False,
        title="Mouse brain H3K27ac-ATAC reconstructed 3D",
    )
    plt.savefig(result_dir / "reconstruction_3d_rec.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="gt_3d_order_plot",
        projection="3d",
        color=color_key,
        s=spot_size_3d,
        linewidths=0,
        show=False,
        title="Mouse brain H3K27ac-ATAC reference 3D",
    )
    plt.savefig(result_dir / "reconstruction_3d_reference.png", dpi=300, bbox_inches="tight")
    plt.close()

    metrics_rows = [
        {
            "metric_group": "basic",
            "metric_name": "n_cells",
            "slice": "ALL",
            "slice_pair": "H3K27ac|ATAC",
            "value": float(adata.n_obs),
            "extra": "",
        },
        {
            "metric_group": "basic",
            "metric_name": "n_genes",
            "slice": "ALL",
            "slice_pair": "H3K27ac|ATAC",
            "value": float(adata.n_vars),
            "extra": "",
        },
    ]
    for _, row in chamfer_df.iterrows():
        metrics_rows.append(
            {
                "metric_group": "chamfer",
                "metric_name": str(row["stage"]),
                "slice": "",
                "slice_pair": "H3K27ac|ATAC",
                "value": float(row["cd_sq_symmetric"]),
                "extra": f"cd_sq_a_to_b={row['cd_sq_a_to_b']};cd_sq_b_to_a={row['cd_sq_b_to_a']}",
            }
        )
    if "Domain" in adata.obs:
        metrics_rows.append(
            {
                "metric_group": "clustering",
                "metric_name": "n_domain_clusters",
                "slice": "ALL",
                "slice_pair": "H3K27ac|ATAC",
                "value": float(adata.obs["Domain"].astype(str).nunique()),
                "extra": "",
            }
        )
        _add_ari_row(metrics_rows, adata, slice_name="ALL", truth_key="truth", pred_key="Domain")
        _add_ari_row(metrics_rows, adata, slice_name="H3K27ac", truth_key="truth", pred_key="Domain")
        _add_ari_row(metrics_rows, adata, slice_name="ATAC", truth_key="truth", pred_key="Domain")
        _add_moran_row(metrics_rows, adata, slice_name="ALL", coord_key="rec_3d", pred_key="Domain", k=6)
        _add_moran_row(metrics_rows, adata, slice_name="H3K27ac", coord_key="spatial", pred_key="Domain", k=6)
        _add_moran_row(metrics_rows, adata, slice_name="ATAC", coord_key="spatial", pred_key="Domain", k=6)
    metrics_rows.append(
        {
            "metric_group": "clustering",
            "metric_name": "cluster_method",
            "slice": "ALL",
            "slice_pair": "H3K27ac|ATAC",
            "value": np.nan,
            "extra": str(adata.uns.get("cluster_method", "unknown")),
        }
    )
    if "alignment_rms_init" in adata.uns:
        metrics_rows.append(
            {
                "metric_group": "alignment",
                "metric_name": "rms_displacement_init_vs_input",
                "slice": "ALL",
                "slice_pair": "H3K27ac|ATAC",
                "value": float(adata.uns["alignment_rms_init"]),
                "extra": "",
            }
        )
    if "alignment_rms_fine" in adata.uns:
        metrics_rows.append(
            {
                "metric_group": "alignment",
                "metric_name": "rms_displacement_fine_vs_input",
                "slice": "ALL",
                "slice_pair": "H3K27ac|ATAC",
                "value": float(adata.uns["alignment_rms_fine"]),
                "extra": "",
            }
        )

    for _, row in other_methods_df.iterrows():
        extra = (
            f"method={row['method']};n_valid={int(row['n_valid'])};"
            f"n_truth={int(row['n_truth'])};n_pred={int(row['n_pred'])}"
        )
        metrics_rows.append(
            {
                "metric_group": "other_methods",
                "metric_name": "ARI",
                "slice": str(row["slice"]),
                "slice_pair": "H3K27ac|ATAC",
                "value": float(row["ARI"]),
                "extra": extra,
            }
        )
        metrics_rows.append(
            {
                "metric_group": "other_methods",
                "metric_name": "Moran's I",
                "slice": str(row["slice"]),
                "slice_pair": "H3K27ac|ATAC",
                "value": float(row["Moran_I"]),
                "extra": extra,
            }
        )

    pd.DataFrame(metrics_rows).to_csv(metric_file, index=False)
    adata.write(final_file)

    print(f"Saved Chamfer distances to: {chamfer_file}")
    print(f"Saved other-method metrics to: {other_methods_file}")
    print(f"Saved metrics summary to: {metric_file}")
    print(f"Saved final AnnData to: {final_file}")
    if spatial_comparison_file is not None:
        print(f"Saved 2D spatial comparison plot to: {spatial_comparison_file}")


if __name__ == "__main__":
    main()
