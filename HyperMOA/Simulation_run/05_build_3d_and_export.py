import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


def _directed_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.nan
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(b)
    dist, _ = nn.kneighbors(a, return_distance=True)
    return float((dist * dist).mean())


def _symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = _directed_chamfer(a, b)
    d_ba = _directed_chamfer(b, a)
    if np.isnan(d_ab) or np.isnan(d_ba):
        return np.nan
    return float(d_ab + d_ba)


def _adjacent_chamfer_table(points: np.ndarray, batches: np.ndarray, order_use) -> pd.DataFrame:
    rows = []
    for i in range(len(order_use) - 1):
        a_name = order_use[i]
        b_name = order_use[i + 1]
        a_pts = points[batches == a_name]
        b_pts = points[batches == b_name]
        rows.append(
            {
                "slice_a": a_name,
                "slice_b": b_name,
                "n_a": int(a_pts.shape[0]),
                "n_b": int(b_pts.shape[0]),
                "cd_sq_a_to_b": _directed_chamfer(a_pts, b_pts),
                "cd_sq_b_to_a": _directed_chamfer(b_pts, a_pts),
                "cd_sq_symmetric": _symmetric_chamfer(a_pts, b_pts),
            }
        )
    return pd.DataFrame(rows)


def _minmax_normalize_by_slice(points: np.ndarray, batches: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    batches = np.asarray(batches).astype(str)
    out = np.zeros_like(points, dtype=np.float64)

    for b in np.unique(batches):
        idx = np.where(batches == b)[0]
        if idx.size == 0:
            continue
        p = points[idx]
        p_min = p.min(axis=0, keepdims=True)
        p_max = p.max(axis=0, keepdims=True)
        span = p_max - p_min
        span[span <= 1e-12] = 1.0
        out[idx] = (p - p_min) / span
    return out


def _pick_hvg_mask(adata, top_fallback: int = 2000) -> np.ndarray:
    if "highly_variable" in adata.var:
        mask = np.asarray(adata.var["highly_variable"].values).astype(bool)
        if int(mask.sum()) > 1:
            return mask

    x = adata.X
    if sparse.issparse(x):
        x2_mean = np.asarray(x.power(2).mean(axis=0)).ravel()
        x_mean = np.asarray(x.mean(axis=0)).ravel()
    else:
        x = np.asarray(x)
        x2_mean = np.mean(x * x, axis=0)
        x_mean = np.mean(x, axis=0)

    var = x2_mean - x_mean * x_mean
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    n_vars = int(var.shape[0])
    top_k = int(min(max(2, top_fallback), n_vars))
    top_idx = np.argpartition(var, -top_k)[-top_k:]
    mask = np.zeros(n_vars, dtype=bool)
    mask[top_idx] = True
    return mask


def _collect_cross_slice_nearest_pairs(coords_3d: np.ndarray, batches: np.ndarray) -> np.ndarray:
    batches = np.asarray(batches).astype(str)
    unique_batches = np.unique(batches)
    directed_pairs = []

    for batch_name in unique_batches:
        idx_src = np.where(batches == batch_name)[0]
        idx_tgt = np.where(batches != batch_name)[0]
        if idx_src.size == 0 or idx_tgt.size == 0:
            continue

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(coords_3d[idx_tgt])
        _, tgt_local = nn.kneighbors(coords_3d[idx_src], return_distance=True)
        tgt_idx = idx_tgt[tgt_local[:, 0]]
        directed_pairs.append(np.column_stack([idx_src, tgt_idx]))

    if not directed_pairs:
        return np.empty((0, 2), dtype=np.int64)

    pairs = np.vstack(directed_pairs).astype(np.int64)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs


def _row_as_dense(x, i: int) -> np.ndarray:
    if sparse.issparse(x):
        return x.getrow(i).toarray().ravel().astype(np.float64)
    return np.asarray(x[i]).ravel().astype(np.float64)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= 1 or y.size <= 1:
        return np.nan

    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if denom <= 0:
        return np.nan
    return float(np.sum(xm * ym) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_corr(rx, ry)


def _pairwise_pearson_for_pairs(matrix, pairs: np.ndarray) -> np.ndarray:
    out = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
    for i, (idx_a, idx_b) in enumerate(pairs):
        vec_a = _row_as_dense(matrix, int(idx_a))
        vec_b = _row_as_dense(matrix, int(idx_b))
        out[i] = _pearson_corr(vec_a, vec_b)
    return out


def _cross_slice_expression_corr_table(
    adata,
    coords_3d_key: str = "rec_3d",
    batch_key: str = "batch",
    hvg_fallback_top: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if coords_3d_key not in adata.obsm:
        raise KeyError(f"{coords_3d_key} not found in adata.obsm")
    if batch_key not in adata.obs:
        raise KeyError(f"{batch_key} not found in adata.obs")

    coords_3d = np.asarray(adata.obsm[coords_3d_key], dtype=np.float64)
    batches = adata.obs[batch_key].astype(str).values
    pairs = _collect_cross_slice_nearest_pairs(coords_3d=coords_3d, batches=batches)

    hvg_mask = _pick_hvg_mask(adata, top_fallback=hvg_fallback_top)
    n_hvg = int(hvg_mask.sum())
    x_hvg = adata[:, hvg_mask].X

    pearson_rna = _pairwise_pearson_for_pairs(x_hvg, pairs)

    spearman_rna = np.full((pairs.shape[0],), np.nan, dtype=np.float64)
    for i, (idx_a, idx_b) in enumerate(pairs):
        vec_a = _row_as_dense(x_hvg, int(idx_a))
        vec_b = _row_as_dense(x_hvg, int(idx_b))
        spearman_rna[i] = _spearman_corr(vec_a, vec_b)

    if "ADT" in adata.obsm:
        pearson_adt = _pairwise_pearson_for_pairs(adata.obsm["ADT"], pairs)
    else:
        pearson_adt = np.full((pairs.shape[0],), np.nan, dtype=np.float64)

    if "HyperMOA" in adata.obsm:
        pearson_hypermoa = _pairwise_pearson_for_pairs(adata.obsm["HyperMOA"], pairs)
    else:
        pearson_hypermoa = np.full((pairs.shape[0],), np.nan, dtype=np.float64)

    pair_df = pd.DataFrame(
        {
            "cell_a": [str(adata.obs_names[int(i)]) for i in pairs[:, 0]] if pairs.size else [],
            "cell_b": [str(adata.obs_names[int(i)]) for i in pairs[:, 1]] if pairs.size else [],
            "slice_a": [str(batches[int(i)]) for i in pairs[:, 0]] if pairs.size else [],
            "slice_b": [str(batches[int(i)]) for i in pairs[:, 1]] if pairs.size else [],
            "distance_3d": [
                float(np.linalg.norm(coords_3d[int(i)] - coords_3d[int(j)])) for i, j in pairs
            ]
            if pairs.size
            else [],
            "pearson_hvg": pearson_rna,
            "spearman_hvg": spearman_rna,
            "pearson_adt": pearson_adt,
            "pearson_hypermoa": pearson_hypermoa,
            "n_hvg": [n_hvg] * pairs.shape[0],
        }
    )

    summary_rows = [
        {
            "level": "global",
            "slice_pair": "ALL",
            "n_pairs": int(pair_df.shape[0]),
            "n_hvg": n_hvg,
            "pearson_mean": float(pair_df["pearson_hvg"].mean()) if not pair_df.empty else np.nan,
            "pearson_median": float(pair_df["pearson_hvg"].median()) if not pair_df.empty else np.nan,
            "spearman_mean": float(pair_df["spearman_hvg"].mean()) if not pair_df.empty else np.nan,
            "spearman_median": float(pair_df["spearman_hvg"].median()) if not pair_df.empty else np.nan,
            "pearson_adt_mean": float(pair_df["pearson_adt"].mean()) if not pair_df.empty else np.nan,
            "pearson_hypermoa_mean": float(pair_df["pearson_hypermoa"].mean()) if not pair_df.empty else np.nan,
            "distance_3d_mean": float(pair_df["distance_3d"].mean()) if not pair_df.empty else np.nan,
        }
    ]

    if not pair_df.empty:
        tmp = pair_df.copy()
        tmp["slice_pair"] = tmp.apply(
            lambda r: "|".join(sorted([str(r["slice_a"]), str(r["slice_b"])])),
            axis=1,
        )
        for pair_name, g in tmp.groupby("slice_pair", sort=True):
            summary_rows.append(
                {
                    "level": "slice_pair",
                    "slice_pair": str(pair_name),
                    "n_pairs": int(g.shape[0]),
                    "n_hvg": n_hvg,
                    "pearson_mean": float(g["pearson_hvg"].mean()),
                    "pearson_median": float(g["pearson_hvg"].median()),
                    "spearman_mean": float(g["spearman_hvg"].mean()),
                    "spearman_median": float(g["spearman_hvg"].median()),
                    "pearson_adt_mean": float(g["pearson_adt"].mean()),
                    "pearson_hypermoa_mean": float(g["pearson_hypermoa"].mean()),
                    "distance_3d_mean": float(g["distance_3d"].mean()),
                }
            )

    return pair_df, pd.DataFrame(summary_rows)


def _clustering_score_rows(adata, truth_key: str = "ground_truth", pred_key: str = "Domain", batch_key: str = "batch"):
    rows = []
    if truth_key not in adata.obs or pred_key not in adata.obs:
        return rows

    scorers = [
        ("ari", adjusted_rand_score),
        ("nmi", normalized_mutual_info_score),
        ("ami", adjusted_mutual_info_score),
    ]

    valid_mask = ~adata.obs[truth_key].isna() & ~adata.obs[pred_key].isna()
    if int(valid_mask.sum()) == 0:
        return rows

    y_true_all = adata.obs.loc[valid_mask, truth_key].astype(str)
    y_pred_all = adata.obs.loc[valid_mask, pred_key].astype(str)
    for metric_prefix, scorer in scorers:
        rows.append(
            {
                "metric_group": "clustering",
                "metric_name": f"{metric_prefix}_global",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(scorer(y_true_all, y_pred_all)),
                "extra": f"truth_key={truth_key};pred_key={pred_key};n_valid={int(valid_mask.sum())}",
            }
        )

    batch_values = adata.obs[batch_key].astype(str)
    for batch_name in sorted(batch_values.unique()):
        mask_batch = valid_mask & (batch_values == batch_name)
        if int(mask_batch.sum()) == 0:
            continue
        y_true_batch = adata.obs.loc[mask_batch, truth_key].astype(str)
        y_pred_batch = adata.obs.loc[mask_batch, pred_key].astype(str)
        for metric_prefix, scorer in scorers:
            rows.append(
                {
                    "metric_group": "clustering",
                    "metric_name": f"{metric_prefix}_by_slice",
                    "slice": str(batch_name),
                    "slice_pair": "",
                    "value": float(scorer(y_true_batch, y_pred_batch)),
                    "extra": f"truth_key={truth_key};pred_key={pred_key};n_valid={int(mask_batch.sum())}",
                }
            )
    return rows


def _domain_coverage_rows(adata, truth_key: str = "ground_truth", pred_key: str = "Domain", batch_key: str = "batch"):
    rows = []
    if pred_key not in adata.obs or batch_key not in adata.obs:
        return rows

    batch_values = adata.obs[batch_key].astype(str)
    pred_all = adata.obs[pred_key].dropna().astype(str)
    rows.append(
        {
            "metric_group": "domain_coverage",
            "metric_name": "n_pred_domains_global",
            "slice": "ALL",
            "slice_pair": "ALL",
            "value": float(pred_all.nunique()),
            "extra": "|".join(sorted(pred_all.unique().tolist())),
        }
    )

    if truth_key in adata.obs:
        truth_all = adata.obs[truth_key].dropna().astype(str)
        rows.append(
            {
                "metric_group": "domain_coverage",
                "metric_name": "n_truth_structures_global",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(truth_all.nunique()),
                "extra": "|".join(sorted(truth_all.unique().tolist())),
            }
        )

    for batch_name in sorted(batch_values.unique()):
        idx = batch_values == batch_name
        pred_values = adata.obs.loc[idx, pred_key].dropna().astype(str)
        rows.append(
            {
                "metric_group": "domain_coverage",
                "metric_name": "n_pred_domains_by_slice",
                "slice": str(batch_name),
                "slice_pair": "",
                "value": float(pred_values.nunique()),
                "extra": "|".join(sorted(pred_values.unique().tolist())),
            }
        )
        if truth_key in adata.obs:
            truth_values = adata.obs.loc[idx, truth_key].dropna().astype(str)
            rows.append(
                {
                    "metric_group": "domain_coverage",
                    "metric_name": "n_truth_structures_by_slice",
                    "slice": str(batch_name),
                    "slice_pair": "",
                    "value": float(truth_values.nunique()),
                    "extra": "|".join(sorted(truth_values.unique().tolist())),
                }
            )
    return rows


def _write_metrics_summary(adata, result_dir: Path, chamfer_adjacent: pd.DataFrame, corr_summary: pd.DataFrame) -> None:
    rows = []
    batches = adata.obs["batch"].astype(str).values
    rows.extend(
        [
            {
                "metric_group": "basic",
                "metric_name": "n_cells",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(adata.n_obs),
                "extra": "",
            },
            {
                "metric_group": "basic",
                "metric_name": "n_genes",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(adata.n_vars),
                "extra": "",
            },
            {
                "metric_group": "basic",
                "metric_name": "n_adt",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(np.asarray(adata.obsm["ADT"]).shape[1]) if "ADT" in adata.obsm else 0.0,
                "extra": "",
            },
        ]
    )

    for s in sorted(np.unique(batches)):
        rows.append(
            {
                "metric_group": "basic",
                "metric_name": "n_cells_per_slice",
                "slice": str(s),
                "slice_pair": "",
                "value": float(np.sum(batches == s)),
                "extra": "",
            }
        )

    if "alignment_rms_init" in adata.uns:
        rows.append(
            {
                "metric_group": "alignment",
                "metric_name": "rms_displacement_init_vs_input",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(adata.uns["alignment_rms_init"]),
                "extra": "",
            }
        )
    if "alignment_rms_fine" in adata.uns:
        rows.append(
            {
                "metric_group": "alignment",
                "metric_name": "rms_displacement_fine_vs_input",
                "slice": "ALL",
                "slice_pair": "ALL",
                "value": float(adata.uns["alignment_rms_fine"]),
                "extra": "",
            }
        )

    for _, row in chamfer_adjacent.iterrows():
        rows.append(
            {
                "metric_group": "chamfer",
                "metric_name": "cd_sq_symmetric_adjacent",
                "slice": "",
                "slice_pair": f"{row['slice_a']}|{row['slice_b']}",
                "value": float(row["cd_sq_symmetric"]),
                "extra": f"n_a={int(row['n_a'])};n_b={int(row['n_b'])}",
            }
        )

    for _, row in corr_summary.iterrows():
        rows.append(
            {
                "metric_group": "cross_slice_corr",
                "metric_name": "pearson_hvg_mean",
                "slice": "",
                "slice_pair": str(row["slice_pair"]),
                "value": float(row["pearson_mean"]),
                "extra": f"level={row['level']};n_pairs={int(row['n_pairs'])};n_hvg={int(row['n_hvg'])}",
            }
        )
        rows.append(
            {
                "metric_group": "cross_slice_corr",
                "metric_name": "pearson_adt_mean",
                "slice": "",
                "slice_pair": str(row["slice_pair"]),
                "value": float(row["pearson_adt_mean"]),
                "extra": f"level={row['level']};n_pairs={int(row['n_pairs'])}",
            }
        )
        rows.append(
            {
                "metric_group": "cross_slice_corr",
                "metric_name": "pearson_hypermoa_mean",
                "slice": "",
                "slice_pair": str(row["slice_pair"]),
                "value": float(row["pearson_hypermoa_mean"]),
                "extra": f"level={row['level']};n_pairs={int(row['n_pairs'])}",
            }
        )

    rows.extend(_clustering_score_rows(adata, truth_key="ground_truth", pred_key="Domain", batch_key="batch"))
    rows.extend(_domain_coverage_rows(adata, truth_key="ground_truth", pred_key="Domain", batch_key="batch"))

    if "metrics_moran_rows_json" in adata.uns:
        try:
            import json

            rows.extend(json.loads(adata.uns["metrics_moran_rows_json"]))
        except Exception:
            pass

    pd.DataFrame(rows).to_csv(result_dir / "metrics_summary.csv", index=False)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    result_dir = root_dir / "Simulation_result"
    export_dir = result_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    processed_file = result_dir / "simulation_processed.h5ad"
    final_file = result_dir / "adata.h5ad"
    order_file = result_dir / "predicted_slice_order.csv"

    if not processed_file.exists():
        raise FileNotFoundError("simulation_processed.h5ad not found. Run 04_location_alignment.py first.")

    adata = sc.read_h5ad(processed_file)
    if "transform_fine" not in adata.obsm:
        raise KeyError("transform_fine not found in adata.obsm. Run 04_location_alignment.py first.")
    if "z_rec" not in adata.obs:
        raise KeyError("z_rec not found in adata.obs. Run 03_slice_order_and_z_reconstruction.py first.")

    if order_file.exists():
        order_df = pd.read_csv(order_file)
        order_use = order_df.sort_values("z_rec")["batch"].astype(str).tolist()
    else:
        order_use = sorted(adata.obs["batch"].astype(str).unique())

    coords = np.asarray(adata.obsm["transform_fine"], dtype=np.float64)
    z_rec = adata.obs["z_rec"].astype(float).values
    batches = adata.obs["batch"].astype(str).values

    adata.obs["x_aligned"] = coords[:, 0]
    adata.obs["y_aligned"] = coords[:, 1]
    adata.obsm["rec_3d"] = np.column_stack([coords, z_rec])
    adata.obsm["rec_3d_norm"] = _minmax_normalize_by_slice(adata.obsm["rec_3d"], batches)

    xy_aligned_plot = _minmax_normalize_by_slice(adata.obsm["transform_fine"], batches)
    xy_input_plot = _minmax_normalize_by_slice(adata.obsm["spatial"], batches)

    adata.obsm["rec_3d_plot"] = np.column_stack([xy_aligned_plot[:, 0], xy_aligned_plot[:, 1], -z_rec])
    adata.obsm["gt_3d_order_plot"] = np.column_stack([xy_input_plot[:, 0], xy_input_plot[:, 1], -z_rec])

    color_key = "Domain" if "Domain" in adata.obs else "batch"
    spot_size_3d = 18
    spot_size_2d = 260
    plt.figure(figsize=(5.8, 5.2))
    sc.pl.embedding(
        adata,
        basis="rec_3d_plot",
        projection="3d",
        color=color_key,
        s=spot_size_3d,
        linewidths=0,
        show=False,
        title="Simulation reconstructed 3D",
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
        title="Simulation reference 3D (x,y,z_rec)",
    )
    plt.savefig(result_dir / "reconstruction_3d_reference.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "spatial" in adata.obsm:
        n_slice = len(order_use)
        n_col = min(4, max(1, n_slice))
        n_row = int(np.ceil(n_slice / n_col))
        fig, axs = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 3.8 * n_row), constrained_layout=True)
        axs = np.array(axs).reshape(n_row, n_col)

        x_min, x_max = adata.obsm["spatial"][:, 0].min(), adata.obsm["spatial"][:, 0].max()
        y_min, y_max = adata.obsm["spatial"][:, 1].min(), adata.obsm["spatial"][:, 1].max()

        idx = 0
        for i in range(n_row):
            for j in range(n_col):
                ax = axs[i, j]
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.axis("off")
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])

                if idx < n_slice:
                    batch_name = order_use[idx]
                    adata_tmp = adata[adata.obs["batch"].astype(str) == batch_name].copy()
                    show_legend = idx == n_slice - 1
                    sc.pl.embedding(
                        adata_tmp,
                        basis="spatial",
                        color=color_key,
                        title=str(batch_name),
                        frameon=False,
                        legend_loc="right margin" if show_legend else None,
                        s=spot_size_2d,
                        linewidths=0,
                        show=False,
                        ax=ax,
                    )
                idx += 1

        plt.savefig(result_dir / "spatial_domains_2d.png", dpi=300, bbox_inches="tight")
        plt.close()

    chamfer_adjacent = _adjacent_chamfer_table(adata.obsm["rec_3d_norm"], batches, order_use)
    chamfer_adjacent.to_csv(result_dir / "chamfer_adjacent_slices.csv", index=False)

    pair_df, summary_df = _cross_slice_expression_corr_table(adata, coords_3d_key="rec_3d_norm")
    pair_df.to_csv(result_dir / "cross_slice_nearest_expr_corr_pairs.csv", index=False)
    summary_df.to_csv(result_dir / "cross_slice_nearest_expr_corr_summary.csv", index=False)

    _write_metrics_summary(adata, result_dir, chamfer_adjacent, summary_df)

    plt.figure(figsize=(5.5, 4.5))
    sc.pl.embedding(
        adata,
        basis="rec_3d_norm",
        color=["batch", "Domain"],
        frameon=False,
        ncols=2,
        show=False,
    )
    plt.savefig(export_dir / "rec_3d_batch_domain.png", dpi=300, bbox_inches="tight")
    plt.close()

    adata.write(processed_file)
    adata.write(final_file)

    print(f"Updated processed data: {processed_file}")
    print(f"Saved final AnnData to: {final_file}")
    print(f"Saved exports to: {export_dir}")


if __name__ == "__main__":
    main()
