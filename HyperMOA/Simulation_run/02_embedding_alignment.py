import warnings
warnings.filterwarnings("ignore")

import json
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from hypermoa.multi_emb_alignment import Multi_Emb_Align
from hypermoa.utils import cluster_func, set_seed


def _moran_i_knn(values: np.ndarray, coords: np.ndarray, k: int = 10) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    n = values.shape[0]
    if n < 3:
        return np.nan

    k_use = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(coords)
    neigh_idx = nbrs.kneighbors(coords, return_distance=False)[:, 1:]

    z = values - values.mean()
    denom = np.sum(z * z)
    if denom <= 0:
        return np.nan

    w = 1.0 / float(k_use)
    num = np.sum(z[:, None] * z[neigh_idx] * w)
    return float(num / denom)


def _compute_slice_moran(adata, domain_key="Domain", spatial_key="spatial", slice_key="batch", k=6):
    rows = []
    for slice_name in sorted(adata.obs[slice_key].astype(str).unique()):
        adata_tmp = adata[adata.obs[slice_key].astype(str) == slice_name].copy()
        coords = adata_tmp.obsm[spatial_key]
        codes = pd.Categorical(adata_tmp.obs[domain_key].astype(str)).codes.astype(np.float64)
        rows.append(
            {
                "metric_group": "moran",
                "metric_name": "moran_i_domain_knn",
                "slice": str(slice_name),
                "slice_pair": "",
                "value": float(_moran_i_knn(codes, coords, k=k)),
                "extra": f"k={k};n_cells={adata_tmp.n_obs}",
            }
        )
    return rows


def _embedding_diagnostics(adata, rep_keys, batch_key="batch", truth_key="ground_truth", pred_key="Domain"):
    rows = []
    labels = [batch_key]
    if truth_key in adata.obs:
        labels.append(truth_key)

    for rep_key in rep_keys:
        if rep_key not in adata.obsm:
            continue
        x = np.asarray(adata.obsm[rep_key], dtype=np.float64)
        x = StandardScaler().fit_transform(x)
        for label_key in labels:
            y = adata.obs[label_key].astype(str).values
            if np.unique(y).shape[0] < 2:
                continue
            try:
                sil = float(silhouette_score(x, y))
            except Exception:
                sil = np.nan
            acc = np.nan
            min_class = pd.Series(y).value_counts().min()
            if min_class >= 2:
                try:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x,
                        y,
                        test_size=0.35,
                        random_state=42,
                        stratify=y,
                    )
                    clf = LogisticRegression(max_iter=300, class_weight="balanced")
                    clf.fit(x_train, y_train)
                    acc = float(clf.score(x_test, y_test))
                except Exception:
                    acc = np.nan

            rows.append(
                {
                    "rep": rep_key,
                    "label": label_key,
                    "silhouette": sil,
                    "holdout_accuracy": acc,
                }
            )

    if truth_key in adata.obs and pred_key in adata.obs:
        rows.append(
            {
                "rep": pred_key,
                "label": truth_key,
                "silhouette": np.nan,
                "holdout_accuracy": np.nan,
                "ari": float(adjusted_rand_score(adata.obs[truth_key].astype(str), adata.obs[pred_key].astype(str))),
            }
        )

    return pd.DataFrame(rows)


def _add_scaled_rep(adata, source_key: str, target_key: str):
    if source_key not in adata.obsm:
        raise KeyError(f"{source_key!r} not found in adata.obsm.")
    x = np.asarray(adata.obsm[source_key], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"adata.obsm[{source_key!r}] contains NaN or Inf values.")
    adata.obsm[target_key] = StandardScaler().fit_transform(x)
    adata.uns[f"{target_key}_source"] = source_key
    adata.uns[f"{target_key}_transform"] = "standard_scale"
    return adata


def main():
    set_seed(42)

    # Best Simulation tuning result: gamma_0.70, ARI=0.9901795047.
    hvg_top = 1000
    ae_epoch = 150
    ae_batch_size = 256
    loss_weight_rna = 1.0
    loss_weight_adt = 5.0
    adt_loss = "mse"

    hgat_epoch = 150
    hgat_batches = 8
    sim_threshold = 0.30
    c_neigh_het = 0.50
    n_neigh_hom = 10
    n_neigh_het = 30
    gamma = 0.70
    dropout_hom = 0.25
    dropout_het = 0.25
    mini_batch = False

    cluster_num = 5
    mclust_source_key = "HyperMOA_bc"
    mclust_rep_key = "HyperMOA_mclust_scaled"
    mclust_model_name = "EEE"

    root_dir = Path(__file__).resolve().parent.parent
    result_dir = root_dir / "Simulation_result"
    embedding_dir = result_dir / "embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    merged_file = result_dir / "simulation_merged.h5ad"
    processed_file = result_dir / "simulation_processed.h5ad"

    if not merged_file.exists():
        raise FileNotFoundError("simulation_merged.h5ad not found. Run 01_prepare_data.py first.")

    adata = sc.read_h5ad(merged_file)
    adata.obs_names_make_unique()

    batch_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )
    print("Detected slice order:", batch_order)

    hvg_top = min(hvg_top, adata.n_vars)
    emb_align = Multi_Emb_Align(
        adata,
        batch_key="batch",
        hvg=hvg_top,
        n_hidden=128,
        n_latent=32,
        likelihood="nb",
        num_workers=0,
        result_path=str(result_dir),
        atac_key="ADT",
        encode_batch=False,
        decode_batch=True,
    )

    emb_align.prepare(count_key=None, lib_size="explog", normalize=True, scale=False)
    emb_align.preprocess(
        epoch_ae=ae_epoch,
        batch_size=ae_batch_size,
        loss_weight_rna=loss_weight_rna,
        loss_weight_atac=loss_weight_adt,
        atac_loss=adt_loss,
    )
    emb_align.latent()
    emb_align.batch_center_obsm(source_key="latent", target_key="latent_bc", batch_key="batch")

    emb_align.prepare_hgat(
        spatial_key="spatial",
        feat_key="latent_bc",
        slice_order=batch_order,
        n_neigh_hom=n_neigh_hom,
        n_neigh_het=n_neigh_het,
        c_neigh_het=c_neigh_het,
        sim_threshold=sim_threshold,
    )

    emb_align.train_hgat(
        gamma=gamma,
        mini_batch=mini_batch,
        epoch_hgat=hgat_epoch,
        batches=hgat_batches,
        dropout_hom=dropout_hom,
        dropout_het=dropout_het,
    )

    adata, attention = emb_align.predict_hgat(mini_batch=mini_batch, batches=hgat_batches)
    attention_file = embedding_dir / "attention.csv"
    attention.to_csv(attention_file)

    emb_align.batch_center_obsm(source_key="HyperMOA", target_key="HyperMOA_bc", batch_key="batch")
    adata = emb_align.adata
    adata = _add_scaled_rep(
        adata,
        source_key=mclust_source_key,
        target_key=mclust_rep_key,
    )

    adata = cluster_func(
        adata,
        clustering="mclust",
        use_rep=mclust_rep_key,
        cluster_num=cluster_num,
        modelNames=mclust_model_name,
        key_add="HyperMOA",
    )
    cluster_method = "mclust"

    adata.obs["Domain_mclust_global"] = adata.obs["HyperMOA"].astype(str)
    adata.obs["Domain"] = adata.obs["Domain_mclust_global"].astype(str)
    adata.uns["cluster_method"] = cluster_method
    adata.uns["mclust_rep_key"] = mclust_rep_key
    adata.uns["mclust_source_key"] = mclust_source_key
    adata.uns["mclust_transform"] = "standard_scale_no_pca"
    adata.uns["mclust_model_name"] = mclust_model_name
    adata.uns["mclust_cluster_num"] = int(cluster_num)
    adata.obs["cluster_method"] = cluster_method

    moran_rows = _compute_slice_moran(
        adata,
        domain_key="Domain",
        spatial_key="spatial",
        slice_key="batch",
        k=6,
    )
    adata.uns["metrics_moran_rows_json"] = json.dumps(moran_rows, ensure_ascii=True)
    pd.DataFrame(moran_rows).to_csv(result_dir / "moran_scores.csv", index=False)

    diagnostics = _embedding_diagnostics(
        adata,
        rep_keys=["latent", "latent_bc", "HyperMOA", "HyperMOA_bc", "HyperMOA_mclust_scaled"],
        batch_key="batch",
        truth_key="ground_truth",
        pred_key="Domain",
    )
    diagnostics_file = embedding_dir / "embedding_diagnostics.csv"
    diagnostics.to_csv(diagnostics_file, index=False)

    sc.pp.neighbors(adata, use_rep="HyperMOA_bc")
    sc.tl.umap(adata, min_dist=0.2)

    adata.write(processed_file)

    print(f"Clustering method: {cluster_method}")
    print(f"mclust clusters: G={cluster_num}")
    print(f"mclust input: {mclust_rep_key} from {mclust_source_key}, transform=standard_scale_no_pca, model={mclust_model_name}")
    print(f"Saved attention to: {attention_file}")
    print(f"Saved embedding diagnostics to: {diagnostics_file}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
