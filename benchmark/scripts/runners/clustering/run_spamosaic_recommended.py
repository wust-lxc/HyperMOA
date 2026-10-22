#!/usr/bin/env python

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import os
import sys
import time
import warnings
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[4]
SPAMOSAIC = ROOT / "baselines/SpaMosaic-main"
OUT = ROOT / "benchmark/results/clustering" / "spamosaic_recommended_clustering"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SPAMOSAIC))

from spamosaic.framework import SpaMosaic  # noqa: E402
from spamosaic.preprocessing import ADT_preprocess, RNA_preprocess  # noqa: E402
from spamosaic import utils as utls  # noqa: E402


DEVICE = "cuda:0"


def ensure_counts_layer(adata):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    return adata


def prefix_obs_names(adata, prefix):
    adata.obs_names = [f"{prefix}_{name}" for name in adata.obs_names]
    return adata


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def moran_i_knn(values, coords, k=6):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    n = values.shape[0]
    if n < 3:
        return np.nan
    k_use = max(1, min(int(k), n - 1))
    neigh_idx = NearestNeighbors(n_neighbors=k_use + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    z = values - values.mean()
    denom = np.sum(z * z)
    if denom <= 0:
        return np.nan
    return float(np.sum(z[:, None] * z[neigh_idx] * (1.0 / k_use)) / denom)


def run_mclust(adata, n_clusters, used_obsm="merged_emb", key="mclust", random_seed=2020, model_names="EEE"):
    import rpy2.robjects as ro

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    np.random.seed(random_seed)
    ro.r["set.seed"](random_seed)
    ro.r.library("mclust")
    r_x = ro.r.matrix(ro.FloatVector(x.ravel(order="F")), nrow=x.shape[0], ncol=x.shape[1])
    ro.globalenv[".spamosaic_mclust_data"] = r_x
    ro.globalenv[".spamosaic_mclust_g"] = ro.IntVector([int(n_clusters)])
    ro.globalenv[".spamosaic_mclust_model"] = ro.StrVector([str(model_names)])
    fit = ro.r(
        "Mclust(data=.spamosaic_mclust_data, "
        "G=.spamosaic_mclust_g, modelNames=.spamosaic_mclust_model)"
    )
    labels = np.asarray(fit.rx2("classification"), dtype=np.int64)
    ro.r(
        "rm(.spamosaic_mclust_data, .spamosaic_mclust_g, "
        ".spamosaic_mclust_model, envir=.GlobalEnv)"
    )
    adata.obs[key] = labels.astype(int).astype(str)
    adata.obs[key] = adata.obs[key].astype("category")
    return adata


def cluster_global(adata, n_clusters, used_obsm, algo, key):
    if algo == "mclust":
        return run_mclust(adata, n_clusters=n_clusters, used_obsm=used_obsm, key=key)
    if algo == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(adata.obsm[used_obsm])
        adata.obs[key] = km.labels_.astype(str)
        adata.obs[key] = adata.obs[key].astype("category")
        return adata
    raise ValueError(f"Unsupported clustering algorithm: {algo}")


def split_global_obs(ad_embs, ad_mosaic, key):
    utls.split_adata_ob(ad_embs, ad_mosaic, "obs", key)
    for adata in ad_embs:
        adata.obs["SpaMosaic"] = adata.obs[key].astype(str).astype("category")


def metric_rows(dataset, ad_embs, truth_key, requested_clusters, config, elapsed):
    rows = []
    for idx, adata in enumerate(ad_embs, start=1):
        truth = adata.obs[truth_key].astype(str)
        pred = adata.obs["SpaMosaic"].astype(str)
        valid = truth.notna() & pred.notna() & (truth != "nan") & (pred != "nan")
        y = truth[valid].to_numpy()
        p = pred[valid].to_numpy()
        coords = np.asarray(adata.obsm["spatial"])[valid.to_numpy()]
        rows.append({
            "dataset": dataset,
            "slice": f"slice{idx}" if dataset == "Human_lymph_node" else f"Simulation{idx}",
            "method": "SpaMosaic",
            "n_obs_input": int(adata.n_obs),
            "n_obs_valid": int(valid.sum()),
            "truth_classes": int(pd.Series(y).nunique()),
            "requested_clusters": int(requested_clusters),
            "pred_clusters": int(pd.Series(p).nunique()),
            "ARI": adjusted_rand_score(y, p),
            "NMI": normalized_mutual_info_score(y, p),
            "AMI": adjusted_mutual_info_score(y, p),
            "Moran's I": moran_i_knn(pd.Categorical(p).codes, coords, k=6),
            "elapsed_seconds_dataset": float(elapsed),
            "config": config,
        })
    return rows


def save_outputs(dataset, ad_embs, ad_mosaic):
    out_dir = OUT / "h5ad" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, adata in enumerate(ad_embs, start=1):
        name = f"slice{i}" if dataset == "Human_lymph_node" else f"Simulation{i}"
        adata.write(out_dir / f"{name}_spamosaic_recommended.h5ad")
    ad_mosaic.write(out_dir / f"{dataset}_spamosaic_recommended_merged.h5ad")


def run_human():
    start = time.time()
    data_dir = ROOT / 'data/datasets/Human_lymph_node'
    ad1_rna = sc.read_h5ad(data_dir / "slice1" / "s1_adata_rna.h5ad")
    ad1_adt = sc.read_h5ad(data_dir / "slice1" / "s1_adata_adt.h5ad")
    ad2_rna = sc.read_h5ad(data_dir / "slice2" / "s2_adata_rna.h5ad")
    ad3_adt = sc.read_h5ad(data_dir / "slice3" / "s3_adata_adt.h5ad")

    input_dict = {
        "rna": [ad1_rna, ad2_rna, None],
        "adt": [ad1_adt, None, ad3_adt],
    }
    input_key = "dimred_bc"
    RNA_preprocess(input_dict["rna"], batch_corr=True, favor="scanpy", n_hvg=5000, batch_key="src", key=input_key)
    ADT_preprocess(input_dict["adt"], batch_corr=True, batch_key="src", key=input_key)

    model = SpaMosaic(
        modBatch_dict=input_dict,
        input_key=input_key,
        batch_key="src",
        intra_knns=10,
        inter_knn_base=10,
        w_g=0.8,
        seed=1234,
        device=DEVICE,
    )
    model.train(net="wlgcn", lr=0.01, T=0.01, n_epochs=100)
    ad_embs = model.infer_emb(input_dict, emb_key="emb", final_latent_key="merged_emb")
    ad_mosaic = sc.concat(ad_embs)
    cluster_global(ad_mosaic, n_clusters=10, used_obsm="merged_emb", algo="mclust", key="mclust")
    split_global_obs(ad_embs, ad_mosaic, "mclust")
    elapsed = time.time() - start
    save_outputs("Human_lymph_node", ad_embs, ad_mosaic)
    config = "SpaMosaic Human demo recommended: mosaic RNA/ADT, dimred_bc, batch_corr=True, intra_knns=10, inter_knn_base=10, w_g=0.8, wlgcn, lr=0.01, T=0.01, n_epochs=100, global mclust=10"
    return metric_rows("Human_lymph_node", ad_embs, "final_annot", 10, config, elapsed)


def run_simulation():
    start = time.time()
    ad1_rna = ensure_counts_layer(sc.read_h5ad(ROOT / 'data/datasets/Simulation/slice1' / "adata_RNA.h5ad"))
    ad1_adt = ensure_counts_layer(sc.read_h5ad(ROOT / 'data/datasets/Simulation/slice1' / "adata_ADT.h5ad"))
    ad2_rna = ensure_counts_layer(sc.read_h5ad(ROOT / 'data/datasets/Simulation/slice2' / "adata_RNA.h5ad"))
    ad3_adt = ensure_counts_layer(sc.read_h5ad(ROOT / 'data/datasets/Simulation/slice3' / "adata_ADT.h5ad"))

    prefix_obs_names(ad1_rna, "Simulation1")
    prefix_obs_names(ad1_adt, "Simulation1")
    prefix_obs_names(ad2_rna, "Simulation2")
    prefix_obs_names(ad3_adt, "Simulation3")

    ad1_rna.obs["src"] = ad1_adt.obs["src"] = ["mult"] * ad1_rna.n_obs
    ad2_rna.obs["src"] = ["rna"] * ad2_rna.n_obs
    ad3_adt.obs["src"] = ["adt"] * ad3_adt.n_obs
    for adata in (ad1_rna, ad1_adt, ad2_rna, ad3_adt):
        adata.X = adata.layers["counts"].copy()

    input_dict = {
        "rna": [ad1_rna, ad2_rna, None],
        "adt": [ad1_adt, None, ad3_adt],
    }
    input_key = "X_pca_be"
    RNA_preprocess(
        input_dict["rna"],
        batch_corr=True,
        favor="scanpy",
        n_hvg=None,
        lognorm=False,
        scale=True,
        batch_key="src",
        key=input_key,
    )
    ADT_preprocess(
        input_dict["adt"],
        batch_corr=True,
        favor="scanpy",
        lognorm=False,
        scale=True,
        batch_key="src",
        key=input_key,
    )

    model = SpaMosaic(
        modBatch_dict=input_dict,
        input_key=input_key,
        batch_key="src",
        intra_knns=10,
        inter_knn_base=10,
        w_g=0.8,
        seed=1234,
        device=DEVICE,
    )
    model.train(net="wlgcn", lr=0.01, T=0.1, n_epochs=100)
    ad_embs = model.infer_emb(input_dict, emb_key="emb", final_latent_key="merged_emb")
    ad_mosaic = sc.concat(ad_embs)
    cluster_global(ad_mosaic, n_clusters=5, used_obsm="merged_emb", algo="kmeans", key="kmeans")
    split_global_obs(ad_embs, ad_mosaic, "kmeans")
    elapsed = time.time() - start
    save_outputs("Simulation", ad_embs, ad_mosaic)
    config = "SpaMosaic Simulation demo recommended: mosaic RNA/ADT, X_pca_be, batch_corr=True, intra_knns=10, inter_knn_base=10, w_g=0.8, wlgcn, lr=0.01, T=0.1, n_epochs=100, global kmeans=5"
    return metric_rows("Simulation", ad_embs, "truth", 5, config, elapsed)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Human_lymph_node", "Simulation"])
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        if dataset == "Human_lymph_node":
            print("[SpaMosaic] Running Human_lymph_node recommended demo config", flush=True)
            rows.extend(run_human())
        elif dataset == "Simulation":
            print("[SpaMosaic] Running Simulation recommended demo config", flush=True)
            rows.extend(run_simulation())
        else:
            raise ValueError(f"No SpaMosaic recommended demo config is available for {dataset!r}.")

        partial = pd.DataFrame(rows)
        partial.to_csv(OUT / "spamosaic_recommended_clustering_per_slice.partial.csv", index=False)
        print(partial.tail(len(rows)).to_string(index=False), flush=True)

    per = pd.DataFrame(rows)
    summary = per.groupby(["dataset", "method"], as_index=False).agg({
        "ARI": "mean",
        "NMI": "mean",
        "AMI": "mean",
        "Moran's I": "mean",
        "pred_clusters": ["min", "max"],
    })
    summary.columns = ["dataset", "method", "ARI", "NMI", "AMI", "Moran's I", "pred_clusters_min", "pred_clusters_max"]

    per_path = OUT / "spamosaic_recommended_clustering_per_slice.csv"
    sum_path = OUT / "spamosaic_recommended_clustering_summary.csv"
    per_round_path = OUT / "spamosaic_recommended_clustering_per_slice_3dec.csv"
    sum_round_path = OUT / "spamosaic_recommended_clustering_summary_3dec.csv"
    per.to_csv(per_path, index=False)
    summary.to_csv(sum_path, index=False)
    round_cols = ["ARI", "NMI", "AMI", "Moran's I"]
    per_round = per.copy()
    summary_round = summary.copy()
    per_round[round_cols] = per_round[round_cols].round(3)
    summary_round[round_cols] = summary_round[round_cols].round(3)
    per_round.to_csv(per_round_path, index=False)
    summary_round.to_csv(sum_round_path, index=False)

    print("\nPER_SLICE_3DEC", flush=True)
    print(per_round[["dataset", "slice", "requested_clusters", "pred_clusters", "ARI", "NMI", "AMI", "Moran's I"]].to_string(index=False), flush=True)
    print("\nSUMMARY_3DEC", flush=True)
    print(summary_round.to_string(index=False), flush=True)
    print("\nSaved:", flush=True)
    for path in (per_path, sum_path, per_round_path, sum_round_path):
        print(path, flush=True)


if __name__ == "__main__":
    main()
