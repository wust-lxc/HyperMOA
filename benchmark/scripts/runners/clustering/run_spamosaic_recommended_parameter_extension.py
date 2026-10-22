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

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[4]
SPAMOSAIC = ROOT / "baselines/SpaMosaic-main"
OUT = ROOT / "benchmark/results/clustering" / "spamosaic_recommended_parameter_extension"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SPAMOSAIC))

from spamosaic.framework import SpaMosaic  # noqa: E402
from spamosaic.preprocessing import ADT_preprocess, Epigenome_preprocess, RNA_preprocess  # noqa: E402
from spamosaic import utils as utls  # noqa: E402


DEVICE = "cuda:0"


def ensure_counts_layer(adata):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    return adata


def prefix_obs_names(adata, prefix):
    adata.obs_names = [f"{prefix}_{name}" for name in adata.obs_names]
    return adata


def align_pair(ad1, ad2):
    common = ad1.obs_names.intersection(ad2.obs_names)
    if len(common) == 0:
        raise ValueError("paired AnnData objects have no shared obs_names")
    return ad1[common].copy(), ad2[common].copy()


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
    ro.globalenv[".spamosaic_ext_mclust_data"] = r_x
    ro.globalenv[".spamosaic_ext_mclust_g"] = ro.IntVector([int(n_clusters)])
    ro.globalenv[".spamosaic_ext_mclust_model"] = ro.StrVector([str(model_names)])
    fit = ro.r(
        "Mclust(data=.spamosaic_ext_mclust_data, "
        "G=.spamosaic_ext_mclust_g, modelNames=.spamosaic_ext_mclust_model)"
    )
    labels = np.asarray(fit.rx2("classification"), dtype=np.int64)
    ro.r(
        "rm(.spamosaic_ext_mclust_data, .spamosaic_ext_mclust_g, "
        ".spamosaic_ext_mclust_model, envir=.GlobalEnv)"
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


def metric_rows(dataset, ad_embs, slice_names, truth_key, requested_clusters, config, elapsed):
    rows = []
    for adata, slice_name in zip(ad_embs, slice_names):
        truth = adata.obs[truth_key].astype(str)
        pred = adata.obs["SpaMosaic"].astype(str)
        valid = truth.notna() & pred.notna() & (truth != "nan") & (pred != "nan")
        y = truth[valid].to_numpy()
        p = pred[valid].to_numpy()
        coords = np.asarray(adata.obsm["spatial"])[valid.to_numpy()]
        rows.append({
            "dataset": dataset,
            "slice": slice_name,
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


def save_outputs(dataset, ad_embs, slice_names, ad_mosaic):
    out_dir = OUT / "h5ad" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for adata, slice_name in zip(ad_embs, slice_names):
        adata.write(out_dir / f"{slice_name}_spamosaic_extension.h5ad")
    ad_mosaic.write(out_dir / f"{dataset}_spamosaic_extension_merged.h5ad")


def run_simulation_all5():
    start = time.time()
    rna_ads, adt_ads, slice_names = [], [], []
    for i in range(1, 6):
        name = f"Simulation{i}"
        directory = ROOT / "data/datasets/Simulation" / f"slice{i}"
        rna = ensure_counts_layer(sc.read_h5ad(directory / "adata_RNA.h5ad"))
        adt = ensure_counts_layer(sc.read_h5ad(directory / "adata_ADT.h5ad"))
        rna, adt = align_pair(rna, adt)
        prefix_obs_names(rna, name)
        prefix_obs_names(adt, name)
        rna.obs["src"] = name
        adt.obs["src"] = name
        rna.X = rna.layers["counts"].copy()
        adt.X = adt.layers["counts"].copy()
        rna_ads.append(rna)
        adt_ads.append(adt)
        slice_names.append(name)

    input_dict = {"rna": rna_ads, "adt": adt_ads}
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
    save_outputs("Simulation", ad_embs, slice_names, ad_mosaic)
    config = "SpaMosaic recommended-parameter extension: Simulation1-5 full RNA+ADT batches, X_pca_be, batch_corr=True, intra_knns=10, inter_knn_base=10, w_g=0.8, wlgcn, lr=0.01, T=0.1, n_epochs=100, global kmeans=5"
    return metric_rows("Simulation", ad_embs, slice_names, "truth", 5, config, elapsed)


def run_mouse_brain():
    start = time.time()
    atac_rna = sc.read_h5ad(ROOT / 'data/datasets/Mouse_Brain/ATAC' / "adata_RNA.h5ad")
    atac_epi = sc.read_h5ad(ROOT / 'data/datasets/Mouse_Brain/ATAC' / "adata_peaks_normalized.h5ad")
    h3_rna = sc.read_h5ad(ROOT / 'data/datasets/Mouse_Brain/H3K27ac' / "adata_RNA.h5ad")
    h3_epi = sc.read_h5ad(ROOT / 'data/datasets/Mouse_Brain/H3K27ac' / "adata_peaks_normalized.h5ad")

    atac_rna, atac_epi = align_pair(atac_rna, atac_epi)
    h3_rna, h3_epi = align_pair(h3_rna, h3_epi)
    prefix_obs_names(atac_rna, "Mouse_Brain_ATAC")
    prefix_obs_names(atac_epi, "Mouse_Brain_ATAC")
    prefix_obs_names(h3_rna, "Mouse_Brain_H3K27ac")
    prefix_obs_names(h3_epi, "Mouse_Brain_H3K27ac")

    atac_rna.obs["src"] = "Mouse_Brain_ATAC"
    atac_epi.obs["src"] = "Mouse_Brain_ATAC"
    h3_rna.obs["src"] = "Mouse_Brain_H3K27ac"
    h3_epi.obs["src"] = "Mouse_Brain_H3K27ac"

    input_dict = {
        "rna": [atac_rna, h3_rna],
        "ATAC": [atac_epi, None],
        "H3K27ac": [None, h3_epi],
    }
    input_key = "dimred_bc"
    RNA_preprocess(input_dict["rna"], batch_corr=True, n_hvg=5000, batch_key="src", key=input_key)
    Epigenome_preprocess(input_dict["ATAC"], batch_corr=False, n_peak=50000, batch_key="src", key=input_key)
    Epigenome_preprocess(input_dict["H3K27ac"], batch_corr=False, n_peak=50000, batch_key="src", key=input_key)

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
    cluster_global(ad_mosaic, n_clusters=18, used_obsm="merged_emb", algo="mclust", key="mclust")
    split_global_obs(ad_embs, ad_mosaic, "mclust")
    elapsed = time.time() - start
    slice_names = ["Mouse_Brain_ATAC", "Mouse_Brain_H3K27ac"]
    save_outputs("Mouse_Brain", ad_embs, slice_names, ad_mosaic)
    config = "SpaMosaic recommended-parameter extension: two mouse brain batches connected by RNA, ATAC/H3K27ac epigenome branches, dimred_bc, RNA n_hvg=5000, Epigenome n_peak=50000, batch_corr RNA=True, intra_knns=10, inter_knn_base=10, w_g=0.8, wlgcn, lr=0.01, T=0.01, n_epochs=100, global mclust=18"
    return metric_rows("Mouse_Brain", ad_embs, slice_names, "truth", 18, config, elapsed)


def write_tables(rows):
    per = pd.DataFrame(rows)
    summary = per.groupby(["dataset", "method"], as_index=False).agg({
        "ARI": "mean",
        "NMI": "mean",
        "AMI": "mean",
        "Moran's I": "mean",
        "pred_clusters": ["min", "max"],
    })
    summary.columns = ["dataset", "method", "ARI", "NMI", "AMI", "Moran's I", "pred_clusters_min", "pred_clusters_max"]
    per_path = OUT / "spamosaic_extension_clustering_per_slice.csv"
    sum_path = OUT / "spamosaic_extension_clustering_summary.csv"
    per_round_path = OUT / "spamosaic_extension_clustering_per_slice_3dec.csv"
    sum_round_path = OUT / "spamosaic_extension_clustering_summary_3dec.csv"
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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Simulation", "Mouse_Brain"])
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        if dataset == "Simulation":
            print("[SpaMosaic extension] Running Simulation1-5 recommended-parameter extension", flush=True)
            rows.extend(run_simulation_all5())
        elif dataset == "Mouse_Brain":
            print("[SpaMosaic extension] Running Mouse_Brain recommended-parameter extension", flush=True)
            rows.extend(run_mouse_brain())
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        pd.DataFrame(rows).to_csv(OUT / "spamosaic_extension_clustering_per_slice.partial.csv", index=False)
        write_tables(rows)


if __name__ == "__main__":
    main()
