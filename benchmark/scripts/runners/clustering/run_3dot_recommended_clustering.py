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

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
THREEDOT = ROOT / "baselines/3d-OT-main"
OUT = ROOT / "benchmark/results/clustering" / "3dot_recommended_clustering"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(THREEDOT))

from lib_3d_OT.multi_modialty import extractModel, train_graph_extractor  # noqa: E402
from lib_3d_OT.utils import prepare_data, set_seed  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


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


def metric_row(dataset, slice_name, n_clusters, adata, truth_key, elapsed, config):
    truth = adata.obs[truth_key].astype(str)
    pred = adata.obs["3d-OT"].astype(str)
    valid = truth.notna() & pred.notna() & (truth != "nan") & (pred != "nan")
    y = truth[valid].to_numpy()
    p = pred[valid].to_numpy()
    coords = np.asarray(adata.obsm["spatial"])[valid.to_numpy()]
    return {
        "dataset": dataset,
        "slice": slice_name,
        "method": "3d-OT",
        "n_obs_input": int(adata.n_obs),
        "n_obs_valid": int(valid.sum()),
        "truth_classes": int(pd.Series(y).nunique()),
        "requested_clusters": int(n_clusters),
        "pred_clusters": int(pd.Series(p).nunique()),
        "ARI": adjusted_rand_score(y, p),
        "NMI": normalized_mutual_info_score(y, p),
        "AMI": adjusted_mutual_info_score(y, p),
        "Moran's I": moran_i_knn(pd.Categorical(p).codes, coords, k=6),
        "elapsed_seconds": float(elapsed),
        "config": config,
    }


def save_adata(adata, dataset, slice_name):
    out_dir = OUT / "h5ad" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slice_name}_3dot_recommended.h5ad"
    adata.write(path)
    return path


def run_3dot_mclust(adata, n_clusters, key="3d-OT", random_seed=2024, n_comp=20, model_names="EEE"):
    if key == "X":
        data = dense(adata.X)
    else:
        if key not in adata.obsm:
            raise KeyError(f"adata.obsm[{key!r}] not found.")
        data = np.asarray(adata.obsm[key])

    n_comp = int(min(n_comp, data.shape[1], data.shape[0] - 1))
    if n_comp < 1:
        raise ValueError("PCA component count for 3d-OT clustering is <1.")
    adata.obsm["emb_pca"] = PCA(n_components=n_comp, random_state=2024).fit_transform(data)

    import rpy2.robjects as ro

    np.random.seed(random_seed)
    ro.r["set.seed"](random_seed)
    ro.r.library("mclust")
    x = StandardScaler().fit_transform(np.asarray(adata.obsm["emb_pca"], dtype=np.float64))
    r_x = ro.r.matrix(ro.FloatVector(x.ravel(order="F")), nrow=x.shape[0], ncol=x.shape[1])
    ro.globalenv[".threedot_mclust_data"] = r_x
    ro.globalenv[".threedot_mclust_g"] = ro.IntVector([int(n_clusters)])
    ro.globalenv[".threedot_mclust_model"] = ro.StrVector([str(model_names)])
    fit = ro.r(
        "Mclust(data=.threedot_mclust_data, "
        "G=.threedot_mclust_g, modelNames=.threedot_mclust_model)"
    )
    labels = np.asarray(fit.rx2("classification"), dtype=np.int64)
    ro.r(
        "rm(.threedot_mclust_data, .threedot_mclust_g, "
        ".threedot_mclust_model, envir=.GlobalEnv)"
    )
    adata.obs["mclust"] = labels.astype(int).astype(str)
    adata.obs["mclust"] = adata.obs["mclust"].astype("category")
    adata.obs["3d-OT"] = adata.obs["mclust"].astype(str).astype("category")
    return adata


def prep_rna_for_multimodal(rna, n_feat):
    rna = rna.copy()
    rna.var_names_make_unique()
    sc.pp.filter_genes(rna, min_cells=min(10, max(1, rna.n_obs // 50)))
    sc.pp.filter_cells(rna, min_genes=min(200, max(1, rna.n_vars // 20)))
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=min(3000, rna.n_vars))
    rna = rna[:, rna.var["highly_variable"]].copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna, zero_center=True, max_value=10)
    n_comp = int(min(n_feat, rna.n_obs - 1, rna.n_vars))
    if n_comp < 2:
        raise ValueError("RNA PCA feature dimension would be <2.")
    rna.obsm["feat"] = PCA(n_components=n_comp, random_state=42).fit_transform(dense(rna.X))
    return rna


def prep_second_modality(mod, obs_names, n_feat):
    mod = mod[obs_names].copy()
    x = dense(mod.X).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.min() < 0:
        x = x - x.min(axis=0, keepdims=True)
    sums = x.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    x = np.log1p(x / sums * 1e4)
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    n_comp = int(min(n_feat, x.shape[0] - 1, x.shape[1]))
    if n_comp < 2:
        raise ValueError("Second modality feature dimension would be <2.")
    if x.shape[1] == n_comp:
        feat = x
    else:
        feat = PCA(n_components=n_comp, random_state=42).fit_transform(x)
    mod.obsm["feat"] = feat.astype(np.float32)
    if "spatial" not in mod.obsm:
        raise KeyError("Second modality lacks obsm['spatial'].")
    return mod


def run_multimodal_slice(dataset, slice_name, rna_path, mod_path, truth_key, n_clusters, n_feat_cap=50, epochs=300):
    start = time.time()
    set_seed(7)
    rna_raw = sc.read_h5ad(rna_path)
    mod_raw = sc.read_h5ad(mod_path)
    common_obs = rna_raw.obs_names.intersection(mod_raw.obs_names)
    if len(common_obs) == 0:
        raise ValueError(f"{slice_name}: no common observations between RNA and second modality.")
    rna_raw = rna_raw[common_obs].copy()
    mod_raw = mod_raw[common_obs].copy()
    if truth_key not in rna_raw.obs:
        raise KeyError(f"{rna_path} lacks obs[{truth_key!r}].")
    rna_raw.obs["truth"] = rna_raw.obs[truth_key].astype(str)
    mod_raw.obs["truth"] = rna_raw.obs["truth"].astype(str).values

    n_feat = int(min(n_feat_cap, max(2, mod_raw.n_vars), rna_raw.n_obs - 1, mod_raw.n_obs - 1))
    rna = prep_rna_for_multimodal(rna_raw, n_feat)
    mod = prep_second_modality(mod_raw, rna.obs_names, rna.obsm["feat"].shape[1])
    if "spatial" not in rna.obsm and "spatial" in mod.obsm:
        rna.obsm["spatial"] = np.asarray(mod.obsm["spatial"])
    if "spatial" not in rna.obsm:
        raise KeyError(f"{slice_name}: RNA lacks obsm['spatial'].")

    graph1 = prepare_data(rna, location="spatial", nb_neighbors=16).to(DEVICE)
    graph2 = prepare_data(mod, location="spatial", nb_neighbors=16).to(DEVICE)
    input_dim = graph1.express.shape[-1]
    if graph2.express.shape[-1] != input_dim:
        raise ValueError(f"{slice_name}: feature dimensions differ: {input_dim} vs {graph2.express.shape[-1]}.")

    model = extractModel(input_dim, hidden_dim=32, n_heads=4, n_layers=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_model = train_graph_extractor(graph1, graph2, model, optimizer, DEVICE, epochs=epochs)
    with torch.no_grad():
        final_model = extractModel(input_dim=input_dim, hidden_dim=32).to(DEVICE)
        final_model.load_state_dict(best_model)
        final_model.eval()
        _, _, mixed = final_model(graph1, graph2)
        rna.obsm["3d-OT"] = mixed.squeeze(0).cpu().detach().numpy()

    run_3dot_mclust(rna, n_clusters=n_clusters, key="3d-OT", random_seed=2024, n_comp=min(20, rna.obsm["3d-OT"].shape[1], rna.n_obs - 1))
    save_adata(rna, dataset, slice_name)
    return metric_row(
        dataset,
        slice_name,
        n_clusters,
        rna,
        "truth",
        time.time() - start,
        f"3d-OT recommended multimodal per-slice: RNA HVG/PCA + second modality PCA to same dim, graph k=16, epochs={epochs}, mclust={n_clusters}",
    )


def run_existing_3dot_slice(dataset, slice_name, path, truth_key, n_clusters):
    start = time.time()
    adata = sc.read_h5ad(path)
    if "3d-OT" not in adata.obsm:
        raise KeyError(f"{path} lacks obsm['3d-OT'].")
    if truth_key not in adata.obs:
        raise KeyError(f"{path} lacks obs[{truth_key!r}].")
    adata.obs["truth"] = adata.obs[truth_key].astype(str)
    run_3dot_mclust(adata, n_clusters=n_clusters, key="3d-OT", random_seed=2024, n_comp=min(20, adata.obsm["3d-OT"].shape[1], adata.n_obs - 1))
    save_adata(adata, dataset, slice_name)
    return metric_row(
        dataset,
        slice_name,
        n_clusters,
        adata,
        "truth",
        time.time() - start,
        f"3d-OT recommended per-slice clustering on existing obsm[3d-OT], mclust={n_clusters}",
    )


def build_jobs():
    jobs = []
    jobs.append(("existing", "Mouse_Brain", "Mouse_Brain_ATAC", ROOT / "data/datasets/Mouse_Brain/ATAC/3d-OT.h5ad", None, "truth", 18))
    jobs.append(("existing", "Mouse_Brain", "Mouse_Brain_H3K27ac", ROOT / "data/datasets/Mouse_Brain/H3K27ac/3d-OT.h5ad", None, "truth", 18))
    for i in range(1, 4):
        jobs.append(
            (
                "multimodal",
                "Human_lymph_node",
                f"slice{i}",
                ROOT / f"data/datasets/Human_lymph_node/slice{i}/s{i}_adata_rna.h5ad",
                ROOT / f"data/datasets/Human_lymph_node/slice{i}/s{i}_adata_adt.h5ad",
                "final_annot",
                10,
            )
        )
    for i in range(1, 6):
        jobs.append(
            (
                "multimodal",
                "Simulation",
                f"Simulation{i}",
                ROOT / f"data/datasets/Simulation/slice{i}/adata_RNA.h5ad",
                ROOT / f"data/datasets/Simulation/slice{i}/adata_ADT.h5ad",
                "truth",
                5,
            )
        )
    return jobs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["Mouse_Brain", "Human_lymph_node", "Simulation"])
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    rows = []
    for mode, dataset, slice_name, rna_or_h5ad, mod_path, truth_key, n_clusters in build_jobs():
        if dataset not in set(args.datasets):
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {dataset} {slice_name} ({mode})", flush=True)
        if mode == "existing":
            row = run_existing_3dot_slice(dataset, slice_name, rna_or_h5ad, truth_key, n_clusters)
        else:
            row = run_multimodal_slice(dataset, slice_name, rna_or_h5ad, mod_path, truth_key, n_clusters, epochs=args.epochs)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "3dot_recommended_clustering_per_slice.partial.csv", index=False)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)

    df = pd.DataFrame(rows)
    per_path = OUT / "3dot_recommended_clustering_per_slice.csv"
    summary_path = OUT / "3dot_recommended_clustering_summary.csv"
    df.to_csv(per_path, index=False)
    summary = (
        df.groupby(["dataset", "method"], sort=False)[["ARI", "NMI", "AMI", "Moran's I"]]
        .mean()
        .reset_index()
    )
    pred_span = df.groupby(["dataset", "method"], sort=False)["pred_clusters"].agg(["min", "max"]).reset_index()
    pred_span = pred_span.rename(columns={"min": "pred_clusters_min", "max": "pred_clusters_max"})
    summary = summary.merge(pred_span, on=["dataset", "method"], how="left")
    summary.to_csv(summary_path, index=False)
    print("\nSUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved: {per_path}\nSaved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
