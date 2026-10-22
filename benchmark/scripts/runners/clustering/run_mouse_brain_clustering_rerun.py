#!/usr/bin/env python

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import os, sys, time, random, warnings
from pathlib import Path
from types import SimpleNamespace

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'benchmark/results/clustering' / 'mouse_brain' / 'rerun_clustering_18'
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = {
    'Mouse_Brain_ATAC': (ROOT/'data/datasets/Mouse_Brain/ATAC/adata_RNA.h5ad', ROOT/'data/datasets/Mouse_Brain/ATAC/adata_peaks_normalized.h5ad'),
    'Mouse_Brain_H3K27ac': (ROOT/'data/datasets/Mouse_Brain/H3K27ac/adata_RNA.h5ad', ROOT/'data/datasets/Mouse_Brain/H3K27ac/adata_peaks_normalized.h5ad'),
}
N_CLUSTERS = 18


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def read_pair(sample):
    rna_path, peak_path = SAMPLES[sample]
    rna = sc.read_h5ad(rna_path)
    peak = sc.read_h5ad(peak_path)
    rna.var_names_make_unique(); peak.var_names_make_unique()
    common = rna.obs_names.intersection(peak.obs_names)
    rna = rna[common].copy(); peak = peak[common].copy()
    if 'truth' not in rna.obs:
        raise ValueError(f'{rna_path} lacks obs[truth]')
    if 'spatial' not in peak.obsm and 'spatial' in rna.obsm:
        peak.obsm['spatial'] = np.asarray(rna.obsm['spatial'])
    if 'spatial' not in rna.obsm and 'spatial' in peak.obsm:
        rna.obsm['spatial'] = np.asarray(peak.obsm['spatial'])
    return rna, peak


def metric_row(sample, method, rna, pred, elapsed, config):
    truth = rna.obs['truth'].astype(str)
    pred = pd.Series(np.asarray(pred).astype(str), index=rna.obs_names)
    valid = truth.notna() & pred.notna() & (truth != 'nan') & (pred != 'nan')
    y = truth[valid].to_numpy(); p = pred[valid].to_numpy()
    coords = np.asarray(rna.obsm['spatial'])[valid.to_numpy()]
    codes = pd.Categorical(p).codes
    return {
        'dataset': sample,
        'method': method,
        'n_obs_input': int(rna.n_obs),
        'n_obs_valid': int(valid.sum()),
        'truth_classes': int(pd.Series(y).nunique()),
        'requested_clusters': N_CLUSTERS,
        'pred_clusters': int(pd.Series(p).nunique()),
        'ARI': adjusted_rand_score(y, p),
        'NMI': normalized_mutual_info_score(y, p),
        'AMI': adjusted_mutual_info_score(y, p),
        "Moran's I": moran_i_knn(codes, coords, k=6),
        'elapsed_seconds': elapsed,
        'config': config,
    }


def moran_i_knn(values, coords, k=6):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    if len(values) < 3:
        return np.nan
    k_use = max(1, min(k, len(values)-1))
    neigh = NearestNeighbors(n_neighbors=k_use+1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    z = values - values.mean(); denom = np.sum(z*z)
    if denom <= 0:
        return np.nan
    return float(np.sum(z[:, None] * z[neigh] / float(k_use)) / denom)


def save_rows(rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'mouse_brain_clustering_rerun_per_slice.csv', index=False)
    if not df.empty:
        summary = df.groupby('method', sort=False)[['ARI','NMI','AMI',"Moran's I"]].mean().reset_index()
        counts = df.groupby('method', sort=False).agg(pred_clusters_min=('pred_clusters','min'), pred_clusters_max=('pred_clusters','max')).reset_index()
        summary = summary.merge(counts, on='method')
        summary.to_csv(OUT/'mouse_brain_clustering_rerun_summary.csv', index=False)
    return df


def run_spatialglue_one(sample):
    set_seed(2022)
    sys.path.insert(0, str(ROOT/'baselines/SpatialGlue-main'))
    import torch
    import torch.nn.functional as F
    import run_spatialglue_benchmark as sg
    from SpatialGlue_main_placeholder import none  # never reached


def run_spatialglue_one(sample):
    set_seed(2022)
    sg_path = str(ROOT/'baselines/SpatialGlue-main')
    if sg_path not in sys.path:
        sys.path.insert(0, sg_path)
    import torch
    import torch.nn.functional as F
    import run_spatialglue_benchmark as sg
    from model import Encoder_omics
    rna, peak = read_pair(sample)
    start = time.time()
    # SpatialGlue recommended branch for Spatial-ATAC-RNA-seq.
    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.filter_cells(rna, min_genes=200)
    peak = peak[rna.obs_names].copy()
    n_top = min(3000, rna.n_vars)
    sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=n_top)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.pca(rna, use_highly_variable=True, n_comps=min(50, rna.n_obs-1, n_top))
    rna.obsm['feat'] = rna.obsm['X_pca'].copy()
    if 'X_lsi' not in peak.obsm:
        raise ValueError(f'{sample} peak file lacks obsm[X_lsi], required by SpatialGlue Spatial-ATAC-RNA-seq preprocessing')
    peak.obsm['feat'] = np.asarray(peak.obsm['X_lsi']).copy()
    n_neighbors = 6
    rna.uns['adj_spatial'] = build_spatial_graph(rna.obsm['spatial'], n_neighbors)
    peak.uns['adj_spatial'] = build_spatial_graph(peak.obsm['spatial'], n_neighbors)
    rna.obsm['adj_feature'], peak.obsm['adj_feature'] = construct_feature_graph(rna), construct_feature_graph(peak)
    cfg = SimpleNamespace(name=f'SpatialGlue_{sample}', epochs=1500, learning_rate=1e-4, weight_decay=0.0, dim_output=64, weight_factors=[1,2.5,1,1])

    class QuietTrain(sg.TrainSpatialGlue):
        def train(self):
            model = Encoder_omics(self.features_omics1.shape[1], self.config.dim_output, self.features_omics2.shape[1], self.config.dim_output).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            w = self.config.weight_factors
            for _ in range(self.config.epochs):
                model.train()
                res = model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
                loss = (w[0]*F.mse_loss(self.features_omics1, res['emb_recon_omics1']) +
                        w[1]*F.mse_loss(self.features_omics2, res['emb_recon_omics2']) +
                        w[2]*F.mse_loss(res['emb_latent_omics1'], res['emb_latent_omics1_across_recon']) +
                        w[3]*F.mse_loss(res['emb_latent_omics2'], res['emb_latent_omics2_across_recon']))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                model.eval()
                res = model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
            return F.normalize(res['emb_latent_combined'], p=2, eps=1e-12, dim=1).cpu().numpy()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    emb = QuietTrain({'adata_omics1': rna, 'adata_omics2': peak}, cfg, device).train()
    rna.obsm['SpatialGlue'] = emb
    n_comps = min(20, emb.shape[1], rna.n_obs-1)
    rna.obsm['SpatialGlue_pca'] = PCA(n_components=n_comps, random_state=2022).fit_transform(emb)
    labels = sg.mclust_labels(rna.obsm['SpatialGlue_pca'], n_clusters=N_CLUSTERS, random_seed=2020).astype(str)
    save_clusters(sample, 'SpatialGlue', rna, labels, emb)
    return metric_row(sample, 'SpatialGlue', rna, labels, time.time()-start, 'SpatialGlue Spatial-ATAC-RNA-seq: RNA min_cells=10, min_genes=200, 3000 HVG, PCA50; peak X_lsi; kNN=6; epochs=1500; mclust=18')


def build_spatial_graph(coords, n_neighbors=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    return pd.DataFrame({'x': x, 'y': y, 'value': np.ones(x.size, dtype=np.float32)})


def construct_feature_graph(adata, k=20):
    return kneighbors_graph(adata.obsm['feat'], min(k, adata.n_obs-1), mode='connectivity', metric='correlation', include_self=False)


def run_miso_one(sample):
    set_seed(100)
    sys.path.insert(0, str(ROOT/'baselines/miso-main'))
    import torch
    from miso import Miso
    from miso.utils import preprocess
    rna, peak = read_pair(sample)
    start = time.time()
    # Mouse_Brain MISO benchmark notebook: RNA min_cells/min_genes + 3000 HVG; peaks 3000 HVG; MISO preprocess rna/protein; cluster 18.
    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=min(3000, rna.n_vars))
    rna = rna[:, rna.var['highly_variable']].copy()
    peak = peak[rna.obs_names].copy()
    sc.pp.highly_variable_genes(peak, flavor='seurat_v3', n_top_genes=min(3000, peak.n_vars))
    peak = peak[:, peak.var['highly_variable']].copy()
    x_rna = preprocess(rna.copy(), modality='rna').astype(np.float32)
    x_peak = preprocess(peak.copy(), modality='protein').astype(np.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Miso([x_rna, x_peak], ind_views='all', combs='all', sparse=False, device=device)
    model.train()
    labels = model.cluster(n_clusters=N_CLUSTERS).astype(str)
    save_clusters(sample, 'MISO', rna, labels, model.emb)
    return metric_row(sample, 'MISO', rna, labels, time.time()-start, 'MISO Mouse_Brain notebook: RNA/peak 3000 HVG, preprocess(rna/protein), train 1000 epochs per modality, KMeans=18')


def run_cellcharter_one(sample):
    set_seed(12345)
    rna, peak = read_pair(sample)
    start = time.time()
    sc.pp.filter_genes(rna, min_cells=min(10, max(1, rna.n_obs//20)))
    sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=min(3000, rna.n_vars))
    rna = rna[:, rna.var['highly_variable']].copy()
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    x_rna = StandardScaler(with_mean=True, with_std=True).fit_transform(dense(rna.X))
    z_rna = PCA(n_components=min(30, x_rna.shape[0]-1, x_rna.shape[1]), random_state=12345).fit_transform(x_rna)
    peak = peak[rna.obs_names].copy()
    if 'X_lsi' in peak.obsm:
        z_peak = np.asarray(peak.obsm['X_lsi'])
    else:
        sc.pp.highly_variable_genes(peak, flavor='seurat_v3', n_top_genes=min(3000, peak.n_vars))
        peak = peak[:, peak.var['highly_variable']].copy()
        z_peak = PCA(n_components=min(30, peak.n_obs-1, peak.n_vars), random_state=12345).fit_transform(StandardScaler().fit_transform(dense(peak.X)))
    rep = StandardScaler(with_mean=True, with_std=True).fit_transform(np.hstack([z_rna, z_peak])).astype(np.float32)
    adj = delaunay_adjacency(rna.obsm['spatial'])
    x_cc = aggregate_neighbors(adj, rep, n_layers=3)
    x_cc = StandardScaler(with_mean=True, with_std=True).fit_transform(x_cc)
    pred = GaussianMixture(n_components=N_CLUSTERS, covariance_type='full', reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', random_state=12345).fit_predict(np.asarray(x_cc, dtype=np.float64)).astype(str)
    save_clusters(sample, 'CellCharter', rna, pred, x_cc)
    return metric_row(sample, 'CellCharter', rna, pred, time.time()-start, 'CellCharter-style defaults: RNA 3000 HVG PCA30 + peak X_lsi, Delaunay graph, neighbor layers=3, GMM=18')


def delaunay_adjacency(coords):
    tri = Delaunay(np.asarray(coords, dtype=np.float64))
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                a,b = int(simplex[i]), int(simplex[j])
                edges.add((a,b)); edges.add((b,a))
    rows, cols = zip(*edges)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(coords), len(coords)))


def setdiag(matrix, value):
    matrix = matrix.tolil(); matrix.setdiag(value); matrix = matrix.tocsr()
    if value == 0: matrix.eliminate_zeros()
    return matrix


def normalize_rows(adj):
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=np.float64), where=deg != 0)
    return sp.diags(inv) @ adj


def aggregate_neighbors(adj, x, n_layers=3):
    adj = setdiag(adj.astype(bool), 0)
    adj_hop = adj.copy(); adj_visited = setdiag(adj.copy(), 1)
    xs=[]
    for i in range(n_layers+1):
        if i == 0:
            xs.append(x)
        else:
            if i > 1:
                adj_hop = adj_hop @ adj
                adj_hop = adj_hop > adj_visited
                adj_visited = adj_visited + adj_hop
            xs.append(normalize_rows(adj_hop) @ x)
    return np.hstack(xs).astype(np.float32)


def run_harmony_all():
    set_seed(0)
    hm_path = str(ROOT/'baselines/harmonypy-master')
    sys.path = [p for p in sys.path if p not in ('', hm_path)] + [hm_path]
    import harmonypy as hm
    start = time.time()
    rnas=[]
    for sample,(rp,pp) in SAMPLES.items():
        rna=sc.read_h5ad(rp); rna.var_names_make_unique(); rna.obs['batch']=sample
        rnas.append(rna)
    common = rnas[0].var_names.intersection(rnas[1].var_names)
    pcs=[]; obs=[]; coords=[]
    for sample,rna in zip(SAMPLES.keys(), rnas):
        rna = rna[:, common].copy()
        sc.pp.filter_genes(rna, min_cells=10)
        sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=min(3000, rna.n_vars))
        rna = rna[:, rna.var['highly_variable']].copy()
        sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
        x = StandardScaler(with_mean=True, with_std=True).fit_transform(dense(rna.X))
        z = PCA(n_components=min(50, x.shape[0]-1, x.shape[1]), random_state=0).fit_transform(x)
        pcs.append(z); obs.append(rna.obs[['truth','batch']].copy()); coords.append(np.asarray(rna.obsm['spatial']))
    # Equalize dimensions after per-slice PCA.
    min_dim = min(z.shape[1] for z in pcs)
    X = np.vstack([z[:, :min_dim] for z in pcs])
    meta = pd.concat(obs, axis=0)
    ho = hm.run_harmony(X, meta, 'batch', theta=2, sigma=0.1, nclust=int(min(round(X.shape[0]/30.0), 100)), max_iter_harmony=10, max_iter_kmeans=4, epsilon_cluster=1e-3, epsilon_harmony=1e-2, alpha=0.2, verbose=False, random_state=0)
    emb = np.asarray(ho.Z_corr)
    pred = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=20).fit_predict(emb).astype(str)
    rows=[]; offset=0
    for sample,z,c in zip(SAMPLES.keys(), pcs, coords):
        n=z.shape[0]
        rna = ad.AnnData(X=np.zeros((n,1)))
        rna.obs_names = obs[list(SAMPLES.keys()).index(sample)].index
        rna.obs['truth'] = obs[list(SAMPLES.keys()).index(sample)]['truth'].values
        rna.obsm['spatial'] = c
        labels=pred[offset:offset+n]
        save_clusters(sample, 'Harmony', rna, labels, emb[offset:offset+n])
        rows.append(metric_row(sample, 'Harmony', rna, labels, time.time()-start, 'Harmony defaults: common RNA genes only across ATAC/H3K27ac; PCA50; batch=sample; Harmony defaults; KMeans=18'))
        offset += n
    return rows


def save_clusters(sample, method, rna, labels, emb):
    d = OUT / method / sample
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'truth': rna.obs['truth'].astype(str).to_numpy(), method: np.asarray(labels).astype(str)}, index=rna.obs_names).to_csv(d/'clusters.csv')
    np.save(d/'embedding.npy', np.asarray(emb, dtype=np.float32))


def main(methods):
    all_rows=[]
    existing = OUT/'mouse_brain_clustering_rerun_per_slice.csv'
    if existing.exists():
        old = pd.read_csv(existing)
        all_rows = old[~old['method'].isin(methods)].to_dict('records')
    for method in methods:
        print(f'[{time.strftime("%H:%M:%S")}] {method}', flush=True)
        if method == 'Harmony':
            rows = run_harmony_all()
            all_rows.extend(rows)
        else:
            for sample in SAMPLES:
                print(f'  {sample}', flush=True)
                if method == 'SpatialGlue': row = run_spatialglue_one(sample)
                elif method == 'MISO': row = run_miso_one(sample)
                elif method == 'CellCharter': row = run_cellcharter_one(sample)
                else: raise ValueError(method)
                all_rows.append(row)
                save_rows(all_rows)
        df = save_rows(all_rows)
        print(df[df.method.eq(method)].to_string(index=False), flush=True)
    print('\nSUMMARY')
    print(pd.read_csv(OUT/'mouse_brain_clustering_rerun_summary.csv').to_string(index=False))

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='*', default=['SpatialGlue','MISO','CellCharter','Harmony'])
    args=ap.parse_args()
    main(args.methods)
