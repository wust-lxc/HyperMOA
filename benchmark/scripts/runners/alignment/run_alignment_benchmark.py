#!/usr/bin/env python

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import os, sys, json, time, math, warnings
from pathlib import Path
from types import SimpleNamespace
warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND','Agg')

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree, distance
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'benchmark/results/alignment'
OUT.mkdir(exist_ok=True)

# Local packages, in priority order matching uploaded repos.
sys.path.insert(0, str(ROOT / 'baselines/paste-main' / 'src'))
sys.path.insert(0, str(ROOT / 'baselines/SLAT-main'))
sys.path.insert(0, str(ROOT / 'baselines/SANTO-main'))
sys.path.insert(0, str(ROOT / 'baselines/3d-OT-main'))

HUMAN = [
    ('Human_slice1', ROOT/'data/datasets/Human_lymph_node/slice1/s1_adata_rna.h5ad'),
    ('Human_slice2', ROOT/'data/datasets/Human_lymph_node/slice2/s2_adata_rna.h5ad'),
    ('Human_slice3', ROOT/'data/datasets/Human_lymph_node/slice3/s3_adata_rna.h5ad'),
]
SIM = [(f'Simulation{i}', ROOT/f'data/datasets/Simulation/slice{i}/adata_RNA.h5ad') for i in range(1,6)]


def dense_x(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def prep_adata(path, max_genes=2000):
    a = ad.read_h5ad(path)
    a.var_names_make_unique()
    # PASTE/SANTO/SLAT are expression based; keep RNA counts but reduce to HVG-like variance top genes for speed.
    X = dense_x(a.X).astype(np.float32)
    # normalize per cell and log1p, standard for these ST alignment tools when raw count matrices are provided.
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    X = np.log1p(X / sums * 1e4)
    if X.shape[1] > max_genes:
        var = X.var(axis=0)
        idx = np.argsort(var)[-max_genes:]
        X = X[:, idx]
        a = a[:, idx].copy()
    else:
        a = a.copy()
    a.X = X.astype(np.float32)
    a.obsm['spatial'] = np.asarray(a.obsm['spatial'], dtype=np.float32)
    if 'truth' not in a.obs:
        for label_key in ['final_annot', 'ground_truth', 'lab']:
            if label_key in a.obs:
                a.obs['truth'] = a.obs[label_key].astype(str).values
                break
    return a


def label_array(a):
    for k in ['truth','ground_truth','final_annot','celltype','Cell_class','layer_guess']:
        if k in a.obs:
            s = a.obs[k].astype(str).str.lower().replace({'vessels':'vessel'})
            return s.to_numpy()
    return np.array(['NA'] * a.n_obs)


def common_expr(a, b):
    genes = a.var_names.intersection(b.var_names)
    if len(genes) == 0:
        genes = a.var_names[:min(a.n_vars, b.n_vars)]
        xa = dense_x(a[:, genes].X) if set(genes).issubset(set(a.var_names)) else dense_x(a.X[:, :len(genes)])
        xb = dense_x(b.X[:, :len(genes)]) if set(genes).issubset(set(b.var_names)) else dense_x(b.X[:, :len(genes)])
    else:
        xa = dense_x(a[:, genes].X)
        xb = dense_x(b[:, genes].X)
    return xa.astype(np.float32), xb.astype(np.float32)


def row_argmax(pi):
    pi = np.asarray(pi)
    return np.argmax(pi, axis=1)


def barycentric_from_pi(pi, target_xy):
    pi = np.asarray(pi, dtype=np.float64)
    rs = pi.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return (pi / rs) @ target_xy


def nearest_match(aligned_src, tgt_xy):
    tree = cKDTree(tgt_xy)
    d, j = tree.query(aligned_src, k=1)
    return j.astype(int), d


def median_nn_radius(xy):
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    return float(np.median(d[:,1]))


def chamfer(a_xy, b_xy):
    ta, tb = cKDTree(a_xy), cKDTree(b_xy)
    da, _ = tb.query(a_xy, k=1)
    db, _ = ta.query(b_xy, k=1)
    return float((da.mean() + db.mean()) / 2.0)


def pcc_matches(xa, xb, pred_j):
    vals = []
    for i, j in enumerate(pred_j):
        r = np.corrcoef(xa[i], xb[j])[0,1]
        if np.isfinite(r): vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def metrics_for_pair(src, tgt, pred_j, aligned_src=None, simulation=False):
    src_xy = np.asarray(src.obsm['spatial'], dtype=float)
    tgt_xy = np.asarray(tgt.obsm['spatial'], dtype=float)
    pred_j = np.asarray(pred_j, dtype=int)
    if aligned_src is None:
        aligned_src = tgt_xy[pred_j]
    aligned_src = np.asarray(aligned_src, dtype=float)
    sl, tl = label_array(src), label_array(tgt)
    xa, xb = common_expr(src, tgt)
    out = {}
    out['PCC'] = pcc_matches(xa, xb, pred_j)
    if simulation:
        out['CI'] = float(np.mean(sl == tl[pred_j]))
        id_to_j = {str(n): j for j, n in enumerate(tgt.obs_names)}
        true_j = np.array([id_to_j.get(str(n), -1) for n in src.obs_names], dtype=int)
        mask = true_j >= 0
        out['PAA'] = float(np.mean(pred_j[mask] == true_j[mask]))
        err = np.linalg.norm(aligned_src[mask] - tgt_xy[true_j[mask]], axis=1)
        out['SMR'] = float(np.mean(err <= median_nn_radius(tgt_xy)))
        # FOSCTTM: fraction of target cells closer than the true counterpart in aligned coordinate space.
        dmat = distance.cdist(aligned_src[mask], tgt_xy)
        true_d = dmat[np.arange(mask.sum()), true_j[mask]]
        out['FOSCTTM'] = float(np.mean((dmat < true_d[:, None]).sum(axis=1) / max(1, tgt.n_obs - 1)))
        out['Chamfer Distance'] = chamfer(aligned_src, tgt_xy)
    else:
        out['Chamfer Distance'] = chamfer(aligned_src, tgt_xy)
    return out


def run_paste(src, tgt):
    import ot
    import paste as pst
    # 3d-OT benchmark recommended: alpha=0.1, norm=True, Torch backend/GPU if available.
    try:
        pi = pst.pairwise_align(src, tgt, use_gpu=True, backend=ot.backend.TorchBackend(), alpha=0.1, norm=True, verbose=False, gpu_verbose=False)
        if hasattr(pi, 'detach'): pi = pi.detach().cpu().numpy()
    except Exception:
        pi = pst.pairwise_align(src, tgt, alpha=0.1, norm=True, verbose=False, gpu_verbose=False)
    return row_argmax(pi), barycentric_from_pi(pi, np.asarray(tgt.obsm['spatial']))


def run_slat(src, tgt):
    import torch
    from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
    a, b = src.copy(), tgt.copy()
    Cal_Spatial_Net(a, k_cutoff=10, model='KNN', verbose=False)
    Cal_Spatial_Net(b, k_cutoff=10, model='KNN', verbose=False)
    edges, features = load_anndatas([a, b], feature='DPCA', join='inner', check_order=False)
    emb0, emb1, _ = run_SLAT(features, edges)
    # spatial_match returns query(second embedding)->reference(first embedding). Reverse args for source->target.
    best, _, _ = spatial_match([emb1, emb0], adatas=[b, a], reorder=False, verbose=False)
    pred_j = np.asarray(best, dtype=int)
    return pred_j, np.asarray(tgt.obsm['spatial'])[pred_j]


def run_santo(src, tgt):
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None
    from SANTO_utils import santo
    import torch
    args = SimpleNamespace(epochs=20, lr=0.001, k=10, alpha=0.9, diff_omics=False, mode='align', dimension=2,
                           device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    aligned, _ = santo(src.copy(), tgt.copy(), args)
    pred_j, _ = nearest_match(aligned, np.asarray(tgt.obsm['spatial']))
    return pred_j, aligned


def run_3dot(src, tgt):
    import torch
    from lib_3d_OT.threeDrecon import pairwise_dpca_and_train, pairwise_align_reverse
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # To align source -> target, train pair order [target, source], then reverse-align graph1 -> graph0.
    corrected, graphs, best_models = pairwise_dpca_and_train([tgt.copy(), src.copy()], epochs=1150, neighbors=8, device=device)
    models = pairwise_align_reverse(graphs, best_models, device=device, nb_epochs=1, simk=5, otk=200)
    model = models[0]
    model.eval()
    with torch.no_grad():
        recon_flow, corr_conf, target_recon, graph, pclouds = model(graphs[1], graphs[0])
    # target_recon is in min-max normalized coordinates of target; unscale to target spatial range.
    tr = target_recon.detach().cpu().numpy().reshape(src.n_obs, 2)
    tgt_xy = np.asarray(tgt.obsm['spatial'], dtype=float)
    mn, mx = tgt_xy.min(axis=0), tgt_xy.max(axis=0)
    aligned = tr * (mx - mn) + mn
    pred_j, _ = nearest_match(aligned, tgt_xy)
    return pred_j, aligned


METHODS = {
    '3d-OT': run_3dot,
    'SLAT': run_slat,
    'PASTE': run_paste,
    'SANTO': run_santo,
}


def pairs(dataset):
    return list(zip(dataset[:-1], dataset[1:]))


def main(method_filter=None, max_genes=2000, dataset_filter=None):
    methods = [m for m in METHODS if method_filter is None or m in method_filter]
    rows = []
    for dname, dataset, sim in [('Simulation', SIM, True), ('Human_lymph_node', HUMAN, False)]:
        if dataset_filter is not None and dname not in dataset_filter:
            continue
        loaded = [(n, prep_adata(p, max_genes=max_genes)) for n, p in dataset]
        for method in methods:
            fn = METHODS[method]
            for (n1, a), (n2, b) in pairs(loaded):
                pair_name = f'{n1}_to_{n2}'
                print(f'[{time.strftime("%H:%M:%S")}] {method} {dname} {pair_name}', flush=True)
                try:
                    pred_j, aligned = fn(a, b)
                    met = metrics_for_pair(a, b, pred_j, aligned, simulation=sim)
                    status = 'ok'
                    err = ''
                except Exception as e:
                    met = {}
                    status = 'failed'
                    err = f'{type(e).__name__}: {e}'
                    print('FAILED', method, pair_name, err, flush=True)
                row = {'method':method, 'dataset':dname, 'pair':pair_name, 'status':status, 'error':err}
                row.update(met)
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT/'alignment_pair_metrics.csv', index=False)
    df = pd.DataFrame(rows)
    ok = df[df.status == 'ok'].copy()
    metric_cols = [c for c in ['PAA','SMR','FOSCTTM','Chamfer Distance','CI','PCC'] if c in ok]
    summary = ok.groupby(['method','dataset'])[metric_cols].mean().reset_index()
    summary.to_csv(OUT/'alignment_summary_metrics.csv', index=False)
    print('\nSUMMARY')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--max-genes', type=int, default=2000)
    ap.add_argument('--datasets', nargs='*')
    args = ap.parse_args()
    main(args.methods, args.max_genes, args.datasets)
