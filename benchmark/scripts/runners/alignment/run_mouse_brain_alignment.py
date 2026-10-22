#!/usr/bin/env python

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import os, sys, time, warnings
from pathlib import Path
from types import SimpleNamespace
warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND','Agg')

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree, distance

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmark/results/alignment'/'mouse_brain'
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT/'baselines/paste-main'/'src'))
sys.path.insert(0, str(ROOT/'baselines/SLAT-main'))
sys.path.insert(0, str(ROOT/'baselines/SANTO-main'))
sys.path.insert(0, str(ROOT/'baselines/3d-OT-main'))

SRC_PATH=ROOT/'data/datasets/Mouse_Brain/ATAC/adata_RNA.h5ad'
TGT_PATH=ROOT/'data/datasets/Mouse_Brain/H3K27ac/adata_RNA.h5ad'


def dense_x(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def prep_pair(src_path, tgt_path, max_features=None):
    src=ad.read_h5ad(src_path)
    tgt=ad.read_h5ad(tgt_path)
    src.var_names_make_unique()
    tgt.var_names_make_unique()
    common=src.var_names.intersection(tgt.var_names)
    if len(common) == 0:
        raise ValueError('Mouse_Brain source/target have no common genes')
    src=src[:, common].copy()
    tgt=tgt[:, common].copy()
    xs=dense_x(src.X).astype(np.float32)
    xt=dense_x(tgt.X).astype(np.float32)
    if max_features and xs.shape[1] > max_features:
        score=xs.var(axis=0) + xt.var(axis=0)
        idx=np.argsort(score)[-max_features:]
        src=src[:, idx].copy()
        tgt=tgt[:, idx].copy()
        xs=xs[:, idx]
        xt=xt[:, idx]
    for a,X in [(src,xs),(tgt,xt)]:
        sums=X.sum(axis=1, keepdims=True)
        sums[sums==0]=1
        a.layers['counts']=X.astype(np.float32)
        a.X=np.log1p(X/sums*1e4).astype(np.float32)
        a.obsm['spatial']=np.asarray(a.obsm['spatial'], dtype=np.float32)
    return src,tgt


def prep(path, max_features=None):
    a=ad.read_h5ad(path)
    a.var_names_make_unique()
    X=dense_x(a.X).astype(np.float32)
    # Spatial alignment packages expect expression-like nonnegative matrix; use library-size log normalization.
    sums=X.sum(axis=1, keepdims=True)
    sums[sums==0]=1
    X=np.log1p(X/sums*1e4)
    if max_features and X.shape[1] > max_features:
        idx=np.argsort(X.var(axis=0))[-max_features:]
        a=a[:,idx].copy(); X=X[:,idx]
    else:
        a=a.copy()
    a.X=X.astype(np.float32)
    a.obsm['spatial']=np.asarray(a.obsm['spatial'], dtype=np.float32)
    return a


def barcode_key(a):
    return np.asarray([str(x) for x in a.obs_names])


def nearest_match(aligned_src, tgt_xy):
    d,j=cKDTree(tgt_xy).query(aligned_src,k=1)
    return j.astype(int), d


def barycentric_from_pi(pi, target_xy):
    pi=np.asarray(pi, dtype=np.float64)
    rs=pi.sum(axis=1, keepdims=True); rs[rs==0]=1
    return (pi/rs) @ target_xy


def chamfer(a,b):
    ta,tb=cKDTree(a),cKDTree(b)
    da,_=tb.query(a,k=1); db,_=ta.query(b,k=1)
    return float((da.mean()+db.mean())/2)


def median_nn(x):
    d,_=cKDTree(x).query(x,k=2)
    return float(np.median(d[:,1]))


def pcc_matches(xa, xb, pred):
    vals=[]
    for i,j in enumerate(pred):
        r=np.corrcoef(xa[i], xb[j])[0,1]
        if np.isfinite(r): vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def metrics(src,tgt,pred_j,aligned_src):
    src_xy=np.asarray(src.obsm['spatial'],float)
    tgt_xy=np.asarray(tgt.obsm['spatial'],float)
    aligned_src=np.asarray(aligned_src,float)
    pred_j=np.asarray(pred_j,int)
    skey,tkey=barcode_key(src),barcode_key(tgt)
    tmap={k:i for i,k in enumerate(tkey)}
    true_j=np.array([tmap.get(k,-1) for k in skey], dtype=int)
    mask=true_j>=0
    xa,xb=dense_x(src.X).astype(np.float32),dense_x(tgt.X).astype(np.float32)
    dmat=distance.cdist(aligned_src[mask], tgt_xy)
    true_d=dmat[np.arange(mask.sum()), true_j[mask]]
    err=np.linalg.norm(aligned_src[mask]-tgt_xy[true_j[mask]],axis=1)
    return {
        'PAA': float(np.mean(pred_j[mask] == true_j[mask])),
        'SMR': float(np.mean(err <= median_nn(tgt_xy))),
        'FOSCTTM': float(np.mean((dmat < true_d[:,None]).sum(axis=1)/max(1,tgt.n_obs-1))),
        'Chamfer Distance': chamfer(aligned_src, tgt_xy),
        'PCC': pcc_matches(xa,xb,pred_j),
        'n_source': int(src.n_obs),
        'n_target': int(tgt.n_obs),
        'n_true_pairs': int(mask.sum()),
    }


def run_paste(src,tgt):
    import ot, paste as pst
    try:
        pi=pst.pairwise_align(src,tgt,use_gpu=True,backend=ot.backend.TorchBackend(),alpha=0.1,norm=True,verbose=False,gpu_verbose=False)
        if hasattr(pi,'detach'): pi=pi.detach().cpu().numpy()
    except Exception:
        pi=pst.pairwise_align(src,tgt,alpha=0.1,norm=True,verbose=False,gpu_verbose=False)
    pred=np.argmax(np.asarray(pi),axis=1)
    aligned=barycentric_from_pi(pi,np.asarray(tgt.obsm['spatial']))
    return pred,aligned


def run_slat(src,tgt):
    from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
    a,b=src.copy(),tgt.copy()
    if 'counts' in a.layers and 'counts' in b.layers:
        a.X=a.layers['counts'].copy()
        b.X=b.layers['counts'].copy()
    Cal_Spatial_Net(a,k_cutoff=10,model='KNN',verbose=False)
    Cal_Spatial_Net(b,k_cutoff=10,model='KNN',verbose=False)
    edges,features=load_anndatas([a,b],feature='DPCA',join='inner',check_order=False)
    emb0,emb1,_=run_SLAT(features,edges)
    # Reverse to get source -> target matches.
    best,_,_=spatial_match([emb1,emb0],adatas=[b,a],reorder=False,verbose=False)
    pred=np.asarray(best,dtype=int)
    return pred,np.asarray(tgt.obsm['spatial'])[pred]


def run_santo(src,tgt):
    import matplotlib.pyplot as plt
    plt.show=lambda *args,**kwargs: None
    import torch
    from SANTO_utils import santo
    args=SimpleNamespace(epochs=20,lr=0.001,k=10,alpha=0.9,diff_omics=False,mode='align',dimension=2,
                         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    aligned,_=santo(src.copy(),tgt.copy(),args)
    pred,_=nearest_match(aligned,np.asarray(tgt.obsm['spatial']))
    return pred,aligned


def run_3dot(src,tgt):
    import torch
    from lib_3d_OT.threeDrecon import pairwise_dpca_and_train, pairwise_align_reverse
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Pair order [target, source] makes reverse align graph1(source) -> graph0(target).
    corrected,graphs,best_models=pairwise_dpca_and_train([tgt.copy(),src.copy()],epochs=1150,neighbors=8,device=device)
    models=pairwise_align_reverse(graphs,best_models,device=device,nb_epochs=1,simk=5,otk=200)
    model=models[0]; model.eval()
    with torch.no_grad():
        recon_flow,corr_conf,target_recon,graph,pclouds=model(graphs[1],graphs[0])
    tr=target_recon.detach().cpu().numpy().reshape(src.n_obs,2)
    tgt_xy=np.asarray(tgt.obsm['spatial'],float)
    mn,mx=tgt_xy.min(axis=0),tgt_xy.max(axis=0)
    aligned=tr*(mx-mn)+mn
    pred,_=nearest_match(aligned,tgt_xy)
    return pred,aligned

METHODS={'3d-OT':run_3dot,'SLAT':run_slat,'PASTE':run_paste,'SANTO':run_santo}


def main(methods=None,max_features=3000):
    methods=methods or list(METHODS)
    src,tgt=prep_pair(SRC_PATH,TGT_PATH,max_features=max_features)
    out_csv=OUT/'mouse_brain_alignment_pair_metrics.csv'
    if out_csv.exists():
        existing=pd.read_csv(out_csv).to_dict('records')
        rows=[r for r in existing if r.get('method') not in set(methods)]
    else:
        rows=[]
    for m in methods:
        print(f'[{time.strftime("%H:%M:%S")}] {m} Mouse_Brain_ATAC_to_H3K27ac', flush=True)
        row={'method':m,'dataset':'Mouse_Brain','pair':'Mouse_Brain_ATAC_to_H3K27ac','status':'ok','error':''}
        try:
            pred,aligned=METHODS[m](src,tgt)
            row.update(metrics(src,tgt,pred,aligned))
        except Exception as e:
            row['status']='failed'; row['error']=f'{type(e).__name__}: {e}'
            print('FAILED',m,row['error'], flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv,index=False)
    df=pd.DataFrame(rows)
    print('\nRESULT')
    print(df.to_string(index=False))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--methods',nargs='*')
    ap.add_argument('--max-features',type=int,default=3000)
    args=ap.parse_args()
    main(args.methods,args.max_features)
