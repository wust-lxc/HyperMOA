import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from torch.utils.data import Dataset


def _to_dense_float32(x):
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _library_size(count, size='explog'):
    if sp.issparse(count):
        sum_use = np.asarray(count.sum(axis=1)).ravel()
    else:
        sum_use = np.asarray(count.sum(axis=1)).ravel()

    sum_use = np.nan_to_num(sum_use, nan=0.0, posinf=0.0, neginf=0.0)
    sum_safe = np.maximum(sum_use, 1.0)

    if size == 'explog':
        lib = np.exp(np.log10(sum_safe))
    elif size == 'sum':
        lib = sum_safe
    elif size == 'median':
        median = np.median(sum_safe)
        if median <= 0:
            median = 1.0
        lib = sum_safe / median
    elif size == 'log':
        lib = np.log1p(sum_safe)
    else:
        raise ValueError("size must be one of {'explog', 'sum', 'median', 'log'}")

    return lib.astype(np.float32).reshape(-1, 1)


class MultiOmicsDataset(Dataset):
    def __init__(
        self,
        adata,
        atac_key='ATAC',
        count_key=None,
        batch_key=None,
        size='explog',
        normalize=True,
        scale=False,
        atac_transform='log1p',
        atac_clip_nonnegative=True,
        atac_standardize=False,
    ):
        super(MultiOmicsDataset, self).__init__()

        # 1. RNA: keep raw counts for NB/ZINB loss, but feed normalized log
        # expression to the encoder, matching the single-omics Dataset behavior.
        if count_key is None:
            count_key = 'counts'
            if count_key not in adata.layers:
                adata.layers[count_key] = adata.X.copy()

        count_rna = adata.layers[count_key].copy()
        self.size_rna = _library_size(count_rna, size=size)

        if normalize:
            sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if scale:
            sc.pp.scale(adata)

        self.feat_rna = _to_dense_float32(adata.X)
        self.count_rna = _to_dense_float32(count_rna)

        # 2. Second modality: protein/ADT or epigenomic ATAC-like features.
        # The default log1p transform keeps MSE targets positive while reducing
        # large outliers that otherwise dominate the fusion branch.
        raw_atac = _to_dense_float32(adata.obsm[atac_key])
        raw_atac = np.nan_to_num(raw_atac, nan=0.0, posinf=0.0, neginf=0.0)
        if atac_clip_nonnegative:
            raw_atac = np.clip(raw_atac, 0.0, None)
        self.feat_atac = raw_atac.copy()
        if atac_transform is None or atac_transform == 'none':
            pass
        elif atac_transform == 'log1p':
            if not atac_clip_nonnegative and np.min(self.feat_atac) < 0:
                raise ValueError("atac_transform='log1p' requires nonnegative ATAC features.")
            self.feat_atac = np.log1p(self.feat_atac).astype(np.float32)
        else:
            raise ValueError("atac_transform must be one of {None, 'none', 'log1p'}")

        if atac_standardize:
            mean = self.feat_atac.mean(axis=0, keepdims=True)
            std = self.feat_atac.std(axis=0, keepdims=True)
            std[std <= 1e-6] = 1.0
            self.feat_atac = ((self.feat_atac - mean) / std).astype(np.float32)

        self.count_atac = raw_atac.copy()
        self.size_atac = np.log1p(np.maximum(self.count_atac.sum(1), 0.0)).astype(np.float32).reshape(-1, 1)

        # 3. 提取批次信息
        self.batch_key = batch_key
        if self.batch_key is not None:
            self.batch_tensor = torch.from_numpy(pd.get_dummies(adata.obs[self.batch_key]).values).float()

    def __len__(self):
        return self.feat_rna.shape[0]

    def __getitem__(self, idx):
        if self.batch_key is not None:
            return (torch.FloatTensor(self.feat_rna[idx]), torch.FloatTensor(self.count_rna[idx]), torch.FloatTensor(self.size_rna[idx]),
                    torch.FloatTensor(self.feat_atac[idx]), torch.FloatTensor(self.count_atac[idx]), torch.FloatTensor(self.size_atac[idx]),
                    self.batch_tensor[idx])
        else:
            return (torch.FloatTensor(self.feat_rna[idx]), torch.FloatTensor(self.count_rna[idx]), torch.FloatTensor(self.size_rna[idx]),
                    torch.FloatTensor(self.feat_atac[idx]), torch.FloatTensor(self.count_atac[idx]), torch.FloatTensor(self.size_atac[idx]))
