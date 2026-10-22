"""Read selected AnnData annotations and coordinates without loading expression."""

from pathlib import Path
import h5py
import numpy as np
import pandas as pd


def read_vector(node):
    if isinstance(node, h5py.Group):
        codes = node['codes'][:]
        categories = read_vector(node['categories'])
        return np.array([categories[i] if i >= 0 else None for i in codes], dtype=object)
    return np.array([v.decode() if isinstance(v, bytes) else v for v in node[:]])


def read_metadata(path: Path, obs_columns=(), obsm_keys=()):
    with h5py.File(path, 'r') as f:
        obs = f['obs']
        index = read_vector(obs[obs.attrs['_index']])
        columns = list(obs) if obs_columns is None else list(obs_columns)
        frame = pd.DataFrame({k: read_vector(obs[k]) for k in columns
                              if k != obs.attrs['_index']}, index=index)
        matrices = {k: np.asarray(f['obsm'][k], dtype=float) for k in obsm_keys}
    return frame, matrices
