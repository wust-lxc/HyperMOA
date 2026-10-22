import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path


def _configure_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key, "").strip()
        if not value.isdigit() or int(value) < 1:
            os.environ[key] = "1"


_configure_thread_env()

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from hypermoa.data_paths import resolve_data_root
from hypermoa.utils import set_seed


SLICE_NAMES = [f"Simulation{i}" for i in range(1, 6)]


def _to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _to_positive_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.nanmin(x, axis=0, keepdims=True)
    scale = np.nanpercentile(x, 99, axis=0, keepdims=True)
    scale[scale <= 0] = 1.0
    x = x / scale
    x = np.clip(x, 0.0, None)
    x = x + 1e-4
    return x.astype(np.float32)


def _read_common_genes(data_dir: Path) -> list[str]:
    gene_sets = []
    for slice_name in SLICE_NAMES:
        rna_file = data_dir / "Simulation" / f"slice{slice_name.removeprefix('Simulation')}" / "adata_RNA.h5ad"
        if not rna_file.exists():
            raise FileNotFoundError(f"Missing RNA file: {rna_file}")
        adata_rna = sc.read_h5ad(rna_file, backed="r")
        gene_sets.append(set(map(str, adata_rna.var_names)))
        adata_rna.file.close()

    common_genes = sorted(set.intersection(*gene_sets))
    if len(common_genes) == 0:
        raise ValueError("No common RNA genes across simulations.")
    return common_genes


def _merge_truth_columns(adata_rna, adata_adt) -> None:
    for col in ("ground_truth", "truth"):
        if col in adata_rna.obs:
            adata_rna.obs[col] = adata_rna.obs[col].astype(str).values
        elif col in adata_adt.obs:
            adata_rna.obs[col] = adata_adt.obs[col].astype(str).values

    if "ground_truth" in adata_rna.obs:
        adata_rna.obs["final_annot"] = adata_rna.obs["ground_truth"].astype(str)
        adata_rna.obs["Domain"] = adata_rna.obs["ground_truth"].astype(str)


def main():
    set_seed(42)

    data_dir = resolve_data_root(ROOT_DIR)
    result_dir = ROOT_DIR / "Simulation_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    common_genes = _read_common_genes(data_dir)
    adata_list = []
    summary_rows = []

    for order, slice_name in enumerate(SLICE_NAMES):
        slice_dir = data_dir / "Simulation" / f"slice{slice_name.removeprefix('Simulation')}"
        rna_file = slice_dir / "adata_RNA.h5ad"
        adt_file = slice_dir / "adata_ADT.h5ad"
        if not adt_file.exists():
            raise FileNotFoundError(f"Missing ADT file: {adt_file}")

        adata_rna = sc.read_h5ad(rna_file)
        adata_adt = sc.read_h5ad(adt_file)

        common_obs = adata_rna.obs_names.intersection(adata_adt.obs_names)
        if len(common_obs) == 0:
            raise ValueError(f"No overlapping cells in {slice_name} between RNA and ADT.")

        adata_rna = adata_rna[common_obs, common_genes].copy()
        adata_adt = adata_adt[common_obs].copy()

        if "spatial" not in adata_rna.obsm:
            if "spatial" not in adata_adt.obsm:
                raise KeyError(f"No 'spatial' found in RNA or ADT for {slice_name}")
            adata_rna.obsm["spatial"] = np.asarray(adata_adt.obsm["spatial"])

        adt_feat = _to_positive_features(_to_dense(adata_adt.X))

        adata_rna.obs["original_barcode"] = adata_rna.obs_names.astype(str)
        adata_rna.obs_names = [f"{slice_name}_{bc}" for bc in adata_rna.obs_names.astype(str)]
        adata_rna.obs_names_make_unique()
        adata_rna.var_names_make_unique()

        adata_rna.obsm["ADT"] = adt_feat
        adata_rna.obs["batch"] = slice_name
        adata_rna.obs["slice_order"] = int(order)

        _merge_truth_columns(adata_rna, adata_adt)

        adata_list.append(adata_rna)
        summary_rows.append(
            {
                "slice": slice_name,
                "n_cells": int(adata_rna.n_obs),
                "n_genes_common": int(adata_rna.n_vars),
                "adt_dim": int(adata_rna.obsm["ADT"].shape[1]),
                "has_ground_truth": bool("ground_truth" in adata_rna.obs),
            }
        )

    merged = ad.concat(adata_list, join="inner", merge="same")
    merged.obs["batch"] = merged.obs["batch"].astype("category")
    merged.obs["batch"] = merged.obs["batch"].cat.set_categories(SLICE_NAMES)

    merged_file = result_dir / "simulation_merged.h5ad"
    processed_file = result_dir / "simulation_processed.h5ad"

    merged.write(merged_file)
    merged.write(processed_file)
    pd.DataFrame(summary_rows).to_csv(result_dir / "input_summary.csv", index=False)

    print(f"Simulation slices: {SLICE_NAMES}")
    print(f"Common genes across slices: {len(common_genes)}")
    print(f"Merged shape: cells={merged.n_obs}, genes={merged.n_vars}")
    print(f"Saved merged data to: {merged_file}")
    print(f"Initialized processed data to: {processed_file}")


if __name__ == "__main__":
    main()
