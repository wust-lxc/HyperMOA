import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from hypermoa.data_paths import resolve_data_root
from hypermoa.utils import set_seed


SLICE_CONFIG = [
    ("H3K27ac", "Mouse_Brain_H3K27ac"),
    ("ATAC", "Mouse_Brain_ATAC"),
]


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


def _read_common_genes(data_root: Path) -> list[str]:
    gene_sets = []
    for _, data_name in SLICE_CONFIG:
        rna_file = data_root / "Mouse_Brain" / data_name.removeprefix("Mouse_Brain_") / "adata_RNA.h5ad"
        if not rna_file.exists():
            raise FileNotFoundError(f"Missing RNA file: {rna_file}")
        adata_rna = sc.read_h5ad(rna_file, backed="r")
        gene_sets.append(set(map(str, adata_rna.var_names)))
        adata_rna.file.close()

    common_genes = sorted(set.intersection(*gene_sets))
    if len(common_genes) == 0:
        raise ValueError("No common RNA genes between H3K27ac and ATAC.")
    return common_genes


def main():
    set_seed(42)

    root_dir = Path(__file__).resolve().parents[2]
    data_root = resolve_data_root(root_dir)
    result_dir = root_dir / "Mouse_brain_result" / "H3K27ac_ATAC"
    result_dir.mkdir(parents=True, exist_ok=True)

    common_genes = _read_common_genes(data_root)
    adata_list = []
    summary_rows = []

    for order, (slice_name, data_name) in enumerate(SLICE_CONFIG):
        slice_dir = data_root / "Mouse_Brain" / data_name.removeprefix("Mouse_Brain_")
        rna_file = slice_dir / "adata_RNA.h5ad"
        peak_file = slice_dir / "adata_peaks_normalized.h5ad"
        if not peak_file.exists():
            raise FileNotFoundError(f"Missing epigenomic file: {peak_file}")

        adata_rna = sc.read_h5ad(rna_file)
        adata_peak = sc.read_h5ad(peak_file)

        common_obs = adata_rna.obs_names.intersection(adata_peak.obs_names)
        if len(common_obs) == 0:
            raise ValueError(f"No overlapping barcodes for {slice_name}.")

        adata_rna = adata_rna[common_obs, common_genes].copy()
        adata_peak = adata_peak[common_obs].copy()

        if "spatial" not in adata_rna.obsm:
            if "spatial" not in adata_peak.obsm:
                raise KeyError(f"No spatial coordinates found for {slice_name}.")
            adata_rna.obsm["spatial"] = np.asarray(adata_peak.obsm["spatial"])

        if "X_lsi" in adata_peak.obsm:
            epi_feat = np.asarray(adata_peak.obsm["X_lsi"], dtype=np.float32)
            epi_source = "X_lsi"
        elif "X_pca" in adata_peak.obsm:
            epi_feat = np.asarray(adata_peak.obsm["X_pca"], dtype=np.float32)
            epi_source = "X_pca"
        else:
            epi_feat = _to_dense(adata_peak.X)
            epi_source = "X"

        epi_feat = _to_positive_features(epi_feat)

        adata_rna.obs["original_barcode"] = adata_rna.obs_names.astype(str)
        adata_rna.obs_names = [f"{slice_name}_{bc}" for bc in adata_rna.obs_names.astype(str)]
        adata_rna.obs_names_make_unique()
        adata_rna.var_names_make_unique()

        adata_rna.obsm["EPI"] = epi_feat
        adata_rna.obs["batch"] = slice_name
        adata_rna.obs["slice_order"] = int(order)
        adata_rna.obs["epigenomic_assay"] = slice_name

        if "RNA_clusters" in adata_rna.obs.columns:
            adata_rna.obs["Domain"] = adata_rna.obs["RNA_clusters"].astype(str)

        adata_list.append(adata_rna)
        summary_rows.append(
            {
                "slice": slice_name,
                "data_dir": data_name,
                "n_cells": int(adata_rna.n_obs),
                "n_genes_common": int(adata_rna.n_vars),
                "epi_dim": int(adata_rna.obsm["EPI"].shape[1]),
                "epi_source": epi_source,
            }
        )

    merged = ad.concat(adata_list, join="inner", merge="same")
    merged.obs["batch"] = merged.obs["batch"].astype("category")
    merged.obs["batch"] = merged.obs["batch"].cat.set_categories([x[0] for x in SLICE_CONFIG])

    merged_file = result_dir / "h3k27ac_atac_merged.h5ad"
    processed_file = result_dir / "h3k27ac_atac_processed.h5ad"
    merged.write(merged_file)
    merged.write(processed_file)

    pd.DataFrame(summary_rows).to_csv(result_dir / "input_summary.csv", index=False)

    print(f"Common RNA genes: {len(common_genes)}")
    print(f"Merged shape: cells={merged.n_obs}, genes={merged.n_vars}")
    print(f"Saved merged data to: {merged_file}")
    print(f"Initialized processed data to: {processed_file}")


if __name__ == "__main__":
    main()
