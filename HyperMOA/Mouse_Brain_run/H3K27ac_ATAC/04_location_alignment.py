import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from hypermoa.loc_alignment import Loc_Align
from hypermoa.final_alignment import finalize_anndata


def main():
    init_num_mnn = 10
    detect_num_domains = 3
    detect_alpha = 45
    fine_max_iterations = 160
    fine_tolerance = 1e-10

    root_dir = Path(__file__).resolve().parents[2]
    result_dir = root_dir / "Mouse_brain_result" / "H3K27ac_ATAC"
    location_dir = result_dir / "location"
    location_dir.mkdir(parents=True, exist_ok=True)

    processed_file = result_dir / "h3k27ac_atac_processed.h5ad"
    if not processed_file.exists():
        raise FileNotFoundError("h3k27ac_atac_processed.h5ad not found. Run 03_slice_order_and_z_reconstruction.py first.")

    adata = sc.read_h5ad(processed_file)
    if "HyperMOA_bc" in adata.obsm:
        emb_key = "HyperMOA_bc"
    elif "HyperMOA" in adata.obsm:
        emb_key = "HyperMOA"
    else:
        emb_key = "HyperMOA"
    if emb_key not in adata.obsm:
        raise KeyError("HyperMOA embedding not found. Run 02_embedding_alignment.py first.")

    if "Domain" not in adata.obs:
        adata.obs["Domain"] = adata.obs["batch"].astype(str)
    uns_keep = {
        key: adata.uns[key]
        for key in [
            "cluster_method",
            "mclust_rep_key",
            "mclust_source_key",
            "mclust_pca_components",
            "mclust_model_name",
            "mclust_cluster_num",
        ]
        if key in adata.uns
    }

    keys_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )

    loc_align = Loc_Align(
        adata,
        batch_key="batch",
        batch_order=keys_order,
        result_path=str(result_dir),
    )

    loc_align.init_align(
        emb_key=emb_key,
        spatial_key="spatial",
        num_mnn=init_num_mnn,
        use_scale=False,
    )

    loc_align.detect_fine_points(
        domain_key="Domain",
        slice_boundary=True,
        domain_boundary=True,
        num_domains=detect_num_domains,
        alpha=detect_alpha,
        return_result=False,
    )
    loc_align.plot_edge(spatial_key="transform_init", figsize=(6, 6), s=1.5)

    adata = loc_align.fine_align(max_iterations=fine_max_iterations, tolerance=fine_tolerance)
    adata.obsm["alignment_input"] = adata.obsm["transform_fine"].copy()
    adata, _ = finalize_anndata(adata, keys_order, embedding_key="HyperMOA_bc", seed=42, copy=False)
    adata.uns.update(uns_keep)

    for basis, filename in [
        ("transform_init", "alignment_init.png"),
        ("transform_fine", "alignment_fine.png"),
    ]:
        plt.figure(figsize=(7.0, 3.8))
        sc.pl.embedding(
            adata,
            basis=basis,
            color=["batch", "Domain"],
            frameon=False,
            ncols=2,
            s=6,
            show=False,
        )
        plt.savefig(location_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    orig = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    init_xy = np.asarray(adata.obsm["transform_init"], dtype=np.float64)
    fine_xy = np.asarray(adata.obsm["transform_fine"], dtype=np.float64)
    adata.uns["alignment_rms_init"] = float(np.sqrt(np.mean(np.sum((init_xy - orig) ** 2, axis=1))))
    adata.uns["alignment_rms_fine"] = float(np.sqrt(np.mean(np.sum((fine_xy - orig) ** 2, axis=1))))

    adata.write(processed_file)
    print(f"Embedding key used for location alignment: {emb_key}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
