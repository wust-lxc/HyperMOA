import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from hypermoa.loc_alignment import Loc_Align
from hypermoa.final_alignment import finalize_anndata


def _remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Human_lymph_node_result")
    location_dir = os.path.join(result_dir, "location")
    os.makedirs(location_dir, exist_ok=True)

    processed_file = os.path.join(result_dir, "human_lymph_node_processed.h5ad")
    order_file = os.path.join(result_dir, "predicted_slice_order.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("human_lymph_node_processed.h5ad not found. Run 03_slice_order_and_z_reconstruction.py first.")
    if not os.path.exists(order_file):
        raise FileNotFoundError("predicted_slice_order.csv not found. Run 03_slice_order_and_z_reconstruction.py first.")

    adata = sc.read_h5ad(processed_file)
    uns_keep = {
        key: adata.uns[key]
        for key in [
            "cluster_method",
            "domain_refinement",
            "mclust_rep_key",
            "mclust_source_key",
            "mclust_pca_components",
            "mclust_model_name",
            "mclust_cluster_num",
            "mclust_observed_cluster_num",
        ]
        if key in adata.uns
    }
    if "HyperMOA" not in adata.obsm:
        raise KeyError("HyperMOA embedding not found. Run 02_embedding_alignment.py first.")
    if "HyperMOA_bc" in adata.obsm:
        emb_key = "HyperMOA_bc"
    else:
        emb_key = "HyperMOA"

    if "Domain" not in adata.obs:
        adata.obs["Domain"] = adata.obs["batch"].astype(str)

    order_df = pd.read_csv(order_file)
    keys_order = order_df.sort_values("z_rec")["batch"].astype(str).tolist()

    loc_align = Loc_Align(
        adata,
        batch_key="batch",
        batch_order=keys_order,
        result_path=result_dir,
    )

    loc_align.init_align(
        emb_key=emb_key,
        spatial_key="spatial",
        num_mnn=10,
        use_scale=True,
    )

    loc_align.detect_fine_points(
        domain_key="Domain",
        slice_boundary=True,
        domain_boundary=True,
        num_domains=3,
        alpha=45,
        return_result=False,
    )

    # 边界可视化继续保留在 location/edge 目录下。
    loc_align.plot_edge(spatial_key="transform_init", figsize=(6, 6), s=1.5)

    adata = loc_align.fine_align(max_iterations=160, tolerance=1e-10)
    adata.obsm["alignment_input"] = adata.obsm["transform_fine"].copy()
    adata, _ = finalize_anndata(adata, keys_order, embedding_key="HyperMOA_bc", seed=42, copy=False)
    adata.uns.update(uns_keep)

    plt.figure(figsize=(6.8, 5.6))
    sc.pl.embedding(
        adata,
        basis="transform_init",
        color=["batch", "Domain"],
        frameon=False,
        ncols=2,
        s=6,
        show=False,
    )
    plt.savefig(os.path.join(location_dir, "alignment_init.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.8, 5.6))
    sc.pl.embedding(
        adata,
        basis="transform_fine",
        color=["batch", "Domain"],
        frameon=False,
        ncols=2,
        s=6,
        show=False,
    )
    plt.savefig(os.path.join(location_dir, "alignment_fine.png"), dpi=300, bbox_inches="tight")
    plt.close()

    orig = adata.obsm["spatial"]
    init_xy = adata.obsm["transform_init"]
    fine_xy = adata.obsm["transform_fine"]

    adata.uns["alignment_rms_init"] = float(np.sqrt(np.mean(np.sum((init_xy - orig) ** 2, axis=1))))
    adata.uns["alignment_rms_fine"] = float(np.sqrt(np.mean(np.sum((fine_xy - orig) ** 2, axis=1))))
    # 位移指标已经会汇总进 metrics_summary.csv，这里不再重复单独导出。
    _remove_if_exists(os.path.join(location_dir, "alignment_displacement.csv"))

    adata.write(processed_file)
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
