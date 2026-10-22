import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from hypermoa.loc_alignment import Loc_Align

from hypermoa.final_alignment import finalize_anndata

def _configure_r_env() -> None:
    os.environ.setdefault(
        "R_DEFAULT_PACKAGES",
        "stats,utils,grDevices,graphics,methods",
    )
    if Path("/usr/bin/Rscript").exists():
        os.environ["R_HOME"] = "/usr/lib/R"
        os.environ["PATH"] = "/usr/bin" + os.pathsep + os.environ.get("PATH", "")
        return

    env_prefix = os.environ.get("CONDA_PREFIX")
    if not env_prefix:
        return
    os.environ["R_HOME"] = str(Path(env_prefix) / "lib" / "R")
    os.environ["PATH"] = str(Path(env_prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = str(Path(env_prefix) / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def main():
    _configure_r_env()

    init_num_mnn = 2
    detect_num_domains = 2
    detect_alpha = 60
    fine_max_iterations = 160
    fine_tolerance = 1e-10
    edge_plot_size = (6, 6)
    edge_plot_s = 1.5
    vis_dot_size = 6

    root_dir = Path(__file__).resolve().parent.parent
    result_dir = root_dir / "Simulation_result"
    location_dir = result_dir / "location"
    location_dir.mkdir(parents=True, exist_ok=True)

    processed_file = result_dir / "simulation_processed.h5ad"
    order_file = result_dir / "predicted_slice_order.csv"

    if not processed_file.exists():
        raise FileNotFoundError("simulation_processed.h5ad not found. Run 03_slice_order_and_z_reconstruction.py first.")
    if not order_file.exists():
        raise FileNotFoundError("predicted_slice_order.csv not found. Run 03_slice_order_and_z_reconstruction.py first.")

    adata = sc.read_h5ad(processed_file)
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
        result_path=str(result_dir),
    )

    loc_align.init_align(
        emb_key=emb_key,
        spatial_key="spatial",
        num_mnn=init_num_mnn,
    )

    try:
        loc_align.detect_fine_points(
            domain_key="Domain",
            slice_boundary=True,
            domain_boundary=True,
            num_domains=detect_num_domains,
            alpha=detect_alpha,
            return_result=False,
        )
    except Exception as exc:
        print("Domain boundary detection failed, fallback to slice-only edges.")
        print(f"Detail: {exc}")
        loc_align.detect_fine_points(
            domain_key="Domain",
            slice_boundary=True,
            domain_boundary=False,
            alpha=detect_alpha,
            return_result=False,
        )

    loc_align.plot_edge(spatial_key="transform_init", figsize=edge_plot_size, s=edge_plot_s)
    adata = loc_align.fine_align(max_iterations=fine_max_iterations, tolerance=fine_tolerance)
    adata.obsm["alignment_input"] = adata.obsm["transform_fine"].copy()
    adata, _ = finalize_anndata(adata, keys_order, embedding_key="HyperMOA_bc", seed=42, copy=False)

    plt.figure(figsize=(6.8, 5.6))
    sc.pl.embedding(
        adata,
        basis="transform_init",
        color=["batch", "Domain"],
        frameon=False,
        ncols=2,
        s=vis_dot_size,
        show=False,
    )
    plt.savefig(location_dir / "alignment_init.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.8, 5.6))
    sc.pl.embedding(
        adata,
        basis="transform_fine",
        color=["batch", "Domain"],
        frameon=False,
        ncols=2,
        s=vis_dot_size,
        show=False,
    )
    plt.savefig(location_dir / "alignment_fine.png", dpi=300, bbox_inches="tight")
    plt.close()

    orig = adata.obsm["spatial"]
    init_xy = adata.obsm["transform_init"]
    fine_xy = adata.obsm["transform_fine"]

    rms_init = float(np.sqrt(np.mean(np.sum((init_xy - orig) ** 2, axis=1))))
    rms_fine = float(np.sqrt(np.mean(np.sum((fine_xy - orig) ** 2, axis=1))))

    adata.uns["alignment_rms_init"] = rms_init
    adata.uns["alignment_rms_fine"] = rms_fine
    pd.DataFrame(
        {
            "metric": ["rms_displacement_vs_input"],
            "init": [rms_init],
            "fine": [rms_fine],
        }
    ).to_csv(location_dir / "alignment_displacement.csv", index=False)

    adata.write(processed_file)
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
