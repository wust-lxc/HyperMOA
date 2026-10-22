import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


FIXED_ORDER = ["H3K27ac", "ATAC"]


def main():
    root_dir = Path(__file__).resolve().parents[2]
    result_dir = root_dir / "Mouse_brain_result" / "H3K27ac_ATAC"
    processed_file = result_dir / "h3k27ac_atac_processed.h5ad"

    if not processed_file.exists():
        raise FileNotFoundError("h3k27ac_atac_processed.h5ad not found. Run 02_embedding_alignment.py first.")

    adata = sc.read_h5ad(processed_file)
    batches_present = set(adata.obs["batch"].astype(str).unique())
    fixed_present = [b for b in FIXED_ORDER if b in batches_present]
    extras = sorted(list(batches_present - set(FIXED_ORDER)))
    final_order = fixed_present + extras

    if len(final_order) == 0:
        raise ValueError("No slices found in adata.obs['batch'].")

    order_df = pd.DataFrame(
        {
            "batch": final_order,
            "score": np.arange(len(final_order), dtype=float),
        }
    )
    order_df["z_rec"] = np.linspace(0.0, float(len(final_order) - 1), len(final_order))

    score_map = dict(zip(order_df["batch"], order_df["score"]))
    z_map = dict(zip(order_df["batch"], order_df["z_rec"]))
    adata.obs["z_rec_raw"] = adata.obs["batch"].astype(str).map(score_map).astype(float)
    adata.obs["z_rec"] = adata.obs["batch"].astype(str).map(z_map).astype(float)

    adata.write(processed_file)

    print(f"Using fixed order: {final_order}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
