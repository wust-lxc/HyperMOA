import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


SLICE_NAMES = [f"Simulation{i}" for i in range(1, 6)]


def main():
    root_dir = Path(__file__).resolve().parent.parent
    result_dir = root_dir / "Simulation_result"
    processed_file = result_dir / "simulation_processed.h5ad"
    order_file = result_dir / "predicted_slice_order.csv"

    if not processed_file.exists():
        raise FileNotFoundError("simulation_processed.h5ad not found. Run 02_embedding_alignment.py first.")

    adata = sc.read_h5ad(processed_file)

    batches_present = set(adata.obs["batch"].astype(str).unique())
    fixed_present = [b for b in SLICE_NAMES if b in batches_present]
    missing = [b for b in SLICE_NAMES if b not in batches_present]
    extras = sorted(list(batches_present - set(SLICE_NAMES)))

    if len(fixed_present) == 0:
        raise ValueError("None of the fixed-order slice names were found in adata.obs['batch'].")

    final_order = fixed_present + extras
    order_df = pd.DataFrame(
        {
            "batch": final_order,
            "score": np.arange(len(final_order), dtype=float),
        }
    )

    if order_df.shape[0] > 1:
        order_df["z_rec"] = np.linspace(0.0, float(order_df.shape[0] - 1), order_df.shape[0])
    else:
        order_df["z_rec"] = 0.0

    score_map = dict(zip(order_df["batch"], order_df["score"]))
    z_map = dict(zip(order_df["batch"], order_df["z_rec"]))

    adata.obs["z_rec_raw"] = adata.obs["batch"].astype(str).map(score_map).astype(float)
    adata.obs["z_rec"] = adata.obs["batch"].astype(str).map(z_map).astype(float)

    order_df.to_csv(order_file, index=False)
    adata.write(processed_file)

    if missing:
        print(f"Warning: missing fixed slices in data: {missing}")
    if extras:
        print(f"Warning: extra slices appended after fixed order: {extras}")
    print(f"Using fixed order: {final_order}")
    print(f"Saved predicted order to: {order_file}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
