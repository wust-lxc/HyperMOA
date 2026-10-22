import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scanpy as sc

from hypermoa.loc_prediction import sort_slices


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "Human_lymph_node_result")

    processed_file = os.path.join(result_dir, "human_lymph_node_processed.h5ad")
    attention_file = os.path.join(result_dir, "embedding", "attention.csv")
    order_file = os.path.join(result_dir, "predicted_slice_order.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError("human_lymph_node_processed.h5ad not found. Run 02_embedding_alignment.py first.")
    if not os.path.exists(attention_file):
        raise FileNotFoundError("attention.csv not found. Run 02_embedding_alignment.py first.")

    adata = sc.read_h5ad(processed_file)
    attention = pd.read_csv(attention_file, index_col=0)
    attention.index = attention.index.astype(str)
    attention.columns = attention.columns.astype(str)

    # Use the earliest known slice by slice_order as anchor to stabilize orientation.
    anchor = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")
        .iloc[0]["batch"]
    )
    dists_pred = sort_slices(attention, start=str(anchor))

    order_df = pd.DataFrame(
        {
            "batch": list(dists_pred.keys()),
            "score": list(dists_pred.values()),
        }
    ).sort_values("score", ascending=True)

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

    print(f"Anchor slice for ordering: {anchor}")
    print(f"Saved predicted order to: {order_file}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
