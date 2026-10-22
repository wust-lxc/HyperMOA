import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hypermoa.multi_emb_alignment import Multi_Emb_Align
from hypermoa.utils import cluster_func, set_seed


def _add_scaled_pca_rep(adata, source_key: str, target_key: str, n_components: int, random_state: int = 2022):
    if source_key not in adata.obsm:
        raise KeyError(f"{source_key!r} not found in adata.obsm.")
    x = np.asarray(adata.obsm[source_key], dtype=np.float64)
    x = StandardScaler().fit_transform(x)
    n_components = int(min(n_components, x.shape[1], x.shape[0] - 1))
    if n_components < 1:
        raise ValueError("n_components must be at least 1 after shape adjustment.")
    adata.obsm[target_key] = PCA(n_components=n_components, random_state=random_state).fit_transform(x)
    adata.uns[f"{target_key}_source"] = source_key
    adata.uns[f"{target_key}_n_components"] = int(n_components)
    return adata


def main():
    set_seed(42)

    hvg_top = 3000
    ae_epoch = 120
    ae_batch_size = 256
    loss_weight_rna = 1.0
    loss_weight_epi = 5.0
    epi_loss = "mse"

    hgat_epoch = 120
    hgat_batches = 6
    sim_threshold = 0.25
    c_neigh_het = 0.35
    n_neigh_hom = 10
    n_neigh_het = 30
    mini_batch = False
    cluster_num = 18
    mclust_source_key = "HyperMOA_bc"
    mclust_rep_key = "HyperMOA_mclust"
    mclust_pca_components = 10
    mclust_model_name = "EEV"

    root_dir = Path(__file__).resolve().parents[2]
    result_dir = root_dir / "Mouse_brain_result" / "H3K27ac_ATAC"
    embedding_dir = result_dir / "embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    merged_file = result_dir / "h3k27ac_atac_merged.h5ad"
    processed_file = result_dir / "h3k27ac_atac_processed.h5ad"
    if not merged_file.exists():
        raise FileNotFoundError("h3k27ac_atac_merged.h5ad not found. Run 01_prepare_data.py first.")

    adata = sc.read_h5ad(merged_file)
    adata.obs_names_make_unique()

    batch_order = (
        adata.obs[["batch", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")["batch"]
        .astype(str)
        .tolist()
    )
    print("Detected slice order:", batch_order)

    hvg_top = min(hvg_top, adata.n_vars)
    emb_align = Multi_Emb_Align(
        adata,
        batch_key="batch",
        hvg=hvg_top,
        n_hidden=128,
        n_latent=32,
        likelihood="nb",
        num_workers=0,
        result_path=str(result_dir),
        atac_key="EPI",
        encode_batch=False,
        decode_batch=True,
    )

    emb_align.prepare(count_key=None, lib_size="explog", normalize=True, scale=False)
    emb_align.preprocess(
        epoch_ae=ae_epoch,
        batch_size=ae_batch_size,
        loss_weight_rna=loss_weight_rna,
        loss_weight_atac=loss_weight_epi,
        atac_loss=epi_loss,
    )
    emb_align.latent()
    emb_align.batch_center_obsm(source_key="latent", target_key="latent_bc", batch_key="batch")

    emb_align.prepare_hgat(
        spatial_key="spatial",
        feat_key="latent_bc",
        slice_order=batch_order,
        n_neigh_hom=n_neigh_hom,
        n_neigh_het=n_neigh_het,
        c_neigh_het=c_neigh_het,
        sim_threshold=sim_threshold,
    )

    set_seed(42)
    emb_align.train_hgat(
        gamma=0.85,
        mini_batch=mini_batch,
        epoch_hgat=hgat_epoch,
        batches=hgat_batches,
        dropout_hom=0.25,
        dropout_het=0.25,
    )

    adata, attention = emb_align.predict_hgat(mini_batch=mini_batch, batches=hgat_batches)
    attention_file = embedding_dir / "attention.csv"
    attention.to_csv(attention_file)

    emb_align.batch_center_obsm(source_key="HyperMOA", target_key="HyperMOA_bc", batch_key="batch")
    adata = emb_align.adata
    adata = _add_scaled_pca_rep(
        adata,
        source_key=mclust_source_key,
        target_key=mclust_rep_key,
        n_components=mclust_pca_components,
        random_state=2022,
    )
    adata = cluster_func(
        adata,
        clustering="mclust",
        use_rep=mclust_rep_key,
        cluster_num=cluster_num,
        modelNames=mclust_model_name,
        key_add="HyperMOA",
    )
    adata.obs["Domain_mclust_global"] = adata.obs["HyperMOA"].astype(str)
    adata.obs["Domain"] = adata.obs["Domain_mclust_global"].astype(str)
    adata.uns["cluster_method"] = "mclust"
    adata.uns["mclust_rep_key"] = mclust_rep_key
    adata.uns["mclust_source_key"] = mclust_source_key
    adata.uns["mclust_pca_components"] = int(mclust_pca_components)
    adata.uns["mclust_model_name"] = mclust_model_name
    adata.uns["mclust_cluster_num"] = int(cluster_num)
    adata.obs["cluster_method"] = "mclust"
    adata.write(processed_file)

    print("Clustering method: mclust")
    print(f"mclust clusters: G={cluster_num}")
    print(f"mclust input: {mclust_rep_key} from {mclust_source_key}, PCA={mclust_pca_components}, model={mclust_model_name}")
    print(f"Saved attention to: {attention_file}")
    print(f"Updated processed data: {processed_file}")


if __name__ == "__main__":
    main()
