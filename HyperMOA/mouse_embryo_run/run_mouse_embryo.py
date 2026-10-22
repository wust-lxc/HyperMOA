#!/usr/bin/env python3
"""Annotation-independent embryonic integration and conditional correspondence."""

from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "8")

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HyperMOA"))
import anndata as ad
import faiss
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import scanpy as sc
import torch
from torch_geometric.data import HeteroData

from hypermoa.emb_alignment import Emb_Align
from hypermoa.embedding.dataset_hgat import calcu_adaptive_hyperedge, calcu_adj
from hypermoa.embedding.module_hgat import HGAT
from hypermoa.final_alignment import finalize_anndata
from hypermoa.loc_alignment import Loc_Align
from hypermoa.utils import set_seed

STAGES = [f"E{x}.5" for x in range(9, 17)]
NODES = {s: s.replace(".", "_") for s in STAGES}
DATA = ROOT / "data/datasets/Stereo_seq_mouse_embryo"
OUT = ROOT / "HyperMOA/mouse_embryo_result"
MARKERS = {
    "Heart": ["Tnnt2", "Myh6", "Actc1"],
    "Liver": ["Afp", "Alb", "Hnf4a"],
    "Brain": ["Sox2", "Tubb3", "Pax6"],
    "Kidney": ["Six2", "Pax2", "Wt1"],
}
CONFIG = {
    "version": "annotation_independent_embryo_v1", "seed": 42,
    "stages": STAGES, "training_spots_per_stage": 4000,
    "alignment_spots_per_stage": 2000, "hvg": 3000,
    "ae_epochs": 100, "ae_batch_size": 256, "n_latent": 32,
    "hgat_epochs": 120, "gamma": .70, "dropout_hom": .25,
    "dropout_het": .25, "hgat_spatial_neighbors": 10,
    "hgat_cross_neighbors": 10, "hyperedge_cosine_cutoff": .25,
    "cross_graph": "adjacent stages only; approximate cosine top-k",
    "hnsw_M": 32, "hnsw_efSearch": 128, "hnsw_efConstruction": 100,
    "matching_k": 10, "matching_spatial_radius_target_NN": 10.,
    "matching_embedding_quantile": .95, "matching_max_cosine_distance": .35,
    "matching_mass": "row-normalized affinity times maximum edge affinity; residual is unmatched",
    "primary_embedding": "HyperMOA_bc", "stage_centering": "record raw and centered; validate using excluded marker genes",
    "stability_seeds": [42, 7, 2022],
    "stability_scope": "alignment refinement only; AE and HGAT frozen at seed42",
    "labels_in_training": False, "domain_boundary": False,
    "excluded_marker_genes": sorted({g for v in MARKERS.values() for g in v}),
    "limitations": ["one sagittal section per stage; no biological replicates",
                    "conditional spot correspondences are not cellular lineage or organ volume",
                    "sampled model training; full-spot inference; sampled affine fitting",
                    "no ground-truth paired spots; do not compute PAA/FOSCTTM"],
}


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strings(dataset):
    return np.asarray(dataset.asstr()[:], dtype=str)


def index_key(group):
    key = group.attrs.get("_index", "_index")
    return key.decode() if isinstance(key, bytes) else key


def read_labels(file):
    entry = file["obs/annotation"]
    if isinstance(entry, h5py.Group):
        categories = strings(entry["categories"])
        codes = entry["codes"][:]
    elif "categories" in entry.attrs:
        categories = strings(file[entry.attrs["categories"]])
        codes = entry[:]
    else:
        values = strings(entry)
        return np.where(np.isin(values, ["", "nan", "None"]), "Unannotated", values)
    values = np.full(len(codes), "Unannotated", dtype=object)
    valid = codes >= 0
    values[valid] = categories[codes[valid]]
    return values.astype(str)


def read_count(file):
    group = file["layers/count"]
    if not isinstance(group, h5py.Group):
        raise ValueError("Expected sparse original count layer")
    shape = tuple(group.attrs["shape"])
    result = sparse.csr_matrix((group["data"][:].astype(np.float32),
                               group["indices"][:], group["indptr"][:]), shape=shape)
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError("Invalid original counts")
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("tables", "models", "pairs", "diagnostics", "inputs"):
        (OUT / name).mkdir(exist_ok=True)
    dump(OUT / "protocol.json", CONFIG)
    metadata, common, selections = [], None, {}
    references, reference_stage = [], []
    for stage in STAGES:
        path = DATA / f"{stage}_E1S1.MOSTA.h5ad"
        with h5py.File(path, "r") as f:
            genes = strings(f["var"][index_key(f["var"])])
            ids = strings(f["obs"][index_key(f["obs"])])
            xy = np.asarray(f["obsm/spatial"][:], dtype=np.float64)
            if len(set(genes)) != len(genes) or len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate genes or spot IDs: {stage}")
            if xy.shape != (len(ids), 2) or not np.isfinite(xy).all():
                raise ValueError(f"Invalid spatial coordinates: {stage}")
            common = set(genes) if common is None else common.intersection(genes)
            rng = np.random.default_rng(CONFIG["seed"] + STAGES.index(stage))
            selections[stage] = np.sort(rng.choice(len(ids), min(len(ids), CONFIG["training_spots_per_stage"]), replace=False))
            metadata.append({"stage": stage, "path": str(path), "n_spots": len(ids),
                             "n_genes": len(genes), "bytes": path.stat().st_size,
                             "mtime_ns": path.stat().st_mtime_ns, "sha256": digest(path)})
        log(f"Validated {stage}: {len(ids)} spots")
    common = sorted(common.difference(CONFIG["excluded_marker_genes"]))
    if len(common) < CONFIG["hvg"]:
        raise ValueError("Too few common training genes")
    for stage in STAGES:
        with h5py.File(DATA / f"{stage}_E1S1.MOSTA.h5ad", "r") as f:
            genes = strings(f["var"][index_key(f["var"])])
            cols = pd.Index(genes).get_indexer(common)
            counts = read_count(f)
            references.append(counts[selections[stage]][:, cols])
            reference_stage.extend([stage] * len(selections[stage]))
        del counts
        gc.collect()
    ref = ad.AnnData(sparse.vstack(references, format="csr"),
                     obs=pd.DataFrame({"stage": pd.Categorical(reference_stage, categories=STAGES)}),
                     var=pd.DataFrame(index=common))
    ref.obs_names = [f"reference_{i}" for i in range(ref.n_obs)]
    sc.pp.highly_variable_genes(ref, flavor="seurat_v3", n_top_genes=CONFIG["hvg"], batch_key="stage")
    selected = ref.var.index[ref.var.highly_variable].tolist()
    pd.DataFrame({"gene": selected}).to_csv(OUT / "inputs/training_genes.csv", index=False)
    dump(OUT / "inputs/input_manifest.json", metadata)
    blocks, spots, marker_blocks, pseudobulk, library_rows = [], [], [], [], []
    marker_genes = CONFIG["excluded_marker_genes"]
    for stage in STAGES:
        with h5py.File(DATA / f"{stage}_E1S1.MOSTA.h5ad", "r") as f:
            counts = read_count(f)
            genes = strings(f["var"][index_key(f["var"])])
            ids = strings(f["obs"][index_key(f["obs"])])
            labels = read_labels(f)
            library = np.asarray(counts.sum(1)).ravel()
            if (library <= 0).any():
                raise ValueError(f"Zero-count spot in {stage}; explicit QC policy required")
            blocks.append(counts[:, pd.Index(genes).get_indexer(selected)])
            markers = np.zeros((len(ids), len(marker_genes)), dtype=np.float32)
            positions = pd.Index(genes).get_indexer(marker_genes)
            available = positions >= 0
            markers[:, available] = counts[:, positions[available]].toarray()
            markers = np.log1p(markers * (1e4 / library[:, None]))
            marker_blocks.append(markers)
            training = np.zeros(len(ids), dtype=bool)
            training[selections[stage]] = True
            spots.append(pd.DataFrame({"spot_id": [f"{stage}:{i}" for i in ids],
                         "original_barcode": ids, "stage": stage, "slice_order": STAGES.index(stage),
                         "stage_numeric": float(stage[1:]), "annotation": labels,
                         "training_spot": training, "full_library_size": library,
                         "spatial_x": f["obsm/spatial"][:, 0], "spatial_y": f["obsm/spatial"][:, 1]}))
            for tissue in sorted(set(labels)):
                mask = labels == tissue
                total = np.asarray(counts[mask].sum(0)).ravel()
                cp10k = total / library[mask].sum() * 1e4
                pseudobulk.append(pd.DataFrame({"stage": stage, "tissue": tissue, "gene": genes,
                                                "pseudobulk_cp10k": cp10k}))
                library_rows.append({"stage": stage, "tissue": tissue, "n_spots": int(mask.sum()),
                                     "total_counts": float(library[mask].sum()),
                                     "section_spot_fraction": float(mask.mean())})
        log(f"Prepared sparse features and original-count summaries: {stage}")
        del counts
        gc.collect()
    frame = pd.concat(spots, ignore_index=True).set_index("spot_id")
    xy = frame[["spatial_x", "spatial_y"]].to_numpy()
    labels = frame[["annotation"]].copy()
    labels.to_csv(OUT / "inputs/annotations.csv.gz")
    frame = frame.drop(columns=["annotation", "spatial_x", "spatial_y"])
    frame["stage"] = pd.Categorical(frame.stage, categories=STAGES, ordered=True)
    data = ad.AnnData(sparse.vstack(blocks, format="csr"), obs=frame, var=pd.DataFrame(index=selected))
    data.obsm["spatial"] = xy
    data.write_h5ad(OUT / "inputs/model_input.h5ad", compression="gzip")
    np.savez_compressed(OUT / "inputs/excluded_marker_expression.npz", expression=np.vstack(marker_blocks), genes=marker_genes)
    pd.concat(pseudobulk, ignore_index=True).to_csv(OUT / "tables/organ_pseudobulk_expression.csv.gz", index=False)
    pd.DataFrame(library_rows).to_csv(OUT / "tables/section_tissue_composition.csv", index=False)
    frame.loc[frame.training_spot].to_csv(OUT / "inputs/training_spots.csv.gz")
    dump(OUT / "inputs/summary.json", {"n_spots": data.n_obs, "common_genes_excluding_markers": len(common),
                                      "training_genes": len(selected), "training_spots": int(frame.training_spot.sum())})


def unit(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def knn(query, reference, k=10, approximate=True):
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite nearest-neighbor input")
    query, reference = unit(query), unit(reference)
    faiss.omp_set_num_threads(8)
    if approximate:
        index = faiss.IndexHNSWFlat(reference.shape[1], CONFIG["hnsw_M"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = CONFIG["hnsw_efConstruction"]
        index.hnsw.efSearch = CONFIG["hnsw_efSearch"]
    else:
        index = faiss.IndexFlatIP(reference.shape[1])
    index.add(reference)
    scores, neighbors = index.search(query, min(k, len(reference)))
    if (neighbors < 0).any():
        raise ValueError("Neighbor search returned missing indices")
    return np.maximum(0, 1 - scores), neighbors


def graph(data, key):
    result = HeteroData()
    masks = {s: np.flatnonzero(data.obs.stage.astype(str).to_numpy() == s) for s in STAGES}
    for stage, indices in masks.items():
        xy, z = data.obsm["spatial"][indices], data.obsm[key][indices]
        result[NODES[stage]].x = torch.from_numpy(z.astype(np.float32))
        result[NODES[stage]].hyperedge_index = calcu_adaptive_hyperedge(xy, z, n_neigh=10, sim_threshold=.25)
        result[NODES[stage], "0", NODES[stage]].edge_index = calcu_adj(xy, n_neigh=10)
    for left, right in zip(STAGES[:-1], STAGES[1:]):
        a, b = masks[left], masks[right]
        _, indices = knn(data.obsm[key][a], data.obsm[key][b], 10)
        edge = np.vstack([np.repeat(np.arange(len(a)), indices.shape[1]), indices.ravel()])
        result[NODES[left], "1", NODES[right]].edge_index = torch.from_numpy(edge).long()
        _, indices = knn(data.obsm[key][b], data.obsm[key][a], 10)
        edge = np.vstack([np.repeat(np.arange(len(b)), indices.shape[1]), indices.ravel()])
        result[NODES[right], "1", NODES[left]].edge_index = torch.from_numpy(edge).long()
        log(f"Sparse HGAT cross graph: {left}/{right}")
    return result, {NODES[s]: v for s, v in masks.items()}


def center(data, key):
    x = np.asarray(data.obsm[key], dtype=np.float32)
    mean = x.mean(0)
    centered = x.copy()
    for stage in STAGES:
        mask = data.obs.stage.astype(str).to_numpy() == stage
        centered[mask] += mean - x[mask].mean(0)
    data.obsm[key + "_bc"] = centered


def embed():
    set_seed(42)
    torch.set_num_threads(8)
    data = ad.read_h5ad(OUT / "inputs/model_input.h5ad")
    assert "annotation" not in data.obs and "Domain" not in data.obs
    hvg_total = np.asarray(data.X.sum(1)).ravel()
    data.obs["model_informative"] = hvg_total > 0
    data.obs["actual_training_spot"] = data.obs.training_spot & data.obs.model_informative
    data.obs.loc[~data.obs.model_informative].to_csv(OUT / "inputs/low_information_spots.csv")
    reference = data[data.obs.actual_training_spot].copy()
    emb = Emb_Align(reference, batch_key="stage", hvg=False, n_latent=32,
                    likelihood="nb", num_workers=0, result_path=str(OUT))
    reference_total = np.asarray(reference.X.sum(1)).ravel()
    normalization_target = float(np.median(reference_total))
    emb.prepare(lib_size="explog", normalize=True, scale=False)
    emb.preprocess(epoch_ae=CONFIG["ae_epochs"], batch_size=CONFIG["ae_batch_size"])
    torch.save(emb.ae.state_dict(), OUT / "models/ae.pt")
    latent = np.empty((data.n_obs, 32), dtype=np.float32)
    onehot_order = pd.get_dummies(reference.obs.stage).columns.tolist()
    with torch.no_grad():
        for start in range(0, data.n_obs, 2048):
            end = min(start + 2048, data.n_obs)
            count = data.X[start:end].toarray().astype(np.float32)
            total = count.sum(1)
            feat = np.log1p(count * (normalization_target / np.maximum(total[:, None], 1.)))
            size = np.exp(np.log10(np.maximum(total, 1.)))[:, None].astype(np.float32)
            batches = data.obs.stage.astype(str).iloc[start:end].to_numpy()
            onehot = (batches[:, None] == np.asarray(onehot_order)[None, :]).astype(np.float32)
            _, _, _, z = emb.ae(torch.from_numpy(feat).to(emb.device),
                                torch.from_numpy(onehot).to(emb.device))
            latent[start:end] = z[:, :CONFIG["n_latent"]].cpu().numpy()
    if not np.isfinite(latent).all():
        raise ValueError("Nonfinite AE inference; stop before graph training")
    data.obsm["latent"] = latent
    center(data, "latent")
    ref = data[data.obs.actual_training_spot].copy()
    g, masks = graph(ref, "latent_bc")
    device = emb.device
    g = g.to(device)
    model = HGAT(32, g.metadata(), gamma=.70, dropout_hom=.25, dropout_het=.25).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    loss_rows = []
    for epoch in range(CONFIG["hgat_epochs"]):
        model.train()
        opt.zero_grad()
        prediction = model(g.x_dict, g.edge_index_dict,
                           {s: g[s].hyperedge_index for s in g.node_types})
        loss = sum(torch.nn.functional.mse_loss(prediction[s], g[s].x) for s in g.node_types)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite HGAT training loss")
        loss.backward()
        opt.step()
        loss_rows.append({"epoch": epoch + 1, "reconstruction_loss": float(loss.detach())})
        if (epoch + 1) % 10 == 0:
            log(f"HGAT {epoch + 1}/{CONFIG['hgat_epochs']}: {float(loss.detach()):.6f}")
    torch.save(model.state_dict(), OUT / "models/hgat.pt")
    pd.DataFrame(loss_rows).to_csv(OUT / "diagnostics/hgat_loss.csv", index=False)
    del g, ref, reference, emb
    gc.collect()
    torch.cuda.empty_cache()
    g, masks = graph(data, "latent_bc")
    g = g.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(g.x_dict, g.edge_index_dict, {s: g[s].hyperedge_index for s in g.node_types})
        z = np.empty_like(latent)
        for node in g.node_types:
            z[masks[node]] = prediction[node].cpu().numpy()
    data.obsm["HyperMOA"] = z
    center(data, "HyperMOA")
    # Expression remains in original-count summaries; integration outputs are a compact coordinate sidecar.
    compact = ad.AnnData(sparse.csr_matrix((data.n_obs, 0)), obs=data.obs.copy())
    for key in ("spatial", "latent", "latent_bc", "HyperMOA", "HyperMOA_bc"):
        compact.obsm[key] = data.obsm[key]
    compact.uns["protocol"] = json.dumps(CONFIG)
    compact.write_h5ad(OUT / "embeddings.h5ad", compression="gzip")
    dump(OUT / "models/normalization.json", {"normalization_target": normalization_target,
          "onehot_order": onehot_order, "training_library": "HVG sum; explog, matching existing Emb_Align.prepare"})
    log("Full-spot embedding inference saved")


def affine_for(raw, fitted):
    design = np.column_stack([raw, np.ones(len(raw))])
    matrix = np.linalg.lstsq(design, fitted, rcond=None)[0]
    error = float(np.max(np.linalg.norm(design @ matrix - fitted, axis=1)))
    if error > 1e-5:
        raise ValueError(f"Landmark result is not a reproducible affine: {error}")
    return matrix, error


def align():
    data = ad.read_h5ad(OUT / "embeddings.h5ad")
    indices = []
    for stage in STAGES:
        candidates = np.flatnonzero((data.obs.stage.astype(str).to_numpy() == stage) & data.obs.model_informative.to_numpy())
        indices.extend(np.random.default_rng(42 + STAGES.index(stage)).choice(candidates, min(2000, len(candidates)), replace=False))
    landmarks = data[np.sort(indices)].copy()
    raw = landmarks.obsm["spatial"].copy()
    loc = Loc_Align(landmarks, batch_key="stage", batch_order=STAGES, result_path=str(OUT))
    loc.init_align(emb_key="HyperMOA_bc", num_mnn=5, use_scale=False)
    spacings = [np.median(cKDTree(x.obsm["transform_init"]).query(x.obsm["transform_init"], k=2)[0][:, 1]) for x in loc.init_adatas]
    alpha = float(max(spacings) * 5)
    loc.detect_fine_points(slice_boundary=True, domain_boundary=False, alpha=alpha)
    if any(len(a) < 3 or len(b) < 3 for a, b in loc.boundary):
        raise ValueError("Slice boundary failed; do not silently substitute annotation domains")
    baseline = loc.fine_align(max_iterations=160, tolerance=1e-10)
    baseline = baseline[landmarks.obs_names].copy()
    baseline.obsm["spatial"] = raw
    assert "Domain" not in baseline.obs and "annotation" not in baseline.obs
    seed_rows, transforms, diagnostics = [], {}, {}
    for seed in CONFIG["stability_seeds"]:
        fitted, result = finalize_anndata(baseline, STAGES, batch_key="stage", seed=seed)
        full = np.empty((data.n_obs, 2), dtype=np.float64)
        stage_transforms = {}
        for stage in STAGES:
            subset = baseline.obs.stage.astype(str).to_numpy() == stage
            matrix, error = affine_for(raw[subset], fitted.obsm["aligned_spatial"][subset])
            mask = data.obs.stage.astype(str).to_numpy() == stage
            full[mask] = np.column_stack([data.obsm["spatial"][mask], np.ones(mask.sum())]) @ matrix
            stage_transforms[stage] = {"matrix": matrix.tolist(), "landmark_error": error}
        np.savez_compressed(OUT / f"diagnostics/aligned_seed{seed}.npz", coordinates=full)
        transforms[str(seed)] = stage_transforms
        diagnostics[str(seed)] = {"chain_accepted": result.chain_accepted,
                                 "pairs": result.pair_diagnostics, "chain": result.chain_diagnostics}
        for record in result.pair_diagnostics:
            seed_rows.append({"seed": seed, "pair": record["pair"], "accepted": record["accepted"],
                              "rejection_reasons": record["rejection_reasons"]})
        if seed == 42:
            data.obsm["aligned_spatial"] = full
            data.obsm["transform_fine"] = full.copy()
    base_full = np.empty_like(data.obsm["spatial"])
    for stage in STAGES:
        subset = baseline.obs.stage.astype(str).to_numpy() == stage
        mask = data.obs.stage.astype(str).to_numpy() == stage
        matrix, _ = affine_for(raw[subset], baseline.obsm["transform_fine"][subset])
        base_full[mask] = np.column_stack([data.obsm["spatial"][mask], np.ones(mask.sum())]) @ matrix
    data.obsm["alignment_input"] = base_full
    data.obsm["rec_3d_time"] = np.column_stack([data.obsm["aligned_spatial"], data.obs.stage_numeric])
    data.uns["time_axis"] = "developmental day; not physical serial-section z"
    data.write_h5ad(OUT / "aligned_label_free.h5ad", compression="gzip")
    baseline.obs.to_csv(OUT / "inputs/alignment_landmarks.csv.gz")
    dump(OUT / "diagnostics/affine_transforms.json", transforms)
    # Some rejected fits contain NaN diagnostics, represented explicitly as JSON null.
    clean = json.loads(json.dumps(diagnostics), parse_constant=lambda _: None)
    dump(OUT / "diagnostics/refinement.json", clean)
    pd.DataFrame(seed_rows).to_csv(OUT / "diagnostics/refinement_decisions.csv", index=False)
    dump(OUT / "diagnostics/alignment_settings.json", {"alpha": alpha, "domain_boundary": False,
          "landmarks_per_stage": 2000, "boundary_icp_iterations": 160})
    log("Label-independent landmark affine alignment propagated to all spots")


def match_arrays(a, b, xy_a, xy_b, forward, reverse, distances, cutoff, radius_factor=1., source_informative=None, target_informative=None):
    rows = np.repeat(np.arange(len(a)), forward.shape[1])
    cols = forward.ravel()
    mutual = np.any(reverse[cols] == rows[:, None], axis=1)
    radius = float(np.median(cKDTree(xy_b).query(xy_b, k=2)[0][:, 1]))
    spatial_distance = np.linalg.norm(xy_a[rows] - xy_b[cols], axis=1) / radius
    cosine_distance = distances.ravel()
    keep = mutual & (spatial_distance <= CONFIG["matching_spatial_radius_target_NN"] * radius_factor) & (cosine_distance <= cutoff)
    if source_informative is not None:
        keep &= source_informative[rows]
    if target_informative is not None:
        keep &= target_informative[cols]
    rows, cols = rows[keep], cols[keep]
    d, sd = cosine_distance[keep], spatial_distance[keep]
    tau = max(float(np.median(d[d > 0])) if np.any(d > 0) else .01, .01)
    affinity = np.exp(-d / tau - sd / 5.)
    total = np.bincount(rows, weights=affinity, minlength=len(a))
    mass = np.zeros(len(a))
    np.maximum.at(mass, rows, affinity)
    weights = affinity / np.maximum(total[rows], 1e-10) * mass[rows]
    return rows, cols, weights, d, sd, mass


def cosine_rows(a, b):
    return np.sum(unit(a) * unit(b), axis=1)


def analyze():
    data = ad.read_h5ad(OUT / "aligned_label_free.h5ad")
    assert "annotation" not in data.obs and "Domain" not in data.obs
    annotations = pd.read_csv(OUT / "inputs/annotations.csv.gz", index_col=0).reindex(data.obs_names).annotation.fillna("Unannotated").to_numpy()
    marker = np.load(OUT / "inputs/excluded_marker_expression.npz")
    marker_expression = marker["expression"]
    stages = data.obs.stage.astype(str).to_numpy()
    summary, flows, stability, recall_rows, null_rows, sensitivity, centering = [], [], [], [], [], [], []
    for left, right in zip(STAGES[:-1], STAGES[1:]):
        pair = f"{left}_to_{right}"
        ia, ib = np.flatnonzero(stages == left), np.flatnonzero(stages == right)
        a, b = data.obsm["HyperMOA_bc"][ia], data.obsm["HyperMOA_bc"][ib]
        da, forward = knn(a, b)
        _, reverse = knn(b, a)
        sample = np.random.default_rng(42).choice(len(a), min(256, len(a)), replace=False)
        _, exact = knn(a[sample], b, approximate=False)
        recall = np.mean([len(set(x).intersection(y)) / len(y) for x, y in zip(forward[sample], exact)])
        recall_rows.append({"pair": pair, "exact_query_count": len(sample), "top10_recall": recall})
        _, centered_nn = knn(a[sample], b, k=1, approximate=False)
        _, raw_nn = knn(data.obsm["HyperMOA"][ia[sample]], data.obsm["HyperMOA"][ib], k=1, approximate=False)
        centered_scores = cosine_rows(marker_expression[ia[sample]], marker_expression[ib[centered_nn[:, 0]]])
        raw_scores = cosine_rows(marker_expression[ia[sample]], marker_expression[ib[raw_nn[:, 0]]])
        centering.append({"pair": pair, "queries": len(sample), "raw_embedding_marker_cosine": float(raw_scores.mean()),
                          "centered_embedding_marker_cosine": float(centered_scores.mean()),
                          "nearest_target_changed_fraction": float(np.mean(raw_nn != centered_nn)),
                          "scope": "exact sampled expression-neighbor probe; no spatial gate; no label-based model selection"})
        rows = np.repeat(np.arange(len(a)), 10)
        cols = forward.ravel()
        mutual = np.any(reverse[cols] == rows[:, None], axis=1)
        observed = da.ravel()[mutual]
        cutoff = min(float(np.quantile(observed, .95)), CONFIG["matching_max_cosine_distance"]) if len(observed) else CONFIG["matching_max_cosine_distance"]
        xy_a, xy_b = data.obsm["aligned_spatial"][ia], data.obsm["aligned_spatial"][ib]
        r, c, w, d, sd, mass = match_arrays(a, b, xy_a, xy_b, forward, reverse, da, cutoff, source_informative=data.obs.model_informative.to_numpy()[ia], target_informative=data.obs.model_informative.to_numpy()[ib])
        edges = pd.DataFrame({"source_spot_id": data.obs_names[ia[r]], "target_spot_id": data.obs_names[ib[c]],
                              "source_local_index": r, "target_local_index": c,
                              "source_stage": left, "target_stage": right,
                              "cosine_distance": d, "spatial_distance_target_NN": sd,
                              "affinity_weight": w})
        edges.to_csv(OUT / f"pairs/{pair}.csv.gz", index=False)
        np.savez_compressed(OUT / f"pairs/{pair}_indices.npz", source=r, target=c, weights=w,
                            forward=forward, reverse=reverse, cosine_distances=da)
        unmatched = pd.DataFrame({"source_spot_id": data.obs_names[ia], "unmatched_weight": 1 - mass,
                                  "has_retained_correspondence": mass > 0})
        unmatched.to_csv(OUT / f"pairs/{pair}_unmatched.csv.gz", index=False)
        labeled = pd.DataFrame({"source_tissue": annotations[ia[r]], "target_tissue": annotations[ib[c]], "weight": w})
        grouped = labeled.groupby(["source_tissue", "target_tissue"]).weight.sum().reset_index()
        unresolved = pd.DataFrame({"source_tissue": annotations[ia], "weight": 1 - mass}).groupby("source_tissue").weight.sum().reset_index()
        unresolved["target_tissue"] = "Unmatched"
        grouped = pd.concat([grouped, unresolved], ignore_index=True)
        denominators = pd.Series(annotations[ia]).value_counts()
        grouped["source_spots"] = grouped.source_tissue.map(denominators)
        grouped["source_fraction"] = grouped.weight / grouped.source_spots
        grouped["pair"], grouped["source_stage"], grouped["target_stage"] = pair, left, right
        flows.append(grouped)
        n_targets = len(np.unique(c))
        record = {"pair": pair, "source_spots": len(a), "target_spots": len(b), "edges": len(r),
                  "source_retained_fraction": float(np.mean(mass > 0)),
                  "unmatched_weight_fraction": float(np.mean(1 - mass)),
                  "target_retained_fraction": n_targets / len(b), "cosine_cutoff": cutoff,
                  "cosine_distance_median": float(np.median(d)) if len(d) else np.nan,
                  "spatial_distance_median_NN": float(np.median(sd)) if len(sd) else np.nan}
        summary.append(record)
        valid = (np.linalg.norm(marker_expression[ia[r]], axis=1) > 0) & (np.linalg.norm(marker_expression[ib[c]], axis=1) > 0)
        if valid.any():
            mr, mc, mw = r[valid], c[valid], w[valid]
            real = float(np.average(cosine_rows(marker_expression[ia[mr]], marker_expression[ib[mc]]), weights=mw))
            rng = np.random.default_rng(2022)
            null = []
            for _ in range(20):
                positive_targets = np.flatnonzero(np.linalg.norm(marker_expression[ib], axis=1) > 0)
                shuffled = np.arange(len(ib))
                shuffled[positive_targets] = rng.permutation(positive_targets)
                null.append(float(np.average(cosine_rows(marker_expression[ia[mr]], marker_expression[ib[shuffled[mc]]]), weights=mw)))
            null_rows.append({"pair": pair, "valid_edges": int(valid.sum()), "excluded_marker_cosine": real,
                              "permuted_target_mean": float(np.mean(null)), "permuted_target_sd": float(np.std(null)),
                              "observed_minus_null": real - float(np.mean(null)),
                              "scope": "descriptive expression concordance; not biological-replicate significance"})
        primary = set(zip(r.tolist(), c.tolist()))
        for seed in CONFIG["stability_seeds"]:
            alternate = np.load(OUT / f"diagnostics/aligned_seed{seed}.npz")["coordinates"]
            rr, cc, _, _, _, mm = match_arrays(a, b, alternate[ia], alternate[ib], forward, reverse, da, cutoff)
            other = set(zip(rr.tolist(), cc.tolist()))
            stability.append({"pair": pair, "seed": seed, "edge_jaccard_to_seed42": len(primary & other) / max(len(primary | other), 1),
                              "source_retained_fraction": float(np.mean(mm > 0)), "scope": "refinement seed only; embeddings frozen"})
        for factor in (.8, 1., 1.2):
            rr, cc, _, _, _, mm = match_arrays(a, b, xy_a, xy_b, forward, reverse, da, cutoff, factor, source_informative=data.obs.model_informative.to_numpy()[ia], target_informative=data.obs.model_informative.to_numpy()[ib])
            sensitivity.append({"pair": pair, "spatial_radius_multiplier": factor, "edges": len(rr),
                                "source_retained_fraction": float(np.mean(mm > 0)), "unmatched_weight_fraction": float(np.mean(1 - mm))})
        log(f"{pair}: {len(r)} edges, retained sources {record['source_retained_fraction']:.3f}, unmatched weight {record['unmatched_weight_fraction']:.3f}")
    for name, rows in (("matching_quality", summary), ("tissue_correspondence", flows),
                       ("refinement_seed_stability", stability), ("neighbor_search_recall", recall_rows),
                       ("excluded_marker_validation", null_rows), ("matching_radius_sensitivity", sensitivity), ("stage_centering_sensitivity", centering)):
        frame = pd.concat(rows, ignore_index=True) if name == "tissue_correspondence" else pd.DataFrame(rows)
        frame.to_csv(OUT / f"tables/{name}.csv", index=False)
    pooled = pd.read_csv(OUT / "tables/organ_pseudobulk_expression.csv.gz")
    genes = [(organ, gene) for organ, values in MARKERS.items() for gene in values]
    trend = pooled.merge(pd.DataFrame(genes, columns=["tissue", "gene"]), on=["tissue", "gene"])
    trend["log1p_pseudobulk_cp10k"] = np.log1p(trend.pseudobulk_cp10k)
    trend.to_csv(OUT / "tables/organ_marker_trends.csv", index=False)
    # Labels are joined only after all model and alignment operations finish.
    data.obs["annotation"] = pd.Categorical(annotations)
    data.write_h5ad(OUT / "mouse_embryo_processed.h5ad", compression="gzip")
    quality = pd.DataFrame(summary)
    null = pd.DataFrame(null_rows)
    checks = {"annotations_absent_from_model_input": True,
              "excluded_markers_absent_from_training": not set(pd.read_csv(OUT / "inputs/training_genes.csv").gene).intersection(CONFIG["excluded_marker_genes"]),
              "all_pairs_have_edges": bool((quality.edges > 0).all()),
              "all_pairs_marker_concordance_above_shuffle": bool(len(null) == 7 and (null.observed_minus_null > 0).all()),
              "note": "A failing concordance check is reported, not used to select alignment parameters."}
    dump(OUT / "diagnostics/quality_assessment.json", checks)
    log("Descriptive downstream tables and full-spot result saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "embed", "align", "analyze", "all"])
    args = parser.parse_args()
    for name, function in (("prepare", prepare), ("embed", embed), ("align", align), ("analyze", analyze)):
        if args.step in (name, "all"):
            dump(OUT / "status.json", {"step": name, "state": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            function()
            dump(OUT / "status.json", {"step": name, "state": "complete"})


if __name__ == "__main__":
    main()
