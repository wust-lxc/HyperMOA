"""All-section and unwarped pair-coordinate views from saved experiment outputs.

Python workflow. No subsampling, model training, or changes to benchmark files.
"""
from pathlib import Path
import sys, os, json, hashlib
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("HYPERMOA_FIGURE_OUTPUT", str(ROOT / "figure/full_manuscript_review")))
OUT.mkdir(parents=True, exist_ok=True)
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 6.2, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
METHODS = ["HyperMOA", "SpatialGlue", "MISO", "CellCharter", "Harmony", "3d-OT", "SpaMosaic"]
ALIGN = ["HyperMOA", "3d-OT", "SLAT", "PASTE", "SANTO"]
PALETTE = [to_hex(c) for c in plt.get_cmap("tab20").colors]
PAPER = pd.read_csv(ROOT / "data/evaluation/fair_benchmark/clustering_paper_metrics_slice.csv")
PAIRS = pd.read_csv(ROOT / "data/evaluation/fair_benchmark/alignment_paper_pair_metrics.csv")
CONFIG = {
    "Human_lymph_node": ("Human lymph node", ["slice1", "slice2", "slice3"], 7.3,
                       ROOT / "HyperMOA/Human_lymph_node_result/adata.h5ad"),
    "Simulation": ("Simulation", [f"Simulation{i}" for i in range(1, 6)], 9.0,
                   ROOT / "HyperMOA/Simulation_result/adata.h5ad"),
    "Mouse_Brain": ("Mouse brain", ["Mouse_Brain_ATAC", "Mouse_Brain_H3K27ac"], 5.5,
                    ROOT / "HyperMOA/Mouse_brain_result/H3K27ac_ATAC/adata.h5ad"),
}

def strings(x):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in x])

def obs_col(f, key):
    obj = f["obs"][key]
    if isinstance(obj, h5py.Dataset):
        return strings(obj[:])
    categories = strings(obj["categories"][:])
    codes = obj["codes"][:]
    return np.asarray([categories[c] if c >= 0 else "" for c in codes])

def obs_ids(f):
    key = f["obs"].attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    return strings(f["obs"][key][:])

def raw_path(dataset, section):
    if dataset == "Human_lymph_node":
        return ROOT / f"data/datasets/{dataset}/{section}/s{section[-1]}_adata_rna.h5ad"
    if dataset == "Simulation":
        return ROOT / f"data/datasets/Simulation/slice{section[-1]}/adata_RNA.h5ad"
    return ROOT / f"data/datasets/Mouse_Brain/{section.removeprefix('Mouse_Brain_')}/adata_RNA.h5ad"

def cluster_dir(dataset, section):
    name = f"slice{section[-1]}" if dataset == "Simulation" else section.removeprefix("Mouse_Brain_")
    return ROOT / f"data/results/clustering/{dataset}/{name}"

def clean_truth(s, dataset):
    s = s.astype("string").str.strip()
    if dataset == "Human_lymph_node":
        s = s.str.lower().str.replace("vessels", "vessel", regex=False)
    return s.mask(s.str.lower().isin(["", "na", "nan", "none", "unknown", "null"]))

def matched_colors(truth, pred, palette):
    if pred is None:
        return truth.map(palette).fillna("#c7cccf").to_numpy()
    pred = pred.astype("string")
    tab = pd.crosstab(truth, pred)
    rr, cc = linear_sum_assignment(-tab.to_numpy())
    mapping = {tab.columns[c]: palette[tab.index[r]] for r, c in zip(rr, cc)}
    available = [c for c in PALETTE if c not in mapping.values()]
    for k in sorted(set(pred.dropna()) - set(mapping)):
        mapping[k] = available.pop(0) if available else "#555555"
    return pred.map(mapping).fillna("#c7cccf").to_numpy()

def square_frame(ax, xy, reverse_y=False):
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    mid = (lo + hi) / 2
    half = max(float((hi - lo).max()), 1e-8) * .53
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    if reverse_y:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

def chamfer(p, q):
    spacing = np.median(cKDTree(q).query(q, k=2)[0][:, 1])
    return (cKDTree(q).query(p)[0].mean() + cKDTree(p).query(q)[0].mean()) / (2 * spacing)

records, geometry = [], []
for dataset, (title, sections, height, hp) in CONFIG.items():
    fig = plt.figure(figsize=(180 / 25.4, height), facecolor="white")
    fig.text(.025, .985, title, fontsize=10, fontweight="bold", va="top")
    fig.text(.025, .946, "a  All sections: reference labels and inferred domains", fontsize=8,
             fontweight="bold", va="top")
    # Explicit square axes ensure that the same-sized panels do not change aspect.
    left, right, width = .07, .985, (.095 if dataset == "Simulation" else .104)
    gap = (right - left - 8 * width) / 7
    axes_h = width * (180 / 25.4) / height
    top = .885
    row_gap = (12 if dataset == "Simulation" else 15) / 72 / height
    group_rows, group_cols = [], [[] for _ in range(8)]
    plot_axes, plot_ids = [], []
    label_records = []
    for r, section in enumerate(sections):
        directory = cluster_dir(dataset, section)
        df = pd.read_csv(directory / "clustering_labels.csv", index_col=0, dtype="string")
        with h5py.File(directory / "clustering_results.h5ad", "r") as f:
            assert np.array_equal(obs_ids(f), df.index.astype(str))
            xy = f["obsm/spatial"][:]
        truth = clean_truth(df["truth"], dataset)
        colors_for_truth = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"] if dataset == "Simulation" else PALETTE
        palette = {v: colors_for_truth[k] for k, v in enumerate(sorted(truth.dropna().unique()))}
        bottom = top - axes_h - r * (axes_h + row_gap)
        short = section.replace("Simulation", "S").replace("slice", "S").replace("Mouse_Brain_", "")
        fig.text(.025, bottom + axes_h / 2, short, fontsize=6, ha="center", va="center", rotation=90, rotation_mode="anchor")
        row_ids = []
        for c, method in enumerate(["Reference"] + METHODS):
            ax = fig.add_axes([left + c * (width + gap), bottom, width, axes_h])
            pid = f"cluster_r{r}_c{c}"
            plot_axes.append(ax); plot_ids.append(pid); row_ids.append(pid); group_cols[c].append(pid)
            pred = None if c == 0 else df["cluster_" + method.replace("3d-OT", "3dOT")]
            color = matched_colors(truth, pred, palette)
            ax.scatter(xy[:, 0], xy[:, 1], c=color, s=.45 if len(xy) > 6000 else .8,
                       linewidths=0, rasterized=True)
            square_frame(ax, xy, reverse_y=True)
            if r == 0:
                fig.text(left + c * (width + gap) + width / 2, top + .012, method,
                         ha="center", va="bottom", fontsize=6)
            if c:
                rec = PAPER[(PAPER.dataset == dataset) & (PAPER["slice"] == section) &
                            (PAPER.method == method)].iloc[0]
                text = f"{rec.ARI:.3f} / {rec.AMI:.3f}"
            else:
                text = f"n = {len(df):,}"
            fig.text(left + c * (width + gap) + width / 2, bottom - .009,
                     text, ha="center", va="top", fontsize=5.5)
        group_rows.append(row_ids)
        label_records.append({"section": section, "n_displayed": len(df),
                              "n_unannotated": int(truth.isna().sum()), "palette": palette})
    b_header = bottom - (.05 if dataset == "Simulation" else .064)
    fig.text(.025, b_header, "b  Registered pairs in a common planar frame", fontsize=8,
             fontweight="bold", va="top")
    fig.text(.025, b_header - .026, "Source: coral     Target: teal     Overlap: dark     Numbers: normalized Chamfer",
             fontsize=6, va="top")
    btop = b_header - (.065 if dataset == "Simulation" else .080)
    bwidth, bleft = .155, .115
    bgap = (.985 - bleft - 5 * bwidth) / 4
    bh = bwidth * (180 / 25.4) / height
    nrows = len(sections) - 1
    # Fit the full pair series on the page while retaining a square coordinate viewport.
    available = btop - .045 - (nrows - 1) * row_gap
    bh = min(bh, available / nrows)
    actual_w = bh * height / (180 / 25.4)
    bcols = [[] for _ in range(5)]
    raw = {}
    for section in sections:
        with h5py.File(raw_path(dataset, section), "r") as f:
            raw[section] = (obs_ids(f), f["obsm/spatial"][:])
    hyper = {}
    with h5py.File(hp, "r") as f:
        batches = obs_col(f, "batch")
        ids = obs_col(f, "original_barcode") if "original_barcode" in f["obs"] else obs_ids(f)
        coords = f["obsm/transform_fine"][:]
        for section in sections:
            batch = section.replace("Simulation", "slice").replace("Mouse_Brain_", "")
            mask = batches == batch
            if not mask.any():
                mask = batches == section
            lookup = pd.Index(ids[mask]).get_indexer(raw[section][0])
            assert (lookup >= 0).all(), (dataset, section, set(batches))
            hyper[section] = coords[mask][lookup]
    for r, (src, tgt) in enumerate(zip(sections[:-1], sections[1:])):
        pair = f"{src}_to_{tgt}"
        bot = btop - bh - r * (bh + row_gap)
        short = f"{r + 1} to {r + 2}" if dataset != "Mouse_Brain" else "ATAC to H3K27ac"
        fig.text(.025, bot + bh / 2, short, fontsize=6, ha="center", va="center", rotation=90, rotation_mode="anchor")
        row_ids = []
        for c, method in enumerate(ALIGN):
            if method == "HyperMOA":
                p, q = hyper[src], hyper[tgt]
                path = hp
            else:
                path = ROOT / f"data/results/alignment/{dataset}/{method}/pair_checkpoints/{pair}.npz"
                with np.load(path) as saved:
                    p = np.asarray(saved["aligned_source"])
                q = raw[tgt][1]
            assert p.shape == raw[src][1].shape and q.shape == raw[tgt][1].shape
            score = chamfer(p, q)
            expected = PAIRS[(PAIRS.dataset == dataset) & (PAIRS.method == method) &
                             (PAIRS.pair == pair)].iloc[0]["Chamfer Distance"]
            assert abs(score - expected) < 1e-7, (dataset, method, pair, score, expected)
            x = bleft + c * (bwidth + bgap) + (bwidth - actual_w) / 2
            ax = fig.add_axes([x, bot, actual_w, bh])
            pid = f"pair_r{r}_c{c}"
            plot_axes.append(ax); plot_ids.append(pid); row_ids.append(pid); bcols[c].append(pid)
            ax.scatter(q[:, 0], q[:, 1], s=.8, color="#008b8b", alpha=.6,
                       linewidths=0, rasterized=True)
            ax.scatter(p[:, 0], p[:, 1], s=.55, color="#e66a50", alpha=.55,
                       linewidths=0, rasterized=True)
            square_frame(ax, np.vstack([p, q]))
            if r == 0:
                fig.text(x + actual_w / 2, btop + .009, method, ha="center", fontsize=6)
            fig.text(x + actual_w / 2, bot - .009, f"{score:.3f}",
                     ha="center", va="top", fontsize=5.5)
            geometry.append({"dataset": dataset, "method": method, "pair": pair,
                             "n_source": len(p), "n_target": len(q), "Chamfer": score,
                             "coordinate_source": str(path.relative_to(ROOT)),
                             "display_transform": "None; common square axes and equal aspect for both sections"})
        group_rows.append(row_ids)
    prefix = OUT / f"{dataset}_all_sections_and_registered_pairs"
    fig.canvas.draw()
    # The author-side layout audit is independent of figure generation.
    fig.savefig(str(prefix) + ".pdf", dpi=600)
    fig.savefig(str(prefix) + ".svg", dpi=600)
    fig.savefig(str(prefix) + ".png", dpi=300)
    plt.close(fig)
    records.append({"dataset": dataset, "file": prefix.name, "cluster_rows": label_records,
                    "all_spots_retained": True, "figure_width_mm": 180,
                    "figure_height_mm": height * 25.4})
    print(dataset, "complete", flush=True)
pd.DataFrame(geometry).to_csv(OUT / "pair_coordinate_verification.csv", index=False)
(OUT / "source_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
