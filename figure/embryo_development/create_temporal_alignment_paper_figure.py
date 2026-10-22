#!/usr/bin/env python3
"""One-page, alignment-focused figure for the frozen mouse embryo result."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manuscript_panel_labels import add_panel_labels_to_figure
from manuscript_typography import apply_manuscript_typography

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "HyperMOA/mouse_embryo_result/mouse_embryo_processed.h5ad"
AUDIT = ROOT / "HyperMOA/mouse_embryo_result/tables/matching_audit_funnel.csv"
DEST = ROOT / "figure/embryo_development/temporal_alignment_paper"
STEM = "mouse_embryo_temporal_alignment_18x22cm"
STAGES = tuple(f"E{day}.5" for day in range(9, 17))
ORGANS = ("Brain", "Heart", "Liver")
COLORS = {"Brain": "#168B8A", "Heart": "#D94354", "Liver": "#8C58A8", "Other": "#CCD3D6"}
PAIR_COLORS = ("#D85A50", "#218F9D")
PAIRS = (("E9.5", "E10.5"), ("E14.5", "E15.5"))
DISPLAY_FRACTION = .20
DISPLAY_SEED = 42




def set_gid(artist, name):
    artist.set_gid(name)
    return artist


def bounds(xy):
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    pad = .035 * max(hi - lo)
    return (lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad)


def draw_native(fig, rect, xy, annotation, stage):
    ax = fig.add_axes(rect)
    ax.set_gid(f"native_{stage.replace('.', '_')}")
    group = np.where(np.isin(annotation, ORGANS), annotation, "Other")
    for key in ("Other", *ORGANS):
        chosen = group == key
        ax.scatter(xy[chosen, 0], xy[chosen, 1], s=.40, c=COLORS[key],
                   linewidths=0, rasterized=True)
    ax.set_xlim(*bounds(xy)[0])
    ax.set_ylim(*bounds(xy)[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(f"{stage}  n={len(xy):,}", fontsize=6.8, pad=1.2)


def draw_time_stack(fig, rect, aligned, labels, stage_index, sampled, focus):
    ax = fig.add_axes(rect, projection="3d")
    ax.set_gid("organ_time_stack" if focus else "whole_embryo_time_stack")
    xy = aligned
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = max(hi - lo)
    for k, stage in enumerate(STAGES):
        stage_mask = sampled & (stage_index == k)
        if not focus:
            other = stage_mask & ~np.isin(labels, ORGANS)
            ax.scatter(xy[other, 0], xy[other, 1], np.full(other.sum(), k),
                       s=.42, c=COLORS["Other"], alpha=.40,
                       depthshade=False, rasterized=True)
        for organ in ORGANS:
            selected = stage_mask & (labels == organ)
            ax.scatter(xy[selected, 0], xy[selected, 1], np.full(selected.sum(), k),
                       s=.82 if focus else .69, c=COLORS[organ], alpha=.95,
                       depthshade=False, rasterized=True)
        ax.text(lo[0] - .085 * span, lo[1] - .045 * span, k, stage,
                ha="right", va="center", fontsize=6.1, color="#343C42")
    ax.set_xlim(lo[0] - .10 * span, hi[0] + .06 * span)
    ax.set_ylim(lo[1] - .05 * span, hi[1] + .04 * span)
    ax.set_zlim(-.35, 7.35)
    ax.set_box_aspect((1.0, 1.45, 1.45), zoom=1.13)
    ax.view_init(elev=20, azim=-65)
    ax.set_axis_off()
    return ax


def viewport(ax, xy, span):
    center = (xy.min(axis=0) + xy.max(axis=0)) / 2
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def draw_pair(fig, raw, aligned, index, sampled, left, right, y, pair_number):
    a = (index == STAGES.index(left)) & sampled
    b = (index == STAGES.index(right)) & sampled
    if not a.any() or not b.any():
        raise ValueError(f"Empty display sample for {left}->{right}")
    raw_a, raw_b = raw[a], raw[b]
    aligned_a, aligned_b = aligned[a], aligned[b]
    registered = np.vstack((aligned_a, aligned_b))
    rects = ((.065, y, .260, .125), (.370, y, .260, .125), (.675, y, .260, .125))
    spans = []
    for part, (rect, points, title, color) in enumerate(zip(
            rects, (raw_a, raw_b, registered),
            (f"{left}  original", f"{right}  original", "HyperMOA aligned"),
            (PAIR_COLORS[0], PAIR_COLORS[1], None))):
        ax = fig.add_axes(rect)
        ax.set_gid(f"pair_{pair_number}_{part}")
        if part == 2:
            colors = np.concatenate((np.repeat(PAIR_COLORS[0], len(aligned_a)),
                                     np.repeat(PAIR_COLORS[1], len(aligned_b))))
            order = np.random.default_rng(DISPLAY_SEED + pair_number).permutation(len(registered))
            ax.scatter(registered[order, 0], registered[order, 1], s=.62,
                       c=colors[order], alpha=.66, linewidths=0, rasterized=True)
        else:
            ax.scatter(points[:, 0], points[:, 1], s=.68, c=color,
                       alpha=.78, linewidths=0, rasterized=True)
        span = float(np.ptp(points, axis=0).max()) * 1.08
        spans.append(span)
        viewport(ax, points, span)
        if part != 2:
            ax.set_title(title, fontsize=6.5, pad=1.2)
    if pair_number == 0:
        set_gid(fig.text(.805, y + .133, "Aligned", ha="center", va="center",
                         fontsize=6.5), "pair_0_aligned_heading")
    set_gid(fig.text(.025, y + .063, f"{left}\nto\n{right}", va="center",
                     ha="center", fontsize=6.0), f"pair_{pair_number}_stage_label")
    return {"pair": f"{left}_to_{right}", "display_spots_source": int(a.sum()),
            "display_spots_target": int(b.sum()), "independent_panel_spans": spans,
            "native_panels_individually_framed": True,
            "aligned_overlay_uses_unscaled_saved_coordinates": True}


def load_coverage(stage_rows):
    audit = pd.read_csv(AUDIT)
    final = audit.loc[audit.step.eq("Spatial (final)"),
                      ["pair", "source_spots", "source_fraction"]].copy()
    expected = [f"{a}_to_{b}" for a, b in zip(STAGES[:-1], STAGES[1:])]
    if final.pair.tolist() != expected:
        raise ValueError("Formal matching-audit pairs differ from stage order")
    for row, stage in zip(final.itertuples(index=False), stage_rows):
        if not np.isclose(row.source_fraction, row.source_spots / stage["n_spots"]):
            raise ValueError(f"Source denominator mismatch: {row.pair}")
    return final




def group_svg(path):
    namespace = "{http://www.w3.org/2000/svg}"
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("width", "18cm")
    root.set("height", "22cm")
    container = root.find(f".//{namespace}g[@id='figure_1']")
    if container is None:
        raise ValueError("Matplotlib SVG figure group missing")
    names = {
        "original_sections": lambda name: name.startswith("native_") or name == "original_sections_heading",
        "whole_embryo_alignment": lambda name: name == "whole_embryo_time_stack" or name == "whole_stack_heading",
        "organ_alignment": lambda name: name == "organ_time_stack" or name == "organ_stack_heading",
        "early_pair_alignment": lambda name: name.startswith("pair_0_"),
        "late_pair_alignment": lambda name: name.startswith("pair_1_"),
        "legends_and_notes": lambda name: name.startswith("legend_") or name.startswith("note_") or name == "pair_section_heading",
    }
    for group_name, belongs in names.items():
        parent = ET.Element(namespace + "g", {"id": group_name})
        parents = {child: node for node in container.iter() for child in node}
        for child in list(container.iter()):
            if belongs(child.attrib.get("id", "")):
                parents[child].remove(child)
                parent.append(child)
        if not len(parent):
            raise ValueError(f"SVG group empty: {group_name}")
        container.append(parent)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"],
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.linewidth": .5})
    data = ad.read_h5ad(SOURCE, backed="r")
    try:
        names = data.obs.stage.astype(str).to_numpy()
        order = data.obs.slice_order.to_numpy(dtype=int)
        annotation = data.obs.annotation.astype(str).to_numpy()
        raw = np.asarray(data.obsm["spatial"], dtype=np.float64)
        aligned = np.asarray(data.obsm["aligned_spatial"], dtype=np.float64)
    finally:
        data.file.close()
    if raw.shape != aligned.shape or raw.shape[1] != 2 or not np.isfinite(aligned).all():
        raise ValueError("Invalid original or model-aligned coordinates")
    if not np.array_equal(names, np.asarray(STAGES)[order]):
        raise ValueError("Stage order differs from the frozen experiment")
    if len(names) != 520815:
        raise ValueError("Unexpected frozen experiment spot count")
    rng = np.random.default_rng(DISPLAY_SEED)
    sampled = rng.random(len(names)) < DISPLAY_FRACTION
    fig = plt.figure(figsize=(18 / 2.54, 22 / 2.54), facecolor="white")
    set_gid(fig.text(.045, .981, "Original annotated sections", fontsize=8.0,
                     weight="bold", va="center"), "original_sections_heading")
    rows = []
    for i, stage in enumerate(STAGES):
        chosen = order == i
        x = .040 + (i % 4) * .240
        y = .858 if i < 4 else .738
        draw_native(fig, [x, y, .214, .100], raw[chosen], annotation[chosen], stage)
        rows.append({"stage": stage, "n_spots": int(chosen.sum()),
                     "n_display_spots": int((chosen & sampled).sum()),
                     "raw_xmin": float(raw[chosen, 0].min()),
                     "raw_xmax": float(raw[chosen, 0].max()),
                     "raw_ymin": float(raw[chosen, 1].min()),
                     "raw_ymax": float(raw[chosen, 1].max()),
                     "aligned_xmin": float(aligned[chosen, 0].min()),
                     "aligned_xmax": float(aligned[chosen, 0].max()),
                     "aligned_ymin": float(aligned[chosen, 1].min()),
                     "aligned_ymax": float(aligned[chosen, 1].max())})
    set_gid(fig.text(.045, .691, "Eight stages in one aligned coordinate frame", fontsize=8.0,
                     weight="bold"), "whole_stack_heading")
    set_gid(fig.text(.535, .691, "Organ locations after alignment", fontsize=8.0,
                     weight="bold"), "organ_stack_heading")
    draw_time_stack(fig, [.032, .365, .455, .312], aligned, annotation, order, sampled, False)
    draw_time_stack(fig, [.515, .365, .455, .312], aligned, annotation, order, sampled, True)
    legend = fig.add_axes([.18, .710, .64, .024])
    legend.set_gid("legend_organs")
    legend.set_axis_off()
    legend.legend([Line2D([], [], marker="o", linestyle="none", color=COLORS[name], markersize=4)
                   for name in (*ORGANS, "Other")],
                  [*ORGANS, "Other tissues"], ncol=4, loc="center",
                  frameon=False, fontsize=6.5, columnspacing=1.6, handletextpad=.35)
    load_coverage(rows)
    set_gid(fig.text(.045, .360, "Adjacent sections: native views and model-aligned overlays",
                     fontsize=8.0, weight="bold"), "pair_section_heading")
    for number, (pair, y) in enumerate(zip(PAIRS, (.198, .045))):
        draw_pair(fig, raw, aligned, order, sampled, *pair, y, number)
    add_panel_labels_to_figure(fig, "embryo")
    apply_manuscript_typography(fig, "embryo")
    base = DEST / STEM
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    svg = DEST / f"{STEM}_grouped.svg"
    fig.savefig(svg, facecolor="white")
    fig.savefig(DEST / f"{STEM}_preview.png", dpi=160, facecolor="white")
    plt.close(fig)
    group_svg(svg)
    print(f"Saved full-page temporal alignment figure in {DEST}", flush=True)


if __name__ == "__main__":
    main()
