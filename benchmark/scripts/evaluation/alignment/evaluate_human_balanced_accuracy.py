#!/usr/bin/env python3
"""Supplemental Human alignment balanced accuracy; leave official scores intact."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))


import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from benchmark.scripts.common import fair_benchmark_common as common
from benchmark.scripts.evaluation.alignment.evaluate_alignment_paper import normalized_labels
from benchmark.scripts.evaluation.alignment.evaluate_fair_alignment import hypermoa_coordinates
from benchmark.scripts.runners.alignment.run_final_alignment_refinement import sha256

OUTPUT = common.ROOT / "data/evaluation/fair_benchmark/human_balanced_accuracy"


def label_scores(source_labels, matched_labels):
    source = normalized_labels(source_labels, "Human_lymph_node")
    target = normalized_labels(matched_labels, "Human_lymph_node")
    valid = common.valid_labels(source) & common.valid_labels(target)
    source, target = source[valid], target[valid]
    classes = sorted(set(source))
    row = {"n_valid_label_pairs": int(valid.sum()), "n_source_classes": len(classes)}
    if not classes:
        row.update({"Balanced Accuracy": np.nan, "CI": np.nan, "Macro-F1": np.nan})
        return row, []
    values = recall_score(source, target, labels=classes, average=None, zero_division=0)
    score = balanced_accuracy_score(source, target, adjusted=False)
    np.testing.assert_allclose(score, values.mean(), atol=1e-12)
    row.update({"Balanced Accuracy": float(score), "CI": float(np.mean(source == target)),
                "Macro-F1": float(f1_score(source, target, labels=sorted(set(source) | set(target)),
                                            average="macro", zero_division=0))})
    regions = [{"label": label, "n_source_spots": int((source == label).sum()),
                "n_correct": int(((source == label) & (target == label)).sum()),
                "recall": float(value)} for label, value in zip(classes, values)]
    return row, regions


def main():
    spec = common.SPECS["Human_lymph_node"]
    slices, inputs = {}, set(spec.raw_paths)
    official_path = common.ROOT / "data/evaluation/fair_benchmark/alignment_paper_pair_metrics.csv"
    inputs.update((official_path, common.HYPERMOA_SOURCE_FILES[spec.name]))
    for method in common.METHODS:
        if method != "HyperMOA":
            for source, target in zip(spec.slices[:-1], spec.slices[1:]):
                inputs.add(spec.result_root / method / "pair_checkpoints" / f"{source}_to_{target}.npz")
    hashes = {str(path): sha256(path) for path in sorted(inputs)}
    for name, path in zip(spec.slices, spec.raw_paths):
        data = ad.read_h5ad(path, backed="r")
        try:
            slices[name] = SimpleNamespace(obs_names=data.obs_names.copy(), labels=common.labels(data),
                                           xy=np.asarray(data.obsm["spatial"], dtype=float))
        finally:
            data.file.close()
    hyper = hypermoa_coordinates(spec, slices)
    official = pd.read_csv(official_path)
    rows, region_rows = [], []
    for method in common.METHODS:
        for source_name, target_name in zip(spec.slices[:-1], spec.slices[1:]):
            source, target = slices[source_name], slices[target_name]
            if method == "HyperMOA":
                source_xy, target_xy = hyper[source_name], hyper[target_name]
                coordinate_path = common.HYPERMOA_SOURCE_FILES[spec.name]
            else:
                coordinate_path = spec.result_root / method / "pair_checkpoints" / f"{source_name}_to_{target_name}.npz"
                with np.load(coordinate_path) as saved:
                    source_xy = np.asarray(saved["aligned_source"], dtype=float)
                target_xy = target.xy
            if source_xy.shape != (len(source.obs_names), 2):
                raise ValueError(f"Source coordinate shape mismatch: {coordinate_path}")
            predicted = common.nearest_match(source_xy, target_xy)
            scores, recalls = label_scores(source.labels, target.labels[predicted])
            pair = f"{source_name}_to_{target_name}"
            reference = official.loc[(official.dataset == spec.name) & (official.method == method) & (official.pair == pair)]
            if len(reference) != 1:
                raise ValueError(f"Missing/duplicate official pair: {method}/{pair}")
            for metric in ("CI", "Macro-F1", "Balanced Accuracy"):
                np.testing.assert_allclose(scores[metric], reference.iloc[0][metric], atol=1e-8, rtol=0)
            metadata = {"dataset": spec.name, "method": method, "pair": pair}
            rows.append({**metadata, **scores, "n_source": len(source.obs_names),
                         "n_target": len(target.obs_names), "coordinate_input": str(coordinate_path)})
            region_rows.extend({**metadata, **record} for record in recalls)
    for path in inputs:
        if sha256(path) != hashes[str(path)]:
            raise RuntimeError(f"Input changed during evaluation: {path}")
    pairs = pd.DataFrame(rows)
    summary = pairs.groupby(["dataset", "method"], sort=False)[["Balanced Accuracy"]].mean().reset_index()
    summary["n_pairs"] = 2
    summary["rank"] = summary["Balanced Accuracy"].rank(method="min", ascending=False).astype(int)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT / "pair_metrics.csv", index=False, float_format="%.12f")
    summary.to_csv(OUTPUT / "overall_metrics.csv", index=False, float_format="%.12f")
    pd.DataFrame(region_rows).to_csv(OUTPUT / "per_region_recall.csv", index=False, float_format="%.12f")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": spec.name,
                "methods": list(common.METHODS), "definition": "mean recall across true source classes; adjusted=False",
                "matching": "same coordinate inputs and source-to-target nearest-neighbor rule as official paper evaluator",
                "labels": "normalized truth; lowercase; medulla vessels -> medulla vessel; no predicted clustering labels",
                "missing_labels": "drop pair if either normalized label is missing, exactly as CI",
                "aggregation": "unweighted arithmetic mean of slice1_to_slice2 and slice2_to_slice3",
                "official_CI_Macro_F1_and_Balanced_Accuracy_reproduced": True, "input_sha256": hashes,
                "code_sha256": {str(path): sha256(path) for path in (Path(__file__),
                    Path(common.__file__), Path(__file__).with_name("evaluate_alignment_paper.py"),
                    Path(__file__).with_name("evaluate_fair_alignment.py"))},
                "sklearn_version": sklearn.__version__, "official_workbook_or_aggregate_modified_by_this_script": False}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pairs[["method", "pair", "Balanced Accuracy", "n_valid_label_pairs", "n_source_classes"]].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
