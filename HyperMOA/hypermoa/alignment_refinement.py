"""Fixed bounded affine refinement with label-free reference-frame rollback.

The implementation operates on arrays, never reads experiment files, and never
uses clustering labels or expression-based evaluation scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


VERSION = "reference_frame_refinement_v2"
EPSILON = 1e-10


def refinement_protocol():
    return {
        "version": VERSION, "default_seed": 42, "mnn_k": 10, "spatial_k": 10,
        "holdout_fraction": .2, "holdout_seed": 42, "stretch_cap": 1.40,
        "ransac_trials": 2000, "ransac_confidence": .99, "ransac_refine_iterations": 10,
        "residual_threshold_target_NN": 3., "max_rotation_deg": 20.,
        "area_ratio_bounds": [.8, 1.25], "max_p95_displacement_target_NN": 20.,
        "minimum_inlier_sources": 100, "minimum_inlier_blocks": 8,
        "training_consensus": "strict majority of assigned one-to-one anchors",
        "local_gate": "heldout median improves, edge support does not decrease, fixed-target Chamfer improves, coverage does not decrease",
        "chain_gate": "cancel registered shared downstream affine; reference Chamfer and coverage must not worsen; otherwise full rollback",
        "inverse_reconstruction_tolerance_NN": 1e-8, "comparison_roundoff_tolerance": EPSILON,
        "context": "concatenate normalized spot embedding and normalized spatial-neighbor mean with equal weight",
        "weights": "exp(-cosine_distance/median_positive_distance)/sqrt(endpoint_degrees), multiplied by 0.25+0.75*clipped_support/support_p95",
        "assignment": "sparse partial one-to-one matching; private unmatched dummy with median observed cost",
        "status": "adopted fixed rule; developed on exposed data, not independently biologically validated",
    }


@dataclass(frozen=True)
class SliceFrame:
    name: str
    spot_ids: np.ndarray
    coordinates: np.ndarray
    embedding: np.ndarray


@dataclass
class PairAnchors:
    training: np.ndarray
    weights: np.ndarray
    heldout: np.ndarray
    heldout_sources: np.ndarray
    partial_assignment: dict


@dataclass
class RefinementResult:
    coordinates: dict[str, np.ndarray]
    pair_diagnostics: list[dict]
    chain_diagnostics: list[dict]
    anchors: dict[str, PairAnchors]
    proposed_transforms: list[tuple[np.ndarray, np.ndarray]]
    retained_transforms: list[tuple[np.ndarray, np.ndarray]]
    chain_accepted: bool
    seed: int


def median_nn_radius(xy):
    return float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))


def coordinate_statistics(source, target):
    spacing = median_nn_radius(target)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Target requires positive median nearest-neighbor spacing")
    forward, matches = cKDTree(target).query(source)
    reverse = cKDTree(source).query(target)[0]
    return {"Chamfer_NN": float((forward.mean()+reverse.mean())/2/spacing),
            "target_coverage": float(len(np.unique(matches))/len(target))}


def spatial_graph(xy):
    neighbors = cKDTree(xy).query(xy, k=11)[1][:, 1:]
    graph = sparse.csr_matrix((np.ones(neighbors.size),
        (np.repeat(np.arange(len(xy)), neighbors.shape[1]), neighbors.ravel())), shape=(len(xy), len(xy)))
    return neighbors, graph


def context_anchors(left, right):
    neighbors_a, graph_a = spatial_graph(left.coordinates)
    neighbors_b, graph_b = spatial_graph(right.coordinates)

    def context(frame, neighbors):
        embedding = frame.embedding
        return normalize(np.concatenate([normalize(embedding), normalize(embedding[neighbors].mean(1))], axis=1))

    a, b = context(left, neighbors_a), context(right, neighbors_b)
    forward = NearestNeighbors(n_neighbors=10, metric="cosine", n_jobs=1).fit(b).kneighbors(a, return_distance=False)
    backward = NearestNeighbors(n_neighbors=10, metric="cosine", n_jobs=1).fit(a).kneighbors(b, return_distance=False)
    reverse = {(int(i), j) for j, row in enumerate(backward) for i in row}
    pairs = np.asarray(sorted((i, int(j)) for i, row in enumerate(forward) for j in row if (i, int(j)) in reverse), dtype=int).reshape(-1, 2)
    heldout_sources = np.zeros(len(a), dtype=bool)
    heldout_sources[np.random.default_rng(42).permutation(len(a))[:int(.2*len(a))]] = True
    if not len(pairs):
        return PairAnchors(pairs, np.empty(0), pairs.copy(), heldout_sources, {"n_unmatched_sources": len(a)})
    normalized_a, normalized_b = normalize(a), normalize(b)
    distance = np.maximum(0, 1-np.sum(normalized_a[pairs[:, 0]]*normalized_b[pairs[:, 1]], axis=1))
    graph = sparse.csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(a), len(b)))
    support = np.asarray((graph_a @ graph @ graph_b.T)[pairs[:, 0], pairs[:, 1]]).ravel()
    tau = max(float(np.median(distance[distance > 0])) if np.any(distance > 0) else 1, 1e-8)
    confidence = np.exp(-distance/tau)
    confidence *= .25+.75*np.minimum(1, support/max(float(np.percentile(support, 95)), 1))
    degree_a = np.bincount(pairs[:, 0], minlength=len(a))
    degree_b = np.bincount(pairs[:, 1], minlength=len(b))
    weights = confidence/np.sqrt(degree_a[pairs[:, 0]]*degree_b[pairs[:, 1]])
    heldout = heldout_sources[pairs[:, 0]]
    candidates, weights = pairs[~heldout], weights[~heldout]
    if not len(candidates):
        return PairAnchors(candidates, weights, pairs[heldout], heldout_sources, {"n_unmatched_sources": len(a)})
    cost = -np.log(np.maximum(weights/weights.max(), 1e-12))+1e-6
    dummy_cost = float(np.median(cost))
    assignment = sparse.csr_matrix((np.r_[cost, np.full(len(a), dummy_cost)],
        (np.r_[candidates[:, 0], np.arange(len(a))], np.r_[candidates[:, 1], len(b)+np.arange(len(a))])),
        shape=(len(a), len(b)+len(a)))
    row, col = min_weight_full_bipartite_matching(assignment)
    matched = col < len(b)
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(candidates)}
    chosen = np.asarray([lookup[(int(i), int(j))] for i, j in zip(row[matched], col[matched])], dtype=int)
    return PairAnchors(candidates[chosen], weights[chosen], pairs[heldout], heldout_sources,
        {"dummy_cost": dummy_cost, "n_unmatched_sources": int((~matched).sum())})


def projected_bounded_affine(moving, fixed, weights):
    w = np.asarray(weights, dtype=float)/np.sum(weights)
    a = np.sum(moving*w[:, None], axis=0)
    b = np.sum(fixed*w[:, None], axis=0)
    matrix = np.linalg.lstsq((moving-a)*np.sqrt(w[:, None]), (fixed-b)*np.sqrt(w[:, None]), rcond=None)[0]
    if np.linalg.det(matrix) <= 0:
        raise ValueError("Affine proposal reflects or collapses space")
    u, singular, vt = np.linalg.svd(matrix)
    bounded = u @ np.diag(np.clip(singular, 1/1.40, 1.40)) @ vt
    u, values, vt = np.linalg.svd(bounded)
    rotation = u @ vt
    angle = np.arctan2(rotation[0, 1], rotation[0, 0])
    limit = np.radians(20.)
    angle = np.clip(angle, -limit+1e-10, limit-1e-10)
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    bounded = rotation @ (vt.T @ np.diag(values) @ vt)
    return bounded, b-a @ bounded, singular


def anchor_statistics(source, target, pairs, spacing):
    if not len(pairs):
        return {"edge_inlier_fraction": np.nan, "edge_residual_median_NN": np.nan}
    residual = np.linalg.norm(source[pairs[:, 0]]-target[pairs[:, 1]], axis=1)/spacing
    return {"edge_inlier_fraction": float(np.mean(residual <= 3.)),
            "edge_residual_median_NN": float(np.median(residual))}


def fit_pair(left, right, anchors, seed):
    record = {"n_training_anchors": len(anchors.training), "partial_assignment": anchors.partial_assignment,
              "correction_applied": False, "fit_status": "no_consensus"}
    matrix, offset = np.eye(2), np.zeros(2)
    if len(anchors.training) < 3:
        return matrix, offset, record
    moving = left.coordinates[anchors.training[:, 0]]
    fixed = right.coordinates[anchors.training[:, 1]]
    spacing = median_nn_radius(right.coordinates)
    threshold = 3.*spacing
    cv2.setRNGSeed(int(seed))
    permutation = np.random.default_rng(seed).permutation(len(moving))
    model, ordered_inliers = cv2.estimateAffine2D(moving[permutation].astype(np.float32), fixed[permutation].astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=2000, confidence=.99, refineIters=10)
    if model is None or ordered_inliers is None:
        return matrix, offset, record
    inliers = np.zeros(len(moving), dtype=bool)
    inliers[permutation] = ordered_inliers.ravel().astype(bool)
    if inliers.sum() < 3:
        return matrix, offset, record
    try:
        attempted, translation, unbounded = projected_bounded_affine(moving[inliers], fixed[inliers], anchors.weights[inliers])
        u, singular, vt = np.linalg.svd(attempted)
        rotation = u @ vt
        angle = float(np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0])))
        determinant = float(np.linalg.det(attempted))
        trial = left.coordinates @ attempted+translation
        residual = np.linalg.norm(trial[anchors.training[:, 0]]-fixed, axis=1)
        sources = anchors.training[residual <= threshold, 0]
        blocks = 4*np.searchsorted(np.quantile(left.coordinates[:, 1], [.25, .5, .75]), left.coordinates[:, 1])
        blocks += np.searchsorted(np.quantile(left.coordinates[:, 0], [.25, .5, .75]), left.coordinates[:, 0])
        displacement = float(np.percentile(np.linalg.norm(trial-left.coordinates, axis=1), 95)/spacing)
        safe = bool(.8 <= determinant <= 1.25 and abs(angle) <= 20. and displacement <= 20.
                    and len(np.unique(sources)) >= 100 and len(np.unique(blocks[sources])) >= 8)
        record.update({"fit_status": "safe_candidate" if safe else "failed_geometry_guard", "correction_applied": safe,
            "singular_values": singular.tolist(), "unbounded_singular_values": unbounded.tolist(),
            "determinant": determinant, "rotation_deg": angle, "p95_displacement_target_NN": displacement,
            "n_inlier_sources": len(np.unique(sources)), "n_inlier_blocks": len(np.unique(blocks[sources])),
            "attempted_matrix": attempted.tolist(), "attempted_translation": translation.tolist()})
        if safe:
            matrix, offset = attempted, translation
    except (ValueError, np.linalg.LinAlgError) as error:
        record.update({"fit_status": "invalid_affine", "error": str(error)})
    return matrix, offset, record


def decide_pair(record, training_after, heldout_before, heldout_after, before, after):
    values = [training_after["edge_inlier_fraction"], heldout_before["edge_residual_median_NN"],
        heldout_after["edge_residual_median_NN"], heldout_before["edge_inlier_fraction"], heldout_after["edge_inlier_fraction"],
        before["Chamfer_NN"], after["Chamfer_NN"], before["target_coverage"], after["target_coverage"]]
    checks = {"finite_diagnostics": bool(np.isfinite(values).all()),
        "original_fit_guard": bool(record["correction_applied"]),
        "training_majority": bool(training_after["edge_inlier_fraction"] > .5),
        "heldout_median_improved": bool(heldout_after["edge_residual_median_NN"] < heldout_before["edge_residual_median_NN"]-EPSILON),
        "heldout_support_not_worse": bool(heldout_after["edge_inlier_fraction"] >= heldout_before["edge_inlier_fraction"]-EPSILON),
        "pair_chamfer_improved": bool(after["Chamfer_NN"] < before["Chamfer_NN"]-EPSILON),
        "pair_coverage_not_worse": bool(after["target_coverage"] >= before["target_coverage"]-EPSILON)}
    return {**checks, "accepted": all(checks.values()),
            "rejection_reasons": ";".join(key for key, value in checks.items() if not value)}


def compose_chain(baseline, transforms, names):
    coordinates = {names[-1]: baseline[names[-1]].copy()}
    suffix, matrix, offset = [None]*len(transforms), np.eye(2), np.zeros(2)
    for index in range(len(transforms)-1, -1, -1):
        suffix[index] = (matrix.copy(), offset.copy())
        a, t = transforms[index]
        offset = t @ matrix+offset
        matrix = a @ matrix
        coordinates[names[index]] = baseline[names[index]] @ matrix+offset
    return {name: coordinates[name] for name in names}, suffix


def reference_chain_checks(coordinates, baseline, transforms, suffix, names):
    checks = []
    for index, (source, target) in enumerate(zip(names[:-1], names[1:])):
        matrix, offset = suffix[index]
        if not np.isfinite(matrix).all() or np.linalg.det(matrix) <= 0:
            raise ValueError("Invalid registered shared transform")
        inverse = np.linalg.inv(matrix)
        source_ref = (coordinates[source]-offset) @ inverse
        target_ref = (coordinates[target]-offset) @ inverse
        a, t = transforms[index]
        expected = baseline[source] @ a+t
        spacing = median_nn_radius(baseline[target])
        source_error = float(np.max(np.linalg.norm(source_ref-expected, axis=1))/spacing)
        target_error = float(np.max(np.linalg.norm(target_ref-baseline[target], axis=1))/spacing)
        if max(source_error, target_error) > 1e-8:
            raise ValueError("Composed coordinates do not reconstruct registered local corrections")
        before = coordinate_statistics(baseline[source], baseline[target])
        after = coordinate_statistics(expected, baseline[target])
        global_after = coordinate_statistics(coordinates[source], coordinates[target])
        geometry_ok = after["Chamfer_NN"] <= before["Chamfer_NN"]+EPSILON
        coverage_ok = after["target_coverage"] >= before["target_coverage"]-EPSILON
        checks.append({"pair": f"{source}_to_{target}", "chain_pair_passed": bool(geometry_ok and coverage_ok),
            "reference_chamfer_not_worse": bool(geometry_ok), "reference_coverage_not_worse": bool(coverage_ok),
            "source_recovery_max_error_NN": source_error, "target_recovery_max_error_NN": target_error,
            **{f"{prefix}_{key}": value for prefix, data in (("baseline", before), ("reference", after), ("global", global_after)) for key, value in data.items()}})
    return checks


def refine_slices(frames, seed=42):
    """Generate anchors, fit and gate corrections from arrays, without file IO."""
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed <= np.iinfo(np.int32).max:
        raise ValueError("Seed must be a nonnegative int32 integer")
    frames = [SliceFrame(str(f.name), np.asarray(f.spot_ids, dtype=str).copy(),
                        np.asarray(f.coordinates, dtype=float).copy(), np.asarray(f.embedding, dtype=float).copy()) for f in frames]
    names = [f.name for f in frames]
    if len(frames) < 2 or len(set(names)) != len(names):
        raise ValueError("Provide at least two distinct slices in explicit alignment order")
    width = frames[0].embedding.shape[1] if frames[0].embedding.ndim == 2 else 0
    for frame in frames:
        n = len(frame.spot_ids)
        if n < 11 or len(set(frame.spot_ids)) != n or frame.coordinates.shape != (n, 2):
            raise ValueError(f"Invalid spot IDs or coordinates: {frame.name}")
        if frame.embedding.ndim != 2 or frame.embedding.shape != (n, width) or width == 0:
            raise ValueError(f"Embedding dimensions differ: {frame.name}")
        if not np.isfinite(frame.coordinates).all() or not np.isfinite(frame.embedding).all():
            raise ValueError(f"Nonfinite input arrays: {frame.name}")
        if median_nn_radius(frame.coordinates) <= 0:
            raise ValueError(f"Nonpositive coordinate spacing: {frame.name}")
    baseline = {f.name: f.coordinates for f in frames}
    pair_diagnostics, all_anchors, proposed, retained = [], {}, [], []
    for left, right in zip(frames[:-1], frames[1:]):
        pair = f"{left.name}_to_{right.name}"
        anchors = context_anchors(left, right)
        a, t, record = fit_pair(left, right, anchors, seed)
        trial = left.coordinates @ a+t
        spacing = median_nn_radius(right.coordinates)
        training_after = anchor_statistics(trial, right.coordinates, anchors.training, spacing)
        heldout_before = anchor_statistics(left.coordinates, right.coordinates, anchors.heldout, spacing)
        heldout_after = anchor_statistics(trial, right.coordinates, anchors.heldout, spacing)
        before, after = coordinate_statistics(left.coordinates, right.coordinates), coordinate_statistics(trial, right.coordinates)
        decision = decide_pair(record, training_after, heldout_before, heldout_after, before, after)
        record.update({"pair": pair, "seed": int(seed), **decision})
        for prefix, values in (("training_after", training_after), ("heldout_before", heldout_before), ("heldout_after", heldout_after),
                               ("geometry_before", before), ("geometry_pair_after", after)):
            record.update({f"{prefix}_{key}": value for key, value in values.items()})
        pair_diagnostics.append(record)
        all_anchors[pair] = anchors
        proposed.append((a, t))
        retained.append((a, t) if decision["accepted"] else (np.eye(2), np.zeros(2)))
    coordinates, suffix = compose_chain(baseline, retained, names)
    chain = reference_chain_checks(coordinates, baseline, retained, suffix, names)
    accepted = all(row["chain_pair_passed"] for row in chain)
    if not accepted:
        coordinates = {name: xy.copy() for name, xy in baseline.items()}
    return RefinementResult(coordinates, pair_diagnostics, chain, all_anchors, proposed, retained, accepted, int(seed))


def refine_anndata(adata, batch_order, *, batch_key="batch", embedding_key="HyperMOA_bc",
                   spatial_key="transform_fine", key_added="transform_refined", seed=42, inplace=False):
    """Add candidate coordinates without changing obs labels or baseline obsm.

    Returns ``(AnnData, RefinementResult)``. Explicit slice order is required.
    By default a copy is returned; even in-place operation never overwrites the
    input coordinate key, an existing output key or embedding array.
    """
    if key_added in adata.obsm or key_added in (spatial_key, embedding_key):
        raise ValueError("Refinement output must use a new coordinate key")
    batches = adata.obs[batch_key].astype(str).to_numpy()
    names = [str(name) for name in batch_order]
    if len(names) != len(set(names)) or set(names) != set(batches):
        raise ValueError("Batch order must contain every observed batch exactly once")
    frames = []
    for name in names:
        mask = batches == name
        ids = np.asarray(adata.obs.loc[mask, "original_barcode"].astype(str)
                         if "original_barcode" in adata.obs else adata.obs_names[mask], dtype=str)
        frames.append(SliceFrame(name, ids, np.asarray(adata.obsm[spatial_key])[mask], np.asarray(adata.obsm[embedding_key])[mask]))
    result = refine_slices(frames, seed=seed)
    output = adata if inplace else adata.copy()
    xy = np.empty((adata.n_obs, 2), dtype=float)
    for name in names:
        xy[batches == name] = result.coordinates[name]
    output.obsm[key_added] = xy
    output.uns[f"{key_added}_protocol"] = json.dumps({**refinement_protocol(), "seed": int(seed),
        "batch_order": names, "embedding_key": embedding_key, "baseline_coordinate_key": spatial_key,
        "candidate_coordinate_key": key_added, "chain_accepted": result.chain_accepted}, sort_keys=True)
    return output, result
