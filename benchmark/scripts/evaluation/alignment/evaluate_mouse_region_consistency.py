"""Evaluate regional partition consistency from saved mouse brain registrations."""

from pathlib import Path
import argparse
import json
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from benchmark.scripts.common.h5_annotations import read_metadata

METHODS = ('HyperMOA', '3d-OT', 'SLAT', 'PASTE', 'SANTO')
OUTPUT = ROOT / 'data/evaluation/fair_benchmark'


def valid(values):
    return np.array([v is not None and str(v).strip().lower() not in
                     ('', 'nan', 'none', 'na', 'n/a', 'null', 'unknown') for v in values])


def compute_region_consistency():
    source, sm = read_metadata(ROOT / 'data/datasets/Mouse_Brain/ATAC/adata_RNA.h5ad',
                               ['truth'], ['spatial'])
    target, tm = read_metadata(ROOT / 'data/datasets/Mouse_Brain/H3K27ac/adata_RNA.h5ad',
                               ['truth'], ['spatial'])
    final_path = ROOT / 'HyperMOA/Mouse_brain_result/H3K27ac_ATAC/adata.h5ad'
    final, fm = read_metadata(final_path, ['batch', 'original_barcode'], ['transform_fine'])
    coords = {}
    aligned = []
    for assay, raw in [('ATAC', source), ('H3K27ac', target)]:
        mask = final['batch'].eq(assay).to_numpy()
        lookup = pd.Index(final.loc[mask, 'original_barcode'])
        assert lookup.is_unique
        indexer = lookup.get_indexer(raw.index)
        assert (indexer >= 0).all()
        aligned.append(fm['transform_fine'][mask][indexer])
    coords['HyperMOA'] = tuple(aligned)
    input_paths = {'HyperMOA': str(final_path)}
    for method in METHODS[1:]:
        path = ROOT / f'data/results/alignment/Mouse_Brain/{method}/pair_checkpoints/Mouse_Brain_ATAC_to_Mouse_Brain_H3K27ac.npz'
        with np.load(path) as saved:
            coords[method] = (np.array(saved['aligned_source'], dtype=float), tm['spatial'])
        input_paths[method] = str(path)

    source_labels = source['truth'].to_numpy()
    target_labels = target['truth'].to_numpy()
    forward = {}; reverse = {}; diagnostics = {}
    known = pd.read_csv(OUTPUT / 'alignment_paper_main_metrics.csv')
    for method, (sx, tx) in coords.items():
        assert sx.shape == (len(source), 2) and tx.shape == (len(target), 2)
        assert np.isfinite(sx).all() and np.isfinite(tx).all()
        distances, nearest = cKDTree(tx).query(sx)
        reverse_distances, reverse_nearest = cKDTree(sx).query(tx)
        spacing = float(np.median(cKDTree(tx).query(tx, k=2)[0][:, 1]))
        chamfer = float((distances.mean() + reverse_distances.mean()) / (2 * spacing))
        expected = float(known.loc[(known.dataset == 'Mouse_Brain') &
                                   (known.method == method), 'Chamfer Distance'].iloc[0])
        assert abs(chamfer - expected) < 1e-8, (method, chamfer, expected)
        forward[method] = target_labels[nearest]
        reverse[method] = source_labels[reverse_nearest]
        diagnostics[method] = {'chamfer_reproduced': chamfer,
                               'forward_valid_before_common_mask': int((valid(source_labels) & valid(forward[method])).sum()),
                               'reverse_valid_before_common_mask': int((valid(target_labels) & valid(reverse[method])).sum())}

    forward_mask = valid(source_labels)
    reverse_mask = valid(target_labels)
    for method in METHODS:
        forward_mask &= valid(forward[method])
        reverse_mask &= valid(reverse[method])
    metric_functions = {'ARI': adjusted_rand_score, 'AMI': adjusted_mutual_info_score,
                        'NMI': normalized_mutual_info_score}
    rows = []
    for method in METHODS:
        row = {'dataset': 'Mouse_Brain', 'method': method,
               'pair': 'Mouse_Brain_ATAC_to_Mouse_Brain_H3K27ac',
               'n_region_valid_spots': int(forward_mask.sum()),
               'n_region_reverse_valid_spots': int(reverse_mask.sum())}
        for name, function in metric_functions.items():
            a = float(function(source_labels[forward_mask].astype(str), forward[method][forward_mask].astype(str)))
            b = float(function(target_labels[reverse_mask].astype(str), reverse[method][reverse_mask].astype(str)))
            row['Region ' + name] = a
            row['Reverse region ' + name] = b
        rows.append(row)
    frame = pd.DataFrame(rows)
    metadata = {
        'main_metrics': ['Region ARI', 'Region AMI', 'Chamfer Distance', 'PCC'],
        'source_direction': 'ATAC to H3K27ac spot matching; target labels transferred onto ATAC observations',
        'matching': 'Euclidean nearest target in saved registration coordinates; no display scaling',
        'mask': 'Intersection of valid supplied source and matched-target labels across all five methods',
        'n_source_spots': len(source), 'n_target_spots': len(target),
        'n_common_annotated_source_spots': int(forward_mask.sum()),
        'n_common_annotated_target_spots_reverse': int(reverse_mask.sum()),
        'labels': 'Independent supplied regional annotations; no mapping fitted to a method or alignment',
        'interpretation': 'Cross-section regional partition consistency; distinct annotation vocabularies retained',
        'reverse': 'Supplementary nearest-source query in the same fitted coordinate frame',
        'aggregate_score': 'Equal-weight mean of normalized method ranks for Region ARI, Region AMI, Chamfer Distance and PCC; each metric has weight 1/4',
        'coordinate_sources': input_paths, 'coordinate_checks': diagnostics,
        'algorithm_runs_repeated': False,
    }
    return frame, metadata


def adopt(frame, metadata):
    frame.to_csv(OUTPUT / 'alignment_mouse_region_consistency.csv', index=False, float_format='%.9f')
    (OUTPUT / 'alignment_mouse_region_manifest.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    for filename in ['alignment_paper_main_metrics.csv', 'alignment_paper_pair_metrics.csv']:
        table = pd.read_csv(OUTPUT / filename)
        for method in METHODS:
            mask = table.dataset.eq('Mouse_Brain') & table.method.eq(method)
            assert mask.sum() == 1
            row = frame.loc[frame.method.eq(method)].iloc[0]
            for column in ['Region ARI', 'Region AMI', 'n_region_valid_spots']:
                table.loc[mask, column] = row[column]
        if filename == 'alignment_paper_main_metrics.csv':
            table.loc[table.dataset.eq('Mouse_Brain'), ['PAA', 'FOSCTTM']] = np.nan
        table.to_csv(OUTPUT / filename, index=False, float_format='%.9f')
    path = OUTPUT / 'alignment_paper_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['main_metrics_by_dataset']['Mouse_Brain'] = metadata['main_metrics']
    manifest['Region ARI'] = 'ARI between supplied ATAC labels and nearest registered H3K27ac labels on 8894 common valid ATAC spots'
    manifest['Region AMI'] = 'AMI between the same two supplied partitions on the same common spots'
    manifest['mouse_proxy_warning'] = 'Grid-ID proxies retained separately for supplementary reference; not in mouse main metrics or an aggregate score'
    manifest['mouse_region_protocol'] = str(OUTPUT / 'alignment_mouse_region_manifest.json')
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    table = pd.read_csv(OUTPUT / 'alignment_paper_main_metrics.csv')
    mouse = table.loc[table.dataset.eq('Mouse_Brain')].set_index('method').reindex(METHODS)
    columns = ['Region ARI', 'Region AMI', 'Chamfer Distance', 'PCC']
    scores = pd.DataFrame(index=mouse.index)
    for name in columns:
        ranks = mouse[name].rank(method='average', ascending=name == 'Chamfer Distance')
        scores[name + ' rank score'] = 1 - (ranks - 1) / (len(METHODS) - 1)
    scores['score_with_regions'] = scores.mean(axis=1)
    scores['score_geometry_expression'] = scores[['Chamfer Distance rank score', 'PCC rank score']].mean(axis=1)
    scores.to_csv(OUTPUT / 'alignment_mouse_region_score_summary.csv', float_format='%.9f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adopt', action='store_true', help='Update source-data tables used by the manuscript figure')
    args = parser.parse_args()
    result, protocol = compute_region_consistency()
    if args.adopt:
        adopt(result, protocol)
    print(result.to_string(index=False))
