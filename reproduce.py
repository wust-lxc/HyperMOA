"""Check the manuscript release, recompute metrics, or regenerate its figures."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLBACKEND','Agg')

def load(name, relative):
    path=ROOT/relative
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def check():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    failed=[]
    for row in manifest['files']:
        p=ROOT/row['path']
        if not p.is_file() or sha256(p)!=row['sha256']:failed.append(row['path'])
    for p in ROOT.rglob('*.py'):
        if 'reproduction_output' not in p.parts:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    with (ROOT/'SOURCE_DATA_INDEX.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if not (ROOT/row['path']).is_file():failed.append(row['path'])
    if failed:raise RuntimeError('Missing or changed release files: '+', '.join(failed))
    print(f"Verified {len(manifest['files'])} release files and all source-data references.")

def compare(actual, expected_path, keys, columns, tolerance=2e-8, neighbor_columns=()):
    import numpy as np
    import pandas as pd
    expected=pd.read_csv(expected_path).set_index(keys).sort_index()
    got=actual.set_index(keys).sort_index()
    if not expected.index.equals(got.index):raise AssertionError(f'Row mismatch: {expected_path}')
    differences={};neighbor_differences={}
    for name in columns:
        a=expected[name].to_numpy(dtype=float);b=got[name].to_numpy(dtype=float)
        if not np.allclose(a,b,rtol=0,atol=tolerance,equal_nan=True):
            if name not in neighbor_columns:
                raise AssertionError(f'{expected_path.name}/{name}: max error {np.nanmax(np.abs(a-b))}')
            neighbor_differences[name]=float(np.nanmax(np.abs(a-b)))
        differences[name]=float(np.nanmax(np.abs(a-b))) if np.isfinite(a-b).any() else 0.0
    return dict(file=expected_path.relative_to(ROOT).as_posix(),rows=len(got),max_absolute_errors=differences,
                neighbor_dependent_differences=neighbor_differences)

def archived_evaluation_inputs(alignment):
    """Read the archived normalization outputs used by the unchanged evaluator."""
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from benchmark.scripts.common import fair_benchmark_common as common
    from benchmark.scripts.evaluation.alignment import evaluate_fair_alignment as original
    from benchmark.scripts.evaluation.alignment import evaluate_mouse_region_consistency as region
    cache=ROOT/'data/analysis_inputs/registration'
    if not cache.is_dir():return
    def slices(spec):
        folder=cache/spec.name
        genes=pd.Index(json.loads((folder/'genes.json').read_text(encoding='utf-8')))
        result={}
        for name in spec.slices:
            with np.load(folder/name/'observations.npz') as saved:
                result[name]=original.RawSlice(name=name,obs_names=pd.Index(saved['barcodes']),
                    xy=saved['spatial'],labels=saved['labels'],
                    expression=sparse.load_npz(folder/name/'rna_logcp10k.npz'),
                    sums=saved['sums'],squared_sums=saved['squared_sums'])
        return result,genes
    alignment.load_raw_slices=slices
    read_original=region.read_metadata
    input_lookup={path.resolve():(spec,name) for spec in common.SPECS.values()
                  for name,path in zip(spec.slices,spec.raw_paths)}
    def metadata(path,obs_keys,obsm_keys):
        info=input_lookup.get(Path(path).resolve())
        if info is None:return read_original(path,obs_keys,obsm_keys)
        spec,name=info
        with np.load(cache/spec.name/name/'observations.npz') as saved:
            frame=pd.DataFrame({'truth':saved['labels']},index=pd.Index(saved['barcodes']))
            coordinates={'spatial':saved['spatial']}
        return frame,coordinates
    region.read_metadata=metadata

def metrics(out):
    import numpy as np
    import pandas as pd
    from benchmark.scripts.evaluation.clustering import evaluate_fair_clustering as fair
    from benchmark.scripts.evaluation.clustering import evaluate_additional_clustering_metrics as spatial
    from benchmark.scripts.evaluation.alignment import evaluate_alignment_paper as alignment
    archived_evaluation_inputs(alignment)
    out.mkdir(parents=True,exist_ok=True)
    slices,macro,pooled,coverage=fair.evaluate()
    extra=[]
    for dataset,section,directory in fair.specs():
        selected,truth,columns=spatial.shared_spots(directory,dataset)
        xy=spatial.spatial_coordinates(directory,selected.index)
        neighbors=spatial.ten_neighbors(xy)
        for method,column in columns.items():
            scores=spatial.spatial_scores(selected[column].to_numpy(dtype=str),xy,neighbors)
            extra.append(dict(dataset=dataset,slice=section,method=method,PAS=scores['PAS'],CHAOS=scores['CHAOS']))
    keys=['dataset','slice','method']
    five=slices[keys+['ARI','AMI',"Moran's I"]].merge(pd.DataFrame(extra),on=keys,validate='one_to_one')
    metrics=['ARI','AMI',"Moran's I",'PAS','CHAOS']
    means=five.groupby(['dataset','method'],sort=False)[metrics].mean().reset_index()
    for frame,groups in [(five,['dataset','slice']),(means,['dataset'])]:
        score=pd.Series(index=frame.index,dtype=float)
        for _,group in frame.groupby(groups,sort=False):
            ranks=pd.DataFrame({m:(group[m].rank(ascending=m not in ('PAS','CHAOS'),method='average')-1)/(len(group)-1) for m in metrics})
            score.loc[group.index]=ranks.mean(axis=1)
        frame['Aggregate score']=score
    five.to_csv(out/'clustering_paper_metrics_slice.csv',index=False,float_format='%.12g')
    means.to_csv(out/'clustering_paper_metrics_macro.csv',index=False,float_format='%.12g')
    report=[compare(five,ROOT/'data/evaluation/fair_benchmark/clustering_paper_metrics_slice.csv',keys,metrics+['Aggregate score'],neighbor_columns=("Moran's I",'PAS','Aggregate score')),
            compare(means,ROOT/'data/evaluation/fair_benchmark/clustering_paper_metrics_macro.csv',['dataset','method'],metrics+['Aggregate score'],neighbor_columns=("Moran's I",'PAS','Aggregate score'))]
    pairs,main,proxy,metadata=alignment.evaluate()
    pairs.to_csv(out/'alignment_paper_pair_metrics.csv',index=False,float_format='%.12g')
    main.to_csv(out/'alignment_paper_main_metrics.csv',index=False,float_format='%.12g')
    expected=ROOT/'data/evaluation/fair_benchmark/alignment_paper_main_metrics.csv'
    names=['Chamfer Distance','PCC','CI','Macro-F1','Balanced Accuracy','PAA','FOSCTTM','SMR','Region ARI','Region AMI']
    report.append(compare(main,expected,['dataset','method'],names))
    report.append(compare(pairs,ROOT/'data/evaluation/fair_benchmark/alignment_paper_pair_metrics.csv',
                          ['dataset','method','pair'],names))
    # These are rank summaries of the existing metric set, not additional experiments.
    all_scores=[]
    for dataset,selected_metrics in alignment.MAIN_METRICS.items():
        frame=main.loc[main.dataset.eq(dataset)].set_index('method')
        ranks=pd.DataFrame({m:1-(frame[m].rank(method='average',ascending=m in ('Chamfer Distance','FOSCTTM'))-1)/(len(frame)-1) for m in selected_metrics})
        for method,value in ranks.mean(axis=1).items():all_scores.append(dict(dataset=dataset,method=method,aggregate_score=value))
    scores=pd.DataFrame(all_scores)
    scores.to_csv(out/'alignment_aggregate_scores.csv',index=False,float_format='%.12g')
    displayed=pd.read_csv(ROOT/'figure/aligment_3d/alignment_metrics_5_methods.csv')
    ordered=scores.merge(displayed[['dataset','method','aggregate_score']],on=['dataset','method'],suffixes=('_full','_display'))
    assert np.all(np.abs(ordered.aggregate_score_full-ordered.aggregate_score_display)<=0.000501)
    deviations=any(row['neighbor_dependent_differences'] for row in report)
    (out/'verification.json').write_text(json.dumps(dict(comparisons=report,aggregate_rows_checked=len(ordered),
                                                       source_tables_modified=False,models_retrained=False,
                                                       all_values_match=not deviations),indent=2),encoding='utf-8')
    print('Registration metrics and rank summaries match the frozen source tables.',flush=True)
    print('Clustering comparison: '+('nearest-neighbor tie differences are recorded in verification.json; frozen tables retained.' if deviations else 'all values match.'),flush=True)

def figures(out, selection):
    out.mkdir(parents=True,exist_ok=True)
    requested=set(selection)
    if requested & {'2','3','4'}:
        m=load('release_clustering','figure/clustering/create_clustering_comparison_with_metrics.py')
        dest=out/'clustering';dest.mkdir(exist_ok=True)
        m.OUTPUT_DIR=dest
        for key in ['METRICS_PATH','SELECTION_PATH','MANIFEST_PATH']:setattr(m,key,dest/getattr(m,key).name)
        m.main()
    if '5' in requested:
        import anndata as ad
        import numpy as np
        import pandas as pd
        m=load('release_marker','figure/mouse_brain_marker_multiomics/create_mouse_brain_marker_multiomics_figure.py')
        source=ROOT/'figure/mouse_brain_marker_multiomics'
        dest=out/'markers';dest.mkdir(exist_ok=True)
        cache=ROOT/'data/analysis_inputs/markers/marker_plot_inputs.npz'
        if cache.is_file():
            from scipy import sparse
            with np.load(cache) as saved:
                domains=saved['domains'];expression=sparse.csr_matrix(saved['expression'])
                xy=saved['spatial'];truth=saved['truth'];genes=saved['genes']
        else:
            rna=ad.read_h5ad(m.ATAC_RNA_PATH)
            processed=ad.read_h5ad(m.PROCESSED_PATH,backed='r')
            try:domains=processed.obs.loc[[f'ATAC_{b}' for b in rna.obs_names],'Domain'].astype(str).to_numpy()
            finally:processed.file.close()
            expression=m.log_normalize_counts(rna.X)
            xy=np.asarray(rna.obsm['spatial'],dtype=float)
            truth=rna.obs.truth.astype(str).to_numpy();genes=np.asarray(rna.var_names.astype(str))
        actual=sorted(np.unique(domains),key=int)
        selected=pd.read_csv(source/'selected_domain_markers.csv',dtype={'domain':str})
        representatives=pd.read_csv(source/'representative_markers.csv',dtype={'domain':str})
        stable=sorted(selected.domain.unique(),key=int)
        composition=pd.read_csv(source/'domain_truth_proportions.csv',index_col=0)
        composition.index=composition.index.astype(str)
        paths,values=m.create_figure(dest,xy,truth,domains,expression,
                                    genes,actual,stable,composition,
                                    selected,representatives)
        values.to_csv(dest/'marker_heatmap_values.csv',index=False)
        del expression
    if '6' in requested:
        m=load('release_alignment','figure/aligment_3d/create_alignment_3d_composite_paper_18x22cm.py')
        dest=out/'alignment';dest.mkdir(exist_ok=True);m.OUTPUT_DIR=dest
        for key in ['PNG_PATH','PDF_PATH','SVG_PATH','PREVIEW_PATH','METRICS_PATH','MANIFEST_PATH']:
            setattr(m,key,dest/getattr(m,key).name)
        m.main()
    if '7' in requested:
        m=load('release_embryo','figure/embryo_development/create_temporal_alignment_paper_figure.py')
        m.DEST=out/'embryo';m.main()
    if 'supplement' in requested:
        os.environ['HYPERMOA_FIGURE_OUTPUT']=str(out/'supplement')
        try:load('release_supplement','figure/full_manuscript_review/build_supplement_figures.py')
        finally:os.environ.pop('HYPERMOA_FIGURE_OUTPUT',None)
    print('Figures saved under '+str(out),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check',help='Check hashes, source-data references and Python syntax')
    m=sub.add_parser('metrics',help='Recompute manuscript metrics from frozen outputs')
    m.add_argument('--output-dir',type=Path,default=ROOT/'reproduction_output/metrics')
    f=sub.add_parser('figures',help='Regenerate figures from the archived results')
    f.add_argument('--output-dir',type=Path,default=ROOT/'reproduction_output/figures')
    f.add_argument('--selection',nargs='+',choices=['2','3','4','5','6','7','supplement'],default=['2','3','4','5','6','7','supplement'])
    t=sub.add_parser('train',help='Show or run the original HyperMOA training entrypoint')
    t.add_argument('dataset',choices=['lymph-node','simulation','mouse-brain','embryo'])
    t.add_argument('--execute',action='store_true',help='Execute training and replace that run folder\'s outputs')
    args=parser.parse_args()
    if args.command=='check':check()
    elif args.command in ('metrics','figures'):
        out=args.output_dir.expanduser().resolve()
        if out==ROOT or out in ROOT.parents or any(out==ROOT/d or ROOT/d in out.parents for d in ['data','HyperMOA','figure','benchmark']):
            parser.error('Choose an output directory outside the frozen source-data and code directories.')
        if args.command=='metrics':metrics(out)
        else:figures(out,args.selection)
    else:
        entries={'lymph-node':'HyperMOA/Human_lymph_node_run/run_all.py',
                 'simulation':'HyperMOA/Simulation_run/run_all.py',
                 'mouse-brain':'HyperMOA/Mouse_Brain_run/H3K27ac_ATAC/run_all.py',
                 'embryo':'HyperMOA/mouse_embryo_run/run_mouse_embryo.py'}
        command=[sys.executable,str(ROOT/entries[args.dataset])]
        if args.dataset=='embryo':command.append('all')
        print('Training entrypoint: '+subprocess.list2cmdline(command))
        if args.execute:subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
