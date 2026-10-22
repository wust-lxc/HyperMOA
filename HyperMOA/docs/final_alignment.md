# Fixed final alignment refinement

The three benchmark run folders use the same post-ICP rule through
`hypermoa.final_alignment.finalize_anndata`. Main seed42, stretch cap1.40,
rotation cap20 degrees, RANSAC radius3 target NN. Failed pairs remain identity;
the registered reference-frame chain gate can roll back the whole refinement.
There is no dataset-specific branch or truth/PCC/rank-based parameter selection.

`obsm['alignment_input']` is the required pre-refinement input, not an obsolete
result to compare or plot. Keeping it prevents repeated runs from applying the
warp twice. `transform_fine` and `aligned_spatial` are the current final x/y.
Expression, embeddings, original spatial coordinates and clustering labels are
unchanged. z represents slice order, not physical thickness.

## Existing frozen results

```bash
python benchmark/scripts/runners/alignment/run_final_alignment_refinement.py run --adopt
python benchmark/scripts/runners/alignment/run_final_alignment_refinement.py stability
python benchmark/scripts/evaluation/alignment/evaluate_alignment_paper.py
python benchmark/scripts/publication/sync_alignment_paper_workbook.py
```

Outputs and provenance are in `data/results/alignment_refinement_final/`. Main scores and
figures use seed42; seeds7/2022 describe refinement computational stability, not
full-model retraining or independent biological replication. The implementation
does not import deleted experiment scripts or cached acceptance decisions.

The original init/ICP settings remain specific to each dataset's existing run
entry. Their step04 now calls the same fixed finalizer after `fine_align`.

## Evaluation and limitations

The formal evaluator uses actual final global coordinates, common-RNA
CP10K+log1p spot-wise PCC, symmetric Chamfer divided by target NN spacing, and
valid nearest-neighbor truth pairs for CI/Macro-F1/Balanced Accuracy. Human
Balanced Accuracy is macro-recall of source truth labels (adjusted=False).
Pair scores are averaged without weighting. Reference-frame diagnostics only decide rollback; they are
not substituted for final global metrics.

The parameters were selected in a finite development-data search using heldout
anchor consistency, not independent anatomical validation. The compact search
record is retained for disclosure; old candidate coordinates/scripts are not
required. Human has regional trade-offs. Mouse shared-grid PAA/FOSCTTM remain
unverified biological proxies. No claim of superiority on every metric follows.
