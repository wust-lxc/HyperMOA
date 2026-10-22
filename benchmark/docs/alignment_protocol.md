# Alignment benchmark: paper protocol

The official alignment table is `benchmark .xlsx`. The full-precision source is
`data/evaluation/fair_benchmark/alignment_paper_main_metrics.csv`; individual adjacent
slice pairs are in `alignment_paper_pair_metrics.csv`. These supersede the older
`alignment_*_fair.csv` audit for paper figures. No registration model is retrained.

## Main tables

| Dataset | Main metrics | Pair count |
| --- | --- | ---: |
| Human_lymph_node | Chamfer / NN spacing (lower), PCC (higher), CI (higher), Macro-F1 (higher), Balanced Accuracy (higher) | 2 |
| Simulation | PAA (higher), FOSCTTM (lower), Chamfer / NN spacing (lower), CI (higher), PCC (higher), SMR (higher) | 4 |
| Mouse_Brain | PAA* (higher), FOSCTTM* (lower), Chamfer / NN spacing (lower), PCC (higher) | 1 |

For every method and adjacent pair, source spots use the saved pairwise continuous
aligned coordinates; targets are in that pair's target reference frame. HyperMOA
uses `transform_fine` for both slices, because it has no separate pair checkpoint.
All matched-spot metrics use the nearest target spot in these continuous coordinates,
including methods that save their own match indices. Pair results are averaged
without weights. This is a common output-coordinate protocol, not a comparison of
the algorithms' internal correspondence matrices.

Chamfer is the symmetric mean nearest-neighbor Euclidean distance divided by the
target slice's median nearest-neighbor spacing. The workbook/figure column is
dimensionless; **it is not the old raw Chamfer Distance**. `Chamfer_raw` and the
normalizing spacing are retained in the pair CSV. PCC is the mean per-spot Pearson
correlation between source and nearest target on the same dataset-wide intersected
raw RNA genes after CP10K and `log1p`. CI is the fraction of valid matched pairs
with equal anatomical labels. Human labels are lowercased and the singular/plural
`medulla vessel(s)` spelling is unified; absent labels are excluded. Mouse Brain
has incompatible ATAC and H3K27ac label ontologies, so no CI is reported.

Human Macro-F1 uses exactly the same nearest-neighbor pairs, label normalization,
and valid-label mask as CI. For each class, F1 = 2 TP / (2 TP + FP + FN);
Macro-F1 is the unweighted class mean with zero-division set to zero. Classes
are the union present in valid paired labels (all ten classes in both Human
pairs). The final value averages the two pair scores equally, not pooled spots.
Human Balanced Accuracy is the unweighted mean of per-class recall, TP / (TP + FN),
over true source classes in valid nearest-neighbor pairs (all ten Human classes).
It uses adjusted=False and exactly the same normalized labels and valid-label
mask as CI. The overall value averages the two directed pair scores equally.
It measures anatomical label-transfer consistency, not exact biological spot
recovery. Balanced Accuracy was added after inspecting these benchmark results;
the expanded five-metric aggregate is an exploratory, post hoc summary, not an
independently prespecified evaluation. CI, Macro-F1 and Balanced Accuracy share
label information and are not independent evidence; their combined weight is 3/5.

Simulation's shared spot IDs provide actual synthetic correspondences. PAA is exact
nearest-target recovery; FOSCTTM is the fraction of other targets closer than the
correct target; SMR is the fraction with geometric error at most one target median
nearest-neighbor spacing. Mouse Brain has shared **grid-position** IDs, but those
are not verified same biological spots. By the user's post hoc choice, PAA* and
FOSCTTM* enter its four-metric aggregate as explicitly marked grid-ID proxies;
they must not be described as validated spot-pair accuracy. SMR remains only in
`alignment_mouse_grid_proxy_metrics.csv` and the `Mouse_Brain_grid_proxy` sheet.
The Mouse score is exploratory and should be accompanied by the two-metric
Chamfer/PCC sensitivity score in `alignment_mouse_score_sensitivity.csv`, not
presented as an independent primary finding.

The aggregate is the equal-weight mean of normalized within-dataset method ranks
over **all** metrics in that dataset's main table. Ties use average rank. It is
not an accuracy and does not mix datasets; Mouse's two grid proxies are included.
Scores are computed at full precision and displayed to three decimals with half-up rounding.
Mouse paper figures alone display PAA multiplied by 1000, with an x10^-3 header,
so nonzero proxy values do not appear as 0.000. The raw workbook/CSV values and
aggregate calculations remain unscaled; no alignment model is rerun.

## Reproduce

Run from the project root with the environment containing `anndata`, `scipy`,
`pandas`, `scikit-learn`, `openpyxl`, `matplotlib` and `Pillow`:

```bash
python benchmark/scripts/evaluation/alignment/evaluate_alignment_paper.py
python benchmark/scripts/publication/sync_alignment_paper_workbook.py
python figure/aligment_3d/create_alignment_3d_dataset_paper_figures.py
python figure/aligment_3d/create_alignment_3d_composite_paper_18x22cm.py
```

The evaluator's manifest records the source coordinate files and gene counts.
The paper figures' manifests record workbook hashes and worksheet cell addresses.
The 3D reconstructions are visualizations of each method's saved reconstruction;
the metrics are computed from adjacent pairwise outputs, not display-normalized
3D coordinates.
