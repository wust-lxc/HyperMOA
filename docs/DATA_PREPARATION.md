# Input data and sample mapping

The release can reproduce the reported metrics and plots from the archived
analysis inputs. Original counts and paired modalities are needed for model
training. DATASETS.csv maps all 18 manuscript samples to their public sources.

| Dataset | Public record | Training input layout |
| --- | --- | --- |
| Simulated RNA–protein | https://doi.org/10.5281/zenodo.15089427 | data/datasets/Simulation/slice1..slice5/adata_RNA.h5ad and adata_ADT.h5ad |
| Human lymph node | SpaMosaic v4 https://doi.org/10.5281/zenodo.18946723; GSE263617 for A1/D1 | data/datasets/Human_lymph_node/slice1..slice3/sN_adata_rna.h5ad and sN_adata_adt.h5ad |
| Mouse brain | SpatialGlue v3 https://doi.org/10.5281/zenodo.10362607; GSE205055 | data/datasets/Mouse_Brain/ATAC and H3K27ac/adata_RNA.h5ad and adata_peaks_normalized.h5ad |
| Mouse embryo | MOSTA https://db.cngb.org/stomics/mosta/download/; STDS0000058; CNP0001543 | data/datasets/Stereo_seq_mouse_embryo/E9.5_E1S1.MOSTA.h5ad through E16.5_E1S1.MOSTA.h5ad |

Human S1 and S2 correspond to A1 and D1; S3 retains the annotation and identity
supplied by the processed release. The brain marker analysis uses ATAC,
whereas the example clustering figure uses H3K27ac. The embryo series uses
E1S1 only, one section per stage; additional specimens are not model inputs.

Within each multi-omics section, the preparation scripts align observations
by barcode and reorder the second modality accordingly. They intersect RNA
genes across sections and perform the preprocessing described in Methods.
The downloaded analysis-ready epigenomic files contain the supplied normalized
features; the runner uses those files rather than substituting a different LSI.

The multi-omics runners support HYPERMOA_DATA_DIR. The full analysis layout,
evaluation and embryo runner use paths relative to the extracted project root.
Historical manifests retain their original server paths as provenance; the
source-data index provides current release-relative paths.
