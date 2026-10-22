# HyperMOA manuscript release 1.0.0

HyperMOA links spatial domain identification with molecularly guided section
registration in spatial multi-omics. This release contains the implementation,
the analysis scripts and the frozen source data used in the manuscript.

## Install the release

Extract `HyperMOA_code_v1.0.0.zip` and `HyperMOA_source_data_v1.0.0.zip` into the
same parent directory. Both archives share the top-level folder
`HyperMOA_v1.0.0`. Run the following commands from that folder.

```bash
python -m pip install -r requirements-evaluation.txt
python reproduce.py check
python reproduce.py metrics --output-dir reproduction_output/metrics
python reproduce.py figures --output-dir reproduction_output/figures
```

The metric and plotting commands use the archived labels, coordinates, normalized
RNA evaluation matrices and plotting inputs. They do not require model training
or downloads of the original counts. Outputs go into the requested directory;
the published metric tables and model outputs are retained.

`SOURCE_DATA_INDEX.csv` connects Fig. 1–7 and Supplementary Figs. S1–S3 and
Tables S1–S3 to their files. Fig. 1 is the final vector artwork. Figures 2–7
and S1–S3 have Python generators. `DATASETS.csv` identifies the 18 analyzed
samples and their public sources. `INPUT_DATA_INDEX.csv` records the original
RNA files and hashes used to prepare the registration evaluation inputs.

## Train HyperMOA

Training uses the original dataset-specific entrypoints. Prepare the RNA and
second-modality inputs according to `docs/DATA_PREPARATION.md`, then use a
compatible PyTorch/Scanpy/R environment described in `docs/TRAINING_ENVIRONMENT.md`.

```bash
python reproduce.py train lymph-node
python reproduce.py train simulation
python reproduce.py train mouse-brain
python reproduce.py train embryo
```

These commands show the training command. Add `--execute` to run it. Training
writes the corresponding `HyperMOA/*_result/` folder; use a separate extraction
for retraining if retaining the archived results. For embryos the entrypoint
supports `prepare`, `embed`, `align`, `analyze` and `all`.

The main run uses seed 42. The archived seeds 7 and 2022 repeat the final
refinement with the embeddings fixed, as reported in the manuscript.

## Evaluation and figure reproduction

The publication tables are in `data/evaluation/fair_benchmark/`; saved labels
and comparison-method coordinates are in `data/results/`. The model outputs
are in the dataset-specific `HyperMOA/*_result/` folders. Table values retain
the original stored precision; figure labels display three decimals.

The archived normalized RNA matrices were computed by the original evaluator:
dataset-common genes in their original order, float64 CP10K normalization and
log1p. Their barcode order, spatial coordinates and labels are retained.
The marker plotting cache contains 15 selected gene columns normalized using
the full original RNA library before column selection. All-gene marker
statistics are supplied separately.

The validation environment and results are recorded under `validation/`.
Registration metrics for 35 pairs and 15 dataset-method summaries reproduce
within 1e-8, as do clustering ARI, AMI and CHAOS. Current Windows reruns of
Moran's I and PAS differ from the archived tables where the nearest-neighbor
cutoff contains equal-distance candidates. The observed maxima and tie-choice
bounds are recorded in the validation reports. Clustering rank summaries can
change with those two values. Figure generators use the archived publication
tables, so the publication figures retain their reported numbers.

The `metrics` command reports such neighbor-dependent deviations explicitly;
other metric mismatches raise an error. Full model training was not repeated
when preparing this release. See `validation/README.md` for the exact checks.

## Method attribution

The implementation follows the STAIR framework for spot- and section-level
attention, molecular initialization and boundary alignment. HyperMOA uses
paired-modality encoding, molecularly filtered spatial hyperedges, two-stage
section-wise mean centering and conditional constrained affine updates within
this workflow. See `THIRD_PARTY_NOTICES.md` and the manuscript references.

## Publication metadata

Repository: https://github.com/wust-lxc/HyperMOA

The manuscript code and source-data packages were prepared on 2026-10-22.
The source-data archive is awaiting a Zenodo record; its download link and DOI
will be added here after publication. Evaluation and figure reproduction require
that archive alongside the code. `PUBLICATION_STATUS.json` records the metadata
currently available. A GitHub version tag and Zenodo DOI have not yet been issued.
