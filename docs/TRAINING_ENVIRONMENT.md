# Training environment

The embryo experiment records Python 3.10.18, PyTorch 1.13.1, CUDA 11.7,
PyTorch Geometric 2.7.0, Scanpy 1.11.4 and FAISS CPU 1.14.2, and an NVIDIA
GeForce RTX 4090 D. These are the recorded experiment versions, not the
environment used for the release's Windows evaluation checks.

`requirements-training.txt` lists the imported training dependencies. Install
PyTorch for the intended CUDA runtime, PyTorch Geometric and its matching
extensions as documented by those projects. Install R and the R package
`mclust` (the original environment records mclust 6.1.2), then configure rpy2
and R_HOME. With the selected environment active:

```bash
python -m pip install -r requirements-training.txt
Rscript -e 'install.packages("mclust", repos="https://cloud.r-project.org")'
python -c "import torch, torch_geometric, scanpy, rpy2; print(torch.__version__)"
```

The original three multi-omics experiments do not contain a complete saved
environment lockfile. The supplied version ranges are an installation list;
the tested release evaluation versions are pinned separately in
`validation/requirements-evaluation-tested.txt`. The preparation checks did
not retrain the encoders or graph model.

Reference installation instructions: https://pytorch.org/get-started/locally/
and https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
