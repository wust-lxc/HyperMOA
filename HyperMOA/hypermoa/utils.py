import numpy as np
import scanpy as sc
import os
import datetime
import torch

def set_seed(seed):
    import random 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def make_seeds(n, per_batch, desired_batches):
    total = per_batch * desired_batches
    base = np.arange(n, dtype=np.int64)
    if total <= n:
        # 如果需要的数量不超过 n，直接取前 total 个
        return base[:total]
    else:
        # 至少放一遍全部节点
        seeds = base.tolist()
        # 还需要补充多少
        remain = total - n
        # 随机补齐
        extra = np.random.choice(base, size=remain, replace=True)
        seeds.extend(extra.tolist())
        return np.array(seeds, dtype=np.int64)



def construct_folder(path_name):
    result_path = path_name+'_'+datetime.datetime.now().strftime('%m-%d-%y %H:%M')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs(result_path + '/embedding')
        os.makedirs(result_path + '/embedding/train')
        os.makedirs(result_path + '/location')
        os.makedirs(result_path + '/location/edge')
    return result_path


class MakeLogClass:
    def __init__(self, log_file):
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    def make(self, *args):
        # print(*args)
        # Write the message to the file
        with open(self.log_file, "a") as f:
            for arg in args:
                f.write("{}\r\n".format(arg))


def mclust_R(adata, num_cluster=10, modelNames='EEE', used_obsm='latent', random_seed=2022, key_add='clusters'):
    import sys

    # Ensure rpy2 binds to the R runtime in the active conda env.
    conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
    if not os.path.exists(os.path.join(conda_prefix, 'lib', 'R')):
        conda_prefix = os.environ.get('CONDA_PREFIX', conda_prefix)

    r_home = os.path.join(conda_prefix, 'lib', 'R')
    env_bin = os.path.join(conda_prefix, 'bin')
    env_lib = os.path.join(conda_prefix, 'lib')

    os.environ['R_HOME'] = r_home
    os.environ['PATH'] = env_bin + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = env_lib + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

    # Defensive fix for hosts exporting invalid thread env values (for example OMP_NUM_THREADS=0).
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(_k) in (None, '', '0'):
            os.environ[_k] = '1'

    # Lazy import so non-R workflows do not fail during module import.
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.rinterface_lib import embedded

    np.random.seed(random_seed)

    # Validate input embedding before crossing Python/R boundary.
    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"adata.obsm[{used_obsm!r}] must be a 2D array, got shape={x.shape}")
    if x.shape[0] == 0:
        raise ValueError(f"adata.obsm[{used_obsm!r}] is empty.")
    if not np.isfinite(x).all():
        raise ValueError(f"adata.obsm[{used_obsm!r}] contains NaN or Inf values.")

    model_name = modelNames
    if x.shape[1] == 1 and modelNames == 'EEE':
        model_name = 'E'

    try:
        robjects.r.library("mclust")
        robjects.conversion.set_conversion(default_converter + numpy2ri.converter)
        robjects.globalenv[".stair_mclust_data"] = x
        robjects.r["set.seed"](random_seed)
        robjects.r(
            """
            .stair_mclust_fit <- Mclust(
                data = .stair_mclust_data,
                G = as.integer(%d),
                modelNames = '%s'
            )
            """ % (int(num_cluster), model_name)
        )
        mclust_res = np.array(robjects.r(".stair_mclust_fit$classification"), dtype=np.int64)
    except embedded.RRuntimeError as exc:
        raise RuntimeError(
            "mclust execution failed in R. Please check R runtime and ensure package 'mclust' and base packages are available."
        ) from exc
    finally:
        try:
            robjects.r(
                "if (exists('.stair_mclust_fit', envir=.GlobalEnv)) rm('.stair_mclust_fit', envir=.GlobalEnv);"
                "if (exists('.stair_mclust_data', envir=.GlobalEnv)) rm('.stair_mclust_data', envir=.GlobalEnv)"
            )
        except embedded.RRuntimeError:
            pass

    adata.obs[key_add] = mclust_res.astype(str)
    return adata

def cluster_func(adata, clustering, use_rep, res=1, cluster_num=None, key_add='cluster', modelNames='EEE'):
    if clustering == 'louvain':
        sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_add)
        sc.tl.louvain(adata, resolution=res, neighbors_key=key_add, key_added=key_add)
    if clustering == 'leiden':
        sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_add)
        sc.tl.leiden(adata, resolution=res, neighbors_key=key_add, key_added=key_add)
    if clustering == 'kmeans':
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=cluster_num, random_state=2022).fit(adata.obsm[use_rep])
        adata.obs[key_add] = km.labels_
    if clustering == 'mclust':
        adata = mclust_R(adata, num_cluster=cluster_num, modelNames=modelNames, used_obsm=use_rep, random_seed=2022, key_add=key_add)
    adata.obs[key_add] = adata.obs[key_add].astype('category')
    return adata

