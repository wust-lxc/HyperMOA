import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from scipy.spatial.distance import cdist

def calcu_adaptive_hyperedge(cord, feat, n_neigh=10, metric='minkowski', sim_threshold=0.3):
    '''
    Scheme B: 自适应空间超边 (Adaptive Spatial Hyperedge)
    逻辑：
    1. 找到空间上的 K 个最近邻居 (Spatial KNN)。
    2. 计算中心点与这些邻居的特征相似度 (Cosine Similarity)。
    3. 仅保留相似度 > sim_threshold 的邻居构成超边。
       (如果所有邻居都不相似，则至少保留自身，避免孤立)
    
    目的：
    - 保证空间局部性 (修复 Scheme A 的对齐漂移问题)。
    - 增强边界锐度 (不跨越不同组织区域进行平滑)。
    '''
    num_nodes = cord.shape[0]
    
    # 1. 空间 KNN 搜索
    # k = n_neigh + 1 (包含自身)
    neigh = NearestNeighbors(n_neighbors=n_neigh+1, metric=metric).fit(cord)
    neigh_dist, neigh_index = neigh.kneighbors(cord, return_distance=True)
    
    # 2. 准备特征用于计算相似度
    # feat: numpy array [N, D]
    # 我们需要计算每个中心点 i 与其邻居 j 的相似度
    
    # 将 numpy 转为 torch 计算更方便 (归一化用于余弦相似度)
    feat_tensor = torch.from_numpy(feat).float()
    feat_norm = F.normalize(feat_tensor, p=2, dim=1)
    
    node_list = []
    hyperedge_list = []
    
    # 为了效率，我们使用向量化操作而不是循环
    # 获取所有邻居的特征索引 [N, k]
    flat_neigh_index = neigh_index.flatten() # [N*k]
    
    # 中心点索引扩展 [N*k]
    center_indices = np.repeat(np.arange(num_nodes), n_neigh+1)
    
    # 提取特征
    center_feats = feat_norm[center_indices] # [N*k, D]
    neigh_feats = feat_norm[flat_neigh_index] # [N*k, D]
    
    # 计算点对相似度 (Cosine)
    # sum(A * B, dim=1)
    similarities = (center_feats * neigh_feats).sum(dim=1).numpy() # [N*k]
    
    # 3. 过滤与构建
    # 保留 (是自身) OR (相似度 > 阈值)
    # neigh_dist.flatten() == 0 通常是自身
    mask = (similarities > sim_threshold) | (np.abs(similarities - 1.0) < 1e-4)
    
    # 应用掩码
    final_nodes = flat_neigh_index[mask]
    final_hyperedges = center_indices[mask]
    
    hyperedge_index = np.vstack((final_nodes, final_hyperedges))
    
    return torch.LongTensor(hyperedge_index)

def calcu_adj(cord, cord2=None, neigh_cal ='knn', n_neigh = 8, n_radius=None, metric='minkowski'):
    '''
    Construct adjacency matrix with coordinates.
    '''
    if cord2 is None:
        cord2 = cord
        n_neigh += 1
    
    if neigh_cal == 'knn':
        neigh = NearestNeighbors(n_neighbors = n_neigh, metric = metric).fit(cord2)
        neigh_index = neigh.kneighbors(cord,return_distance=False)
        index = torch.LongTensor(np.vstack((np.repeat(range(cord.shape[0]),n_neigh),neigh_index.ravel())) ) 
    
    if neigh_cal == 'radius':
        neigh = NearestNeighbors(radius=n_radius, metric = metric).fit(cord2)
        neigh_index = neigh.radius_neighbors(cord, return_distance=False)
        index = np.array([[],[]], dtype=int)
        
        for it in range(cord.shape[0]):
            index = np.hstack(((index, np.vstack((np.array([it]*neigh_index[it].shape[0]), neigh_index[it])))))  
        index = torch.LongTensor(index)
    
    return index 

def get_high_sim_indices_blocked(x, y, threshold=0.9, block_size=10000):
    """
    分块计算两个矩阵之间的余弦相似度
    """
    n, d = x.shape
    m, _ = y.shape
    x_norm = x.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    y_norm = y.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    x_normalized = x / x_norm
    y_normalized = y / y_norm
    indices = []
    for i in range(0, n, block_size):
        start_i = i
        end_i = min(i + block_size, n)
        for j in range(0, m, block_size):
            start_j = j
            end_j = min(j + block_size, m)
            x_block = x_normalized[start_i:end_i]
            y_block = y_normalized[start_j:end_j]
            sim_block = torch.mm(x_block, y_block.t())
            high_sim_mask = sim_block > threshold
            i_indices, j_indices = high_sim_mask.nonzero().unbind(dim=1)
            i_indices += start_i
            j_indices += start_j
            indices.append(torch.stack([i_indices, j_indices], dim=1))
    if len(indices) > 0:
        indices = torch.cat(indices, dim=0)
    else:
        indices = torch.tensor([], dtype=torch.long).reshape(0, 2)
    return indices


def get_topk_sim_indices_blocked(x, y, top_k=30, threshold=None, block_size=10000):
    """
    Build cross-slice anchors by keeping at most top_k cosine-similar nodes in y
    for each node in x. If threshold is set and a node has no neighbor above it,
    keep its single best match so every node has at least one cross-slice anchor.
    """
    n, _ = x.shape
    m, _ = y.shape
    k_use = max(1, min(int(top_k), int(m)))

    x_norm = x.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    y_norm = y.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    x_normalized = x / x_norm
    y_normalized = y / y_norm

    indices = []
    for i in range(0, n, block_size):
        start_i = i
        end_i = min(i + block_size, n)
        x_block = x_normalized[start_i:end_i]
        sim_block = torch.mm(x_block, y_normalized.t())
        vals, cols = torch.topk(sim_block, k=k_use, dim=1)

        if threshold is None:
            mask = torch.ones_like(vals, dtype=torch.bool)
        else:
            mask = vals > threshold
            empty_rows = ~mask.any(dim=1)
            if bool(empty_rows.any()):
                mask[empty_rows, 0] = True

        rows = torch.arange(start_i, end_i, dtype=torch.long).unsqueeze(1).expand_as(cols)
        indices.append(torch.stack([rows[mask], cols[mask]], dim=1))

    if len(indices) > 0:
        return torch.cat(indices, dim=0)
    return torch.tensor([], dtype=torch.long).reshape(0, 2)

def hgat_data(adata, 
              batch_key, 
              batch_order = None,
              spatial_key ='spatial', 
              feat_key='latent',
              n_neigh_hom = 10, 
              n_neigh_het = None,
              c_neigh_het = 0.9, 
              kernal_thresh = 0.,
              sim_threshold=0.3):
    
    if batch_order is None:
        batch_order = list(adata.obs[batch_key].value_counts().sort_index().index)

    feat_dict = {}
    adj_dict = {}
    hyperedge_dict = {} 
    coord_dict = {}
    index_dict = {}

    for batch_tmp in batch_order:
        adata_tmp = adata[adata.obs[batch_key]==batch_tmp].copy()
        feat_tmp = adata_tmp.obsm[feat_key]
        coord_tmp = adata_tmp.obsm[spatial_key]
        
        # 1. 普通 KNN 边 (保持不变，用于 GAT)
        adj_tmp = calcu_adj(coord_tmp, 
                            neigh_cal = 'knn', 
                            n_neigh = n_neigh_hom, 
                            metric ='minkowski')
        
        # 2. 修改：使用自适应空间超边 (Scheme B)
        # 结合了空间邻近性和特征相似性
        # sim_threshold=0.0 表示只要有点正相关就保留，可适当调高如 0.3
        hyper_tmp = calcu_adaptive_hyperedge(coord_tmp, feat_tmp, 
                                             n_neigh=n_neigh_hom, 
                                             sim_threshold=sim_threshold)
        
        feat_dict[batch_tmp] = feat_tmp
        coord_dict[batch_tmp] = coord_tmp
        adj_dict[batch_tmp, '0', batch_tmp] = adj_tmp
        hyperedge_dict[batch_tmp] = hyper_tmp 
        index_dict[batch_tmp] = list(adata_tmp.obs_names)

    cross_adj_dict = {}
    for target_tmp in batch_order:
        for source_tmp in batch_order:
            if target_tmp != source_tmp:
                if (source_tmp, '1', target_tmp) in list(feat_dict.keys()):
                    cross_adj_dict[target_tmp, '1', source_tmp] = cross_adj_dict[source_tmp, '1', target_tmp][[1,0],:]
                else:    
                    if n_neigh_het is None:
                        indices = get_high_sim_indices_blocked(
                            torch.from_numpy(feat_dict[target_tmp]),
                            torch.from_numpy(feat_dict[source_tmp]),
                            threshold=c_neigh_het,
                            block_size=10000,
                        ).T
                    else:
                        indices = get_topk_sim_indices_blocked(
                            torch.from_numpy(feat_dict[target_tmp]),
                            torch.from_numpy(feat_dict[source_tmp]),
                            top_k=n_neigh_het,
                            threshold=c_neigh_het,
                            block_size=10000,
                        ).T
                    num_cols = indices.size(1)
                    if num_cols > 1000000:
                        cols_to_remove = torch.randperm(num_cols)[:num_cols-1000000]
                        indices = torch.index_select(indices, 1, torch.from_numpy(np.setdiff1d(np.arange(num_cols), cols_to_remove)))
                    cross_adj_dict[target_tmp, '1', source_tmp] = indices 

    adj_dict.update(cross_adj_dict)

    data = HeteroData()
    for ii in list(feat_dict.keys()):
        data[ii].x = torch.from_numpy(feat_dict[ii]).float()
        data[ii].hyperedge_index = hyperedge_dict[ii]

    for jj in list(adj_dict.keys()):
        data[jj].edge_index = adj_dict[jj]

    if kernal_thresh != 0:
        kernals = dict()
        for node_key, node_coord in coord_dict.items():
            dist_tmp = np.sqrt(cdist(node_coord, node_coord, metric='euclidean'))
            dist_tmp = dist_tmp/np.percentile(dist_tmp, kernal_thresh, axis=0)
            kernal_tmp = np.exp(-dist_tmp)
            kernal_tmp = torch.from_numpy(kernal_tmp).float()    
            kernal_tmp = 0.5 * (kernal_tmp + kernal_tmp.T)
            kernals[node_key] = kernal_tmp
    else:
        kernals = dict()

    return data, kernals, index_dict
