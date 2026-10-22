import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

# 继承你原有的对齐主类
from hypermoa.emb_alignment import Emb_Align
from hypermoa.embedding.loss import nll_loss

# 引入我们新建的多组学组件
from hypermoa.embedding.multi_module_ae import MultiOmics_ZINB_AE
from hypermoa.embedding.multi_dataset_ae import MultiOmicsDataset

class Multi_Emb_Align(Emb_Align):
    def __init__(self, adata, batch_key=None, hvg=False, n_hidden=128, n_latent=32, 
                 dropout_rate=0.2, likelihood="nb", device=None, num_workers=4, 
                 result_path=None, make_log=True, atac_key="ATAC",
                 encode_batch=False, decode_batch=True,
                 atac_decoder_activation='exp'):
        if hvg:
            hvg = min(int(hvg), adata.shape[1])
        
        # 调用父类初始化，自动配置日志、设备、基础信息
        super().__init__(adata, batch_key, hvg, n_hidden, n_latent, 
                         dropout_rate, likelihood, device, num_workers, result_path, make_log)
        
        # 补充多组学特定参数
        self.atac_key = atac_key
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.encode_batch = encode_batch
        self.decode_batch = decode_batch
        self.atac_decoder_activation = atac_decoder_activation
        n_input_rna = self.n_input # 父类已提取 RNA 维度
        n_input_atac = adata.obsm[atac_key].shape[1]
        
        # 覆盖父类的单组学 self.ae，替换为我们全新的多组学模型
        self.ae = MultiOmics_ZINB_AE(
            input_dim_rna=n_input_rna, 
            input_dim_atac=n_input_atac, 
            hidden_dim=n_hidden, 
            latent_dim=n_latent, 
            n_batch=self.n_batch, 
            dropout=dropout_rate,
            encode_batch=self.encode_batch,
            decode_batch=self.decode_batch,
            atac_decoder_activation=self.atac_decoder_activation,
        ).to(self.device)

    # 覆盖父类方法：使用多组学 Dataset
    def prepare(
        self,
        count_key=None,
        lib_size='explog',
        normalize=True,
        scale=False,
        atac_transform='log1p',
        atac_clip_nonnegative=True,
        atac_standardize=False,
    ):
        if self.hvg:
            if count_key is None:
                count_key = 'counts'
                if count_key not in self.adata.layers:
                    self.adata.layers[count_key] = self.adata.X.copy()

            hvg_top = min(int(self.hvg), self.adata.n_vars)
            try:
                sc.pp.highly_variable_genes(
                    self.adata,
                    n_top_genes=hvg_top,
                    layer=count_key,
                    flavor='seurat_v3',
                    batch_key=self.batch_key,
                    subset=False,
                )
                hvg_mask = np.asarray(self.adata.var['highly_variable'].values).astype(bool)
                hvg_flavor = 'seurat_v3'
            except Exception as exc:
                if self.make_log:
                    self.makeLog(
                        "  HVG selection with seurat_v3 failed; "
                        f"fallback to cell_ranger on log-normalized RNA. Detail: {exc}"
                    )

                adata_hvg = self.adata.copy()
                adata_hvg.X = adata_hvg.layers[count_key].copy()
                if normalize:
                    sc.pp.normalize_total(adata_hvg)
                sc.pp.log1p(adata_hvg)
                if scale:
                    sc.pp.scale(adata_hvg)
                sc.pp.highly_variable_genes(
                    adata_hvg,
                    n_top_genes=hvg_top,
                    flavor='cell_ranger',
                    batch_key=self.batch_key,
                    subset=False,
                )
                hvg_mask = np.asarray(adata_hvg.var['highly_variable'].values).astype(bool)
                hvg_flavor = 'cell_ranger'

            if int(hvg_mask.sum()) < 1:
                raise ValueError("HVG selection returned no genes.")

            self.adata = self.adata[:, hvg_mask].copy()
            self.n_input = self.adata.n_vars
            if self.ae.input_dim_rna != self.n_input:
                self.ae = MultiOmics_ZINB_AE(
                    input_dim_rna=self.n_input,
                    input_dim_atac=self.adata.obsm[self.atac_key].shape[1],
                    hidden_dim=self.n_hidden,
                    latent_dim=self.n_latent,
                    n_batch=self.n_batch,
                    dropout=self.dropout_rate,
                    encode_batch=self.encode_batch,
                    decode_batch=self.decode_batch,
                    atac_decoder_activation=self.atac_decoder_activation,
                ).to(self.device)
            if self.make_log:
                self.makeLog(f"  RNA HVG selection: {self.n_input} genes")
                self.makeLog(f"  RNA HVG flavor: {hvg_flavor}")
                self.makeLog(f"  RNA HVG batch key: {self.batch_key}")
        self.data_ae = MultiOmicsDataset(
            self.adata,
            atac_key=self.atac_key,
            count_key=count_key,
            batch_key=self.batch_key,
            size=lib_size,
            normalize=normalize,
            scale=scale,
            atac_transform=atac_transform,
            atac_clip_nonnegative=atac_clip_nonnegative,
            atac_standardize=atac_standardize,
        )
        if self.make_log:
            self.makeLog(f"  Library size: {lib_size}")
            self.makeLog(f"  Input normalize: {normalize}")
            self.makeLog(f"  Input scale: {scale}")
            self.makeLog(f"  Second modality key: {self.atac_key}")
            self.makeLog(f"  Second modality transform: {atac_transform}")
            self.makeLog(f"  Second modality clip nonnegative: {atac_clip_nonnegative}")
            self.makeLog(f"  Second modality standardize: {atac_standardize}")
            self.makeLog(f"  Second modality decoder activation: {self.atac_decoder_activation}")
            self.makeLog(f"  Encode batch in AE: {self.encode_batch}")
            self.makeLog(f"  Decode batch in AE: {self.decode_batch}")

    # 覆盖父类方法：联合训练双组学，纯手动控制损失权重
    def preprocess(self, lr=0.001, weight_decay=0, epoch_ae=100, batch_size=128, plot=False, 
                   loss_weight_rna=1.0, loss_weight_atac=1.0, atac_loss='mse'):
        """
        :param loss_weight_rna: RNA 的损失权重，默认 1.0
        :param loss_weight_atac: ATAC 的损失权重，默认 1.0。若 ATAC 损失数值较小被淹没，可手动调大（如 5.0 或 10.0）
        :param atac_loss: ATAC 分支损失类型，'mse'（默认，适合 LSI/PCA 连续特征）或 'nb'/'zinb'（适合原始计数）
        """
        data_loader = DataLoader(self.data_ae, shuffle=True, batch_size=batch_size, num_workers=self.num_workers)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr, weight_decay=weight_decay)   

        if self.make_log:
            self.makeLog(f"  [Manual Weighting] RNA Loss Weight: {loss_weight_rna}")
            self.makeLog(f"  [Manual Weighting] ATAC Loss Weight: {loss_weight_atac}")
            self.makeLog(f"  [ATAC Loss Type] {atac_loss}")

        train_loss = []
        epoch_loss_records = []
        for epoch in tqdm(range(0, epoch_ae)):
            loss_tmp = 0
            loss_rna_tmp = 0
            loss_atac_tmp = 0
            loss_atac_weighted_tmp = 0
            for i, batch_data in enumerate(data_loader):
                if self.batch_key is not None:
                    feat_r, count_r, size_r, feat_a, count_a, size_a, batch_t = [x.to(self.device) for x in batch_data]
                    out_rna, out_atac, _ = self.ae(feat_r, feat_a, batch_t)
                else:
                    feat_r, count_r, size_r, feat_a, count_a, size_a = [x.to(self.device) for x in batch_data]
                    out_rna, out_atac, _ = self.ae(feat_r, feat_a)

                optimizer.zero_grad()
                
                # --- RNA 损失计算 ---
                rate_scaled_r, logi_r, drop_r = out_rna
                rate_r = rate_scaled_r * size_r
                mean_r = rate_r * logi_r
                loss_rna = nll_loss(count_r, mean_r, rate_r, drop_r, dist=self.likelihood).mean()
                
                # --- ATAC 损失计算 ---
                rate_scaled_a, logi_a, drop_a = out_atac
                if atac_loss == 'mse':
                    # LSI/PCA 等连续表征不满足 NB/ZINB 计数分布，默认使用 MSE。
                    loss_atac = F.mse_loss(rate_scaled_a, feat_a)
                elif atac_loss in ('nb', 'zinb'):
                    rate_a = rate_scaled_a * size_a
                    mean_a = rate_a * logi_a
                    loss_atac = nll_loss(count_a, mean_a, rate_a, drop_a, dist=atac_loss).mean()
                else:
                    raise ValueError("atac_loss must be one of {'mse', 'nb', 'zinb'}")
                
                # =========================================================
                # 核心机制：纯手动静态加权计算总损失
                # =========================================================
                loss_train = (loss_weight_rna * loss_rna) + (loss_weight_atac * loss_atac)
                loss_train.backward()
                
                optimizer.step()
                loss_tmp += loss_train.item()
                loss_rna_tmp += loss_rna.item()
                loss_atac_tmp += loss_atac.item()
                loss_atac_weighted_tmp += (loss_weight_atac * loss_atac).item()
                
            n_batch = len(data_loader)
            epoch_total = loss_tmp / n_batch
            epoch_rna = loss_rna_tmp / n_batch
            epoch_atac = loss_atac_tmp / n_batch
            epoch_atac_weighted = loss_atac_weighted_tmp / n_batch
            train_loss.append(epoch_total)

            epoch_loss_records.append(
                {
                    'epoch': int(epoch + 1),
                    'loss_total': float(epoch_total),
                    'loss_rna_raw': float(epoch_rna),
                    'loss_atac_raw': float(epoch_atac),
                    'loss_atac_weighted': float(epoch_atac_weighted),
                    'loss_weight_rna': float(loss_weight_rna),
                    'loss_weight_atac': float(loss_weight_atac),
                    'atac_loss_type': str(atac_loss),
                }
            )

            if self.make_log:
                self.makeLog(
                    f"  AE Epoch {epoch + 1:03d}: total={epoch_total:.6f}, "
                    f"rna_raw={epoch_rna:.6f}, atac_raw={epoch_atac:.6f}, "
                    f"atac_weighted={epoch_atac_weighted:.6f}"
                )

        self.ae_loss_history = pd.DataFrame(epoch_loss_records)

        if self.result_path is not None:
            train_dir = os.path.join(self.result_path, 'embedding', 'train')
            os.makedirs(train_dir, exist_ok=True)

            csv_file = os.path.join(train_dir, 'ae_loss_components.csv')
            self.ae_loss_history.to_csv(csv_file, index=False)

            # Plot raw/weighted components together to check magnitude balance quickly.
            plt.figure(figsize=(7.0, 4.5))
            plt.plot(self.ae_loss_history['epoch'], self.ae_loss_history['loss_rna_raw'], label='loss_rna_raw')
            plt.plot(self.ae_loss_history['epoch'], self.ae_loss_history['loss_atac_raw'], label='loss_atac_raw')
            plt.plot(self.ae_loss_history['epoch'], self.ae_loss_history['loss_atac_weighted'], label='loss_atac_weighted')
            plt.plot(self.ae_loss_history['epoch'], self.ae_loss_history['loss_total'], label='loss_total', linestyle='--')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('AE pretraining loss components')
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(train_dir, 'ae_loss_components.png'), dpi=220)
            plt.close()

            if self.make_log:
                self.makeLog(f"  Saved AE loss components to: {csv_file}")
        self.ae.eval()

    # 覆盖父类方法：获取融合特征 Z
    def latent(self, batch_size=10000, return_data=False):
        self.ae.eval()
        batch_size = batch_size if batch_size is not None else len(self.data_ae)
        dataloader = DataLoader(self.data_ae, shuffle=False, batch_size=batch_size, num_workers=self.num_workers)

        z_list = []
        with torch.no_grad():
            for _, batch_data in enumerate(dataloader):
                if self.batch_key is not None:
                    feat_r, count_r, lib_r, feat_a, count_a, lib_a, batch_t = [x.to(self.device) for x in batch_data]
                    _, _, z_tmp = self.ae(feat_r, feat_a, batch_t)
                else:
                    feat_r, count_r, lib_r, feat_a, count_a, lib_a = [x.to(self.device) for x in batch_data]
                    _, _, z_tmp = self.ae(feat_r, feat_a)
                z_list.append(z_tmp.cpu()[:, :self.n_latent])

        z = torch.cat(z_list)
        # 将提取出的多组学融合特征放入 'latent'，实现无缝衔接。
        self.adata.obsm['latent'] = z.detach().cpu().numpy()

        if return_data:
            return self.adata

    def batch_center_obsm(self, source_key='HyperMOA', target_key=None, batch_key=None):
        if batch_key is None:
            batch_key = self.batch_key
        if target_key is None:
            target_key = f"{source_key}_bc"
        if batch_key is None:
            raise ValueError("batch_key is required for batch centering.")
        if source_key not in self.adata.obsm:
            raise KeyError(f"{source_key!r} not found in adata.obsm.")

        x = np.asarray(self.adata.obsm[source_key], dtype=np.float32)
        batches = self.adata.obs[batch_key].astype(str).values
        out = x.copy()
        global_mean = x.mean(axis=0, keepdims=True)
        for batch_name in np.unique(batches):
            idx = batches == batch_name
            out[idx] = x[idx] - x[idx].mean(axis=0, keepdims=True) + global_mean

        self.adata.obsm[target_key] = out
        if self.make_log:
            self.makeLog(f"  Batch-centered embedding: {source_key} -> {target_key}")
        return self.adata
