import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接引入你原有的全连接层组件，不破坏原有文件
from hypermoa.embedding.module_ae import FC_Layer

class MultiOmics_ZINB_AE(nn.Module):
    def __init__(
        self,
        input_dim_rna,
        input_dim_atac,
        hidden_dim,
        latent_dim,
        n_batch=None,
        dropout=0.2,
        encode_batch=False,
        decode_batch=True,
        atac_decoder_activation='exp',
    ):
        super(MultiOmics_ZINB_AE, self).__init__()
        
        self.n_batch = n_batch
        self.input_dim_rna = input_dim_rna
        self.input_dim_atac = input_dim_atac
        self.encode_batch = bool(encode_batch and n_batch is not None)
        self.decode_batch = bool(decode_batch and n_batch is not None)
        self.atac_decoder_activation = atac_decoder_activation
        
        input_dim_rna_all = input_dim_rna + n_batch if self.encode_batch else input_dim_rna
        input_dim_atac_all = input_dim_atac + n_batch if self.encode_batch else input_dim_atac
        latent_dim_all = latent_dim + n_batch if self.decode_batch else latent_dim
        
        # --- 1. 双分支编码器 (Dual Encoders) ---
        self.enc_rna_1 = FC_Layer(input_dim_rna_all, hidden_dim, bn=True, activate='relu', dropout=dropout)
        self.enc_rna_2 = FC_Layer(hidden_dim, hidden_dim, bn=True, activate='relu', dropout=dropout)
        
        self.enc_atac_1 = FC_Layer(input_dim_atac_all, hidden_dim, bn=True, activate='relu', dropout=dropout)
        self.enc_atac_2 = FC_Layer(hidden_dim, hidden_dim, bn=True, activate='relu', dropout=dropout)
        
        # --- 2. 跨组学特征融合 (Fusion MLP) ---
        self.fusion_layer = FC_Layer(hidden_dim * 2, latent_dim, bn=True, activate='relu', dropout=dropout)
        
        # --- 3. 双分支解码器 (Dual Decoders) ---
        self.dec_rna_1 = FC_Layer(latent_dim_all, hidden_dim, bn=False, activate='relu')
        self.dec_rna_disp = FC_Layer(hidden_dim, input_dim_rna, bn=False, activate='exp')
        self.dec_rna_drop = FC_Layer(hidden_dim, input_dim_rna, bn=False, activate='sigmoid')
        
        self.dec_atac_1 = FC_Layer(latent_dim_all, hidden_dim, bn=False, activate='relu')
        self.dec_atac_disp = FC_Layer(hidden_dim, input_dim_atac, bn=False, activate=atac_decoder_activation)
        self.dec_atac_drop = FC_Layer(hidden_dim, input_dim_atac, bn=False, activate='sigmoid')
        
        if self.n_batch is not None:
            self.layer_logi_rna = torch.nn.Parameter(torch.randn(input_dim_rna, self.n_batch))
            self.layer_logi_atac = torch.nn.Parameter(torch.randn(input_dim_atac, self.n_batch))
        else:
            self.layer_logi_rna = torch.nn.Parameter(torch.randn(input_dim_rna))
            self.layer_logi_atac = torch.nn.Parameter(torch.randn(input_dim_atac))

    def forward(self, x_rna, x_atac, batch_tensor=None):
        if self.encode_batch:
            x_rna = torch.cat((x_rna, batch_tensor), dim=-1)
            x_atac = torch.cat((x_atac, batch_tensor), dim=-1)
            
        h_rna = self.enc_rna_2(self.enc_rna_1(x_rna))
        h_atac = self.enc_atac_2(self.enc_atac_1(x_atac))
        
        # 特征拼接与融合
        h_concat = torch.cat((h_rna, h_atac), dim=-1)
        z = self.fusion_layer(h_concat)
        
        z_decode = torch.cat((z, batch_tensor), dim=-1) if self.decode_batch else z
        
        # 重构 RNA
        x3_rna = self.dec_rna_1(z_decode)
        rate_scaled_rna = self.dec_rna_disp(x3_rna)
        dropout_rna = self.dec_rna_drop(x3_rna)
        logi_rna = F.linear(batch_tensor, self.layer_logi_rna) if self.n_batch is not None else self.layer_logi_rna
        
        # 重构 ATAC
        x3_atac = self.dec_atac_1(z_decode)
        rate_scaled_atac = self.dec_atac_disp(x3_atac)
        dropout_atac = self.dec_atac_drop(x3_atac)
        logi_atac = F.linear(batch_tensor, self.layer_logi_atac) if self.n_batch is not None else self.layer_logi_atac
        
        return (rate_scaled_rna, logi_rna.exp(), dropout_rna), (rate_scaled_atac, logi_atac.exp(), dropout_atac), z
