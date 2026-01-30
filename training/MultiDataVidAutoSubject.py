# -*- coding: utf-8 -*-
"""睡眠分期 Cross-Modal Transformer 训练脚本（模块化注释版）

本文件在不改变你原有逻辑的前提下，补充了“模块化备注”：
- 通过大标题把脚本拆成：依赖/配置、通道定义、模型组件、数据与预处理、训练与验证、入口等模块；
- 在关键函数/关键逻辑处增加“为什么这么做/输入输出是什么/容易踩坑点”等注释；
- 尽量保持注释“少而关键”，避免把每一行都写满说明，影响可读性。

快速定位（按模块）：
  0) 环境配置与依赖导入
  1) 通道与临床通道定义
  2) 日志/输出重定向
  3) 张量重排/基础组件（Attention/PosEnc/Embedding）
  4) 跨模态显式 9 路 Cross-Attention
  5) 主模型 Epoch_Cross_Transformer_Network
  6) 指标函数
  7) Dataset 与数据读取预处理
  8) 可视化与混淆矩阵
  9) 训练：subject-level 10-fold 7:2:1 CV
 10) 验证/指标汇总
 11) 命令行参数与 main 入口

生成时间：2025-12-13 08:10:02
"""

# ==========================================================================================
# 模块 0：环境配置与依赖导入（warnings / torch / numpy / sklearn 等）
# ==========================================================================================

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch import optim as optim
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from torch.utils import data
import math
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import argparse
import glob
import os
import pandas as pd
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.nn import ModuleList, Linear, ReLU, Dropout, LayerNorm, MultiheadAttention
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, GroupKFold
import sys
import csv
from scipy import stats
import logging
import shutil
import traceback
from scipy.signal import welch, butter, filtfilt
from scipy.interpolate import interp1d
logging.basicConfig(level=logging.INFO)

# ==========================================================================================
# 模块 0.1：多进程 DataLoader 相关设置（sharing_strategy / start_method）
# ==========================================================================================

import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    try:
        mp.set_sharing_strategy('file_descriptor')
    except Exception:
        pass

# 可选：在调试时强制 spawn 起始方式（更稳定，开销略大）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ==========================================================================================
# 模块 1：通道与临床通道集合（用于通道选择/解释）
# ==========================================================================================

CHANNEL_NAMES = [
    'EEG Fp1-LER','EEG Fp2-LER','EEG F7-LER','EEG F8-LER','EEG F3-LER',
    'EEG F4-LER','EEG T3-LER','EEG T4-LER','EEG C3-LER','EEG C4-LER',
    'EEG T5-LER','EEG T6-LER','EEG P3-LER','EEG P4-LER','EEG O1-LER',
    'EEG O2-LER','EEG Fz-LER','EEG Cz-LER','EEG Pz-LER','EEG Oz-LER',
    'EOG Left Horiz', 'EOG Right Horiz', 'EMG Chin1', 'EMG Chin2', 'EMG Chin3'
]

CLINICAL_CHANNELS = {'EEG C3-LER', 'EEG C4-LER', 'EEG F3-LER',
                     'EEG F4-LER', 'EEG O1-LER', 'EEG O2-LER'}

subject_results = {}
# 重定向标准输出流到文件

# ==========================================================================================
# 模块 2：日志与输出重定向（把 stdout/stderr 写入 train_log.txt）
# ==========================================================================================

class Logger(object):
    """
    Robust stdout-to-file logger used to capture console output into train_log.txt.
    Methods: write, flush, close. Always flush after write to ensure data is on disk.
    """
    def __init__(self, filename="train_log.txt", mode="a", encoding="utf-8"):
        # keep a reference to the original terminal so prints still appear on console
        try:
            self.terminal = sys.stdout if sys.stdout is not None else sys.__stdout__
        except Exception:
            self.terminal = sys.__stdout__
        # open log file in append mode
        self.log = open(filename, mode, encoding=encoding)
        self._closed = False

    def write(self, message):
        # write to both terminal and file; guard against broken pipes
        try:
            if self.terminal is not None:
                try:
                    self.terminal.write(message)
                except Exception:
                    # fallback to system stdout
                    try:
                        sys.__stdout__.write(message)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            self.log.write(message)
        except Exception:
            # If write fails, ignore but keep running
            pass

    def flush(self):
        # flush both terminal and file
        try:
            if self.terminal is not None:
                try:
                    self.terminal.flush()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.log.flush()
            os.fsync(self.log.fileno())
        except Exception:
            pass

    def close(self):
        # flush and close file safely
        if getattr(self, "_closed", False):
            return
        try:
            self.flush()
        except Exception:
            pass
        try:
            self.log.close()
        except Exception:
            pass
        self._closed = True



# ==========================================================================================
# 模块 3：张量重排工具（注意：此处自定义 Rearrange 会遮蔽上面从 einops.layers.torch 导入的同名 Rearrange）
# ==========================================================================================

class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        return rearrange(x, self.pattern)

# Attention layer (simple)

# ==========================================================================================
# 模块 4：注意力/位置编码/嵌入等基础网络组件
# ==========================================================================================

class AttentionLayerSimple(nn.Module):
    def __init__(self, input_dim, att_dim, dropout=0.3):
        super().__init__()
        self.W1 = nn.Linear(input_dim, att_dim)
        self.W2 = nn.Linear(att_dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        e = self.W1(x)
        e = F.relu(e)
        e = self.dropout(e)
        e = self.W2(e)
        e = torch.tanh(e)
        e = self.v(e)  # [B, T, 1]
        alpha = F.softmax(e / 0.5, dim=1)  # [B, T, 1]
        return alpha


# 位置编码

# ==========================================================================================
# 模块 4.1：位置编码（sin/cos）
# ==========================================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) > self.pe.size(0):
            position = torch.arange(x.size(0)).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(2), 2) * (-math.log(10000.0) / self.pe.size(2)))
            new_pe = torch.zeros(x.size(0), 1, self.pe.size(2))
            new_pe[:, 0, 0::2] = torch.sin(position * div_term)
            new_pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', new_pe)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# 窗口嵌入

# ==========================================================================================
# 模块 4.2：窗口嵌入（多分支 1D-CNN + 全局融合 + CLS + PE）
# ==========================================================================================

class Window_Embedding(nn.Module):
    def __init__(self, in_channels: int = 25, window_size: int = 50, emb_size: int = 256, pool_size=128):
        super(Window_Embedding, self).__init__()
        self.in_channels = in_channels
        self.pool_size = pool_size
        self.emb_size = emb_size

        # 三个卷积分支
        self.projection_1 = nn.Sequential(
            nn.Conv1d(1, emb_size//4, kernel_size=window_size, stride=window_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(emb_size//4),
            nn.AdaptiveAvgPool1d(pool_size),
        )
        self.projection_2 = nn.Sequential(
            nn.Conv1d(1, emb_size//8, kernel_size=5, stride=5),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//8, emb_size//4, kernel_size=5, stride=5),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//4, (emb_size - emb_size//4)//2, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d((emb_size - emb_size//4)//2),
            nn.AdaptiveAvgPool1d(pool_size),
        )
        self.projection_3 = nn.Sequential(
            nn.Conv1d(1, emb_size//4, kernel_size=25, stride=25),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//4, (emb_size - emb_size//4)//2, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d((emb_size - emb_size//4)//2),
            nn.AdaptiveAvgPool1d(pool_size),
        )

        # 全局特征融合（保持）
        self.global_projection = nn.Sequential(
            nn.Conv1d(in_channels * emb_size, emb_size, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(emb_size),
            Rearrange('b e s -> b s e'),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.arrange1 = Rearrange('b s e -> s b e')
        self.pos = PositionalEncoding(d_model=emb_size)
        self.arrange2 = Rearrange('s b e -> b s e')

    def forward(self, x):
        # ---- 1) 规范化到 3D 张量 ----
        orig_shape = tuple(x.shape)
        if x.dim() == 3:
            # 可能是 [B, C, seq] 或 [B, seq, C]（后者要检测）
            # 不立刻假设，后面会检查 channels 与 self.in_channels
            pass
        elif x.dim() == 4:
            b, d1, d2, d3 = x.size()
            # 常见情形： [B,1,C,seq]
            if d1 == 1 and d2 > 1 and d3 > 1:
                x = x.squeeze(1)  # -> [B, C, seq]
            # [B,C,1,seq]
            elif d2 == 1 and d1 > 1:
                x = x.squeeze(2)  # -> [B, C, seq]
            # [B,C,seq,1]
            elif d3 == 1 and d2 > 1:
                x = x.squeeze(3)
            else:
                # 兜底：把中间维度合并为 channels（非常规，但避免空）
                B = x.size(0)
                seq = x.size(-1)
                rem = x.shape[1:-1]
                channels = 1
                for r in rem:
                    channels *= r
                x = x.reshape(B, channels, seq)
        elif x.dim() == 2:
            # [B, seq] -> [B, 1, seq]
            x = x.unsqueeze(1)
        else:
            # 更高维或异常，尝试合并中间维度
            try:
                B = x.size(0)
                seq = x.size(-1)
                rem = x.shape[1:-1]
                channels = 1
                for r in rem:
                    channels *= r
                x = x.reshape(B, channels, seq)
            except Exception as e:
                raise ValueError(f"Window_Embedding.forward: unsupported input shape {orig_shape}") from e

        # ---- 2) 现在应当是 [B, C, seq]，进一步自检与修正 ----
        batch_size, channels, seq_length = x.size()

        # 如果 channels 与预期 self.in_channels 不一致，尝试判断是否输入实际上为 [B, seq, C]
        if channels != getattr(self, "in_channels", None):
            # 如果最后一维等于预期通道数，则可能是 [B, seq, C]
            expected = getattr(self, "in_channels", None)
            if expected is not None and seq_length == expected:
                # 交换维度
                x = x.permute(0, 2, 1)  # -> [B, C, seq]
                batch_size, channels, seq_length = x.size()
            else:
                # 如果 channels==1 而最后一维等于 expected*something，也可能是错位，尝试兜底但打印信息
                if channels == 1 and seq_length >= expected:
                    # 可能是 [B,1, C*L] -> 尝试按 expected 拆分为 channels
                    if seq_length % expected == 0:
                        # 拆分：假设 seq_length = L * expected, 视为每通道有更短的序列
                        new_seq = seq_length // expected
                        x = x.view(batch_size, expected, new_seq)
                        batch_size, channels, seq_length = x.size()
                # 如果仍然不匹配，打印警告（不立即报错，给更友好的调试信息）
                if channels != expected:
                    # 输出调试信息，便于你快速定位
                    print("WARNING: Window_Embedding input channels mismatch.")
                    print(f"  original input shape: {orig_shape}")
                    print(f"  post-normalize shape: {tuple(x.shape)}")
                    print(f"  expected in_channels: {expected}")
                    # 继续执行，但如果 channels==0 会报错下文
        # ---- 3) 主处理逻辑 ----
        if channels <= 0:
            raise ValueError(f"Window_Embedding: after normalization channels=={channels}. Original shape: {orig_shape}. Check your data loader and slicing (e.g., eeg=x[:, :20, :])")

        channel_features = []
        for i in range(channels):
            # 单通道输入必须是 [B,1,seq] 才能给 Conv1d(in_channels=1)
            channel_data = x[:, i:i+1, :]  # [B,1,seq]
            feat1 = self.projection_1(channel_data)
            feat2 = self.projection_2(channel_data)
            feat3 = self.projection_3(channel_data)
            channel_feat = torch.cat([feat1, feat2, feat3], dim=1)  # [B, emb, pool]
            channel_features.append(channel_feat)

        # 防御：如果列表仍为空，抛出明确异常
        if len(channel_features) == 0:
            raise RuntimeError(f"Window_Embedding: no channel features created (channels={channels}, input_shape={orig_shape}).")

        # [B, C, emb, pool]
        channel_features = torch.stack(channel_features, dim=1)
        # 每通道平均时间维 => [B, C, emb]
        channel_emb = channel_features.mean(dim=-1)

        # 重构用于 global_projection
        channel_features = channel_emb.unsqueeze(-1).expand(-1, -1, -1, self.pool_size)
        combined_features = channel_features.reshape(batch_size, channels*self.emb_size, self.pool_size)

        x_out = self.global_projection(combined_features)  # [B, pool, emb]

        # cls token + pos encoding
        cls_tokens = repeat(self.cls_token, '() s e -> b s e', b=batch_size)
        x_out = torch.cat([cls_tokens, x_out], dim=1)  # [B, pool+1, emb]
        x_out = self.arrange1(x_out)
        x_out = self.pos(x_out)
        x_out = self.arrange2(x_out)

        return x_out, channel_emb, self.pool_size


# ==========================================================================================
# 模块 4.3：跨模态显式 9 路 Cross-Attention（EEG/EOG/EMG）
# ==========================================================================================

class ExplicitCrossAttention9Way(nn.Module):
    """
    对 EEG / EOG / EMG 三个模态做显式 9 路 cross-attention：

        - 每个模态有自己的一套 Q 投影；
        - 每个模态有自己的一套 K/V 投影；
        - 对于每个 Query 模态，都分别和 3 个 Key/Value 模态做一次 attention：
              EEG-Q  : (EEG-K,V), (EOG-K,V), (EMG-K,V)
              EOG-Q  : (EEG-K,V), (EOG-K,V), (EMG-K,V)
              EMG-Q  : (EEG-K,V), (EOG-K,V), (EMG-K,V)
          共 9 个 attention；
        - 每个 Query 模态用一个 learnable softmax 权重，把 3 条分支加权求和，
          再加 residual + FFN（标准 Transformer block 结构）。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # --- Q 投影（按 Query 模态区分）---
        self.q_eeg = nn.Linear(d_model, d_model)
        self.q_eog = nn.Linear(d_model, d_model)
        self.q_emg = nn.Linear(d_model, d_model)

        # --- K / V 投影（按 Key/Value 源模态区分）---
        self.k_eeg = nn.Linear(d_model, d_model)
        self.v_eeg = nn.Linear(d_model, d_model)

        self.k_eog = nn.Linear(d_model, d_model)
        self.v_eog = nn.Linear(d_model, d_model)

        self.k_emg = nn.Linear(d_model, d_model)
        self.v_emg = nn.Linear(d_model, d_model)

        # --- 每个 Query 模态自己的输出投影 ---
        self.out_eeg = nn.Linear(d_model, d_model)
        self.out_eog = nn.Linear(d_model, d_model)
        self.out_emg = nn.Linear(d_model, d_model)

        # --- 3 源模态的 mixing 权重（softmax 后相当于可学习的加权平均）---
        # 顺序约定为：from_eeg, from_eog, from_emg
        self.mix_eeg = nn.Parameter(torch.ones(3))
        self.mix_eog = nn.Parameter(torch.ones(3))
        self.mix_emg = nn.Parameter(torch.ones(3))

        self.dropout = nn.Dropout(dropout)

        # --- 每个模态自己的 LayerNorm + FFN（Transformer block）---
        self.norm_eeg_attn = nn.LayerNorm(d_model)
        self.norm_eog_attn = nn.LayerNorm(d_model)
        self.norm_emg_attn = nn.LayerNorm(d_model)

        self.ffn_eeg = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_eog = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_emg = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_eeg_ffn = nn.LayerNorm(d_model)
        self.norm_eog_ffn = nn.LayerNorm(d_model)
        self.norm_emg_ffn = nn.LayerNorm(d_model)

    # --------- 一些内部工具函数 ---------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] -> [B, H, S, d_head]
        B, S, D = x.shape
        H = self.n_heads
        d_h = self.d_head
        return x.view(B, S, H, d_h).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, S, d_head] -> [B, S, D]
        B, H, S, d_h = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * d_h)

    def _attend(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        Q: [B, S_q, D], K,V: [B, S_k, D]
        返回:
            out: [B, S_q, D]
            attn_weights: [B, H, S_q, S_k]
        """
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,S_q,S_k]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, Vh)                                             # [B,H,S_q,d_head]
        out = self._merge_heads(out)                                            # [B,S_q,D]
        return out, attn

    # --------- 前向：显式 9 路 cross-attention ---------
    def forward(self, eeg_seq, eog_seq, emg_seq):
        """
        eeg_seq, eog_seq, emg_seq: [B, S_m, D]
        返回:
            eeg_out, eog_out, emg_out: [B, S_m, D]  (已做 cross-attention + FFN)
            attn_info: dict，包含 9 个注意力矩阵
        """
        # 1) pre-norm（Transformer pre-norm 风格）
        eeg_in = eeg_seq
        eog_in = eog_seq
        emg_in = emg_seq

        eeg = self.norm_eeg_attn(eeg_in)
        eog = self.norm_eog_attn(eog_in)
        emg = self.norm_emg_attn(emg_in)

        # 2) 投影到各自的 Q/K/V
        Q_eeg = self.q_eeg(eeg)
        Q_eog = self.q_eog(eog)
        Q_emg = self.q_emg(emg)

        K_eeg = self.k_eeg(eeg)
        V_eeg = self.v_eeg(eeg)

        K_eog = self.k_eog(eog)
        V_eog = self.v_eog(eog)

        K_emg = self.k_emg(emg)
        V_emg = self.v_emg(emg)

        attn_info = {}

        # --- EEG 作为 Query ---
        eeg_from_eeg, att_ee_qe_ke = self._attend(Q_eeg, K_eeg, V_eeg)
        eeg_from_eog, att_ee_qe_ko = self._attend(Q_eeg, K_eog, V_eog)
        eeg_from_emg, att_ee_qe_km = self._attend(Q_eeg, K_emg, V_emg)

        # --- EOG 作为 Query ---
        eog_from_eeg, att_eo_qo_ke = self._attend(Q_eog, K_eeg, V_eeg)
        eog_from_eog, att_eo_qo_ko = self._attend(Q_eog, K_eog, V_eog)
        eog_from_emg, att_eo_qo_km = self._attend(Q_eog, K_emg, V_emg)

        # --- EMG 作为 Query ---
        emg_from_eeg, att_em_qm_ke = self._attend(Q_emg, K_eeg, V_eeg)
        emg_from_eog, att_em_qm_ko = self._attend(Q_emg, K_eog, V_eog)
        emg_from_emg, att_em_qm_km = self._attend(Q_emg, K_emg, V_emg)

        # 3) 对每个 Query 模态，把来自 3 个源模态的结果加权融合
        w_eeg = F.softmax(self.mix_eeg, dim=0)  # [3]
        w_eog = F.softmax(self.mix_eog, dim=0)
        w_emg = F.softmax(self.mix_emg, dim=0)

        eeg_comb = (
            w_eeg[0] * eeg_from_eeg
            + w_eeg[1] * eeg_from_eog
            + w_eeg[2] * eeg_from_emg
        )
        eog_comb = (
            w_eog[0] * eog_from_eeg
            + w_eog[1] * eog_from_eog
            + w_eog[2] * eog_from_emg
        )
        emg_comb = (
            w_emg[0] * emg_from_eeg
            + w_emg[1] * emg_from_eog
            + w_emg[2] * emg_from_emg
        )

        # 4) 输出投影 + residual
        eeg_attn_out = eeg_in + self.dropout(self.out_eeg(eeg_comb))
        eog_attn_out = eog_in + self.dropout(self.out_eog(eog_comb))
        emg_attn_out = emg_in + self.dropout(self.out_emg(emg_comb))

        # 5) 各自 FFN（完整 Transformer block）
        eeg_ffn_in = self.norm_eeg_ffn(eeg_attn_out)
        eog_ffn_in = self.norm_eog_ffn(eog_attn_out)
        emg_ffn_in = self.norm_emg_ffn(emg_attn_out)

        eeg_ffn_out = self.ffn_eeg(eeg_ffn_in)
        eog_ffn_out = self.ffn_eog(eog_ffn_in)
        emg_ffn_out = self.ffn_emg(emg_ffn_in)

        eeg_out = eeg_attn_out + self.dropout(eeg_ffn_out)
        eog_out = eog_attn_out + self.dropout(eog_ffn_out)
        emg_out = emg_attn_out + self.dropout(emg_ffn_out)

        # 6) 把 9 个注意力矩阵打包到 dict 里，方便之后分析
        attn_info.update({
            "eeg_q_eeg_k": att_ee_qe_ke,  # [B, H, S_eeg, S_eeg]
            "eeg_q_eog_k": att_ee_qe_ko,  # [B, H, S_eeg, S_eog]
            "eeg_q_emg_k": att_ee_qe_km,  # [B, H, S_eeg, S_emg]
            "eog_q_eeg_k": att_eo_qo_ke,  # [B, H, S_eog, S_eeg]
            "eog_q_eog_k": att_eo_qo_ko,  # [B, H, S_eog, S_eog]
            "eog_q_emg_k": att_eo_qo_km,  # [B, H, S_eog, S_emg]
            "emg_q_eeg_k": att_em_qm_ke,  # [B, H, S_emg, S_eeg]
            "emg_q_eog_k": att_em_qm_ko,  # [B, H, S_emg, S_eog]
            "emg_q_emg_k": att_em_qm_km,  # [B, H, S_emg, S_emg]
        })

        return eeg_out, eog_out, emg_out, attn_info


# ==========================================================================================
# 模块 5：主模型 Epoch_Cross_Transformer_Network（分模态嵌入 -> 通道注意力 -> 跨模态注意力 -> 分类）
# ==========================================================================================

class Epoch_Cross_Transformer_Network(nn.Module):
    """
    Add channel gates (per modality) + LLNM-like gated fusion (rho-regularizable).
    """
    def __init__(self, d_model=256, dim_feedforward=1024, window_size=50, eeg_channels=20, eog_channels=2, emg_channels=3, nhead_channel=4, nhead_time=8, num_classes=5, attn_dropout=0.1,):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Per-modality embeddings (kept from your original)
        self.eeg_embedding = Window_Embedding(in_channels=eeg_channels, window_size=window_size, emb_size=d_model)
        self.eog_embedding = Window_Embedding(in_channels=eog_channels, window_size=window_size, emb_size=d_model)
        self.emg_embedding = Window_Embedding(in_channels=emg_channels, window_size=window_size, emb_size=d_model)

        # Channel-level self-attention
        eeg_heads = pick_valid_nheads(d_model, nhead_channel)
        eog_heads_cap = max(1, min(nhead_channel, eog_channels))
        emg_heads_cap = max(1, min(nhead_channel, emg_channels))
        eog_heads = pick_valid_nheads(d_model, eog_heads_cap)
        emg_heads = pick_valid_nheads(d_model, emg_heads_cap)

        self.eeg_channel_attn = ChannelAttention(d_model=d_model, nhead=eeg_heads)
        self.eog_channel_attn = ChannelAttention(d_model=d_model, nhead=eog_heads)
        self.emg_channel_attn = ChannelAttention(d_model=d_model, nhead=emg_heads)


        # Intra-modal temporal attention
        self.intra_atten = Intra_model_atten(d_model=d_model, nhead=nhead_time)
        # 显式 9 路 cross-attention，用在 EEG/EOG/EMG token 序列上
        self.explicit_cross_attn = ExplicitCrossAttention9Way(
            d_model=d_model,      # 这里的 embed_dim 换成你 token 的维度
            n_heads=nhead_time,      # 和 self.intra_atten 用的一致
            dropout=attn_dropout,
        )

        self.spindle_fc = None
        cls_input_dim = d_model


        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x, n2_probs=None):
        # ---- 1) normalize input -> [B, C, T] ----
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() > 3:
            for d in reversed(range(1, x.dim() - 1)):
                if x.size(d) == 1:
                    x = x.squeeze(d)
        if x.dim() == 3:
            b, c, s = x.size()
            total_expected = (
                self.eeg_embedding.in_channels
                + self.eog_embedding.in_channels
                + self.emg_embedding.in_channels
            )
            if c not in (total_expected,):
                # 有些数据是 [B, T, C_total]
                if s == total_expected:
                    x = x.permute(0, 2, 1)
        elif x.dim() == 2:
            # [B, T] -> [B, 1, T]
            x = x.unsqueeze(1)

        # ---- 1.5) 初始化 attn_dict，后面所有统计信息都写进来 ----
        attn_dict = {}

        # ---- 2) split by modality ----
        C_eeg = self.eeg_embedding.in_channels
        C_eog = self.eog_embedding.in_channels
        C_emg = self.emg_embedding.in_channels
        total_ch = x.size(1)
        if total_ch < (C_eeg + C_eog + C_emg):
            C_eeg = min(C_eeg, total_ch)
            C_eog = max(0, min(C_eog, total_ch - C_eeg))
            C_emg = max(0, total_ch - C_eeg - C_eog)

        eeg = x[:, :C_eeg, :]
        eog = x[:, C_eeg:C_eeg + C_eog, :] if C_eog > 0 else None
        emg = x[:, C_eeg + C_eog:C_eeg + C_eog + C_emg, :] if C_emg > 0 else None

        # ---- 3) per-modality embeddings ----
        # Window_Embedding: 返回 (token_seq, channel_emb, pooled)
        eeg_seq, eeg_channel_emb, eeg_pool = self.eeg_embedding(eeg)  # [B, S_eeg, D], [B, C_eeg, D]

        if eog is not None and C_eog > 0:
            eog_seq, eog_channel_emb, eog_pool = self.eog_embedding(eog)  # [B, S_eog, D], [B, C_eog, D]
        else:
            eog_seq = torch.zeros(
                (eeg_seq.size(0), 1, eeg_seq.size(-1)),
                device=eeg_seq.device,
                dtype=eeg_seq.dtype,
            )
            eog_channel_emb = None

        if emg is not None and C_emg > 0:
            emg_seq, emg_channel_emb, emg_pool = self.emg_embedding(emg)  # [B, S_emg, D], [B, C_emg, D]
        else:
            emg_seq = torch.zeros(
                (eeg_seq.size(0), 1, eeg_seq.size(-1)),
                device=eeg_seq.device,
                dtype=eeg_seq.dtype,
            )
            emg_channel_emb = None

        # 记录 token 长度和 channel emb，方便之后的通道重要性分析
        attn_dict["eeg_seq_len"] = int(eeg_seq.size(1))
        attn_dict["eog_seq_len"] = int(eog_seq.size(1))
        attn_dict["emg_seq_len"] = int(emg_seq.size(1))
        attn_dict["eeg_channel_emb"] = eeg_channel_emb.detach()
        attn_dict["eog_channel_emb"] = (
            eog_channel_emb.detach() if eog_channel_emb is not None else None
        )
        attn_dict["emg_channel_emb"] = (
            emg_channel_emb.detach() if emg_channel_emb is not None else None
        )

        # ---- 4) channel self-attn & channel gates (per modality) ----
        eeg_chan_upd, attn_eeg = self.eeg_channel_attn(eeg_channel_emb)
        eog_chan_upd, attn_eog = (
            self.eog_channel_attn(eog_channel_emb)
            if eog_channel_emb is not None
            else (None, None)
        )
        emg_chan_upd, attn_emg = (
            self.emg_channel_attn(emg_channel_emb)
            if emg_channel_emb is not None
            else (None, None)
        )


        # ---- 5) 显式 9 路 cross-attention（EEG/EOG/EMG 三模态，两两交互） ----
        # 这里假定：
        #   eeg_seq: [B, S_eeg, D]，第0个 token 为 EEG 的 CLS
        #   eog_seq: [B, S_eog, D]，第0个 token 为 EOG 的 CLS
        #   emg_seq: [B, S_emg, D]，第0个 token 为 EMG 的 CLS
        eeg_feat_seq, eog_feat_seq, emg_feat_seq, cross_attn_info = self.explicit_cross_attn(
            eeg_seq, eog_seq, emg_seq
        )

        # 把 9 路 cross-attn 的权重全部记录下来（detach 防止显存和梯度问题）
        attn_dict["cross_attn"] = {k: v.detach() for k, v in cross_attn_info.items()}
        # 兼容旧 key（可选）
        if "eeg_q_eog_k" in cross_attn_info:
            attn_dict["cross_eeg2eog"] = cross_attn_info["eeg_q_eog_k"].detach()
        if "eog_q_eeg_k" in cross_attn_info:
            attn_dict["cross_eog2eeg"] = cross_attn_info["eog_q_eeg_k"].detach()

        # ---- 5.1) 根据 9 个注意力块拼一个完整的 fused time-attn 矩阵，方便通道重要性分析 ----
        try:
            S_eeg = eeg_feat_seq.size(1)
            S_eog = eog_feat_seq.size(1)
            S_emg = emg_feat_seq.size(1)
            S_total = S_eeg + S_eog + S_emg

            sample_att = next(iter(cross_attn_info.values()))
            B_att, H_att, _, _ = sample_att.shape
            full_time = sample_att.new_zeros(B_att, H_att, S_total, S_total)

            sl_eeg = slice(0, S_eeg)
            sl_eog = slice(S_eeg, S_eeg + S_eog)
            sl_emg = slice(S_eeg + S_eog, S_total)

            # EEG 作为 Query
            full_time[:, :, sl_eeg, sl_eeg] = cross_attn_info["eeg_q_eeg_k"]
            full_time[:, :, sl_eeg, sl_eog] = cross_attn_info["eeg_q_eog_k"]
            full_time[:, :, sl_eeg, sl_emg] = cross_attn_info["eeg_q_emg_k"]
            # EOG 作为 Query
            full_time[:, :, sl_eog, sl_eeg] = cross_attn_info["eog_q_eeg_k"]
            full_time[:, :, sl_eog, sl_eog] = cross_attn_info["eog_q_eog_k"]
            full_time[:, :, sl_eog, sl_emg] = cross_attn_info["eog_q_emg_k"]
            # EMG 作为 Query
            full_time[:, :, sl_emg, sl_eeg] = cross_attn_info["emg_q_eeg_k"]
            full_time[:, :, sl_emg, sl_eog] = cross_attn_info["emg_q_eog_k"]
            full_time[:, :, sl_emg, sl_emg] = cross_attn_info["emg_q_emg_k"]

            # 这两个 key 是 analyze_channel_importance 里用到的
            attn_dict["time"] = full_time.detach()  # [B, H, S_total, S_total]
            fused_tokens = torch.cat([eeg_seq, eog_seq, emg_seq], dim=1)
            attn_dict["pre_intra_tokens"] = fused_tokens.detach()
        except Exception:
            # 分析用的，不影响训练，出问题就跳过
            pass

        # ---- 6) 从 cross-attention 后的 token 序列池化出模态级时间表示 ----
        def pooled_non_cls(seq: torch.Tensor) -> torch.Tensor:
            # 严格按照“去掉 CLS 后对剩余时序平均”的方式
            return seq[:, 1:, :].mean(dim=1) if seq.size(1) > 1 else seq.mean(dim=1)

        eeg_repr_time = pooled_non_cls(eeg_feat_seq)   # [B, D]
        eog_repr_time = pooled_non_cls(eog_feat_seq)   # [B, D]
        emg_repr_time = pooled_non_cls(emg_feat_seq)   # [B, D]

        # 通道表示 + 时间表示 合成最终模态表示
        eeg_repr = eeg_repr_time
        eog_repr = eog_repr_time
        emg_repr = emg_repr_time

        feats = torch.stack([eeg_repr, eog_repr, emg_repr], dim=1)  # [B,3,D]
        fused_repr = feats.mean(dim=1)                              # [B,D]
        modality_gate = torch.ones( eeg_repr.size(0), 3, device=eeg_repr.device, dtype=eeg_repr.dtype ) / 3.0
        clf_in = fused_repr

        # ---- 7.6) 主分类头 ----
        logits = self.classifier(clf_in)

        # ---- 9) pack attn_dict（注意：这里用 update，不覆盖前面的 cross_attn/time 等）----
        base_attn = {
            "channel_eeg_attn": attn_eeg,
            "channel_eog_attn": attn_eog,
            "channel_emg_attn": attn_emg,
            "eeg_repr": eeg_repr.detach(),
            "eog_repr": eog_repr.detach(),
            "emg_repr": emg_repr.detach(),
            "modality_gate": modality_gate,  # [B,3]
        }
        attn_dict.update(base_attn)

        return logits, attn_dict



# ==========================================================================================
# 模块 5.1：头数选择辅助函数（确保 embed_dim % nheads == 0）
# ==========================================================================================

def pick_valid_nheads(embed_dim: int, cap: int) -> int:
    """
    Pick a valid number of heads h s.t. 1 <= h <= cap and embed_dim % h == 0.
    Preference: largest such h; fallback to 1.
    Works well with embed_dim like 256 (divisors: 1,2,4,8,16,32,64,128,256).
    """
    if cap < 1:
        return 1
    best = 1
    # 优先挑最大的有效因子（稳）
    for h in range(1, cap + 1):
        if embed_dim % h == 0:
            best = h
    return best



# ==========================================================================================
# 模块 5.2：通道维度 Self-Attention（每个模态内通道交互）
# ==========================================================================================

class ChannelAttention(nn.Module):
    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super().__init__()
        # batch_first=True: 输入为 [B, C, emb]
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, channel_emb):
        """
        channel_emb: [B, C, emb]
        返回:
            updated_emb: [B, C, emb]
            attn_weights: [B, num_heads, C, C]  (query_len = C, key_len = C)
        """
        attn_output, attn_weights = self.self_attn(channel_emb, channel_emb, channel_emb, need_weights=True, average_attn_weights=False)
        x = self.norm1(channel_emb + attn_output)
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + ff)
        return x, attn_weights  # attn_weights: [B, num_heads, C, C]

# 模态内注意力

# ==========================================================================================
# 模块 5.3：时序 Self-Attention（MultiheadAttention, 返回 per-head 权重）
# ==========================================================================================

class Intra_model_atten(nn.Module):
    def __init__(self, d_model=64, nhead=8, dropout=0.1, layer_norm_eps=1e-5, window_size=25, First=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, **factory_kwargs)
        self.linear1 = Linear(d_model, 4 * d_model, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(4 * d_model, d_model, **factory_kwargs)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src):
        # src: [batch, seq_len, d_model]
        src = src.permute(1, 0, 2)  # → [seq_len, batch, d_model]

        # Get attention with per-head weights
        attn_output, attn_scores = self.self_attn(src, src, src, average_attn_weights=False)
        # attn_scores: [num_heads, batch, query_len, key_len]

        # Residual + LayerNorm
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(attn_output))))
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Transpose back to [batch, seq_len, d_model]
        src = src.permute(1, 0, 2)

        return src, attn_scores  # attn_scores: [num_heads, batch, query_len, key_len]

# 前馈网络

# ==========================================================================================
# 模块 5.4：前馈网络（FFN Block）
# ==========================================================================================

class Feed_forward(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, dim_feedforward=512, layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout2 = Dropout(dropout)

    def forward(self, src: torch.Tensor):
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm(src)
        return src


# ==========================================================================================
# 模块 6：训练/评估指标（accuracy / kappa / g-mean 等）
# ==========================================================================================

def accuracy(pred, labels):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == labels).sum().item()
    total = labels.size(0)
    return correct / total

def kappa(pred, labels):
    preds = torch.argmax(pred, 1)
    return cohen_kappa_score(labels, preds)

def g_mean(sens, spec):
    return (sens * spec) ** 0.5

# 数据集类 - 修改为根据实际数据通道数处理

# ==========================================================================================
# 模块 7：数据集定义（标准化 + 多通道格式兼容）
# ==========================================================================================

class MASS_MultiChan_Dataset(Dataset):
    def __init__(self, data, labels, device, transform=None, target_transform=None):
        # 检查数据的实际通道数
        self.num_channels = data.shape[1]
        print(f"Data has {self.num_channels} channels")

        # 允许 6 通道（纯 EEG）、11 通道（三模态 6+2+3）和你现有的几种格式
        if self.num_channels in (3, 6, 11, 20, 22, 23, 25):
            self.data = data
        else:
            raise ValueError(
                f"Unexpected number of channels: {self.num_channels}. "
                f"Expected 3 / 6 / 11 / 20 / 22 / 23 / 25."
            )

        self.labels = labels
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

        # 计算数据统计信息用于标准化
        self.mean = np.mean(self.data, axis=(0, 2), keepdims=True)
        self.std = np.std(self.data, axis=(0, 2), keepdims=True)
        # 防止除零
        self.std[self.std < 1e-6] = 1e-6

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取样本并标准化
        eeg = (self.data[idx] - self.mean) / self.std  # 标准化
        label = self.labels[idx]

        eeg = torch.tensor(eeg, dtype=torch.float32).to(self.device)
        label = torch.tensor(label, dtype=torch.long).to(self.device)

        return eeg, label


# ==========================================================================================
# 模块 8：数据读取与预处理（h5 读取、下采样、通道裁剪、标签映射）
# ==========================================================================================

def load_all_epochs_one_subject(file_path, return_indices=False):
    """
    从单个 h5 里把该 subject 的所有 epoch 都读出来：
      data: [N, C, L]
      labels: [N]，已经：
        - 去掉 label==0
        - 映射 1..5 -> 0..4
      同时按照你现在的通道裁剪规则做通道选择
    """
    with h5py.File(file_path, "r") as f:
        keys = set(f.keys())
        if {"all_data", "all_labels"}.issubset(keys):
            data = f["all_data"][:]
            labels = f["all_labels"][:].flatten()
        elif {"train_val_data", "train_val_labels", "test_data", "test_labels"}.issubset(keys):
            data = np.concatenate(
                [f["train_val_data"][:], f["test_data"][:]],
                axis=0
            )
            labels = np.concatenate(
                [f["train_val_labels"][:], f["test_labels"][:]],
                axis=0
            ).flatten()
        elif {"data", "labels"}.issubset(keys):
            data = f["data"][:]
            labels = f["labels"][:].flatten()
        else:
            raise RuntimeError(f"No recognizable data/label keys in {file_path}")
    N_all = len(labels)
    orig_idx = np.arange(N_all, dtype=np.int64)
    # ---------- 统一时间长度：只允许 7680 或 15360 ----------
    L = data.shape[2]
    if L == 15360:
        data = data[:, :, ::2]
        print(f"[INFO] {os.path.basename(file_path)}: downsample 15360 -> {data.shape[2]} by step=2")
    elif L == 7680:
        pass
    else:
        raise ValueError(
            f"Unexpected epoch length {L} in {file_path}, "
            f"only support 7680 or 15360 (15360 will be downsampled to 7680)"
        )
    C = data.shape[1]
    print(f"[DEBUG] {os.path.basename(file_path)} raw channels = {C}")

    if C == 23:
        # NOTE: 这里的索引列表包含 23/24（0-based），如果你的原始 C==23(0..22) 会越界；
        #       需要确认原始 h5 的通道数到底是 23 还是 25，以及索引映射是否一致。
        # 你代码里的 0-based 索引：EEG6 + EOG2 + EMG3
        eeg_idx = [4, 5, 8, 9, 14, 15]        # F3,F4,C3,C4,O1,O2
        eog_idx = [20, 21]                    # EOG L,R
        emg_idx = [22, 23, 24]                # EMG 3 通道
        selected_idx = eeg_idx + eog_idx + emg_idx
        data = data[:, selected_idx, :]
        print(f"[INFO] {os.path.basename(file_path)} -> select 6 EEG + 2 EOG + 3 EMG")
    elif C in (6, 11):
        print(f"[INFO] {os.path.basename(file_path)} already {C} channels, use as is.")
    elif C in (20, 22, 25):
        print(f"[INFO] {os.path.basename(file_path)} {C} channels, keep full.")
    else:
        raise ValueError(
            f"Unexpected channels {C} in {file_path}, expected 6/11/20/22/23/25"
        )

    # ---------- 标签处理：去掉 0，映射 1..5 -> 0..4 ----------
    labels = np.array(labels, dtype=int)
    valid = labels != 0
    data = data[valid]
    labels = labels[valid]
    orig_idx = orig_idx[valid]
    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    labels = np.array([label_mapping.get(int(l), int(l)) for l in labels], dtype=int)

    if return_indices:
        return data, labels, orig_idx
    else:
        return data, labels


# ==========================================================================================
# 模块 9：可视化（混淆矩阵绘制与保存）
# ==========================================================================================

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True, save_path=None):
    import itertools

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))

    if normalize:
        # 按行归一化（计算每个真实类别的预测分布）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 按行求和
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=20)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=15)
        plt.yticks(tick_marks, target_names, fontsize=15)

    # 为每个单元格添加样本计数和百分比
    thresh = cm.max() / 2 if not normalize else 50  # 用于文本颜色区分
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            # 显示样本计数和百分比（确保每行总和为100%）
            cell_text = f"{int(cm[i, j])}\n{cm_normalized[i, j]:.1f}%"
        else:
            cell_text = f"{int(cm[i, j])}"

        plt.text(j, i, cell_text,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if (not normalize and cm[i, j] > thresh) or (normalize and cm_normalized[i, j] > thresh) else "black",
                 fontsize=13)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predictions', fontsize=18)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # 关闭图形，防止显示
    else:
        plt.show()

    return fig


# 自定义混淆矩阵函数

# ==========================================================================================
# 模块 9.1：混淆矩阵与派生指标计算（敏感度/特异度/F1/Precision）
# ==========================================================================================

def custom_confusion_matrix(pred, labels, num_classes, print_conf_mat=False):
    # 检查 pred 是否已经是类别索引
    if pred.dim() == 1:
        preds = pred
    else:
        preds = torch.argmax(pred, 1)

    conf_matrix = torch.zeros(num_classes, num_classes)
    avg_sensitivity = 0
    avg_specificity = 0
    avg_F1_score = 0
    avg_precision = 0
    sens_list = []
    spec_list = []
    F1_list = []
    precision_list = []

    for p, t in zip(preds, labels):
        if torch.is_tensor(p):
            p = p.item()
            t = int(t.item())
        conf_matrix[t, p] += 1
    if print_conf_mat == True:
        # print(conf_matrix)
        plot_confusion_matrix(cm=conf_matrix,
                              normalize=True,
                              target_names=['WAKE', 'N1', 'N2', 'N3', 'REM'],
                              title="Confusion Matrix (5-Class)")

        plt.show()

    TP = conf_matrix.diag()
    for c in range(num_classes):
        idx = torch.ones(num_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        if (TP[c] + FN) != 0:
            sensitivity = (TP[c] / (TP[c] + FN))
        else:
            sensitivity = 0

        if (TN + FP) != 0:
            specificity = (TN / (TN + FP))
        else:
            specificity = 0

        if ((2 * TP[c]) + (FN + FP)) != 0:
            F1_score = (2 * TP[c]) / ((2 * TP[c]) + (FN + FP))
        else:
            F1_score = 0

        if (TP[c] + FP) != 0:
            precision = (TP[c] / (TP[c] + FP))
        else:
            precision = 0

        sens_list.append(float(sensitivity))
        spec_list.append(float(specificity))
        F1_list.append(float(F1_score))
        precision_list.append(float(precision))

        avg_sensitivity += float(sensitivity)
        avg_specificity += float(specificity)
        avg_F1_score += float(F1_score)
        avg_precision += float(precision)

    label_accuracies = TP / conf_matrix.sum(dim=0)
    return conf_matrix, sens_list, spec_list, F1_list, precision_list, avg_sensitivity / 5, avg_specificity / 5, avg_F1_score / 5, avg_precision / 5, label_accuracies


# ==========================================================================================
# 模块 10：训练工具（滑动统计均值）
# ==========================================================================================

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ==========================================================================================
# 模块 10.1：Dataset 包装器（为每个样本附加 index，便于对齐 subject ids 等）
# ==========================================================================================

class DatasetWithIdx(torch.utils.data.Dataset):
    """
    Wrap existing dataset so that __getitem__ returns (x, label, idx, *rest)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        if isinstance(item, (tuple, list)):
            # ensure at least (x,label)
            x = item[0]
            lbl = item[1] if len(item) > 1 else None
            rest = item[2:] if len(item) > 2 else ()
            return (x, lbl, index, *rest)
        else:
            return (item, None, index)

    def __getattr__(self, name):
        """
        除了 __init__ / __len__ / __getitem__ 自己实现的属性外，
        其他属性都往 base_dataset 上转发，比如 num_channels / sampling_rate 等。
        """
        return getattr(self.base, name)


# ==========================================================================================
# 模块 11：训练主流程（subject-level 10-fold 7:2:1 CV + 重采样 + early stopping）
# ==========================================================================================

def subject_level_10fold_cv(device, args, n_folds=10):
    """
    在若干 h5 (每个 h5 视作一个 subject) 上做 10 折 7:2:1 的 subject-level CV：
      - 先把 subject 随机打乱分成 10 折，每折 3 个 subject（这里假定正好 30 个）
      - 每一轮：7 折 train, 2 折 val, 1 折 test

    本版本集成了：
      1) 训练集按类别做上下采样，使各类样本数接近平均（缓解类别不平衡）
      2) 交叉熵加入 class weight（进一步照顾 N1 等少数类）
      3) 训练轮数由 args.n_epochs 控制，并带 early stopping（基于验证集 acc）
      4) 测试阶段可选按 subject 做滑动窗口多数投票平滑预测（args.smooth_window > 1）
    """
    # ===================== 1. 找到要用的 30 个 subject 文件 =====================
    file_pattern = os.path.join(args.data_path, "01-03-00??-*.h5")
    all_files = glob.glob(file_pattern)

    def get_file_number(file):
        filename = os.path.basename(file)
        parts = filename.split('-')
        return int(parts[2]) if len(parts) >= 3 else 0

    all_files = sorted(all_files, key=get_file_number)

    # 可选：根据 start_file / end_file 进行粗筛
    if getattr(args, "start_file", None) is not None:
        try:
            start_num = int(args.start_file)
            all_files = [f for f in all_files if get_file_number(f) >= start_num]
        except Exception:
            pass
    if getattr(args, "end_file", None) is not None:
        try:
            end_num = int(args.end_file)
            all_files = [f for f in all_files if get_file_number(f) <= end_num]
        except Exception:
            pass

    # 这里假定筛完正好 30 个（50 岁以下那批）
    valid_files = all_files[:30]
    assert len(valid_files) == 30, f"期望 30 个 subject，目前找到 {len(valid_files)} 个"

    print(f"[INFO] subject-level CV on {len(valid_files)} subjects")
    for i, f in enumerate(valid_files):
        print(f"  [{i}] {os.path.basename(f)}")

    # ===================== 2. 把 30 个 subject 分成 10 折 =====================
    rng = np.random.RandomState(getattr(args, "seed", 42))
    indices = np.arange(len(valid_files))
    rng.shuffle(indices)

    folds = np.array_split(indices, n_folds)  # 每折大约 3 个 subject

    all_test_cms = []
    all_fold_metrics = []

    # ---------- 小工具：拼接若干 subject ----------
    def concat_subjects(file_list):
        """
        把若干 subject (h5 文件) 拼成一个大的 (data, labels, subj_ids)：
          - data: [N, C, T]
          - labels: [N]
          - subj_ids: [N]，每个 epoch 对应的“本轮里的 subject 序号”（0..len(file_list)-1）
        """
        data_list, label_list, subj_list = [], [], []
        for local_sid, fp in enumerate(file_list):
            x, y = load_all_epochs_one_subject(fp)
            data_list.append(x)
            label_list.append(y)
            subj_list.append(np.full_like(y, fill_value=local_sid))
        return (np.concatenate(data_list, axis=0),
                np.concatenate(label_list, axis=0),
                np.concatenate(subj_list, axis=0))

    def concat_subjects_with_n2(file_list, args, device):
        """
        返回:
        data: [N, C, T]
        labels: [N]
        subj_ids: [N]  当前 fold 内的 subject 编号 0...(len(file_list)-1)
        n2_probs: [N]  与 data/labels 对齐的 N2 概率
        """
        data_list, label_list, subj_list = [], [], []

        for local_sid, fp in enumerate(file_list):
            # 改用 return_indices=True
            x, y, idx_in_all = load_all_epochs_one_subject(fp, return_indices=True)
            data_list.append(x)
            label_list.append(y)
            subj_list.append(np.full_like(y, fill_value=local_sid))

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        subj_ids = np.concatenate(subj_list, axis=0)

        return data, labels, subj_ids

    # ---------- 小工具：按 subject 做多数投票平滑 ----------
    def smooth_predictions_by_subject(y_true, y_pred, subj_ids, num_classes=5, window=5):
        """
        对同一 subject 内的预测做滑动窗口多数投票平滑。
        - y_true 只用来对齐长度，不参与投票
        - window 为奇数更合理；若给偶数，会自动向下取最近的奇数
        """
        if window is None or window <= 1:
            return y_pred

        window = int(window)
        if window < 3:
            return y_pred
        if window % 2 == 0:
            window -= 1
        half = window // 2

        y_pred = np.asarray(y_pred, dtype=int)
        subj_ids = np.asarray(subj_ids, dtype=int)
        smoothed = y_pred.copy()

        unique_sids = np.unique(subj_ids)
        for sid in unique_sids:
            idx = np.where(subj_ids == sid)[0]
            if idx.size == 0:
                continue
            seq = y_pred[idx]
            sm_seq = seq.copy()
            for i in range(len(seq)):
                left = max(0, i - half)
                right = min(len(seq), i + half + 1)
                window_vals = seq[left:right]
                counts = np.bincount(window_vals, minlength=num_classes)
                sm_seq[i] = int(np.argmax(counts))
            smoothed[idx] = sm_seq
        return smoothed

    # ===================== 3. 10 折循环：7 折 train / 2 折 val / 1 折 test =====================
    for k in range(n_folds):
        test_fold = k
        val_folds = [(k + 1) % n_folds, (k + 2) % n_folds]       # 2 折验证
        train_folds = [i for i in range(n_folds)
                       if i not in ([test_fold] + val_folds)]    # 剩下 7 折训练

        train_subj_idx = np.concatenate([folds[i] for i in train_folds])
        val_subj_idx   = np.concatenate([folds[i] for i in val_folds])
        test_subj_idx  = folds[test_fold]

        train_files = [valid_files[i] for i in train_subj_idx]
        val_files   = [valid_files[i] for i in val_subj_idx]
        test_files  = [valid_files[i] for i in test_subj_idx]

        print(f"\n===== Subject-level CV round {k+1}/{n_folds} =====")
        print("Train subjects:", [os.path.basename(f) for f in train_files])
        print("Val   subjects:", [os.path.basename(f) for f in val_files])
        print("Test  subjects:", [os.path.basename(f) for f in test_files])

        # ---------- 4. 生成这一轮的 train/val/test ----------
        train_data, train_labels, train_subj_ids = concat_subjects_with_n2(train_files, args, device)
        val_data,   val_labels,   val_subj_ids   = concat_subjects_with_n2(val_files,   args, device)
        test_data,  test_labels,  test_subj_ids  = concat_subjects_with_n2(test_files,  args, device)

        print(f"[DEBUG] train {train_data.shape}, val {val_data.shape}, test {test_data.shape}")

        # ---------- 4.1 训练集类别重采样（上下采样到平均样本数） ----------
        train_labels_arr = np.asarray(train_labels, dtype=int)
        classes = np.unique(train_labels_arr)
        class_counts = np.array([(train_labels_arr == c).sum() for c in classes], dtype=np.int64)

        print(f"[INFO] Train class counts (before resample): {dict(zip(classes.tolist(), class_counts.tolist()))}")

        # 使用中位数作为参考
        median_count = float(np.median(class_counts[class_counts > 0]))

        max_up_factor   = getattr(args, "max_up_factor",   2.0)
        max_down_factor = getattr(args, "max_down_factor", 0.7)

        target_counts = {}
        for c, cnt in zip(classes, class_counts):
            cnt = float(cnt)
            if cnt < median_count:
                target = min(median_count, cnt * max_up_factor)
            else:
                target = max(median_count, cnt * max_down_factor)
            target_counts[c] = int(round(target))

        print("[INFO] target_counts:", target_counts)

        new_train_chunks = []
        new_label_chunks = []

        for c, cnt in zip(classes, class_counts):
            idxs = np.where(train_labels_arr == c)[0]
            tgt = target_counts[c]

            if cnt < tgt:
                extra = np.random.choice(idxs, size=(tgt - cnt), replace=True)
                chosen = np.concatenate([idxs, extra])
            elif cnt > tgt:
                chosen = np.random.choice(idxs, size=tgt, replace=False)
            else:
                chosen = idxs

            new_train_chunks.append(train_data[chosen])
            new_label_chunks.append(train_labels_arr[chosen])

        # 合并
        train_data = np.concatenate(new_train_chunks, axis=0)
        train_labels = np.concatenate(new_label_chunks, axis=0)

        print(f"[INFO] After balanced resample: train_data {train_data.shape}, train_labels {train_labels.shape}")
        # ---------- 5. 构造 Dataset / DataLoader ----------
        train_dataset = MASS_MultiChan_Dataset(train_data, train_labels, device)
        val_dataset   = MASS_MultiChan_Dataset(val_data,   val_labels,   device)
        test_dataset  = MASS_MultiChan_Dataset(test_data,  test_labels,  device)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                                shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                                shuffle=False, drop_last=False)

        # val/test 的 global_indices 就简单用 [0..len-1]
        val_global_indices  = np.arange(len(val_labels), dtype=np.int64)
        test_global_indices = np.arange(len(test_labels), dtype=np.int64)

        # ---------- 6. 初始化模型 ----------
        Net = Epoch_Cross_Transformer_Network(
            d_model=args.d_model,
            dim_feedforward=args.dim_feedforward,
            window_size=args.window_size,
        ).to(device)


        # ---------- 6.1 构造 class weight（改良版：原始分布 + gamma + clip） ----------

        num_classes = int(getattr(args, "num_classes", 5))

        # 使用“原始未重采样的 train_labels”来计算权重（避免平衡策略二次放大）
        train_labels_np = np.asarray(train_labels, dtype=int)

        counts_full = np.bincount(train_labels_np, minlength=num_classes).astype(np.float32)
        print("[INFO] Resampled train class counts :", counts_full.tolist())

        total = max(counts_full.sum(), 1.0)
        freq = counts_full / total  # 各类占比

        # gamma 越小，权重越平滑（不让少数类权重爆炸）
        gamma = float(getattr(args, "class_weight_gamma", 0.5))

        # 避免除零
        freq_safe = freq.copy()
        freq_safe[freq_safe == 0] = 1e-6

        # 平滑反比频率
        class_weights = (freq_safe ** (-gamma))

        # clip 避免极端爆炸
        w_min = float(getattr(args, "class_weight_min", 0.5))
        w_max = float(getattr(args, "class_weight_max", 3.0))
        class_weights = np.clip(class_weights, w_min, w_max)

        # 归一化：平均权重 = 1
        class_weights = class_weights / class_weights.mean()

        print("[INFO] Class weights used in CE    :", class_weights.tolist())

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)


        # ---------- 6.2 优化器 & 学习率调度 ----------
        optimizer = optim.Adam(Net.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(getattr(args, "step_size", 30)),
            gamma=getattr(args, "gamma", 0.5)
        )

        # ===================== 7. 训练 + Early Stopping =====================
        best_val_acc = 0.0
        best_state_dict = None
        patience = int(getattr(args, "early_stop_patience", 10))
        no_improve = 0

        for epoch in range(args.n_epochs):
            Net.train()
            train_loss_meter = AverageMeter()
            train_acc_meter = AverageMeter()

            for batch_idx, batch in enumerate(train_loader):
                eeg_batch, labels = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()
                outputs, _ = Net(eeg_batch.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_meter.update(loss.item(), n=eeg_batch.size(0))
                train_acc_meter.update(accuracy(outputs, labels),
                                       n=eeg_batch.size(0))

                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch+1}/{args.n_epochs}] "
                          f"Batch [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {train_loss_meter.val:.4f} (Avg {train_loss_meter.avg:.4f}) "
                          f"Acc: {train_acc_meter.val:.4f} (Avg {train_acc_meter.avg:.4f})")

            # ----- 验证 -----
            Net.eval()
            (val_loss, val_acc,
             macro_prec, macro_rec, macro_f1,
             macro_sens, macro_spec,
             micro_prec, micro_rec, micro_f1, micro_sens,
             weighted_prec, weighted_rec, weighted_f1,
             weighted_sens, weighted_spec,
             val_targets, val_preds) = validate(
                Net, val_loader, criterion, device,
                use_n2_input=False,
                n2_probs_all=None,
                val_global_indices=None,
                capture_attn=False
            )

            print(f"[VAL] Round {k+1} Epoch {epoch+1}/{args.n_epochs} "
                  f"Loss {val_loss:.4f}, Acc {val_acc:.4f}, Macro F1 {macro_f1:.4f}")

            # Early stopping：按验证集 acc 判断
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state_dict = Net.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            scheduler.step()

            if no_improve >= patience:
                print(f"[Early Stop] Round {k+1}: val_acc 连续 {patience} 个 epoch 无提升，停止训练。")
                break

        # 恢复最佳验证精度对应的参数
        if best_state_dict is not None:
            Net.load_state_dict(best_state_dict)

        # ===================== 7.1 在测试集上评估（可选平滑） =====================
        Net.eval()
        all_test_labels = []
        all_test_preds = []
        all_test_subj_ids = []

        with torch.no_grad():
            for batch in test_loader:
                # NOTE: 此处假设 batch 里含有 idx（batch[2]）用于对齐 test_subj_ids。
                #       但 MASS_MultiChan_Dataset.__getitem__ 当前只返回 (eeg, label)。
                #       如果你确实需要 idx，请用 DatasetWithIdx 包一层：
                #           test_loader = DataLoader(DatasetWithIdx(test_dataset), ...)
                eeg_batch, labels, idx = batch[0].to(device), batch[1].to(device), batch[2]
                outputs, _ = Net(eeg_batch.float())
                preds = torch.argmax(outputs, dim=1)

                all_test_labels.append(labels.cpu().numpy())
                all_test_preds.append(preds.cpu().numpy())
                idx_np = idx.cpu().numpy().astype(np.int64)
                all_test_subj_ids.append(test_subj_ids[idx_np])

        all_test_labels = np.concatenate(all_test_labels, axis=0)
        all_test_preds  = np.concatenate(all_test_preds, axis=0)
        all_test_subj_ids = np.concatenate(all_test_subj_ids, axis=0)

        # 测试阶段：若设置了平滑窗口，则对同一 subject 内的预测做多数投票平滑
        smooth_w = int(getattr(args, "smooth_window", 0))
        if smooth_w > 1:
            print(f"[INFO] Apply subject-wise smoothing with window={smooth_w}")
            all_test_preds = smooth_predictions_by_subject(
                all_test_labels, all_test_preds, all_test_subj_ids,
                num_classes=num_classes,
                window=smooth_w
            )

        # 计算这一折的混淆矩阵和指标
        test_cm = confusion_matrix(all_test_labels, all_test_preds, labels=list(range(num_classes)))
        total_samples = test_cm.sum()
        fold_acc = float(np.trace(test_cm) / total_samples) if total_samples > 0 else 0.0
        metrics_dict = calculate_metrics(test_cm)
        metrics_dict['accuracy'] = fold_acc

        print(f"[ROUND {k+1}] Test Confusion Matrix:\n{test_cm}")
        all_test_cms.append(test_cm)
        all_fold_metrics.append(metrics_dict)

    # ===================== 8. 汇总 10 折的混淆矩阵和指标 =====================
    total_cm = np.sum(all_test_cms, axis=0)
    print("\n===== 10-fold Subject-level CV Summary =====")
    print("Total Confusion Matrix:")
    print(total_cm)

    save_cm(total_cm, args.project_path, "subject10fold_total")

    accs         = [m.get('accuracy', 0.0) for m in all_fold_metrics]
    macro_f1s    = [m.get('macro', {}).get('f1', 0.0) for m in all_fold_metrics]
    weighted_f1s = [m.get('weighted', {}).get('f1', 0.0) for m in all_fold_metrics]

    print("\n===== 10-fold Subject-level Metrics Summary =====")
    print(f"Accuracy    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1    : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"Weighted F1 : {np.mean(weighted_f1s):.4f} ± {np.std(weighted_f1s):.4f}")

    os.makedirs(args.project_path, exist_ok=True)
    with open(os.path.join(args.project_path, "subject10fold_metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write("===== 10-fold Subject-level Metrics Summary =====\n")
        f.write(f"Accuracy    : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"Macro F1    : {np.mean(macro_f1s):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"Weighted F1 : {np.mean(weighted_f1s):.4f} ± {np.std(weighted_f1s):.4f}\n")

    return total_cm, all_fold_metrics


# ==========================================================================================
# 模块 12：结果落盘（混淆矩阵 txt/png）
# ==========================================================================================

def save_cm(cm, project_path, prefix):
    """保存混淆矩阵到文件和图像"""
    np.savetxt(f"{project_path}/{prefix}_confusion_matrix.txt", cm, fmt='%d')
    plot_confusion_matrix(cm,
                         target_names=['WAKE', 'N1', 'N2', 'N3', 'REM'],
                         title=f'{prefix.capitalize()} Confusion Matrix',
                         normalize=True,
                         save_path=f"{project_path}/{prefix}_confusion_matrix.png")
    plot_confusion_matrix(cm,
                         target_names=['WAKE', 'N1', 'N2', 'N3', 'REM'],
                         title=f'{prefix.capitalize()} Confusion Matrix (Non-normalized)',
                         normalize=False,
                         save_path=f"{project_path}/{prefix}_confusion_matrix_non_normalized.png")


# 验证函数

# ==========================================================================================
# 模块 13：验证流程（输出 loss/acc/宏微加权指标；可选收集 attention）
# ==========================================================================================


# NOTE: 当前实现无论 capture_attn True/False 都只返回 base_returns（18 项）。
#       若需要额外返回 attn_summary，需要在函数末尾拼接并 return。

def validate(Net, val_data_loader, criterion, device, use_n2_input=False, n2_probs_all=None, val_global_indices=None, capture_attn=False):
    """
    validate 现在支持 capture_attn 标志（默认 False）：
      - capture_attn=False: 返回之前相同的 18 项 tuple（不变）
      - capture_attn=True: 在原有返回的基础上，额外返回 attn_summary 字典作为最后一项
    """
    Net.eval()
    val_losses = AverageMeter()
    val_accuracy = AverageMeter()
    all_labels = []
    all_preds = []

    # 用于累积 attention（若需要）
    eeg_attn_list = []   # 存 per-batch EEG channel attn tensor
    eog_attn_list = []   # 存 per-batch EOG channel attn tensor
    generic_channel_attn_list = []  # 若模型只返回 'channel'
    cross_eeg2eog_list = []
    cross_eog2eeg_list = []

    with torch.no_grad():
        for batch in val_data_loader:
            # ---- 统一解包 ----
            if isinstance(batch, (list, tuple)):
                eeg = batch[0]
                labels = batch[1] if len(batch) >= 2 else None
                local_idx = batch[2] if (use_n2_input and len(batch) >= 3) else None
            else:
                eeg, labels, local_idx = batch, None, None

            eeg = eeg.to(device)
            labels = (labels.to(device) if labels is not None
                      else torch.zeros(eeg.shape[0], dtype=torch.long, device=device))

            # ---- 准备 n2（仅在需要时）----
            n2_tensor = None
            if (use_n2_input and n2_probs_all is not None and
                val_global_indices is not None and local_idx is not None):
                if isinstance(local_idx, torch.Tensor):
                    local_idx_np = local_idx.detach().cpu().numpy().astype(int)
                else:
                    local_idx_np = np.array(local_idx, dtype=int)
                global_idx = val_global_indices[local_idx_np]
                n2_vals = n2_probs_all[global_idx]
                n2_tensor = torch.from_numpy(n2_vals).float().to(device)

            # ---- 前向 ----
            if use_n2_input:
                outputs, attn_scores = Net(eeg.float(), n2_tensor)
            else:
                outputs, attn_scores = Net(eeg.float())

            loss = criterion(outputs, labels)
            # 记录
            val_losses.update(loss.item())
            val_accuracy.update(accuracy(outputs.cpu(), labels.cpu()))
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            # 收集 attention（如果 caller 要求）
            if capture_attn and isinstance(attn_scores, dict):
                # 支持多种 key 名称组合
                # 1) 如果有 'channel_eeg' 和 'channel_eog'
                if 'channel_eeg' in attn_scores and 'channel_eog' in attn_scores:
                    try:
                        eeg_attn_list.append(attn_scores['channel_eeg'].detach().cpu())
                        eog_attn_list.append(attn_scores['channel_eog'].detach().cpu())
                    except Exception:
                        pass
                # 2) 如果只有 'channel'（原始实现）
                elif 'channel' in attn_scores:
                    try:
                        generic_channel_attn_list.append(attn_scores['channel'].detach().cpu())
                    except Exception:
                        pass
                # 3) 跨模态 attn（如果存在）
                if 'cross_eeg2eog' in attn_scores:
                    try: cross_eeg2eog_list.append(attn_scores['cross_eeg2eog'].detach().cpu())
                    except Exception: pass
                if 'cross_eog2eeg' in attn_scores:
                    try: cross_eog2eeg_list.append(attn_scores['cross_eog2eeg'].detach().cpu())
                    except Exception: pass

    # 校验标签
    assert set(np.unique(all_labels)).issubset({0, 1, 2, 3, 4}), "Unmapped labels found!"
    assert set(np.unique(all_preds)).issubset({0, 1, 2, 3, 4}), "Unmapped predictions found!"

    # 混淆矩阵与指标
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
    metrics = calculate_metrics(conf_matrix)

    base_returns = (
        val_losses.avg,          #1
        val_accuracy.avg,        #2
        # Macro指标（3-7）
        metrics['macro']['precision'],  #3
        metrics['macro']['recall'],     #4
        metrics['macro']['f1'],         #5
        metrics['macro']['sensitivity'],#6
        metrics['macro']['specificity'],#7
        # Micro指标（8-11）
        metrics['micro']['precision'],  #8
        metrics['micro']['recall'],     #9
        metrics['micro']['f1'],         #10
        metrics['micro']['sensitivity'],#11
        # Weighted指标（12-16）
        metrics['weighted']['precision'],#12
        metrics['weighted']['recall'],   #13
        metrics['weighted']['f1'],       #14
        metrics['weighted']['sensitivity'],#15
        metrics['weighted']['specificity'],#16
        # 标签和预测（17-18）
        np.array(all_labels),   #17
        np.array(all_preds)     #18
    )
    return base_returns



# ==========================================================================================
# 模块 13.1：指标计算（macro/micro/weighted + per-class）
# ==========================================================================================

def calculate_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]
    sens_list = []
    spec_list = []
    F1_list = []
    precision_list = []
    support_list = []  # 每个类别的支持度（真实样本数）

    # 计算每个类别的指标
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        tn = np.sum(conf_matrix) - tp - fp - fn
        support = tp + fn  # 支持度（该类别的真实样本数）

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        sens_list.append(sensitivity)
        spec_list.append(specificity)
        F1_list.append(F1)
        precision_list.append(precision)
        support_list.append(support)
    total_tp = np.sum(np.diag(conf_matrix))
    total_fp = np.sum(np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))
    total_fn = np.sum(np.sum(conf_matrix, axis=1) - np.diag(conf_matrix))
    total_tn = np.sum(conf_matrix) - total_tp - total_fp - total_fn

    micro_sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_spec = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = micro_sens
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    total_support = np.sum(support_list)
    weighted_sens = np.sum([sens * support for sens, support in zip(sens_list, support_list)]) / total_support if total_support > 0 else 0
    weighted_spec = np.sum([spec * support for spec, support in zip(spec_list, support_list)]) / total_support if total_support > 0 else 0
    weighted_precision = np.sum([precision * support for precision, support in zip(precision_list, support_list)]) / total_support if total_support > 0 else 0
    weighted_recall = weighted_sens
    weighted_f1 = np.sum([f1 * support for f1, support in zip(F1_list, support_list)]) / total_support if total_support > 0 else 0
    macro_sens = np.mean(sens_list)
    macro_spec = np.mean(spec_list)
    macro_precision = np.mean(precision_list)
    macro_recall = macro_sens
    macro_f1 = np.mean(F1_list)

    # 返回值：在原有结构中补充sens/spec的三类平均（不改动原有key）
    return {
        'per_class': {
            'sensitivity': sens_list,
            'specificity': spec_list,
            'precision': precision_list,
            'f1': F1_list,
            'support': support_list
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
            'sensitivity': macro_sens,
            'specificity': macro_spec
        },
        'micro': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1,
            'sensitivity': micro_sens,
            'specificity': micro_spec
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1,
            'sensitivity': weighted_sens,
            'specificity': weighted_spec
        }
    }


# 解析命令行参数

# ==========================================================================================
# 模块 14：命令行参数解析（训练超参与路径）
# ==========================================================================================

def parse_option():
    parser = argparse.ArgumentParser('Argument for training')
    parser.add_argument('--project_path', type=str, default='c:\\Users\\WIN11\\Desktop\\Research\\CrossModel\\results\\subject',
                        help='Path to store project results')
    parser.add_argument('--data_path', type=str, default='D:\\data\\emgselected',
                        help='Path to the dataset directory')
    parser.add_argument('--model_type', type=str, default='Epoch', choices=['Epoch', 'Seq'], help='Model type')
    parser.add_argument('--d_model', type=int, default=256, help='Embedding size of the CMT')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='No of neurons feed forward block')
    parser.add_argument('--window_size', type=int, default=50, help='Size of non - overlapping window')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta 1 for adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta 2 for adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-9, help='eps for adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay for adam optimizer')
    parser.add_argument('--n_epochs', type=int, default=60, help='No of training epochs')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='验证集 acc 连续多少个 epoch 不提升就提前停止')
    parser.add_argument('--step_size', type=float, default=30, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for LR scheduler')
    parser.add_argument('--save_model_freq', type=int, default=10, help='Frequency to save the model')
    parser.add_argument('--start_file', type=int, default=3, help='Start processing from this file number (e.g., 9 for 01-01-0009-CLE.h5)')
    parser.add_argument('--end_file', type=int, default=59, help='End processing from this file number (e.g., 9 for 01-01-0009-CLE.h5)')
    parser.add_argument('--subject_level_cv', action='store_true',default=True, help='Use subject-level 10-fold 7:2:1 cross validation')
    parser.add_argument('--smooth_window', type=int, default=0, help='subject-level CV 测试阶段的滑动窗口平滑宽度(>1 时启用；0/1 表示不平滑)')
    parser.add_argument('--mode', type=str, default='train', choices=['startup', 'train'],
                        help='startup: 只跑启动期数据对齐/QC检查；train: 正常训练')
    parser.add_argument('--data_root', type=str, default='',
                        help='启动期用：数据根目录（用于自动生成 cohort_table.csv，可选）')
    parser.add_argument('--cohort_table', type=str, default='cohort_table.csv',
                        help='启动期用：cohort_table.csv 路径')
    parser.add_argument('--startup_out_dir', type=str, default='startup_reports',
                        help='启动期报告输出目录')
    opt = parser.parse_args()
    return opt


# ==========================================================================================
# 模块 15：主函数与脚本入口（日志、路径、调用 subject CV）
# ==========================================================================================

def main():
    args = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================== 启动期模式：生成 cohort_table + 跑对齐检查（训练前） =====================
    # 兼容：即使 parse_option 里还没加这些参数，也不会报错
    mode = getattr(args, "mode", "train")  # 'startup' or 'train'
    data_root = getattr(args, "data_root", "")
    cohort_table = getattr(args, "cohort_table", "cohort_table.csv")
    startup_out_dir = getattr(args, "startup_out_dir", "startup_reports")

    if mode == "startup":
        import subprocess
        from pathlib import Path

        # ---- 1) 解析路径：尽量兼容你把脚本放在不同目录的情况 ----
        this_dir = Path(__file__).resolve().parent

        def find_script(candidates):
            for p in candidates:
                p = (this_dir / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
                if p.exists() and p.is_file():
                    return str(p)
            return None

        # 允许脚本放在：同目录 / ../startup/tools / ../../startup/tools（你把 training/ 分开时也能找到）
        gen_script = find_script([
            "generate_cohort_table.py",
            "../startup/tools/generate_cohort_table.py",
            "../../startup/tools/generate_cohort_table.py",
        ])
        chk_script = find_script([
            "check_alignment.py",
            "../startup/tools/check_alignment.py",
            "../../startup/tools/check_alignment.py",
        ])

        if chk_script is None:
            raise FileNotFoundError(
                "找不到 check_alignment.py。请把它放到脚本同目录，或放到 startup/tools/ 下。"
            )

        # ---- 2) 输出目录 ----
        Path(startup_out_dir).mkdir(parents=True, exist_ok=True)

        # ---- 3) 若提供 data_root 且存在 generate_cohort_table.py，则先生成最小 cohort_table（默认10人）----
        if data_root:
            if gen_script is None:
                raise FileNotFoundError(
                    "你传了 --data_root，但找不到 generate_cohort_table.py。请把它放到同目录或 startup/tools/ 下。"
                )
            cmd_gen = [
                sys.executable, gen_script,
                "--data_root", str(data_root),
                "--out_csv", str(cohort_table),
                "--limit", "10"
            ]
            print("[STARTUP] generating cohort_table:\n  " + " ".join(cmd_gen))
            subprocess.check_call(cmd_gen)

        # ---- 4) 跑对齐检查（抽样10人）----
        cmd_chk = [
            sys.executable, chk_script,
            "--cohort_table", str(cohort_table),
            "--out_dir", str(startup_out_dir),
            "--n_samples", "10"
        ]
        print("[STARTUP] running alignment check:\n  " + " ".join(cmd_chk))
        subprocess.check_call(cmd_chk)

        print("[STARTUP] done. Reports saved to:", startup_out_dir)
        return
    # ===================== 启动期模式结束，下面是原训练逻辑（不变） =====================

    base_project_path = os.path.abspath(args.project_path)
    base_results_folder = os.path.dirname(os.path.dirname(base_project_path))
    os.makedirs(base_results_folder, exist_ok=True)

    subject_root = os.path.join(base_results_folder, "subjecttest")
    subject10fold_dir = os.path.join(subject_root, "subject10fold")
    os.makedirs(subject10fold_dir, exist_ok=True)

    # 固定项目路径为 subject-level 输出路径
    args.project_path = subject10fold_dir

    # 日志重定向
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    def safe_console_print(*p_args, **p_kwargs):
        try:
            print(*p_args, **p_kwargs, file=sys.__stdout__)
        except Exception:
            try:
                sys.__stdout__.write(" ".join(map(str, p_args)) + "\n")
            except Exception:
                pass

    log_path = os.path.join(args.project_path, "train_log.txt")
    logger_obj = None

    safe_console_print(f"[MAIN] device = {device}")
    safe_console_print(f"[MAIN] data_path = {args.data_path}")
    safe_console_print(f"[MAIN] project_path = {args.project_path}")
    safe_console_print(f"[MAIN] log_path = {log_path}")

    try:
        # 尝试重定向 stdout/stderr 到日志
        try:
            logger_obj = Logger(log_path)  # 你文件里已有 Logger 实现
            sys.stdout = logger_obj
            sys.stderr = logger_obj
        except Exception as e:
            safe_console_print(f"Warning: Could not redirect stdout to logger ({e}), proceeding with original stdout.")
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            logger_obj = None

        args.subject_level_cv = True

        print("\n===== Starting SUBJECT-LEVEL 10-fold CV =====")
        print(f"device: {device}")
        print(f"data_path: {args.data_path}")
        print(f"project_path: {args.project_path}")

        total_cm, all_metrics = subject_level_10fold_cv(device, args)

        print("\n===== Finished SUBJECT-LEVEL 10-fold CV =====")
        print("Done subject-level 10-fold CV.")
        return total_cm, all_metrics

    except Exception as e:
        safe_console_print(f"[MAIN-ERROR] {e}")
        try:
            traceback.print_exc(file=sys.__stdout__)
        except Exception:
            pass
        raise

    finally:
        try:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
        except Exception:
            pass

        if logger_obj:
            try:
                logger_obj.close()
            except Exception:
                pass


if __name__=="__main__":
    main()
