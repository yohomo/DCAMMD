import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.distances.mmd_distance import MMDDistance
from modules.layers.attention import Attention, Multi_Cross_Attention
from modules.utils.utils import Metaprompt, _l2norm, centering, triplet_loss, SupConLoss

# 【新增模块】跨图像双重交叉注意力权重生成模块 (AWGM)
class CrossImageAWGM(nn.Module):
    def __init__(self, dim, tau=0.5):
        super().__init__()
        self.tau = tau
        self.lam = nn.Parameter(torch.tensor(0.1)) # 可学习的残差控制标量 \lambda
        
        # 通道交叉注意力投影 (深度卷积)
        self.cca_q = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.cca_k = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.cca_v = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        
        # 空间交叉注意力投影 (深度卷积)
        self.sca_q = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.sca_k = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)

    def forward(self, f_q, f_s):
        # f_q: [B*Nq, C, H, W], f_s: [B*Ns, C, H, W]
        B_Nq, C, H, W = f_q.shape
        B_Ns = f_s.shape[0]
        P = H * W
        
        # 1. 通道交叉注意力块
        qc = self.cca_q(f_q).view(B_Nq, C, P) 
        kc = self.cca_k(f_s).view(B_Ns, C, P)
        vc = self.cca_v(f_s).view(B_Ns, C, P)
        
        # 跨图交互：此处使用全局均值作为引导键值
        kc_mean = kc.mean(dim=0, keepdim=True).expand(B_Nq, -1, -1)
        vc_mean = vc.mean(dim=0, keepdim=True).expand(B_Nq, -1, -1)
        
        # 通道注意力响应矩阵 Ac
        Ac = F.softmax(torch.bmm(qc, kc_mean.transpose(1, 2)) / P, dim=-1) 
        f_cca_q = torch.bmm(Ac, vc_mean).view(B_Nq, C, H, W)
        
        # 2. 空间交叉注意力块
        qs = self.sca_q(f_cca_q).view(B_Nq, C, P)
        ks = self.sca_k(f_s).view(B_Ns, C, P)
        ks_mean = ks.mean(dim=0, keepdim=True).expand(B_Nq, -1, -1)
        
        # 跨图空间匹配相似度矩阵 ℳ
        M = torch.bmm(qs.transpose(1, 2), ks_mean) / (self.tau * C) 
        
        # 3. 自适应离散概率质量生成 (联合行列池化)
        alpha = F.softmax(M.mean(dim=-1), dim=-1) # Query概率质量 [B_Nq, P]
        beta = F.softmax(M.mean(dim=-2), dim=-1)  # Support概率质量 [B_Nq, P]
        
        # 为适配后续接口，计算出平均 Support 分布
        beta_mean = beta.mean(dim=0, keepdim=True).expand(B_Ns, -1) # [B_Ns, P]

        # 4. 正交残差映射机制 (锚定度量空间)
        Fq_out = f_q  + self.lam * f_cca_q
        
        # 返回残差保护特征以及对应的归一化权重
        return Fq_out, alpha, beta_mean

class MMD(nn.Module):
    # (保留原有基类结构，以便兼容代码库其他调用)
    def __init__(self, in_channels, cfg, loss="ce", kernel="linear"):
        super().__init__()
        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.kernel = kernel
        self.l2_norm = cfg.model.mmd.l2norm
        self.loss = loss
        self.cfg = cfg
        self.feat_dim = in_channels
        self.num_groups = cfg.model.mmd.num_groups
        self.mmd = MMDDistance(cfg, kernel=self.kernel)
        
        if self.loss in ['ce', 'ce_triplet']:
            self.temperature = cfg.model.temperature
        if self.loss in ['triplet', 'ce_triplet']:
            self.threshold = cfg.model.tri_thres

    def compute_loss(self, mmd_dis, query_y):
        if self.loss == 'ce':
            loss = F.cross_entropy(-mmd_dis / self.temperature, query_y)
        elif self.loss == 'triplet':
            loss = triplet_loss(mmd_dis, query_y, thres=self.threshold)
        elif self.loss == 'ce_triplet':
            loss = 0.5 * F.cross_entropy(-mmd_dis / self.temperature, query_y) + triplet_loss(mmd_dis, query_y, thres=self.threshold)
        else:
            raise KeyError("loss is not supported")
        return loss

    def inference(self, support_xf, query_xf, query_y, beta=None, gamma=None):
        b, nq, nf, c = query_xf.size()
        support_xf, query_xf = centering(support_xf, query_xf)
        if self.l2_norm:
            support_xf = _l2norm(support_xf, dim=-1)
            query_xf = _l2norm(query_xf, dim=-1)

        if self.cfg.model.mmd.switch == "all_supports":
            support_xf = support_xf.reshape(b, self.n_way, -1, c) 
            mmd_dis = self.mmd(support_xf, query_xf, beta=beta, gamma=gamma).view(b * nq, -1)
        elif self.cfg.model.mmd.switch == "per_shot":
            mmd_dis = self.mmd(support_xf, query_xf, beta, gamma)
            mmd_dis = mmd_dis.view(b * nq, self.n_way, self.k_shot).mean(-1) 
        
        query_y = query_y.view(b * nq)
        if self.training:
            loss = self.compute_loss(mmd_dis, query_y)
            return {"MMD_loss": loss}
        else:
            _, predict_labels = torch.min(mmd_dis, 1)
            rewards = [1 if predict_labels[j] == query_y[j] else 0 for j in range(len(query_y))]
            return rewards

# 【核心修改模块】将原本的单图自注意力替换为 DCA AWGM
class AttentiveMMDPrompt(MMD):
    def __init__(self, in_channels, cfg, loss="ce_triplet", kernel="linear"):
        super().__init__(in_channels, cfg, loss, kernel)
        
        # 实例化基于初稿设计的跨图像注意力模块
        self.awgm = CrossImageAWGM(dim=self.feat_dim)

    def forward(self, support_xf, support_y, query_xf, query_y):
        b, ns, c, fh, fw = support_xf.shape
        nq = query_xf.shape[1]
        
        # 调整形状以适应卷积输入 [B, C, H, W]
        support_xf_in = support_xf.reshape(b * ns, c, fh, fw)
        query_xf_in = query_xf.reshape(b * nq, c, fh, fw)
        
        # 通过 AWGM 进行跨图交互、计算离散分布，并使用残差保护特征
        # alpha 为 Query 的分布质量，beta 为 Support 的分布质量
        query_xf_res, alpha, beta = self.awgm(query_xf_in, support_xf_in)
        
        # 特征恢复原形状并送入底层核距离计算
        query_xf_out = query_xf_res.view(b, nq, c, -1).permute(0, 1, 3, 2)
        support_xf_out = support_xf_in.view(b, ns, c, -1).permute(0, 1, 3, 2)
        
        # 调整分布权重的形状：[B, N, P, 1] 以对接底层 MMDDistance
        # 注意：此处源码的 beta 参数对应原版的 Support 权重，gamma 对应 Query 权重
        # 为了对应，我们传入 gamma=alpha.unsqueeze(-1), beta=beta.unsqueeze(-1)
        gamma_weight = alpha.reshape(b, nq, 1, -1, 1)
        way = len(torch.unique(support_y))
        shot = ns // way
        beta_weight = beta.reshape(b, 1, way, -1, 1) / shot
        
        return self.inference(support_xf_out, query_xf_out, query_y, beta=beta_weight, gamma=gamma_weight)

# 底层池化工具函数保持原样（兼容其他代码）
def adaptive_pool(features, attn_from):
    assert features.size() == attn_from.size()
    B, N, C, H, W = features.size()
    attention = torch.einsum('bnchw,bnc->bnhw',
                             [attn_from, nn.functional.adaptive_avg_pool2d(attn_from, (1, 1)).view(B, N, C)])
    attention = attention / attention.view(B, N, -1).sum(2).view(B, N, 1, 1).repeat(1, 1, H, W)
    attention = attention.view(B, N, 1, H, W)
    return (features * attention).view(B, N, C, -1).sum(3)

def adaptive_pool_new(features, attn_from):
    assert features.size() == attn_from.size()
    B, N, C, H, W = features.size()
    attention = torch.einsum('bnchw,bnc->bnhw',
                             [attn_from, nn.functional.adaptive_avg_pool2d(attn_from, (1, 1)).view(B, N, C)])
    attention = attention / (C ** 0.5)
    attention = F.softmax(attention, dim=-2)
    attention = attention.view(B, N, 1, H, W)
    return (features * attention).view(B, N, C, -1).sum(3)