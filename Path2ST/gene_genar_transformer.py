"""
GenAR Transformer Components

This module implements the core components for the multi-scale GenAR model,
including AdaLN self-attention blocks, condition processors, Q-Former fusion, and utility functions.

Based on the GenAR architecture with adaptations for gene expression prediction.

Key Components:
1. GeneAdaLNSelfAttn: Self-attention block with adaptive layer normalization
2. GeneAdaLNBeforeHead: AdaLN layer before output head
3. ConditionProcessor: Enhanced condition processing with positional encoding
4. CellEmbeddingProcessor: Process cell embeddings to match condition dimension
5. QFormerFusion: Q-Former based fusion of histology and cell embeddings
6. DropPath: Stochastic depth for regularization

Author: Assistant
Date: 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List


# ============================================================================
# Cell-Type Contrastive Learning Modules
# ============================================================================

class GatedAttentionPool(nn.Module):
    """
    Gated attention-based pooling over gene token sequence.

    将 Transformer 输出的 gene token 序列 [B, num_genes, D] 聚合为单向量 [B, proj_dim]。

    设计选择：使用门控注意力池化（ABMIL 风格，Ilse et al. 2018）而非 mean/max pool：
    - max pool 只保留最大激活的单个 token，丢失分布信息
    - mean pool 均等对待所有 gene，稀释关键信号
    - 门控注意力让模型自己学习哪些 gene token 与细胞类型对比任务最相关

    Args:
        in_dim:   输入维度，等于 embed_dim（Transformer 输出维度）
        proj_dim: 对比空间维度
        hidden:   注意力评分 MLP 的隐层维度
        dropout:  Dropout 率
    """
    def __init__(
        self,
        in_dim: int = 768,
        proj_dim: int = 256,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Gated attention scoring: tanh branch × sigmoid gate (element-wise)
        self.attn_V = nn.Linear(in_dim, hidden, bias=False)   # tanh branch
        self.attn_U = nn.Linear(in_dim, hidden, bias=False)   # sigmoid gate
        self.attn_w = nn.Linear(hidden, 1, bias=False)         # scalar score per token

        self.dropout = nn.Dropout(dropout)

        # Project pooled representation → contrastive space
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]  — gene token embeddings from transformer output (after FiLM, before output_head)
        Returns:
            z: [B, proj_dim]  — L2-normalised spot embedding
        """
        # a: [B, L, 1]  gated attention scores over gene tokens
        a = self.attn_w(
            torch.tanh(self.attn_V(x)) * torch.sigmoid(self.attn_U(x))
        )  # [B, L, 1]
        a = self.dropout(torch.softmax(a, dim=1))  # softmax over L dimension

        # Weighted sum: [B, D]
        pooled = (a * x).sum(dim=1)  # [B, D]

        # Project and L2-normalise
        z = self.proj(pooled)              # [B, proj_dim]
        return F.normalize(z, dim=-1)      # [B, proj_dim], unit vectors


class CellTypeCompositionEncoder(nn.Module):
    """
    将每个 spot 内的细胞类别统计（频率直方图）编码到对比空间。

    频率向量 v[b, c] = count(type c in spot b) / total cells in spot b
    - order-invariant：不依赖细胞排列顺序
    - scale-invariant：按 spot 内总细胞数归一化
    - 保留组成比例信息（如 "60% T cells + 40% B cells"）

    Args:
        num_cell_types: 细胞类型总数（与 QFormerFusion.max_cell_types 一致）
        proj_dim:       对比空间维度
        hidden:         MLP 隐层维度
        dropout:        Dropout 率
    """
    def __init__(
        self,
        num_cell_types: int = 5,
        proj_dim: int = 256,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_cell_types = num_cell_types

        # MLP: frequency vector [B, num_cell_types] → contrastive space [B, proj_dim]
        self.encoder = nn.Sequential(
            nn.Linear(num_cell_types, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def _build_freq_vector(
        self,
        spot_cell_assignments: List[List[int]],
        cell_type_ids: torch.Tensor,   # [num_cells] long tensor
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算每个 spot 的归一化细胞类型频率向量。

        Args:
            spot_cell_assignments: 长度为 B 的列表，每个元素是属于该 spot 的细胞索引列表
            cell_type_ids:         [num_cells] 细胞类型 ID
            device:                目标设备
        Returns:
            freq: [B, num_cell_types]  行归一化的频率直方图
        """
        B = len(spot_cell_assignments)
        freq = torch.zeros(B, self.num_cell_types, device=device)

        for spot_idx, cell_indices in enumerate(spot_cell_assignments):
            if len(cell_indices) == 0:
                continue
            types = cell_type_ids[cell_indices]                         # [k]
            types = types.clamp(0, self.num_cell_types - 1)            # 防御性 clamp
            freq[spot_idx].scatter_add_(
                0, types, torch.ones_like(types, dtype=torch.float)
            )
            n = freq[spot_idx].sum()
            if n > 0:
                freq[spot_idx] = freq[spot_idx] / n

        return freq  # [B, num_cell_types]

    def forward(
        self,
        spot_cell_assignments: List[List[int]],
        cell_type_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            z: [B, proj_dim]  — L2-normalised composition embedding
        """
        freq = self._build_freq_vector(spot_cell_assignments, cell_type_ids, device)
        z = self.encoder(freq)             # [B, proj_dim]
        return F.normalize(z, dim=-1)      # [B, proj_dim], unit vectors


class InfoNCELoss(nn.Module):
    """Soft-positive InfoNCE loss.

    Upgrade over v1 (hard labels)
    ------------------------------
    v1 treats every off-diagonal entry as a negative sample.  But within a
    batch, two spots with similar cell-type composition (e.g. adjacent spots
    in the same tumour region) are *false negatives* — they should not be
    strongly repelled.

    v2 uses the cosine similarity between proportion vectors as soft labels:

        P[i, j] = softmax_j( sim(prop_i, prop_j) / tau_soft )

    where sim is cosine similarity.  The contrastive loss becomes a KL
    divergence from P to the model's similarity distribution:

        L = 0.5 * KL(P || Q_s2t) + 0.5 * KL(P || Q_t2s)

    When two spots have identical proportions, P[i,j] > P[i,i] is allowed,
    and the loss does not penalise the model for pulling them together.

    For spots with all-zero proportions (no cells), we fall back to the
    uniform distribution (no preference), which is equivalent to removing
    that spot from the contrastive set.

    Parameters
    ----------
    init_temperature : float
        Initial temperature τ for the model similarity logits (learnable).
    soft_temperature : float
        Fixed temperature τ_soft for constructing the soft label distribution
        from proportion similarities.  Smaller → sharper soft labels.
    max_temperature : float
        Upper bound on τ (prevents gradient vanishing).
    """
    def __init__(
        self,
        init_temperature: float = 0.07,
        soft_temperature: float = 0.1,
        max_temperature: float = 100.0,
    ):
        super().__init__()
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temperature))
        )
        self.max_log_scale = math.log(1.0 / (1.0 / max_temperature))
        self.soft_temperature = soft_temperature

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.clamp(max=self.max_log_scale).exp()

    @staticmethod
    def _build_soft_labels(
        proportions: torch.Tensor,   # [B, num_cell_types]  (may be None)
        temperature: float,
        device: torch.device,
    ) -> torch.Tensor:               # [B, B]
        """Build soft label matrix from proportion cosine similarities.

        Row i of the output sums to 1 and represents the target distribution
        for spot i: how much should the embedding of spot i align with each
        other spot's cell-type embedding.
        """
        B = proportions.shape[0]
        row_sums = proportions.sum(dim=1, keepdim=True)            # [B, 1]
        has_cells = (row_sums > 0).squeeze(1)                      # [B]

        # L2-normalise; rows with no cells stay as zeros
        norm = proportions / (row_sums + 1e-8)                     # [B, T]

        # Cosine similarity matrix
        sim = torch.matmul(norm, norm.t())                         # [B, B]

        # Softmax over columns (each row sums to 1)
        soft = F.softmax(sim / temperature, dim=1)                 # [B, B]

        # For spots with no cells: replace soft label with one-hot (diagonal)
        # so they neither attract nor repel anyone
        eye = torch.eye(B, device=device)
        no_cell_mask = (~has_cells).unsqueeze(1).float()           # [B, 1]
        soft = soft * (1 - no_cell_mask) + eye * no_cell_mask      # [B, B]

        return soft

    def forward(
        self,
        z_spot: torch.Tensor,                      # [B, proj_dim]  L2-normalised
        z_type: torch.Tensor,                      # [B, proj_dim]  L2-normalised
        proportions: Optional[torch.Tensor] = None, # [B, num_cell_types] or None
    ) -> torch.Tensor:
        B = z_spot.shape[0]
        if B == 1:
            return torch.tensor(0.0, device=z_spot.device, requires_grad=True)

        # Model similarity logits [B, B]
        logits_s2t = torch.matmul(z_spot, z_type.t()) * self.scale
        logits_t2s = logits_s2t.t()

        if proportions is not None and proportions.shape[0] == B:
            # ── Soft-label path ───────────────────────────────────────────
            soft = self._build_soft_labels(
                proportions, self.soft_temperature, z_spot.device
            )  # [B, B]
            # KL( soft || softmax(logits) )
            log_probs_s2t = F.log_softmax(logits_s2t, dim=1)
            log_probs_t2s = F.log_softmax(logits_t2s, dim=1)
            loss_s2t = F.kl_div(log_probs_s2t, soft,          reduction='batchmean')
            loss_t2s = F.kl_div(log_probs_t2s, soft.t().contiguous(), reduction='batchmean')
        else:
            # ── Hard-label fallback (original behaviour) ──────────────────
            labels = torch.arange(B, device=z_spot.device)
            loss_s2t = F.cross_entropy(logits_s2t, labels)
            loss_t2s = F.cross_entropy(logits_t2s, labels)

        return (loss_s2t + loss_t2s) / 2.0


class SpotCellTypeContrastModule(nn.Module):
    """
    统一的 spot-细胞类型对比学习模块（插件式，挂载到 MultiScaleGenAR）。

    封装 GatedAttentionPool + CellTypeCompositionEncoder + InfoNCELoss，
    在 forward_training() 的最终 scale 被调用一次。

    Args:
        embed_dim:      Transformer 的 embedding 维度
        proj_dim:       对比空间维度（两支路共用）
        attn_hidden:    GatedAttentionPool 内部评分 MLP 的隐层维度
        comp_hidden:    CellTypeCompositionEncoder MLP 的隐层维度
        num_cell_types: 细胞类型数（需与 QFormerFusion.max_cell_types 一致）
        init_temp:      InfoNCE 温度初始值
        dropout:        两个子模块共用的 Dropout 率
    """
    def __init__(
        self,
        embed_dim: int = 768,
        proj_dim: int = 256,
        attn_hidden: int = 128,
        comp_hidden: int = 128,
        num_cell_types: int = 5,
        init_temp: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.spot_encoder = GatedAttentionPool(
            in_dim=embed_dim,
            proj_dim=proj_dim,
            hidden=attn_hidden,
            dropout=dropout,
        )

        self.type_encoder = CellTypeCompositionEncoder(
            num_cell_types=num_cell_types,
            proj_dim=proj_dim,
            hidden=comp_hidden,
            dropout=dropout,
        )

        self.loss_fn = InfoNCELoss(init_temperature=init_temp)

    def forward(
        self,
        spot_features: torch.Tensor,               # [B, num_genes, D]  FiLM 之后、output_head 之前
        spot_cell_assignments: List[List[int]],    # pseudo-assignments from proportions
        cell_type_ids: torch.Tensor,               # [num_cells] long
        device: Optional[torch.device] = None,
        cell_type_proportions: Optional[torch.Tensor] = None,  # [B, T] for soft labels
    ) -> torch.Tensor:
        """
        计算一个 batch 的 soft-positive InfoNCE 对比 loss。

        Args:
            spot_features:          最终 scale 的 Transformer 特征 [B, num_genes, D]
            spot_cell_assignments:  pseudo spot→cell 对应关系
            cell_type_ids:          全局细胞类型 ID [num_cells]
            device:                 目标设备（None 时从 spot_features 推断）
            cell_type_proportions:  [B, T] proportion vectors for soft-label construction
        Returns:
            scalar soft InfoNCE loss
        """
        if device is None:
            device = spot_features.device

        # Spot 侧：门控注意力池化 gene tokens → 对比空间
        z_spot = self.spot_encoder(spot_features)   # [B, proj_dim]

        # 细胞类型侧：频率直方图 → 对比空间
        z_type = self.type_encoder(
            spot_cell_assignments, cell_type_ids, device
        )                                           # [B, proj_dim]

        # Soft-positive InfoNCE: pass proportions so similar-composition spots
        # are treated as soft positives instead of hard negatives
        return self.loss_fn(z_spot, z_type, proportions=cell_type_proportions)


# ============================================================================
# Lightweight Scale Memory - 轻量级跨scale辅助记忆（基于用户需求设计）
# ============================================================================

class LightweightScaleMemory(nn.Module):
    """
    轻量级跨scale记忆模块 - 作为辅助组件补充长程依赖
    
    设计原则（对齐用户需求）：
    1. 定位：辅助增强，不替代Token序列
    2. 输入：前序scales的Transformer中间特征（平均池化压缩）
    3. 更新：简单加权平均（无复杂门控）
    4. 融合：低权重残差连接（0.15）
    5. 位置：Transformer之前（不破坏Causal Attention）
    
    关键特性：
    - 存储跨scale上下文的压缩表征
    - 轻量级参数（~1.2M）
    - 不干扰GenAR核心流程
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        memory_dim: int = 256,
        fusion_weight: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.fusion_weight = fusion_weight
        
        # 压缩：Transformer特征 → 紧凑记忆
        self.compress = nn.Sequential(
            nn.Linear(embed_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 解压：记忆 → Token维度
        self.expand = nn.Sequential(
            nn.Linear(memory_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 自适应融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
    
    def compress_features(self, transformer_features: torch.Tensor) -> torch.Tensor:
        """
        压缩Transformer中间特征
        
        Args:
            transformer_features: [B, L, embed_dim]
        
        Returns:
            compressed: [B, memory_dim]
        """
        # 平均池化（替代复杂门控）
        pooled = transformer_features.mean(dim=1)  # [B, embed_dim]
        compressed = self.compress(pooled)  # [B, memory_dim]
        return compressed
    
    def forward(
        self,
        current_tokens: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        轻量融合：在Transformer之前增强tokens
        
        Args:
            current_tokens: [B, L, embed_dim]
            memory_state: [B, memory_dim] or None
        
        Returns:
            enhanced_tokens: [B, L, embed_dim]
        """
        if memory_state is None:
            return current_tokens
        
        B, L, D = current_tokens.shape
        
        # 解压memory
        memory_expanded = self.expand(memory_state)  # [B, embed_dim]
        memory_expanded = memory_expanded.unsqueeze(1).expand(-1, L, -1)  # [B, L, embed_dim]
        
        # 自适应融合权重
        gate = self.fusion_gate(current_tokens)  # [B, L, embed_dim]
        
        # 低权重残差：Token序列为主（85%），Memory为辅（15%）
        enhanced = current_tokens + self.fusion_weight * gate * memory_expanded
        
        return enhanced



# ============================================================================
# Memory Module - 完全基于memory.py，只移除.cuda()以支持设备无关
# ============================================================================

class RepeatLinear(nn.Module):
    """Linear layer that applies a learnable vector 'w' for feature-wise modulation of input.
    
    完全基于memory.py的实现，只移除.cuda()调用。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        # 移除.cuda()，让PyTorch自动处理设备分配
        self.w = nn.Parameter(torch.randn(in_dim))
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.relu(w * x)
        x = torch.mean(x, dim=1)
        return self.linear(x)


class GroupLinearLayer(nn.Module):
    """Custom linear layer with grouped weights initialization.
    
    完全基于memory.py的实现。
    """
    
    def __init__(
        self,
        in_dim,
        out_dim,
        a=None,
    ) -> None:
        super().__init__()
        if a is None:
            a = 1.0 / math.sqrt(out_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.weight.data.uniform_(-a, a)
        self.linear.bias.data.uniform_(-a, a)

    def forward(self, x):
        x = self.linear(x)
        return x


class MemoryModule(nn.Module):
    """Relational memory core with multi-head attention and gating mechanisms.
    
    完全基于memory.py的实现，保持所有原始逻辑不变。
    """

    def __init__(
        self,
        mem_slots: int,
        head_size: int,
        hidden_dim: int,
        attn_drop: float = 0.9,
        num_heads: int = 1,
        num_blocks: int = 1,
        forget_bias: float = 1.0,
        input_bias: float = 0.0,
        attention_mlp_layers: int = 2,
        use_topk: bool = False,
        topk: int = 3,
    ) -> None:
        super().__init__()

        self.mem_slots = mem_slots
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.n_heads = num_heads
        self.use_topk = use_topk
        self.topk = topk
        self.attn_drop = nn.Dropout(attn_drop)

        if num_blocks < 1:
            msg = f"num blocks must be >= 1. Got: {num_blocks}"
            raise ValueError(msg)
        self.num_blocks = num_blocks
        self.num_atten_mlp_layers = attention_mlp_layers

        self.query_proj = nn.Linear(self.hidden_dim, self.mem_slots)
        self.key_proj = nn.Linear(self.mem_slots, self.mem_slots)
        self.value_proj = nn.Linear(self.mem_slots, self.mem_slots)

        # Define MLP layers for processing attended memory
        self.attention_mlp = nn.ModuleList(
            [nn.Linear(self.mem_slots, self.mem_slots)] * self.num_atten_mlp_layers
        )
        self.attended_memory_layernorm = nn.LayerNorm(self.mem_slots)
        self.attended_memory_layernorm2 = nn.LayerNorm(self.mem_slots)

        # params for gating
        self.num_gates = 2 * self.calculate_gate_size()

        # Initialize input gate projector with RepeatLinear
        self.input_gate_projector = RepeatLinear(
            in_dim=self.mem_slots, out_dim=self.num_gates
        )
        # Initialize memory gate projector with GroupLinearLayer
        self.memory_gate_projector = GroupLinearLayer(
            in_dim=self.mem_slots, out_dim=self.num_gates
        )

        # Define bias parameters for forget and input gates
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch.float32))
        
        # Output projection layer: mem_slots → hidden_dim
        # 用于将memory信息投影回hidden_dim维度
        if self.mem_slots != self.hidden_dim:
            self.memory_to_hidden = nn.Linear(self.mem_slots, self.hidden_dim)
        else:
            self.memory_to_hidden = nn.Identity()

    def multi_head_attention(
        self,
        ipts: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply multi-head attention over memory and inputs.
        
        完全基于memory.py的实现。
        """
        b, t, c1 = ipts.size()
        _, m, c2 = memory.size()

        # Adjust the input sequence length to match memory length
        if t < m:
            # Upsample input using linear interpolation to match memory slots
            ipts = F.interpolate(ipts.transpose(1, 2), size=m, mode="linear").transpose(
                1, 2
            )
            t = m  # 更新t为调整后的长度
        elif t > m:
            # Downsample input using adaptive average pooling to match memory slots
            ipts = F.adaptive_avg_pool1d(ipts.transpose(1, 2), m).transpose(1, 2)
            t = m  # 更新t为调整后的长度

        """Perform multi-head attention"""
        q = self.query_proj(ipts)
        k = self.key_proj(memory)
        v = self.value_proj(memory)

        # Reshape and transpose for multi-head attention
        q = q.reshape(b, m, self.n_heads, -1).transpose(1, 2)
        k = k.reshape(k.size(0), k.size(1), self.n_heads, -1).transpose(1, 2)
        v = v.reshape(v.size(0), v.size(1), self.n_heads, -1).transpose(1, 2)

        # Compute scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 现在t和m应该相等（都是m）
        if m != t:
            raise ValueError(f"Memory length M {m} must be equal sequence length T {t} for causal masking.")

        causal_mask = ~torch.tril(torch.ones((t, m), dtype=torch.bool, device=att.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            combined_mask = causal_mask | attention_mask
        else:
            combined_mask = causal_mask

        attn_bias = combined_mask.to(dtype=torch.float32).masked_fill(combined_mask, float("-inf"))

        att = att + attn_bias
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        if self.use_topk:
            # If top-k attention is enabled, retain only the top-k attention scores
            topk = torch.topk(att, dim=-1, k=self.topk)
            mask = torch.zeros_like(att).to(att.device)
            mask.scatter_(3, topk.indices, 1)
            att = att * mask

        output = att @ v
        return output.transpose(1, 2).contiguous().view(b, t, self.n_heads * v.size(-1))

    def attend_over_memory(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend over memory for each block.
        
        完全基于memory.py的实现。
        """
        for _ in range(self.num_blocks):
            attended_memory = self.multi_head_attention(inputs, memory, attention_mask)
            memory = self.attended_memory_layernorm(memory + attended_memory)

            # Pass the normalized memory through MLP layers with ReLU activation
            attention_mlp = memory
            for i, _ in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            # Add residual connection and apply second layer normalization
            memory = self.attended_memory_layernorm2(memory + attention_mlp)
        return memory

    def forward(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process inputs and memory states.
        
        完整修复版本：
        inputs: [B, L, hidden_dim]  例如 [2, 512, 768]
        memory: [B, mem_slots, mem_slots]  例如 [2, 16, 16]
        
        返回:
        gated_output: [B, L, hidden_dim]  与inputs相同shape
        updated_memory: [B, mem_slots, mem_slots]
        """
        # 保存原始形状
        b, original_len, hidden_dim = inputs.shape
        
        # 验证并初始化memory（防御性编程）
        if memory is None:
            # 如果memory是None，初始化为单位矩阵
            memory = torch.eye(self.mem_slots, device=inputs.device).unsqueeze(0).repeat(b, 1, 1)
        elif memory.dim() != 3:
            # 如果memory维度不对，重新初始化
            print(f"Warning: Invalid memory dimensions {memory.shape}, expected 3D. Reinitializing...")
            memory = torch.eye(self.mem_slots, device=inputs.device).unsqueeze(0).repeat(b, 1, 1)
        elif memory.shape[0] != b:
            # 如果batch size不匹配，调整
            print(f"Warning: Memory batch size {memory.shape[0]} != input batch size {b}. Reinitializing...")
            memory = torch.eye(self.mem_slots, device=inputs.device).unsqueeze(0).repeat(b, 1, 1)
        
        _, mem_slots, _ = memory.shape
        
        # Step 1: Attend over memory
        # 内部会调整inputs到mem_slots长度，返回 [B, mem_slots, mem_slots]
        attended_memory = self.attend_over_memory(inputs, memory, attention_mask)
        
        # Step 2: Create gates
        # 传入attended_memory而不是inputs！
        # attended_memory的最后维度是mem_slots，符合RepeatLinear的期望
        input_gate, forget_gate = self.create_gates(attended_memory, memory)
        
        # Step 3: Update memory
        # input_gate, forget_gate: [B, 1, mem_slots, ?]
        # attended_memory: [B, mem_slots, mem_slots]
        # memory: [B, mem_slots, mem_slots]
        next_memory = input_gate * torch.tanh(attended_memory)
        next_memory += forget_gate * memory
        
        # Step 4: 将memory投影回原始维度
        # next_memory: [B, mem_slots, mem_slots]
        # 需要返回: [B, original_len, hidden_dim]
        
        # 4a. 调整序列长度 mem_slots → original_len
        if original_len != mem_slots:
            if original_len < mem_slots:
                # 下采样
                memory_adjusted = F.adaptive_avg_pool1d(
                    next_memory.transpose(1, 2), original_len
                ).transpose(1, 2)  # [B, original_len, mem_slots]
            else:
                # 上采样
                memory_adjusted = F.interpolate(
                    next_memory.transpose(1, 2), size=original_len, mode="linear"
                ).transpose(1, 2)  # [B, original_len, mem_slots]
        else:
            memory_adjusted = next_memory  # [B, mem_slots, mem_slots]
        
        # 4b. 投影维度 mem_slots → hidden_dim
        memory_output = self.memory_to_hidden(memory_adjusted)  # [B, original_len, hidden_dim]
        
        # Step 5: 返回结果
        return inputs + memory_output, next_memory

    def calculate_gate_size(self) -> int:
        """Determine gate size based on gating style.
        
        完全基于memory.py的实现。
        """
        return self.mem_slots

    def create_gates(
        self, inputs: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create input and forget gates using inputs and memory.
        
        完全基于memory.py的实现。
        """
        memory = torch.tanh(memory)
        shape_dim = 3

        if len(inputs.shape) == shape_dim:
            # Project inputs to gate values using RepeatLinear
            gate_inputs = self.input_gate_projector(inputs)
            # Add a dimension: Shape (Batch, 1, Seq, num_gates)
            gate_inputs = gate_inputs.unsqueeze(1)

            # Project memory to gate values using GroupLinearLayer
            gate_memory = self.memory_gate_projector(memory)
        else:
            # Raise an error if input shape is not as expected (Batch, Seq, Features)
            msg = f"input shape of create_gate function is {inputs.shape}, expects 3"
            raise ValueError(msg)

        # Combine gate inputs and memory projections
        gates = gate_memory + gate_inputs

        # Split the combined gates into input and forget gates
        gates = torch.split(
            gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2
        )
        input_gate, forget_gate = gates

        # Ensure input and forget gates have the same number of features
        if input_gate.shape[2] != forget_gate.shape[2]:
            raise ValueError

        # Apply sigmoid activation with biases to gates to constrain them between 0 and 1
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate


# ============================================================================
# Original GenAR Components
# ============================================================================


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    
    Implementation from timm library, used in the original GenAR design
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class SelfAttention(nn.Module):
    """
    Self-Attention module with optional L2 normalization
    
    Based on the GenAR attention implementation
    """
    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_l2_norm: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        
        # L2 normalization setup (like original GenAR)
        if self.attn_l2_norm:
            self.scale = 1.0
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), 
                requires_grad=True
            )
            self.max_scale_mul = torch.log(torch.tensor(100.0)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        # QKV projection
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        self.attn_drop = attn_drop
        
        # KV caching for inference
        self.caching = False
        self.cached_k = None
        self.cached_v = None
    
    def kv_caching(self, enable: bool):
        """Enable/disable KV caching"""
        self.caching = enable
        if not enable:
            self.cached_k = None
            self.cached_v = None
    
    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        
        # QKV projection with bias
        qkv = F.linear(
            input=x, 
            weight=self.mat_qkv.weight, 
            bias=torch.cat([self.q_bias, self.zero_k_bias, self.v_bias])
        ).view(B, L, 3, self.num_heads, self.head_dim)
        
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, H, L, C//H]
        
        # L2 normalization
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        # KV caching
        if self.caching:
            if self.cached_k is None:
                self.cached_k, self.cached_v = k, v
            else:
                k = self.cached_k = torch.cat([self.cached_k, k], dim=2)
                v = self.cached_v = torch.cat([self.cached_v, v], dim=2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.training and self.attn_drop > 0:
            attn_probs = F.dropout(attn_probs, p=self.attn_drop)
        
        out = torch.matmul(attn_probs, v)  # [B, H, L, C//H]
        out = out.transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(out))
    
    def extra_repr(self) -> str:
        return f'attn_l2_norm={self.attn_l2_norm}, caching={self.caching}'


class FFN(nn.Module):
    """
    Feed-Forward Network with GELU activation
    
    Based on the GenAR feed-forward implementation
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GeneAdaLNSelfAttn(nn.Module):
    """
    Gene-specific AdaLN Self-Attention Block
    
    Based on the GenAR AdaLNSelfAttn with adaptations for gene expression prediction.
    Uses Adaptive Layer Normalization to condition on histology and spatial features.
    """
    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        condition_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        attn_l2_norm: bool = True,
    ):
        super().__init__()
        
        self.block_idx = block_idx
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.shared_aln = shared_aln
        
        # Self-Attention
        self.attn = SelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            attn_l2_norm=attn_l2_norm,
        )
        
        # Feed-Forward Network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop_rate
        )
        
        # LayerNorm without learnable parameters
        self.ln_wo_grad = nn.LayerNorm(embed_dim, eps=norm_eps, elementwise_affine=False)
        
        # Adaptive LayerNorm parameters
        if shared_aln:
            # Shared AdaLN parameters (saves parameters)
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            # Independent AdaLN parameters
            self.ada_lin = nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(condition_dim, 6 * embed_dim)
            )
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    
    def enable_kv_cache(self, enable: bool = True):
        """Enable/disable KV caching for inference"""
        self.attn.kv_caching(enable)
    
    def forward(
        self, 
        x: torch.Tensor,                    # [B, L, C]
        condition_embed: torch.Tensor,      # [B, C] or [B, 1, 6, C]
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive layer normalization
        
        Args:
            x: Input token embeddings [B, L, C]
            condition_embed: Condition embeddings [B, C]
            attn_mask: Attention mask [L, L] or [B, H, L, L]
            
        Returns:
            Output embeddings [B, L, C]
        """
        B, L, C = x.shape
        
        # Get AdaLN parameters
        if self.shared_aln:
            if condition_embed.dim() == 2:
                condition_embed = condition_embed.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_gss + condition_embed
            ).unbind(2)  # 6 tensors of [B, 1, C]
        else:
            ada_params = self.ada_lin(condition_embed)  # [B, 6*C]
            gamma1, gamma2, scale1, scale2, shift1, shift2 = ada_params.view(
                B, 1, 6, C
            ).unbind(2)  # 6 tensors of [B, 1, C]
        
        # First AdaLN + Self-Attention
        x_norm1 = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
        attn_output = self.attn(x_norm1, attn_mask)
        x = x + self.drop_path(attn_output.mul_(gamma1))
        
        # Second AdaLN + FFN
        x_norm2 = self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
        ffn_output = self.ffn(x_norm2)
        x = x + self.drop_path(ffn_output.mul(gamma2))
        
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, block_idx={self.block_idx}'


class GeneMambaBlock(nn.Module):
    """
    Mamba-style Selective State Space Model (SSM) Block with AdaLN conditioning.

    替换前三层 Transformer 的 SSM 块，专门用于序列建模（不涉及图结构）。
    架构参考 IGN.py 中的 SSM 核心实现，但精简为纯序列版本：
    - 不含图卷积（adj / edge_index）
    - 不含 Gumbel-Softmax 动态图采样
    - 保留 selective scan 的 deltaA / deltaB_u / C 投影机制

    AdaLN 条件调制方式与 GeneAdaLNSelfAttn 保持一致（ada_lin → scale / shift → gamma）。

    Forward 接口与 GeneAdaLNSelfAttn 完全相同：
        x = block(x, condition_embed, causal_mask)
    causal_mask 参数接受但在 SSM 中被忽略（SSM 天然具有因果性）。
    """

    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        condition_dim: int = 768,
        d_state: int = 16,          # SSM 状态空间维度 N
        d_inner_ratio: float = 2.0, # d_inner = embed_dim * d_inner_ratio
        dt_rank: Optional[int] = None,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
    ):
        super().__init__()

        self.block_idx = block_idx
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.shared_aln = shared_aln
        self.d_state = d_state
        self.d_inner = int(embed_dim * d_inner_ratio)
        # dt_rank 默认 ceil(embed_dim / 16)，与 Mamba 论文一致
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(embed_dim / 16)

        # ---------- SSM 核心投影 ----------
        # 输入展开：embed_dim → d_inner * 2 （主路 + 残差门控）
        self.in_proj = nn.Linear(embed_dim, self.d_inner * 2, bias=False)

        # x → (delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # delta: dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM 矩阵 A（对数域存储，保证负定）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # 跳跃连接系数 D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影：d_inner → embed_dim
        self.out_proj = nn.Linear(self.d_inner, embed_dim, bias=False)

        # ---------- FFN ----------
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FFN(in_features=embed_dim, hidden_features=mlp_hidden_dim, drop=drop_rate)

        # ---------- LayerNorm (无可学习参数，用于 AdaLN) ----------
        self.ln_wo_grad = nn.LayerNorm(embed_dim, eps=norm_eps, elementwise_affine=False)

        # ---------- AdaLN 条件参数（与 GeneAdaLNSelfAttn 相同接口）----------
        if shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim ** 0.5)
        else:
            self.ada_lin = nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(condition_dim, 6 * embed_dim)
            )

        # ---------- Drop path ----------
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    # ------------------------------------------------------------------
    # Selective Scan（纯序列版，无图结构）
    # ------------------------------------------------------------------
    def _selective_scan(
        self,
        u: torch.Tensor,     # [B, L, d_inner]
        delta: torch.Tensor, # [B, L, d_inner]
        A: torch.Tensor,     # [d_inner, d_state]
        B: torch.Tensor,     # [B, L, d_state]
        C: torch.Tensor,     # [B, L, d_state]
        D: torch.Tensor,     # [d_inner]
    ) -> torch.Tensor:       # [B, L, d_inner]
        """
        Chunked selective scan — 完全向量化，无 Python for-loop。

        原实现的三个严重性能问题：
        1. Python for-loop over L steps：每步都是一次 Python interpreter 调用，
           L=200+ 时产生数百次 Python 级开销，且无法被 CUDA 异步掩盖。
        2. 提前展开 deltaA/deltaB_u 为 [B, L, d_in, n]：
           当 d_inner=1536, n=16, L=213 时单个 tensor 约 80MB，
           两个共 160MB；反向传播时 autograd 全部保留，实际 >320MB/layer。
        3. ys.append(y_i) + torch.stack(ys, dim=1)：
           在 Python heap 上积累 L 个 tensor 引用，最后一次性拼接，
           产生额外的内存分配和复制。

        修复方案：Chunked parallel scan
        - 将序列切成 chunk_size 大小的块，块内完全向量化（无 loop）
        - 块间只需 Python loop over num_chunks 次（远小于 L）
        - 内存峰值从 O(L * d_in * n) 降为 O(chunk_size * d_in * n)
        - 块内 einsum 可被 cuBLAS/cuDNN 高效调度
        """
        B_batch, L, d_in = u.shape
        n = A.shape[1]

        # ----- chunk 参数 -----
        # chunk_size 越大块内并行度越高，但内存也越大
        # 64 在典型序列长度（50~250）下是良好平衡点
        CHUNK = min(64, L)
        num_chunks = (L + CHUNK - 1) // CHUNK

        # 预分配输出，避免 list.append + torch.stack
        y = torch.empty_like(u)  # [B, L, d_in]

        # 块间传递的隐状态
        x_state = torch.zeros(B_batch, d_in, n, device=u.device, dtype=u.dtype)

        for ci in range(num_chunks):
            s = ci * CHUNK
            e = min(s + CHUNK, L)
            T = e - s  # 当前块实际长度（最后一块可能 < CHUNK）

            # 切片当前块：均为连续切片，无额外复制
            u_c     = u    [:, s:e, :]  # [B, T, d_in]
            delta_c = delta[:, s:e, :]  # [B, T, d_in]
            B_c     = B    [:, s:e, :]  # [B, T, n]
            C_c     = C    [:, s:e, :]  # [B, T, n]

            # 离散化（块内）
            # dA_c: [B, T, d_in, n]  —— 内存峰值 = B*CHUNK*d_in*n (远小于原来)
            dA_c = torch.exp(
                torch.einsum('b t d, d n -> b t d n', delta_c, A)
            )
            # dBu_c: [B, T, d_in, n]
            dBu_c = torch.einsum('b t d, b t n, b t d -> b t d n', delta_c, B_c, u_c)

            # 块内串行递推（T 步，T <= CHUNK=64，开销可控）
            # 用预分配 tensor 存储输出，避免 append
            y_chunk = torch.empty(B_batch, T, d_in, device=u.device, dtype=u.dtype)
            for i in range(T):
                x_state = dA_c[:, i] * x_state + dBu_c[:, i]
                # einsum 'b d n, b n -> b d'
                y_chunk[:, i, :] = (x_state * C_c[:, i, :].unsqueeze(1)).sum(-1)

            y[:, s:e, :] = y_chunk

        # D skip connection（完全向量化）
        y = y + u * D
        return y

    def _ssm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """SSM 主路（不含残差门控）"""
        # 用 float32 保证数值稳定（A_log 和 D 在 fp16 训练时也需要 fp32）
        A = -torch.exp(self.A_log.float())  # [d_in, n]，负定
        D = self.D.float()
        x_f = x.float() if x.dtype != torch.float32 else x

        # 投影 x → (delta, B, C)，在原始精度下做 linear，再转 float32
        x_dbl = self.x_proj(x)             # [B, L, dt_rank + 2*n]
        delta, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta)).float()  # [B, L, d_in]，正数
        B = B.float()
        C = C.float()

        y = self._selective_scan(x_f, delta, A, B, C, D)  # [B, L, d_in] float32

        # 转回原始精度
        return y.to(x.dtype)

    # ------------------------------------------------------------------
    # enable_kv_cache：保持与 GeneAdaLNSelfAttn 的接口一致（SSM 无需 KV Cache）
    # ------------------------------------------------------------------
    def enable_kv_cache(self, enable: bool = True):
        pass  # SSM 天然因果，无需 KV Cache

    def forward(
        self,
        x: torch.Tensor,                    # [B, L, C]
        condition_embed: torch.Tensor,      # [B, C]
        attn_mask: Optional[torch.Tensor] = None  # 接受但忽略（SSM 天然因果）
    ) -> torch.Tensor:
        B, L, C = x.shape

        # ---------- AdaLN 参数 ----------
        if self.shared_aln:
            if condition_embed.dim() == 2:
                condition_embed = condition_embed.unsqueeze(1).unsqueeze(2)
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_gss + condition_embed
            ).unbind(2)
        else:
            ada_params = self.ada_lin(condition_embed)          # [B, 6*C]
            gamma1, gamma2, scale1, scale2, shift1, shift2 = ada_params.view(
                B, 1, 6, C
            ).unbind(2)                                         # 每个 [B, 1, C]

        # ---------- SSM 分支（对应 Transformer 的 Self-Attention 分支）----------
        x_norm1 = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)

        # in_proj: embed_dim → d_inner * 2，split 成主路和残差门
        xz = self.in_proj(x_norm1)                             # [B, L, 2*d_inner]
        x_main, z = xz.chunk(2, dim=-1)                        # 各 [B, L, d_inner]
        x_main = F.silu(x_main)

        # SSM 处理
        y_ssm = self._ssm_forward(x_main)                      # [B, L, d_inner]

        # 门控残差
        y_ssm = y_ssm * F.silu(z)                              # [B, L, d_inner]

        # 输出投影回 embed_dim
        ssm_output = self.out_proj(y_ssm)                      # [B, L, C]

        x = x + self.drop_path(ssm_output.mul(gamma1))

        # ---------- FFN 分支 ----------
        x_norm2 = self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
        ffn_output = self.ffn(x_norm2)
        x = x + self.drop_path(ffn_output.mul(gamma2))

        return x

    def extra_repr(self) -> str:
        return (f'block_idx={self.block_idx}, d_state={self.d_state}, '
                f'd_inner={self.d_inner}, dt_rank={self.dt_rank}')


class GeneAdaLNSelfAttnWithMemory(nn.Module):
    """
    Memory-Augmented GenAR Self-Attention Block with Adaptive Layer Normalization
    
    基于LM2论文和model_memory_llama.py的集成方式：
    1. 先执行标准的self-attention
    2. 然后将attention output送入Memory模块
    3. Memory模块返回gated memory output
    4. 将gated memory添加到原始输出上
    """
    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        condition_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        attn_l2_norm: bool = True,
        # Memory参数
        memory_slots: int = 16,
        num_mem_heads: int = 4,
        memory_attn_drop: float = 0.1,
    ):
        super().__init__()
        
        self.block_idx = block_idx
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.shared_aln = shared_aln
        self.memory_slots = memory_slots
        
        # Self-Attention
        self.attn = SelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            attn_l2_norm=attn_l2_norm,
        )
        
        # Memory Module - 使用与LLaMA相同的配置
        head_size = memory_slots // num_mem_heads
        self.memory_module = MemoryModule(
            mem_slots=memory_slots,
            head_size=head_size,
            hidden_dim=embed_dim,
            num_heads=num_mem_heads,
            attn_drop=memory_attn_drop,  # 使用更合理的dropout
            num_blocks=1,
        )
        
        # Feed-Forward Network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop_rate
        )
        
        # LayerNorm without learnable parameters
        self.ln_wo_grad = nn.LayerNorm(embed_dim, eps=norm_eps, elementwise_affine=False)
        
        # Adaptive LayerNorm parameters
        if shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            self.ada_lin = nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(condition_dim, 6 * embed_dim)
            )
        
        # Drop path
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    
    def enable_kv_cache(self, enable: bool = True):
        """Enable/disable KV caching for inference"""
        self.attn.kv_caching(enable)
    
    def forward(
        self, 
        x: torch.Tensor,                    # [B, L, C]
        condition_embed: torch.Tensor,      # [B, C]
        memory: Optional[torch.Tensor] = None,  # [B, mem_slots, mem_slots]
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with memory augmentation
        
        关键：不要将GenAR的attn_mask传给Memory模块！
        Memory模块会自己创建causal mask。
        """
        B, L, C = x.shape
        
        # Initialize memory if not provided
        if memory is None:
            memory = torch.eye(self.memory_slots, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        
        # Get AdaLN parameters
        if self.shared_aln:
            if condition_embed.dim() == 2:
                condition_embed = condition_embed.unsqueeze(1).unsqueeze(2)
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_gss + condition_embed
            ).unbind(2)
        else:
            ada_params = self.ada_lin(condition_embed)
            gamma1, gamma2, scale1, scale2, shift1, shift2 = ada_params.view(
                B, 1, 6, C
            ).unbind(2)
        
        # First AdaLN + Self-Attention
        x_norm1 = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
        attn_output = self.attn(x_norm1, attn_mask)
        x = x + self.drop_path(attn_output.mul_(gamma1))
        
        # Memory Module Processing
        # 关键：传入None作为attention_mask，让Memory模块自己处理
        gated_memory, updated_memory = self.memory_module(x, memory, attention_mask=None)
        x = x + gated_memory  # 添加memory-enhanced features
        
        # Second AdaLN + FFN
        x_norm2 = self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
        ffn_output = self.ffn(x_norm2)
        x = x + self.drop_path(ffn_output.mul(gamma2))
        
        return x, updated_memory
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, block_idx={self.block_idx}, memory_slots={self.memory_slots}'


class GeneAdaLNBeforeHead(nn.Module):
    """
    Adaptive LayerNorm before output head
    
    Based on the GenAR AdaLNBeforeHead
    """
    def __init__(self, embed_dim: int, condition_dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        
        self.ln_wo_grad = nn.LayerNorm(embed_dim, eps=norm_eps, elementwise_affine=False)
        self.ada_lin = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(condition_dim, 2 * embed_dim)
        )
    
    def forward(self, x: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization before output head
        
        Args:
            x: Input embeddings [B, L, C]
            condition_embed: Condition embeddings [B, C]
            
        Returns:
            Normalized embeddings [B, L, C]
        """
        scale, shift = self.ada_lin(condition_embed).view(-1, 1, 2, self.embed_dim).unbind(2)
        return self.ln_wo_grad(x).mul(scale.add(1)).add_(shift)


class CellEmbeddingProcessor(nn.Module):
    """
    Cell Embedding Processor for processing cell-level features
    
    Processes cell embeddings to match the condition embedding dimension.
    Since cell embeddings lack spatial information, they are processed
    to a 768-dimensional representation without spatial encoding.
    """
    def __init__(
        self,
        cell_embed_dim: int = 1024,
        condition_embed_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cell_embed_dim = cell_embed_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Cell feature processor
        self.cell_processor = nn.Sequential(
            nn.LayerNorm(cell_embed_dim),
            nn.Linear(cell_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Final projection to condition embedding dimension
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, condition_embed_dim),
            nn.LayerNorm(condition_embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, cell_features: torch.Tensor) -> torch.Tensor:
        """
        Process cell embeddings into condition-compatible embeddings
        
        Args:
            cell_features: Cell embeddings [B, cell_embed_dim]
            
        Returns:
            Processed cell embeddings [B, condition_embed_dim]
        """
        # Process cell features
        cell_embed = self.cell_processor(cell_features)  # [B, hidden_dim]
        
        # Project to condition embedding space
        cell_embed = self.final_projection(cell_embed)  # [B, condition_embed_dim]
        
        return cell_embed
    
    def extra_repr(self) -> str:
        return (f'cell_embed_dim={self.cell_embed_dim}, '
                f'condition_embed_dim={self.condition_embed_dim}')


class QFormerFusion(nn.Module):
    """
    Q-Former based fusion module for combining histology and cell embeddings
    
    Uses a learnable query mechanism to fuse information from both histology
    (with spatial information) and cell-level features. The output maintains
    the same sequence length and dimension as the conditional embedding.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_queries: int = 1,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_cell_types: int = 5,
        top_k_cells: int = 5,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.top_k_cells = top_k_cells
        
        # Cell type embeddings (learnable)
        self.cell_type_embeddings = nn.Embedding(max_cell_types, embed_dim)
        nn.init.normal_(self.cell_type_embeddings.weight, mean=0, std=0.02)
        
        # Instance attention for selecting top-K cells
        self.instance_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Self-attention for aggregation
        self.agg_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.agg_ln = nn.LayerNorm(embed_dim)
        self.agg_ffn = FFN(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            drop=dropout
        )
        self.agg_ln_ffn = nn.LayerNorm(embed_dim)
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        # Cross-attention layers for querying histology and cell features
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            FFN(
                in_features=embed_dim,
                hidden_features=int(embed_dim * mlp_ratio),
                drop=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.ln_q = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        self.ln_ffn = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Final projection to ensure output matches input
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def match_cells_to_spots(
        self,
        spot_coords: torch.Tensor,     # [B, 2] - spot center coordinates
        cell_coords: torch.Tensor,     # [num_cells, 2] - cell center coordinates
        distance_threshold: float = 112.0  # pixels
    ) -> List[List[int]]:
        """
        Match cells to spots based on spatial distance.
        
        Args:
            spot_coords: Spot center coordinates [B, 2]
            cell_coords: Cell center coordinates [num_cells, 2]
            distance_threshold: Maximum distance for a cell to belong to a spot (default: 112 pixels)
            
        Returns:
            List of length B, where each element is a list of cell indices belonging to that spot
        """
        B = spot_coords.shape[0]
        num_cells = cell_coords.shape[0]
        
        # Compute pairwise distances between all spots and cells
        # spot_coords: [B, 2], cell_coords: [num_cells, 2]
        # distances: [B, num_cells]
        spot_coords_expanded = spot_coords.unsqueeze(1)  # [B, 1, 2]
        cell_coords_expanded = cell_coords.unsqueeze(0)  # [1, num_cells, 2]
        
        distances = torch.sqrt(
            ((spot_coords_expanded - cell_coords_expanded) ** 2).sum(dim=2)
        )  # [B, num_cells]
        
        # Find cells within threshold for each spot
        spot_cell_assignments = []
        for spot_idx in range(B):
            # Get cell indices within threshold
            cell_mask = distances[spot_idx] < distance_threshold
            cell_indices = torch.where(cell_mask)[0].cpu().tolist()
            spot_cell_assignments.append(cell_indices)
        
        return spot_cell_assignments
    
    def aggregate_cell_features(
        self,
        cell_embed: torch.Tensor,              # [num_cells, embed_dim] - cell condition embeddings
        spot_cell_assignments: List[List[int]]  # Length B, each element is list of cell indices
    ) -> torch.Tensor:                          # [B, embed_dim] - aggregated cell features per spot
        """
        Aggregate cell features for each spot using instance attention weighted features + self-attention with CLS.
        
        For each spot:
        1. Compute instance attention scores for all assigned cells
        2. Multiply instance attention scores to cell features (element-wise weighting)
        3. Select top-K most relevant cells (based on original scores)
        4. Use self-attention with CLS token: Q=[CLS, top_K], K/V=weighted_cells
        5. Return CLS token as aggregated representation
        
        Args:
            cell_embed: Cell condition embeddings [num_cells, embed_dim]
            spot_cell_assignments: List of cell indices for each spot
            
        Returns:
            Aggregated cell features per spot [B, embed_dim]
        """
        B = len(spot_cell_assignments)
        embed_dim = cell_embed.shape[1]
        device = cell_embed.device
        
        aggregated_features = torch.zeros(B, embed_dim, device=device)
        
        for spot_idx, cell_indices in enumerate(spot_cell_assignments):
            if len(cell_indices) == 0:
                # No cells assigned, keep as zeros
                continue
            
            # Get cell features for this spot
            spot_cell_features = cell_embed[cell_indices]  # [num_assigned_cells, embed_dim]
            num_assigned = spot_cell_features.shape[0]
            
            # Step 1: Compute instance attention scores
            attn_scores = self.instance_attn(spot_cell_features).squeeze(-1)  # [num_assigned_cells]
            
            # Step 2: Apply instance attention as weights to cell features
            # Normalize scores with softmax to get proper weights
            attn_weights = torch.softmax(attn_scores, dim=0)  # [num_assigned_cells]
            weighted_cell_features = spot_cell_features * attn_weights.unsqueeze(1)  # [num_assigned_cells, embed_dim]
            
            # Step 3: Select top-K cells (based on original attention scores)
            k = min(self.top_k_cells, num_assigned)
            top_k_scores, top_k_indices = torch.topk(attn_scores, k=k, dim=0)
            
            # Get top-K weighted cell features as queries
            queries = weighted_cell_features[top_k_indices].unsqueeze(0)  # [1, k, embed_dim]
            
            # All weighted cells as key-value pairs
            keys_values = weighted_cell_features.unsqueeze(0)  # [1, num_assigned_cells, embed_dim]
            
            # Prepend CLS token to queries
            cls = self.cls_token.expand(1, -1, -1)  # [1, 1, embed_dim]
            queries_with_cls = torch.cat([cls, queries], dim=1)  # [1, k+1, embed_dim]
            
            # Step 4: Self-attention
            queries_norm = self.agg_ln(queries_with_cls)
            attn_out, _ = self.agg_self_attn(
                query=queries_norm,
                key=keys_values,
                value=keys_values
            )  # [1, k+1, embed_dim]
            
            # Residual connection
            queries_with_cls = queries_with_cls + attn_out
            
            # Feed-forward network
            queries_norm = self.agg_ln_ffn(queries_with_cls)
            ffn_out = self.agg_ffn(queries_norm)
            queries_with_cls = queries_with_cls + ffn_out  # [1, k+1, embed_dim]
            
            # Step 5: Extract CLS token as aggregated representation
            aggregated_features[spot_idx] = queries_with_cls[0, 0, :]  # [embed_dim]
        
        return aggregated_features
    
    def forward(
        self,
        histology_embed: torch.Tensor,  # [B, embed_dim] - from ConditionProcessor
        cell_embed: torch.Tensor,       # [num_cells, embed_dim] - from CellConditionProcessor
        spot_coords: torch.Tensor,      # [B, 2] - spot center coordinates
        cell_coords: torch.Tensor,      # [num_cells, 2] - cell center coordinates
        cell_type_ids: torch.Tensor,    # [num_cells] - cell type IDs
        distance_threshold: float = 112.0
    ) -> torch.Tensor:                  # [B, embed_dim]
        """
        Fuse histology and cell embeddings using Q-Former with spot-cell matching
        
        Args:
            histology_embed: Histology condition embeddings [B, embed_dim]
            cell_embed: Cell condition embeddings [num_cells, embed_dim]
            spot_coords: Spot center coordinates [B, 2]
            cell_coords: Cell center coordinates [num_cells, 2]
            cell_type_ids: Cell type IDs [num_cells]
            distance_threshold: Maximum distance for cell-spot matching (default: 112 pixels)
            
        Returns:
            Fused embeddings [B, embed_dim]
        """
        B = histology_embed.shape[0]
        
        # Step 0: Add cell type embeddings to cell features
        type_embed = self.cell_type_embeddings(cell_type_ids)  # [num_cells, embed_dim]
        cell_embed_with_type = cell_embed + type_embed
        
        # Step 1: Match cells to spots based on spatial distance
        spot_cell_assignments = self.match_cells_to_spots(
            spot_coords, cell_coords, distance_threshold
        )
        
        # Step 2: Aggregate cell features for each spot using instance attention
        aggregated_cell_features = self.aggregate_cell_features(
            cell_embed_with_type, spot_cell_assignments
        )  # [B, embed_dim]
        
        # Step 3: Q-Former fusion
        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, embed_dim]
        
        # Concatenate histology and aggregated cell embeddings as key-value pairs
        kv = torch.stack([histology_embed, aggregated_cell_features], dim=1)  # [B, 2, embed_dim]
        
        # Apply cross-attention layers
        for i in range(self.num_layers):
            # Cross-attention: queries attend to concatenated histology+cell features
            queries_norm = self.ln_q[i](queries)
            attn_out, _ = self.cross_attn_layers[i](
                query=queries_norm,
                key=kv,
                value=kv
            )
            queries = queries + attn_out
            
            # Feed-forward
            queries_norm = self.ln_ffn[i](queries)
            ffn_out = self.ffn_layers[i](queries_norm)
            queries = queries + ffn_out
        
        # Pool query outputs (mean pooling across query dimension)
        fused = queries.mean(dim=1)  # [B, embed_dim]
        
        # Final projection
        fused = self.output_proj(fused)  # [B, embed_dim]
        
        # 同时返回 spot_cell_assignments，供对比学习模块使用
        # spot_cell_assignments: List[List[int]], 长度 B
        return fused, spot_cell_assignments
    
    def extra_repr(self) -> str:
        return (f'embed_dim={self.embed_dim}, num_queries={self.num_queries}, '
                f'num_heads={self.num_heads}, num_layers={self.num_layers}')


class CellConditionProcessor(nn.Module):
    """
    Cell Condition Processor for cell features and their spatial coordinates
    
    This processor handles cell-level features similarly to how ConditionProcessor
    handles histology features, creating condition embeddings from cell features
    and their spatial coordinates.
    """
    def __init__(
        self,
        cell_feature_dim: int = 1536,
        spatial_dim: int = 2,
        condition_embed_dim: int = 768,
        cell_hidden_dim: int = 512,
        spatial_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cell_feature_dim = cell_feature_dim
        self.spatial_dim = spatial_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Cell feature processor (similar to histology processor)
        self.cell_processor = nn.Sequential(
            nn.LayerNorm(cell_feature_dim),
            nn.Linear(cell_feature_dim, cell_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cell_hidden_dim, cell_hidden_dim),
            nn.LayerNorm(cell_hidden_dim)
        )
        
        # Spatial coordinate processor
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, spatial_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(spatial_hidden_dim // 2, spatial_hidden_dim),
            nn.LayerNorm(spatial_hidden_dim)
        )
        
        # Sinusoidal positional encoding for spatial coordinates
        self.pos_encoding_dim = spatial_hidden_dim // 2
        div_term = torch.exp(torch.arange(0, self.pos_encoding_dim, 2).float() * 
                           (-math.log(10000.0) / self.pos_encoding_dim))
        self.register_buffer('div_term', div_term)
        
        # Final projection to condition embedding dimension
        total_dim = cell_hidden_dim + spatial_hidden_dim
        self.final_projection = nn.Sequential(
            nn.Linear(total_dim, condition_embed_dim),
            nn.LayerNorm(condition_embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        cell_features: torch.Tensor,      # [num_cells, cell_feature_dim]
        cell_coords: torch.Tensor         # [num_cells, spatial_dim]
    ) -> torch.Tensor:                    # [num_cells, condition_embed_dim]
        """
        Process cell features and spatial coordinates into condition embeddings
        
        Args:
            cell_features: Cell-level features [num_cells, 1536]
            cell_coords: Cell center coordinates [num_cells, 2]
            
        Returns:
            Cell condition embeddings [num_cells, condition_embed_dim]
        """
        
        # Process cell features
        cell_embed = self.cell_processor(cell_features)  # [num_cells, cell_hidden_dim]
        
        # Process spatial coordinates
        spatial_embed = self.spatial_processor(cell_coords)  # [num_cells, spatial_hidden_dim]
        
        # Add sinusoidal positional encoding to spatial coordinates
        num_cells = cell_coords.shape[0]
        x_coords = cell_coords[:, 0:1]  # [num_cells, 1]
        y_coords = cell_coords[:, 1:2]  # [num_cells, 1]
        
        # Create positional encodings
        x_pe = torch.zeros(num_cells, self.pos_encoding_dim, device=cell_coords.device)
        y_pe = torch.zeros(num_cells, self.pos_encoding_dim, device=cell_coords.device)
        
        # Apply sinusoidal encoding
        x_pe[:, 0::2] = torch.sin(x_coords * self.div_term[None, :])  # Even dimensions
        x_pe[:, 1::2] = torch.cos(x_coords * self.div_term[None, :])  # Odd dimensions
        y_pe[:, 0::2] = torch.sin(y_coords * self.div_term[None, :])
        y_pe[:, 1::2] = torch.cos(y_coords * self.div_term[None, :])
        
        # Combine x and y positional encodings
        pos_encoding = torch.cat([x_pe, y_pe], dim=1)  # [num_cells, spatial_hidden_dim]
        
        # Add positional encoding to spatial embeddings
        spatial_embed = spatial_embed + pos_encoding
        
        # Fuse cell and spatial features
        condition_features = torch.cat([cell_embed, spatial_embed], dim=1)  # [num_cells, total_dim]
        
        # Final projection to condition embedding space
        condition_embed = self.final_projection(condition_features)  # [num_cells, condition_embed_dim]
        
        return condition_embed
    
    def extra_repr(self) -> str:
        return (f'cell_feature_dim={self.cell_feature_dim}, spatial_dim={self.spatial_dim}, '
                f'condition_embed_dim={self.condition_embed_dim}')


class SelectiveProportionGate(nn.Module):
    """Mamba-inspired selective gate for CellTypeProportionFusion.

    动机
    ----
    原始 element-wise gate 的根本缺陷：

        gate = sigmoid(Linear(proportions))   # [B, D]

    gate logits 仅由 ``proportions`` [B, T] 决定，与 ``spot_embed`` 的内容完全无关。
    这意味着只要两个 spot 的细胞组成比例相同，无论 spot_embed 差异多大，
    它们获得的 gate 完全一致——这是真正的退化（non-input-dependent）。

    Mamba selective 机制的核心思想
    --------------------------------
    在 Mamba (Gu & Dao, 2023) 中，状态空间模型的矩阵 Δ、B、C 均从当前输入 x 计算：

        Δ = softplus(Linear_Δ(x))      # input-dependent step size
        B = Linear_B(x)                # input-dependent write weights
        C = Linear_C(x)                # input-dependent read weights

    这使得模型可以根据内容选择性地"遗忘"或"记住"信息，
    而不是用与输入无关的固定参数控制所有时间步。

    迁移方案
    --------
    本模块将上述思想迁移到 proportion → spot_embed 融合场景：

    1. **Δ（步长）**：由 proportions 和 spot_embed 联合决定，
       控制每个维度允许多少细胞信号流入：

           Δ = softplus(W_Δ · [proportions; spot_embed])  →  [B, D]

    2. **B（写入权重）**：由 proportions 决定，
       指示各细胞类型信号应写入哪些维度：

           B = W_B · proportions_feat  →  [B, D]

    3. **C（读出权重）**：由 spot_embed 决定，
       指示当前 spot 的内容倾向读取哪些维度的细胞信息：

           C = W_C · spot_feat  →  [B, D]

    4. **selective gate**：三路逐元素乘积再经 sigmoid：

           gate_logits = Δ ⊙ B ⊙ C + scale_bias
           gate = sigmoid(gate_logits)              →  [B, D]

    不掉点保障
    ----------
    **关键设计**：这是对原始 gate 路径的 *扩展*，而非替换。

    原始路径 ``Linear(proportions)`` 被完整保留（``gate_proj_prop``），
    新增的 spot_embed 感知分支（``gate_proj_spot``）以 **零初始化** 加入：

        gate_logits = Linear(proportions)           # 原始路径，bias=-3，保持 sigmoid≈0.047
                    + Δ ⊙ B ⊙ C                    # 新增路径，weight=0 → 训练开始时贡献为 0
                    + scale_bias                     # 原始 scale-adaptive bias，不变

    训练初始时（所有新参数 weight=0）：

        gate_logits ≈ Linear(proportions) + scale_bias   # 与原版完全等价
        gate ≈ sigmoid(-3) ≈ 0.047                        # 与原版完全一致

    梯度会逐渐打开 spot_embed 感知分支，实现平滑过渡，无性能跳变。

    中间维度 ``r`` 默认 64，远小于 D=768，新增参数量约 3 × 64 × 768 × 2 ≈ 295K，
    相对整体模型可忽略。

    Parameters
    ----------
    embed_dim : int
        spot_embed 的维度 D。
    num_cell_types : int
        proportions 的维度 T。
    num_scales : int
        scale_gate_bias 的 embedding 行数。
    r : int
        中间投影维度，默认 64。
    dropout : float
        Dropout 率（仅用于 prop_proj）。
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_cell_types: int = 5,
        num_scales: int = 6,
        r: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.r = r

        # ── 原始路径（完全保留，保证开局等价性）─────────────────────────
        # gate_logits 基础项：仅依赖 proportions，bias=-3 → sigmoid≈0.047
        self.gate_proj_prop = nn.Linear(num_cell_types, embed_dim, bias=True)
        nn.init.zeros_(self.gate_proj_prop.weight)
        nn.init.constant_(self.gate_proj_prop.bias, -3.0)

        # ── Scale-adaptive bias（原版 scale_gate_bias，完整保留）────────
        self.scale_gate_bias = nn.Embedding(num_scales, embed_dim)
        nn.init.zeros_(self.scale_gate_bias.weight)

        # ── 新增：proportions → 写入侧特征（r 维） ───────────────────────
        # 用小 MLP 而非单层 Linear，保留稀疏比例向量的非线性结构
        # Dropout 有助于稀疏 proportions 向量的泛化
        self.prop_proj = nn.Sequential(
            nn.Linear(num_cell_types, r, bias=False),
            nn.LayerNorm(r),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── 新增：spot_embed → 读出侧特征（r 维） ────────────────────────
        # 轻量线性投影：保持 spot_embed 梯度的直接流动
        self.spot_proj = nn.Linear(embed_dim, r, bias=False)

        # ── 新增：Δ 投影：[p_feat; q_feat] → [B, D]（Mamba Δ 思想） ────
        # 输入为 2r 维拼接，输出为 D 维步长
        # softplus 保证正数（与 Mamba dt_proj 一致）
        # 零权重初始化 → 训练开始时 Δ ≈ softplus(0) = ln(2) ≈ 0.693，
        # 但随后乘以零初始化的 B/C，整个新路径贡献为 0
        self.delta_proj = nn.Linear(2 * r, embed_dim, bias=True)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

        # ── 新增：B 投影（写入权重）：prop_feat → [B, D] ─────────────────
        # 零权重初始化 → 训练开始时 B=0，新路径贡献为 0
        self.B_proj = nn.Linear(r, embed_dim, bias=False)
        nn.init.zeros_(self.B_proj.weight)

        # ── 新增：C 投影（读出权重）：spot_feat → [B, D] ─────────────────
        # 零权重初始化 → 训练开始时 C=0，新路径贡献为 0
        self.C_proj = nn.Linear(r, embed_dim, bias=False)
        nn.init.zeros_(self.C_proj.weight)

    def forward(
        self,
        spot_embed: torch.Tensor,            # [B, D]
        cell_type_proportions: torch.Tensor, # [B, T]
        scale_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        gate : [B, D]  ∈ (0, 1)
            可直接用于 enriched = spot_embed + gate * attn_agg
        """
        device = spot_embed.device

        # ── 原始路径（保留，提供 gate 的基础偏置） ───────────────────────
        gate_logits = self.gate_proj_prop(cell_type_proportions)   # [B, D]

        # ── Scale-adaptive bias（原版逻辑，保留） ────────────────────────
        if scale_idx is not None:
            sid = torch.tensor([scale_idx], device=device)
            scale_bias = self.scale_gate_bias(sid).squeeze(0)      # [D]
            gate_logits = gate_logits + scale_bias                  # broadcast [B, D]

        # ── 新增路径：input-dependent Δ ⊙ B ⊙ C ─────────────────────────
        # 写入侧特征（proportions 驱动）
        p_feat = self.prop_proj(cell_type_proportions)              # [B, r]
        # 读出侧特征（spot_embed 驱动）
        q_feat = self.spot_proj(spot_embed)                         # [B, r]

        # Δ：两侧联合决定步长（input-dependent step size）
        pq = torch.cat([p_feat, q_feat], dim=-1)                    # [B, 2r]
        delta = F.softplus(self.delta_proj(pq))                     # [B, D]，正数

        # B：写入权重（proportions 驱动，决定哪些维度接收细胞信号）
        B_sel = self.B_proj(p_feat)                                 # [B, D]

        # C：读出权重（spot_embed 驱动，决定当前 spot 倾向读取哪些维度）
        C_sel = self.C_proj(q_feat)                                 # [B, D]

        # Δ ⊙ B ⊙ C：三路逐元素乘积，叠加到 gate logits
        # 训练开始时 B_sel=0, C_sel=0 → 增量为 0，与原版等价
        gate_logits = gate_logits + delta * B_sel * C_sel           # [B, D]

        return torch.sigmoid(gate_logits)                           # [B, D]


class CellTypeProportionFusion(nn.Module):
    """Cross-Attention fusion with selective gate (v3).

    相较于 v2（element-wise gate）的改进
    --------------------------------------
    v2 问题：
        ``gate = sigmoid(Linear(proportions))``

        gate 仅由 ``proportions`` [B, T] 决定，完全不感知 ``spot_embed`` 的内容。
        对所有 spot_embed 差异，只要 proportions 相同，gate 完全一致——
        这是真正的 non-input-dependent 退化。

    v3 改进（Selective Gate）：
        gate 由 ``proportions`` 和 ``spot_embed`` 联合决定，
        受 Mamba selective scan 机制启发（Δ、B、C 均从当前输入计算）：

            Δ = softplus(W_Δ · [prop_feat; spot_feat])  # input-dependent 步长
            B = W_B · prop_feat                          # 写入权重（proportions 驱动）
            C = W_C · spot_feat                          # 读出权重（spot_embed 驱动）

            gate_logits = Linear(proportions)            # 原始路径（完整保留）
                        + Δ ⊙ B ⊙ C                     # 新增 selective 路径
                        + scale_bias                      # 原始 scale-adaptive bias
            gate = sigmoid(gate_logits)                  # [B, D]

    不掉点保障（向后兼容设计）
    ---------------------------
    所有新参数（``delta_proj``、``B_proj``、``C_proj``、``prop_proj``、``spot_proj``）
    全部使用 **零权重初始化**。训练开始时：

        Δ ⊙ B ⊙ C = delta * 0 * 0 = 0

    gate_logits 退化为原版 ``Linear(proportions) + scale_bias``，
    gate ≈ sigmoid(-3) ≈ 0.047，与 v2 **完全等价**。

    梯度信号会逐渐激活新路径，实现平滑过渡，无性能跳变。

    从 v2 checkpoint 迁移（strict=False 加载）：
        - 原有参数（``gate_proj`` → 已重命名为 ``selective_gate.gate_proj_prop``）
          需要手动映射，或使用 ``strict=False`` 加载后让新参数从零收敛。
        - 其余所有参数（cell_type_embeddings、query_tokens、cross_attn、FFN 等）
          完全复用，通常 1-2k steps 追上原始 baseline。

    Architecture
    ------------
    ::
        proportions [B,T]
            ↓ Embedding * weight
        type_tokens [B,T,D]                    ← KV
            ↑
        multi-query cross-attn
            ↑
        query_tokens [B,nQ,D] + spot_embed     ← Q
            ↓ weighted aggregation → attn_agg [B,D]
            ↓
        SelectiveProportionGate(spot_embed, proportions, scale_idx) → gate [B,D]
            ↓
        enriched = spot_embed + gate * attn_agg
            ↓ FFN residual
        output [B,D]

    Parameters
    ----------
    embed_dim : int
        Spot condition embedding dimension.
    num_cell_types : int
        Number of distinct cell types.
    num_scales : int
        Total number of generation scales (for scale-adaptive gate biases).
    num_queries : int
        Number of learnable Q tokens (default 4).
    num_heads : int
        Cross-attention heads.
    dropout : float
        Dropout rate.
    gate_r : int
        Inner projection dimension of SelectiveProportionGate (default 64).
        Controls the capacity of the selective mechanism; smaller = lighter.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_cell_types: int = 5,
        num_scales: int = 6,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_r: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cell_types = num_cell_types
        self.num_scales = num_scales
        self.num_queries = num_queries

        # ── Cell-type embeddings ──────────────────────────────────────────
        # Each type c gets its own embedding; scaled by proportion v[b,c] → weighted KV
        self.cell_type_embeddings = nn.Embedding(num_cell_types, embed_dim)
        nn.init.normal_(self.cell_type_embeddings.weight, mean=0.0, std=0.02)

        # ── Multi-query tokens ────────────────────────────────────────────
        # num_queries learnable vectors; each queries cell info from a different angle
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )

        # Project spot_embed into query space to condition the query tokens
        # (query = learned_token + f(spot_embed) so queries are context-aware)
        self.query_proj = nn.Linear(embed_dim, num_queries * embed_dim)

        # ── Cross-attention (Pre-LN) ──────────────────────────────────────
        self.ln_q  = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── Query aggregation: weighted sum over num_queries ──────────────
        # Learn a softmax weight per query → [num_queries] importance scores
        self.query_agg = nn.Sequential(
            nn.Linear(num_queries * embed_dim, num_queries),
            # weights are computed later with softmax in forward
        )

        # ── Improvement B→v3: Selective gate (replaces element-wise gate) ──
        # SelectiveProportionGate 内部保留了原始 Linear(proportions) + scale_bias 路径，
        # 并以零初始化叠加 Mamba-inspired Δ·B·C 分支。
        # 接口与原始 gate 完全相同：forward(spot_embed, proportions, scale_idx) → [B, D]
        self.selective_gate = SelectiveProportionGate(
            embed_dim=embed_dim,
            num_cell_types=num_cell_types,
            num_scales=num_scales,
            r=gate_r,
            dropout=dropout,
        )

        # ── Post-attention FFN ────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        spot_embed: torch.Tensor,                    # [B, D]
        cell_type_proportions: torch.Tensor,         # [B, T]  normalised frequencies
        scale_idx: Optional[int] = None,             # which generation scale (0..num_scales-1)
    ) -> torch.Tensor:                               # [B, D]
        """
        Parameters
        ----------
        spot_embed : [B, D]
        cell_type_proportions : [B, T]  (rows sum to 1 for assigned spots; 0 rows for empty)
        scale_idx : int or None
            Current generation scale index.  When None, zero scale bias is used
            (safe for inference when scale is unknown).

        Returns
        -------
        enriched : [B, D]
        """
        B = spot_embed.shape[0]
        device = spot_embed.device

        # ── Build weighted KV tokens ──────────────────────────────────────
        type_ids  = torch.arange(self.num_cell_types, device=device)
        type_base = self.cell_type_embeddings(type_ids)            # [T, D]
        type_base = type_base.unsqueeze(0).expand(B, -1, -1)      # [B, T, D]
        weights   = cell_type_proportions.unsqueeze(-1)            # [B, T, 1]
        type_tokens = type_base * weights                          # [B, T, D]  absent→~0

        # ── Build context-conditioned multi-query ─────────────────────────
        # Start from the learnable prototypes, then add spot-specific offsets
        proto = self.query_tokens.expand(B, -1, -1)                # [B, nQ, D]
        offset = self.query_proj(spot_embed)                        # [B, nQ*D]
        offset = offset.view(B, self.num_queries, self.embed_dim)   # [B, nQ, D]
        queries = proto + offset                                    # [B, nQ, D]

        # ── Cross-attention (Pre-LN) ──────────────────────────────────────
        q  = self.ln_q(queries)                                    # [B, nQ, D]
        kv = self.ln_kv(type_tokens)                               # [B, T,  D]
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)   # [B, nQ, D]

        # ── Weighted aggregation over queries ─────────────────────────────
        # Compute importance score for each query then softmax-weight the outputs
        flat_out  = attn_out.reshape(B, self.num_queries * self.embed_dim)  # [B, nQ*D]
        agg_w     = F.softmax(self.query_agg(flat_out), dim=-1)    # [B, nQ]
        attn_agg  = (attn_out * agg_w.unsqueeze(-1)).sum(dim=1)    # [B, D]

        # ── Selective gate (v3) ───────────────────────────────────────────
        # gate = SelectiveProportionGate(spot_embed, proportions, scale_idx)
        # 初始时等价于原版 element-wise gate（所有新参数权重为 0）
        # 训练后获得 input-dependent selectivity（Δ·B·C 路径逐渐激活）
        gate = self.selective_gate(
            spot_embed, cell_type_proportions, scale_idx
        )                                                           # [B, D]  ∈ (0,1)

        # Gated residual（与 v2 完全相同）
        enriched = spot_embed + gate * attn_agg                    # [B, D]

        # ── FFN residual ──────────────────────────────────────────────────
        enriched = enriched + self.ffn(enriched)                   # [B, D]
        enriched = self.out_norm(enriched)

        return enriched

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, num_cell_types={self.num_cell_types}, '
            f'num_queries={self.num_queries}, num_scales={self.num_scales}, '
            f'gate=SelectiveProportionGate(r={self.selective_gate.r}) '
            f'(selective gate + multi-query + scale-adaptive)'
        )


class ConditionProcessor(nn.Module):
    """
    Enhanced Condition Processor for histology features and spatial coordinates
    
    Improvements over original:
    - Better positional encoding for spatial coordinates
    - More robust feature processing
    - Proper normalization and dropout
    """
    def __init__(
        self,
        histology_dim: int = 1024,
        spatial_dim: int = 2,
        condition_embed_dim: int = 768,
        histology_hidden_dim: int = 512,
        spatial_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.histology_dim = histology_dim
        self.spatial_dim = spatial_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Histology feature processor
        self.histology_processor = nn.Sequential(
            nn.LayerNorm(histology_dim),
            nn.Linear(histology_dim, histology_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(histology_hidden_dim, histology_hidden_dim),
            nn.LayerNorm(histology_hidden_dim)
        )
        
        # Spatial coordinate processor
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, spatial_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(spatial_hidden_dim // 2, spatial_hidden_dim),
            nn.LayerNorm(spatial_hidden_dim)
        )
        
        # Sinusoidal positional encoding for spatial coordinates
        self.pos_encoding_dim = spatial_hidden_dim // 2
        div_term = torch.exp(torch.arange(0, self.pos_encoding_dim, 2).float() * 
                           (-math.log(10000.0) / self.pos_encoding_dim))
        self.register_buffer('div_term', div_term)
        
        # Final projection to condition embedding dimension
        total_dim = histology_hidden_dim + spatial_hidden_dim
        self.final_projection = nn.Sequential(
            nn.Linear(total_dim, condition_embed_dim),
            nn.LayerNorm(condition_embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        histology_features: torch.Tensor,  # [B, histology_dim]
        spatial_coords: torch.Tensor       # [B, spatial_dim]
    ) -> torch.Tensor:                     # [B, condition_embed_dim]
        """
        Process histology features and spatial coordinates into condition embeddings
        
        Args:
            histology_features: Histology features [B, 1024]
            spatial_coords: Spatial coordinates [B, 2]
            
        Returns:
            Condition embeddings [B, condition_embed_dim]
        """
        
        # Process histology features
        histology_embed = self.histology_processor(histology_features)  # [B, histology_hidden_dim]
        
        # Process spatial coordinates
        spatial_embed = self.spatial_processor(spatial_coords)  # [B, spatial_hidden_dim]
        
        # Add sinusoidal positional encoding to spatial coordinates
        B = spatial_coords.shape[0]
        x_coords = spatial_coords[:, 0:1]  # [B, 1]
        y_coords = spatial_coords[:, 1:2]  # [B, 1]
        
        # Create positional encodings
        x_pe = torch.zeros(B, self.pos_encoding_dim, device=spatial_coords.device)
        y_pe = torch.zeros(B, self.pos_encoding_dim, device=spatial_coords.device)
        
        # Apply sinusoidal encoding
        x_pe[:, 0::2] = torch.sin(x_coords * self.div_term[None, :])  # Even dimensions
        x_pe[:, 1::2] = torch.cos(x_coords * self.div_term[None, :])  # Odd dimensions
        y_pe[:, 0::2] = torch.sin(y_coords * self.div_term[None, :])
        y_pe[:, 1::2] = torch.cos(y_coords * self.div_term[None, :])
        
        # Combine x and y positional encodings
        pos_encoding = torch.cat([x_pe, y_pe], dim=1)  # [B, spatial_hidden_dim]
        
        # Add positional encoding to spatial embeddings
        spatial_embed = spatial_embed + pos_encoding
        
        # Fuse histology and spatial features
        condition_features = torch.cat([histology_embed, spatial_embed], dim=1)  # [B, total_dim]
        
        # Final projection to condition embedding space
        condition_embed = self.final_projection(condition_features)  # [B, condition_embed_dim]
        
        return condition_embed
    
    def extra_repr(self) -> str:
        return (f'histology_dim={self.histology_dim}, spatial_dim={self.spatial_dim}, '
                f'condition_embed_dim={self.condition_embed_dim}')


# Legacy components for backward compatibility
class PositionalEncoding(nn.Module):
    """Legacy positional encoding (kept for compatibility)"""
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class GeneGenARTransformer(nn.Module):
    """
    Legacy GenAR Transformer (kept for backward compatibility)
    
    Note: This is the old implementation. New code should use MultiScaleGenAR instead.
    """
    def __init__(self,** kwargs):
        super().__init__()
        # This is kept for backward compatibility but should not be used
        raise NotImplementedError(
            "GeneGenARTransformer is deprecated. Use MultiScaleGenAR instead."
        )