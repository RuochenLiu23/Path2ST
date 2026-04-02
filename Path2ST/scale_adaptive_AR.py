import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class GatedAttentionPool(nn.Module):

    def __init__(
        self,
        in_dim: int = 768,
        proj_dim: int = 256,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn_V = nn.Linear(in_dim, hidden, bias=False)
        self.attn_U = nn.Linear(in_dim, hidden, bias=False)
        self.attn_w = nn.Linear(hidden, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        a = self.attn_w(
            torch.tanh(self.attn_V(x)) * torch.sigmoid(self.attn_U(x))
        )
        a = self.dropout(torch.softmax(a, dim=1))

        pooled = (a * x).sum(dim=1)

        z = self.proj(pooled)
        return F.normalize(z, dim=-1)

class CellTypeCompositionEncoder(nn.Module):

    def __init__(
        self,
        num_cell_types: int = 5,
        proj_dim: int = 256,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_cell_types = num_cell_types

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
        cell_type_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:

        B = len(spot_cell_assignments)
        freq = torch.zeros(B, self.num_cell_types, device=device)

        for spot_idx, cell_indices in enumerate(spot_cell_assignments):
            if len(cell_indices) == 0:
                continue
            types = cell_type_ids[cell_indices]
            types = types.clamp(0, self.num_cell_types - 1)
            freq[spot_idx].scatter_add_(
                0, types, torch.ones_like(types, dtype=torch.float)
            )
            n = freq[spot_idx].sum()
            if n > 0:
                freq[spot_idx] = freq[spot_idx] / n

        return freq

    def forward(
        self,
        spot_cell_assignments: List[List[int]],
        cell_type_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:

        freq = self._build_freq_vector(spot_cell_assignments, cell_type_ids, device)
        z = self.encoder(freq)
        return F.normalize(z, dim=-1)

class InfoNCELoss(nn.Module):

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
        proportions: torch.Tensor,
        temperature: float,
        device: torch.device,
    ) -> torch.Tensor:

        B = proportions.shape[0]
        row_sums = proportions.sum(dim=1, keepdim=True)
        has_cells = (row_sums > 0).squeeze(1)

        norm = proportions / (row_sums + 1e-8)

        sim = torch.matmul(norm, norm.t())

        soft = F.softmax(sim / temperature, dim=1)

        eye = torch.eye(B, device=device)
        no_cell_mask = (~has_cells).unsqueeze(1).float()
        soft = soft * (1 - no_cell_mask) + eye * no_cell_mask

        return soft

    def forward(
        self,
        z_spot: torch.Tensor,
        z_type: torch.Tensor,
        proportions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = z_spot.shape[0]
        if B == 1:
            return torch.tensor(0.0, device=z_spot.device, requires_grad=True)

        logits_s2t = torch.matmul(z_spot, z_type.t()) * self.scale
        logits_t2s = logits_s2t.t()

        if proportions is not None and proportions.shape[0] == B:

            soft = self._build_soft_labels(
                proportions, self.soft_temperature, z_spot.device
            )

            log_probs_s2t = F.log_softmax(logits_s2t, dim=1)
            log_probs_t2s = F.log_softmax(logits_t2s, dim=1)
            loss_s2t = F.kl_div(log_probs_s2t, soft,          reduction='batchmean')
            loss_t2s = F.kl_div(log_probs_t2s, soft.t().contiguous(), reduction='batchmean')
        else:

            labels = torch.arange(B, device=z_spot.device)
            loss_s2t = F.cross_entropy(logits_s2t, labels)
            loss_t2s = F.cross_entropy(logits_t2s, labels)

        return (loss_s2t + loss_t2s) / 2.0

class SpotCellTypeContrastModule(nn.Module):

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
        spot_features: torch.Tensor,
        spot_cell_assignments: List[List[int]],
        cell_type_ids: torch.Tensor,
        device: Optional[torch.device] = None,
        cell_type_proportions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if device is None:
            device = spot_features.device

        z_spot = self.spot_encoder(spot_features)

        z_type = self.type_encoder(
            spot_cell_assignments, cell_type_ids, device
        )

        return self.loss_fn(z_spot, z_type, proportions=cell_type_proportions)

class DropPath(nn.Module):

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
        

        if self.attn_l2_norm:
            self.scale = 1.0
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), 
                requires_grad=True
            )
            self.max_scale_mul = torch.log(torch.tensor(100.0)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        self.attn_drop = attn_drop
        

        self.caching = False
        self.cached_k = None
        self.cached_v = None
    
    def kv_caching(self, enable: bool):

        self.caching = enable
        if not enable:
            self.cached_k = None
            self.cached_v = None
    
    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        

        qkv = F.linear(
            input=x, 
            weight=self.mat_qkv.weight, 
            bias=torch.cat([self.q_bias, self.zero_k_bias, self.v_bias])
        ).view(B, L, 3, self.num_heads, self.head_dim)
        
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        

        if self.caching:
            if self.cached_k is None:
                self.cached_k, self.cached_v = k, v
            else:
                k = self.cached_k = torch.cat([self.cached_k, k], dim=2)
                v = self.cached_v = torch.cat([self.cached_v, v], dim=2)
        

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.training and self.attn_drop > 0:
            attn_probs = F.dropout(attn_probs, p=self.attn_drop)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(out))
    
    def extra_repr(self) -> str:
        return f'attn_l2_norm={self.attn_l2_norm}, caching={self.caching}'

class FFN(nn.Module):

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
        

        self.attn = SelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            attn_l2_norm=attn_l2_norm,
        )
        

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop_rate
        )
        

        self.ln_wo_grad = nn.LayerNorm(embed_dim, eps=norm_eps, elementwise_affine=False)
        

        if shared_aln:

            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:

            self.ada_lin = nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(condition_dim, 6 * embed_dim)
            )
        

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    
    def enable_kv_cache(self, enable: bool = True):

        self.attn.kv_caching(enable)
    
    def forward(
        self, 
        x: torch.Tensor,
        condition_embed: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        B, L, C = x.shape
        

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
        

        x_norm1 = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
        attn_output = self.attn(x_norm1, attn_mask)
        x = x + self.drop_path(attn_output.mul_(gamma1))
        

        x_norm2 = self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
        ffn_output = self.ffn(x_norm2)
        x = x + self.drop_path(ffn_output.mul(gamma2))
        
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, block_idx={self.block_idx}'

class GeneAdaLNBeforeHead(nn.Module):

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

        scale, shift = self.ada_lin(condition_embed).view(-1, 1, 2, self.embed_dim).unbind(2)
        return self.ln_wo_grad(x).mul(scale.add(1)).add_(shift)

class SelectiveProportionGate(nn.Module):

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

        self.gate_proj_prop = nn.Linear(num_cell_types, embed_dim, bias=True)
        nn.init.zeros_(self.gate_proj_prop.weight)
        nn.init.constant_(self.gate_proj_prop.bias, -3.0)

        self.scale_gate_bias = nn.Embedding(num_scales, embed_dim)
        nn.init.zeros_(self.scale_gate_bias.weight)

        self.prop_proj = nn.Sequential(
            nn.Linear(num_cell_types, r, bias=False),
            nn.LayerNorm(r),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.spot_proj = nn.Linear(embed_dim, r, bias=False)

        self.delta_proj = nn.Linear(2 * r, embed_dim, bias=True)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

        self.B_proj = nn.Linear(r, embed_dim, bias=False)
        nn.init.zeros_(self.B_proj.weight)

        self.C_proj = nn.Linear(r, embed_dim, bias=False)
        nn.init.zeros_(self.C_proj.weight)

    def forward(
        self,
        spot_embed: torch.Tensor,
        cell_type_proportions: torch.Tensor,
        scale_idx: Optional[int] = None,
    ) -> torch.Tensor:

        device = spot_embed.device

        gate_logits = self.gate_proj_prop(cell_type_proportions)

        if scale_idx is not None:
            sid = torch.tensor([scale_idx], device=device)
            scale_bias = self.scale_gate_bias(sid).squeeze(0)
            gate_logits = gate_logits + scale_bias

        p_feat = self.prop_proj(cell_type_proportions)

        q_feat = self.spot_proj(spot_embed)

        pq = torch.cat([p_feat, q_feat], dim=-1)
        delta = F.softplus(self.delta_proj(pq))

        B_sel = self.B_proj(p_feat)

        C_sel = self.C_proj(q_feat)

        gate_logits = gate_logits + delta * B_sel * C_sel

        return torch.sigmoid(gate_logits)

class CellTypeProportionFusion(nn.Module):

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

        self.cell_type_embeddings = nn.Embedding(num_cell_types, embed_dim)
        nn.init.normal_(self.cell_type_embeddings.weight, mean=0.0, std=0.02)

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )

        self.query_proj = nn.Linear(embed_dim, num_queries * embed_dim)

        self.ln_q  = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.query_agg = nn.Sequential(
            nn.Linear(num_queries * embed_dim, num_queries),

        )

        self.selective_gate = SelectiveProportionGate(
            embed_dim=embed_dim,
            num_cell_types=num_cell_types,
            num_scales=num_scales,
            r=gate_r,
            dropout=dropout,
        )

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
        spot_embed: torch.Tensor,
        cell_type_proportions: torch.Tensor,
        scale_idx: Optional[int] = None,
    ) -> torch.Tensor:

        B = spot_embed.shape[0]
        device = spot_embed.device

        type_ids  = torch.arange(self.num_cell_types, device=device)
        type_base = self.cell_type_embeddings(type_ids)
        type_base = type_base.unsqueeze(0).expand(B, -1, -1)
        weights   = cell_type_proportions.unsqueeze(-1)
        type_tokens = type_base * weights

        proto = self.query_tokens.expand(B, -1, -1)
        offset = self.query_proj(spot_embed)
        offset = offset.view(B, self.num_queries, self.embed_dim)
        queries = proto + offset

        q  = self.ln_q(queries)
        kv = self.ln_kv(type_tokens)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)

        flat_out  = attn_out.reshape(B, self.num_queries * self.embed_dim)
        agg_w     = F.softmax(self.query_agg(flat_out), dim=-1)
        attn_agg  = (attn_out * agg_w.unsqueeze(-1)).sum(dim=1)

        gate = self.selective_gate(
            spot_embed, cell_type_proportions, scale_idx
        )

        enriched = spot_embed + gate * attn_agg

        enriched = enriched + self.ffn(enriched)
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
        

        self.histology_processor = nn.Sequential(
            nn.LayerNorm(histology_dim),
            nn.Linear(histology_dim, histology_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(histology_hidden_dim, histology_hidden_dim),
            nn.LayerNorm(histology_hidden_dim)
        )
        

        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, spatial_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(spatial_hidden_dim // 2, spatial_hidden_dim),
            nn.LayerNorm(spatial_hidden_dim)
        )
        

        self.pos_encoding_dim = spatial_hidden_dim // 2
        div_term = torch.exp(torch.arange(0, self.pos_encoding_dim, 2).float() * 
                           (-math.log(10000.0) / self.pos_encoding_dim))
        self.register_buffer('div_term', div_term)
        

        total_dim = histology_hidden_dim + spatial_hidden_dim
        self.final_projection = nn.Sequential(
            nn.Linear(total_dim, condition_embed_dim),
            nn.LayerNorm(condition_embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        histology_features: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> torch.Tensor:

        

        histology_embed = self.histology_processor(histology_features)
        

        spatial_embed = self.spatial_processor(spatial_coords)
        

        B = spatial_coords.shape[0]
        x_coords = spatial_coords[:, 0:1]
        y_coords = spatial_coords[:, 1:2]
        

        x_pe = torch.zeros(B, self.pos_encoding_dim, device=spatial_coords.device)
        y_pe = torch.zeros(B, self.pos_encoding_dim, device=spatial_coords.device)
        

        x_pe[:, 0::2] = torch.sin(x_coords * self.div_term[None, :])
        x_pe[:, 1::2] = torch.cos(x_coords * self.div_term[None, :])
        y_pe[:, 0::2] = torch.sin(y_coords * self.div_term[None, :])
        y_pe[:, 1::2] = torch.cos(y_coords * self.div_term[None, :])
        

        pos_encoding = torch.cat([x_pe, y_pe], dim=1)
        

        spatial_embed = spatial_embed + pos_encoding
        

        condition_features = torch.cat([histology_embed, spatial_embed], dim=1)
        

        condition_embed = self.final_projection(condition_features)
        
        return condition_embed
    
    def extra_repr(self) -> str:
        return (f'histology_dim={self.histology_dim}, spatial_dim={self.spatial_dim}, '
                f'condition_embed_dim={self.condition_embed_dim}')