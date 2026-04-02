"""
GenAR: Multi-Scale Gene Autoregressive for Spatial Transcriptomics

This module implements a GenAR model for spatial transcriptomics
based on the GenAR architecture. The model uses cumulative multi-scale
generation to predict gene expressions from histology features, spatial coordinates,
and cell-level features.

Key Features:
- Multi-scale cumulative generation (like original GenAR)
- AdaLN conditioning for deep feature fusion
- Q-Former fusion for histology and cell embeddings
- Residual accumulation across scales
- KV caching for efficient inference
- FiLM-based dynamic gene identity modulation

Author: Assistant
Date: 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import partial

# Soft-target type annotations
SoftTarget = Dict[str, torch.Tensor]
HierarchicalTargets = List[Union[torch.Tensor, SoftTarget]]

from .gene_genar_transformer import (
    GeneAdaLNSelfAttn, 
    GeneAdaLNSelfAttnWithMemory,  # 保留向后兼容（未使用）
    GeneAdaLNBeforeHead, 
    ConditionProcessor,
    CellEmbeddingProcessor,       # 保留向后兼容（未使用）
    CellConditionProcessor,       # 保留向后兼容（未使用）
    QFormerFusion,                # 保留向后兼容（未使用）
    CellTypeProportionFusion,     # NEW: proportion-vector cross-attention fusion
    DropPath,
    SpotCellTypeContrastModule,
)
from .film_layer import FiLMLayer
from .gene_identity_pooling import GeneIdentityPooling

logger = logging.getLogger(__name__)


class ZINBLoss(nn.Module):
    """
    Improved Zero-Inflated Negative Binomial (ZINB) Loss with better numerical stability.
    
    Key improvements:
    1. L2 regularization to prevent parameter explosion
    2. Gradient clipping for stability
    3. More robust parameter clamping
    4. Detailed error checking
    """
    
    def __init__(self, eps: float = 1e-8, scale_factor: float = 1.0, ridge_lambda: float = 0.0):
        super().__init__()
        self.eps = eps
        self.scale_factor = scale_factor
        self.ridge_lambda = ridge_lambda
    
    def forward(
        self, 
        mu: torch.Tensor,           # [B, seq_len] - mean parameter
        theta: torch.Tensor,        # [B, seq_len] - dispersion parameter  
        pi: torch.Tensor,           # [B, seq_len] - zero-inflation probability
        target: torch.Tensor        # [B, seq_len] - observed counts
    ) -> torch.Tensor:
        """
        Compute ZINB negative log-likelihood with numerical stability.
        
        Returns:
            Negative log-likelihood loss (scalar)
        """
        # 1. Apply scale factor and clamp parameters
        mu = mu * self.scale_factor
        mu = torch.clamp(mu, min=self.eps, max=1e4)  # Prevent extreme values
        theta = torch.clamp(theta, min=self.eps, max=1e4)
        pi = torch.clamp(pi, min=self.eps, max=1 - self.eps)
        
        # 2. Compute NB log-likelihood
        theta_mu_sum = theta + mu
        
        # For numerical stability
        log_theta = torch.log(theta + self.eps)
        log_mu = torch.log(mu + self.eps)
        log_theta_mu_sum = torch.log(theta_mu_sum + self.eps)
        
        # Log probability for y > 0
        log_nb_positive = (
            torch.lgamma(theta + target + self.eps)
            - torch.lgamma(theta + self.eps)
            - torch.lgamma(target + 1.0 + self.eps)
            + theta * (log_theta - log_theta_mu_sum)
            + target * (log_mu - log_theta_mu_sum)
        )
        
        # Log probability for y = 0
        log_nb_zero = theta * (log_theta - log_theta_mu_sum)
        
        # 3. ZINB log-likelihood
        zero_mask = (target < 0.5).float()
        
        log_pi = torch.log(pi + self.eps)
        log_1_minus_pi = torch.log(1.0 - pi + self.eps)
        
        # For y = 0: use log-sum-exp trick
        log_prob_zero = torch.logsumexp(
            torch.stack([log_pi, log_1_minus_pi + log_nb_zero], dim=0),
            dim=0
        )
        
        # For y > 0
        log_prob_positive = log_1_minus_pi + log_nb_positive
        
        # 4. Combine
        log_prob = zero_mask * log_prob_zero + (1.0 - zero_mask) * log_prob_positive
        
        # 5. Negative log-likelihood
        nll = -log_prob
        
        # 6. Clamp to prevent gradient explosion
        nll = torch.clamp(nll, min=-100, max=100)
        
        # 7. Add L2 regularization (optional)
        if self.ridge_lambda > 0:
            l2_reg = self.ridge_lambda * (
                torch.mean(mu.pow(2)) + 
                torch.mean(theta.pow(2)) + 
                torch.mean(pi.pow(2))
            )
            nll = nll + l2_reg
        
        # 8. Return mean loss
        return nll.mean()


class GeneGroupUpsampling(nn.Module):
    """Group-aware upsampling module respecting gene hierarchy."""
    
    def __init__(self, embed_dim: int, scale_dims: Tuple[int, ...], num_genes: int = 200):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_dims = scale_dims
        self.num_genes = num_genes
        
        # Pre-compute group mappings for each scale transition
        self.group_mappings = self._compute_group_mappings()

        # Learnable upsampling transforms between adjacent scales
        self.upsample_transforms = nn.ModuleDict()
        for i in range(len(scale_dims) - 1):
            self.upsample_transforms[f'scale_{i}_to_{i+1}'] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1)
            )
        
        logger.info("Gene Group Upsampling initialised")
        logger.info(f"   Scale dims: {scale_dims}")
        logger.info(f"   Number of upsampling transforms: {len(self.upsample_transforms)}")
        for key in self.group_mappings:
            logger.info(f"   {key}: {len(self.group_mappings[key])} mappings")
    
    def _compute_group_mappings(self):
        """Compute mapping tables between source and target scales."""
        mappings = {}
        
        for i in range(len(self.scale_dims) - 1):
            source_dim = self.scale_dims[i]
            target_dim = self.scale_dims[i + 1]
            
            # Determine how many target tokens each source token maps to
            genes_per_source = self.num_genes // source_dim
            genes_per_target = self.num_genes // target_dim
            targets_per_source = genes_per_source // genes_per_target
            
            mapping = []
            for source_idx in range(source_dim):
                start_target = source_idx * targets_per_source
                end_target = start_target + targets_per_source
                target_indices = list(range(start_target, min(end_target, target_dim)))
                mapping.append(target_indices)
            
            mappings[f'scale_{i}_to_{i+1}'] = mapping
            
        return mappings
    
    def forward(self, source_embeddings: torch.Tensor, source_scale_idx: int, target_scale_idx: int):
        """Group-aware upsampling between scales."""
        if source_embeddings is None:
            raise ValueError("source_embeddings must not be None for upsampling")
        
        B, source_dim, embed_dim = source_embeddings.shape
        target_dim = self.scale_dims[target_scale_idx]
        
        # Lookup the precomputed mapping
        mapping_key = f'scale_{source_scale_idx}_to_{target_scale_idx}'
        if mapping_key not in self.group_mappings:
            raise ValueError(f"Missing group mapping for {mapping_key}")
        
        mapping = self.group_mappings[mapping_key]
        
        # Group-aware upsampling
        upsampled = torch.zeros(B, target_dim, embed_dim, device=source_embeddings.device)
        
        for source_idx, target_indices in enumerate(mapping):
            if source_idx < source_dim:
                source_emb = source_embeddings[:, source_idx, :]

                # Optional learned transform
                if mapping_key in self.upsample_transforms:
                    transformed_emb = self.upsample_transforms[mapping_key](source_emb)
                else:
                    transformed_emb = source_emb

                # Copy into target indices
                for target_idx in target_indices:
                    if target_idx < target_dim:
                        upsampled[:, target_idx, :] = transformed_emb

        return upsampled
    
    def _interpolate_upsample(self, source_embeddings, target_dim):
        """Interpolation-based upsampling for non-adjacent scales."""
        _, source_dim, _ = source_embeddings.shape
        
        if source_dim == 1:
            return source_embeddings.expand(-1, target_dim, -1)
        
        # Linear interpolation across the sequence dimension
        source_embeddings_t = source_embeddings.transpose(1, 2)  # [B, embed_dim, source_dim]
        upsampled = F.interpolate(source_embeddings_t, size=target_dim, mode='linear', align_corners=False)
        return upsampled.transpose(1, 2)  # [B, target_dim, embed_dim]


class MultiScaleGenAR(nn.Module):
    """
    Hierarchical GenAR for Spatial Transcriptomics with Semantic-Aware Embeddings
    
    This model implements a hierarchical generation process with improved position embeddings
    that address semantic mismatches between different scales. Key improvements include:
    
    Architecture:
    - Condition Processor: Encodes histology and spatial features
    - Cell Embedding Processor: Encodes cell-level features
    - Q-Former Fusion: Fuses histology and cell embeddings
    - Hierarchical Position Embeddings: Scale-specific embeddings that preserve semantic meaning
    - Hierarchical Generation: Sequentially refines predictions across scales (e.g., 1 → 4 → 8 → 40 → 100 → 200)
    - AdaLN Transformer: Core computation block with deep conditioning
    - Teacher Forcing: Uses ground truth averages at each scale to guide the next
    
    Key Features:
    - Multi-scale cumulative generation (like original GenAR)
    - Semantic-aware position embeddings for each scale
    - Q-Former based fusion of histology and cell embeddings
    - AdaLN conditioning for deep feature fusion
    - Residual accumulation across scales
    - KV caching for efficient inference
    - Soft label training for information preservation
    
    Embedding Innovation:
    - Each scale has dedicated position embeddings with appropriate semantic meaning
    - Intermediate scales use pool position embeddings (representing gene groups)
    - Final scale uses gene identity embeddings (representing individual genes)
    - This eliminates semantic confusion where same position represents different biology
    """
    
    def __init__(
        self,
        # Gene-related parameters
        vocab_size: int,
        num_genes: int = 200,
        scale_dims: Tuple[int, ...] = (1, 4, 8, 40, 100, 200),
        
        # Model architecture parameters
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        
        # Dropout parameters
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        
        # Condition-related parameters
        histology_feature_dim: int = 1024,
        spatial_coord_dim: int = 2,
        condition_embed_dim: int = 768,
        cond_drop_rate: float = 0.1,
        
        # Cell-type proportion fusion parameters  (NEW – replaces cell_embed_dim / use_cell_emb)
        use_cell_type_proportion: bool = True,    # whether to fuse cell-type proportions
        num_cell_types: int = 5,                  # number of distinct cell types
        cell_type_fusion_heads: int = 8,          # cross-attention heads in CellTypeProportionFusion
        cell_type_fusion_dropout: float = 0.1,
        
        # Other parameters
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        attn_l2_norm: bool = True,
        device: str = 'cuda',
        adaptive_sigma_alpha: float = 0.1,  # Proportional factor for adaptive sigma
        adaptive_sigma_beta: float = 1.0,   # Base value for adaptive sigma
        
        # ZINB hybrid configuration
        use_zinb_loss: bool = True,           # Whether to use ZINB loss
        zinb_loss_weight: float = 0.3,        # ZINB loss weight (0-1), α in paper
        use_zinb_for_inference: bool = False, # Use ZINB prediction in inference
        zinb_ridge_lambda: float = 1e-5,      # L2 regularization for ZINB params
        
        # Cell-type contrastive learning configuration
        use_cell_type_contrast: bool = True,  # Whether to use InfoNCE contrastive loss
        contrast_weight: float = 0.1,         # λ · L_InfoNCE 的权重
        contrast_proj_dim: int = 256,         # 对比空间维度
        # NOTE: contrast_num_cell_types is intentionally removed as a separate parameter.
        # It is always forced equal to num_cell_types to guarantee consistency between
        # CellTypeProportionFusion, virtual_cell_type_ids, and SpotCellTypeContrastModule.
    ):
        super().__init__()
        
        # Enforce: contrast module must see the same number of cell types as the fusion module
        contrast_num_cell_types = num_cell_types
        
        # Store key parameters
        self.num_genes = num_genes
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cond_drop_rate = cond_drop_rate
        self.device = device
        self.adaptive_sigma_alpha = adaptive_sigma_alpha
        self.adaptive_sigma_beta = adaptive_sigma_beta
        # NEW: cell-type proportion fusion flags
        self.use_cell_type_proportion = use_cell_type_proportion
        self.num_cell_types = num_cell_types
        
        # ZINB hybrid configuration
        self.use_zinb_loss = use_zinb_loss
        self.zinb_loss_weight = zinb_loss_weight
        self.use_zinb_for_inference = use_zinb_for_inference
        self.zinb_ridge_lambda = zinb_ridge_lambda
        
        # Cell-type contrastive learning configuration
        self.use_cell_type_contrast = use_cell_type_contrast
        self.contrast_weight = contrast_weight
        self.contrast_proj_dim = contrast_proj_dim
        self.contrast_num_cell_types = contrast_num_cell_types
        
        
        # Hierarchical scale configuration
        self.scale_dims = scale_dims
        self.num_scales = len(scale_dims)
        
        # Store other config parameters for checkpointing
        self.histology_feature_dim = histology_feature_dim
        self.spatial_coord_dim = spatial_coord_dim
        self.condition_embed_dim = condition_embed_dim
        # cell_embed_dim retained as alias for backward-compat checkpoint loading
        self.cell_embed_dim = num_cell_types
        
        # Log the new hierarchical configuration
        logger.info(f"Hierarchical scale dimensions: {self.scale_dims}")
        logger.info(f"Number of scales: {self.num_scales}")
        logger.info(f"Use cell-type proportion fusion: {self.use_cell_type_proportion}")
        
        # Condition processor for histology features
        self.condition_processor = ConditionProcessor(
            histology_dim=histology_feature_dim,
            spatial_dim=spatial_coord_dim,
            condition_embed_dim=condition_embed_dim
        )
        
        # Cell-type proportion fusion (v3: selective gate + multi-query + scale-adaptive)
        if self.use_cell_type_proportion:
            self.cell_proportion_fusion = CellTypeProportionFusion(
                embed_dim=condition_embed_dim,
                num_cell_types=num_cell_types,
                num_scales=len(scale_dims),          # scale-adaptive gate biases
                num_queries=4,                       # multi-query fusion
                num_heads=cell_type_fusion_heads,
                dropout=cell_type_fusion_dropout,
                gate_r=64,                           # NEW v3: SelectiveProportionGate inner dim
            )
            logger.info("Cell-type proportion fusion v3 enabled (selective gate):")
            logger.info(f"   - num_cell_types: {num_cell_types}")
            logger.info(f"   - num_queries: 4 (multi-query fusion)")
            logger.info(f"   - num_scales: {len(scale_dims)} (scale-adaptive gate biases)")
            logger.info(f"   - cross-attention heads: {cell_type_fusion_heads}")
            logger.info(f"   - Design: Q=multi-query(spot-conditioned), KV=type_tokens*proportions")
            logger.info(f"   - Gate: SelectiveProportionGate(r=64) — Mamba-inspired Δ·B·C")
            logger.info(f"     Δ=softplus(W_Δ·[prop_feat;spot_feat]), B=W_B·prop_feat, C=W_C·spot_feat")
            logger.info(f"     gate_logits = Linear(proportions) + Δ⊙B⊙C + scale_bias[s]")
            logger.info(f"     Init: W_Δ=W_B=W_C=0 → gate≈sigmoid(-3)≈0.047, equiv. to v2 at start")
        
        # Gene token embedding (for expression counts)
        self.gene_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # NEW: Unified gene identity embedding as modulation condition
        self.gene_identity_embedding = nn.Embedding(num_genes, embed_dim)
        
        # NEW: FiLM layer for dynamic gene-specific modulation
        self.film_layer = FiLMLayer(
            condition_dim=embed_dim,
            feature_dim=embed_dim,
            hidden_dim=embed_dim // 2
        )
        
        # NEW: Gene identity pooling for multi-scale modulation (conservative approach)
        self.gene_identity_pooling = GeneIdentityPooling(
            num_genes=num_genes,
            scale_dims=scale_dims,
            embed_dim=embed_dim,
            enable_pooling=True  # Required in strict mode
        )
        
        # NEW: Gene group upsampling module for intelligent target position initialization
        self.gene_upsampling = GeneGroupUpsampling(
            embed_dim=embed_dim,
            scale_dims=scale_dims,
            num_genes=num_genes
        )
        
        # Hierarchical position embedding - separate embedding for each scale
        # Updated to support cumulative input from all previous scales
        self.hierarchical_pos_embedding = nn.ModuleDict()
        for i, dim in enumerate(self.scale_dims):
            # Calculate maximum sequence length for this scale:
            # start_token + all previous scales + current scale
            if dim == self.num_genes:
                # For the final scale, add extra positions for all target genes
                # This follows GenAR's approach: cumulative_context + new_target_positions
                max_cumulative_length = 1 + sum(self.scale_dims[:i]) + self.num_genes
            else:
                # For intermediate scales, use normal cumulative length
                max_cumulative_length = 1 + sum(self.scale_dims[:i+1])
            self.hierarchical_pos_embedding[f'scale_{i}'] = nn.Embedding(max_cumulative_length, embed_dim)
        
        logger.info("Created hierarchical position embeddings for GenAR-style input:")
        for i, dim in enumerate(self.scale_dims):
            if dim == self.num_genes:
                max_length = 1 + sum(self.scale_dims[:i]) + self.num_genes
                logger.info(f"   Scale {i} (dim={dim}): max {max_length} positions (GenAR-style: context + targets)")
            else:
                max_length = 1 + sum(self.scale_dims[:i+1])
                logger.info(f"   Scale {i} (dim={dim}): max {max_length} positions (cumulative)")
        
        logger.info("GenAR improvements applied:")
        logger.info("   - Gene Group Upsampling: intelligent target position initialisation")
        logger.info("   - Scale Embedding Storage: progressive information transfer")
        logger.info("   - Weighted Identity Fusion: final scale mixes upsampling + identities")
        logger.info("   - Multi-Scale Gene Modulation: pooling enabled across scales")
        logger.info("   - Conservative Design: feature flags fixed in strict mode")
        
        # Scale embedding to distinguish different scales
        self.scale_embedding = nn.Embedding(self.num_scales, embed_dim)
        
        # Single start token for initiating the generation process
        self.start_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer backbone with AdaLN
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        self.transformer_blocks = nn.ModuleList([
            GeneAdaLNSelfAttn(
                block_idx=i,
                embed_dim=embed_dim,
                condition_dim=condition_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_eps=norm_eps,
                shared_aln=shared_aln,
                attn_l2_norm=attn_l2_norm,
            )
            for i in range(num_layers)
        ])
        
        # Output head with AdaLN
        self.head_norm = GeneAdaLNBeforeHead(embed_dim, condition_embed_dim, norm_eps)
        
        # IMPORTANT: Keep the classification output head (for all scales)
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # ZINB heads (only for final scale, as auxiliary supervision)
        if self.use_zinb_loss:
            logger.info("=" * 60)
            logger.info("HYBRID ARCHITECTURE: Classification + ZINB")
            logger.info("=" * 60)
            
            # Shared feature extractor for ZINB parameters
            # This helps stabilize training and reduce parameters
            self.zinb_feature_extractor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            
            # Three separate heads for ZINB parameters
            # mu: mean (positive real number)
            self.zinb_mu_head = nn.Sequential(
                nn.Linear(embed_dim // 2, 1),
                nn.Softplus(beta=1)  # beta=1 for sufficient gradients
            )
            
            # theta: dispersion (positive real number, higher = less dispersed)
            self.zinb_theta_head = nn.Sequential(
                nn.Linear(embed_dim // 2, 1),
                nn.Softplus(beta=1)
            )
            
            # pi: zero-inflation probability [0, 1]
            self.zinb_pi_head = nn.Sequential(
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
            
            # Update ZINB loss with L2 regularization
            self.zinb_loss = ZINBLoss(
                eps=1e-8, 
                scale_factor=1.0,
                ridge_lambda=self.zinb_ridge_lambda
            )
            
            logger.info("ZINB configuration:")
            logger.info(f"   - Loss weight (α): {self.zinb_loss_weight:.2f}")
            logger.info(f"   - Classification weight: {1 - self.zinb_loss_weight:.2f}")
            logger.info(f"   - Use ZINB for inference: {self.use_zinb_for_inference}")
            logger.info(f"   - L2 regularization: {self.zinb_ridge_lambda}")
            logger.info(f"   - Shared feature extractor: {embed_dim} -> {embed_dim // 2}")
            logger.info("=" * 60)
        else:
            logger.info("ZINB loss disabled - using pure classification")
        
        # ── Cell-type contrastive learning module ────────────────────────────
        if self.use_cell_type_contrast and self.use_cell_type_proportion:
            self.cell_type_contrast = SpotCellTypeContrastModule(
                embed_dim=embed_dim,
                proj_dim=contrast_proj_dim,
                num_cell_types=contrast_num_cell_types,
                init_temp=0.07,
                dropout=0.1,
            )
            logger.info("Cell-type contrastive module enabled:")
            logger.info(f"   - proj_dim: {contrast_proj_dim}")
            logger.info(f"   - contrast_weight (λ): {contrast_weight}")
            logger.info(f"   - num_cell_types: {contrast_num_cell_types}")
            logger.info(f"   - Spot encoder: GatedAttentionPool (embed_dim={embed_dim} → proj_dim={contrast_proj_dim})")
            logger.info(f"   - Type encoder: CellTypeCompositionEncoder (num_types={contrast_num_cell_types} → proj_dim={contrast_proj_dim})")
        elif self.use_cell_type_contrast and not self.use_cell_type_proportion:
            logger.warning("use_cell_type_contrast=True but use_cell_type_proportion=False; contrastive loss will be skipped.")
            self.use_cell_type_contrast = False
        # ─────────────────────────────────────────────────────────────────────
        
        # ── Improvement E: CellFiLM head for final-scale direct conditioning ──
        # At the final scale (200 genes), directly modulate each gene token with
        # the cell-type proportion vector via FiLM: γ,β = Linear(proportions).
        # This is a lightweight 'last-mile' injection that bypasses potential
        # signal dilution through 8 transformer layers.
        if self.use_cell_type_proportion:
            self.cell_film_head = nn.Sequential(
                nn.Linear(num_cell_types, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim * 2),
            )
            # Init to near-identity: γ≈1, β≈0
            nn.init.zeros_(self.cell_film_head[-1].weight)
            nn.init.zeros_(self.cell_film_head[-1].bias)
            logger.info("CellFiLM head enabled (final-scale direct conditioning)")
        
        # Log detailed parameter information
        total_params = self._count_parameters()
        identity_params = self.gene_identity_embedding.num_embeddings * self.gene_identity_embedding.embedding_dim
        film_params = sum(p.numel() for p in self.film_layer.parameters())
        
        logger.info("Hierarchical GenAR initialised successfully")
        logger.info("FiLM layer for dynamic gene identity modulation:")
        logger.info(f"   - Gene identity embedding: [{num_genes}, {embed_dim}]")
        logger.info(f"   - FiLM hidden dimension: {embed_dim // 2}")
        logger.info("Parameter breakdown:")
        logger.info(f"   - Total parameters: ~{total_params/1e6:.1f}M")
        logger.info(f"   - Gene identity embedding: {identity_params:,} ({identity_params/1e3:.1f}K)")
        logger.info(f"   - FiLM layer: {film_params:,} ({film_params/1e3:.1f}K)")
        logger.info(f"   - New parameters ratio: {(identity_params + film_params)/total_params*100:.2f}%")
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _get_hierarchical_position_embedding(self, scale_idx: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return semantic position embeddings for a given scale."""
        # Select the scale-specific embedding table
        embedding_layer = self.hierarchical_pos_embedding[f'scale_{scale_idx}']

        # Generate indices [0, 1, ..., seq_len-1]
        pos_indices = torch.arange(seq_len, device=device)

        # Lookup position embeddings
        pos_embed = embedding_layer(pos_indices)

        # Add batch dimension
        return pos_embed.unsqueeze(0)
    
    def init_weights(self, init_std: float = 0.02):
        """Initialize model weights following GenAR initialization"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=init_std)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
        
        self.apply(_init_weights)
        logger.info("Model weights initialised with hierarchical position embeddings")

    def _create_hierarchical_targets(self, target_genes: torch.Tensor) -> HierarchicalTargets:
        """
        Creates hierarchical ground truth targets using soft labels for intermediate scales.
        
        This method solves the information loss problem in the original implementation by:
        1. Using soft labels (floor/ceil + weights) for intermediate scales instead of hard rounding
        2. Preserving the complete information from floating-point pooled values
        3. Only using hard labels for the final, full-resolution scale

        Args:
            target_genes (torch.Tensor): The ground truth gene expressions, shape [B, 200].

        Returns:
            HierarchicalTargets: A list containing either hard targets (torch.Tensor) for the final scale
                               or soft targets (Dict[str, torch.Tensor]) for intermediate scales.
                               Soft targets contain:
                               - 'floor_targets': Lower bound token IDs
                               - 'ceil_targets': Upper bound token IDs  
                               - 'weights': Interpolation weights (0.0 to 1.0)
        """
        hierarchical_targets = []

        # Ensure target_genes is float for pooling operations
        if torch.any(target_genes < 0) or torch.any(target_genes >= self.vocab_size):
            raise ValueError("Target genes are out of vocabulary range")
        target_genes_float = target_genes.float().unsqueeze(1) # -> [B, 1, 200]

        for _, dim in enumerate(self.scale_dims):
            if dim == self.num_genes:
                # The final scale uses hard labels (original behavior)
                hard_targets = target_genes.long()
                hard_targets = torch.clamp(hard_targets, 0, self.vocab_size - 1)
                hierarchical_targets.append(hard_targets)
            else:
                # Intermediate scales use soft labels to preserve information
                pooled_targets = F.adaptive_avg_pool1d(target_genes_float, output_size=dim)
                pooled_targets = pooled_targets.squeeze(1) # -> [B, dim]
                
                if pooled_targets.min() < 0 or pooled_targets.max() > (self.vocab_size - 1):
                    raise ValueError("Pooled targets out of vocabulary range")

                # Generate soft labels: floor + ceil + interpolation weight
                floor_targets = torch.floor(pooled_targets).long()
                ceil_targets = torch.ceil(pooled_targets).long()
                weights = pooled_targets - floor_targets.float()  # Interpolation weights [0.0, 1.0]
                
                soft_target = {
                    'floor_targets': floor_targets,
                    'ceil_targets': ceil_targets,
                    'weights': weights
                }
                
                hierarchical_targets.append(soft_target)
            
        return hierarchical_targets

    def _create_gaussian_target_distribution(self, target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create adaptive Gaussian target distribution for final scale loss computation.
        
        This method constructs a Gaussian probability distribution centered at the true gene 
        expression values with adaptive sigma that scales with expression level.
        High expression genes get larger sigma (more tolerance), low expression genes get 
        smaller sigma (stricter requirements).
        
        Args:
            target (torch.Tensor): Hard target labels, shape [B, seq_len]
            device (torch.device): Device to create tensors on
            
        Returns:
            torch.Tensor: Gaussian target distribution, shape [B, seq_len, vocab_size]
        """
        vocab_size = self.vocab_size
        if torch.any(target < 0) or torch.any(target >= vocab_size):
            raise ValueError("Target values are out of vocabulary range")
        
        # Create vocabulary indices tensor [vocab_size]
        vocab_indices = torch.arange(vocab_size, device=device, dtype=torch.float32)
        
        # Expand target to [B, seq_len, 1] for broadcasting
        mu = target.float().unsqueeze(-1)  # [B, seq_len, 1]
        
        # --- ADAPTIVE SIGMA COMPUTATION ---
        # sigma = alpha * mu + beta
        # This allows high expression genes to have larger tolerance
        sigma = self.adaptive_sigma_alpha * mu + self.adaptive_sigma_beta
        
        if torch.any(sigma <= 0):
            raise ValueError("Adaptive sigma must be positive")
        # --- END ADAPTIVE SIGMA ---
        
        # Expand vocab_indices to [1, 1, vocab_size] for broadcasting
        x = vocab_indices.view(1, 1, -1)  # [1, 1, vocab_size]
        
        # Compute Gaussian probabilities: exp(-(x - mu)^2 / (2 * sigma^2))
        # Broadcasting: mu [B, seq_len, 1], sigma [B, seq_len, 1], x [1, 1, vocab_size]
        # Result: [B, seq_len, vocab_size]
        squared_diff = (x - mu) ** 2
        gaussian_probs = torch.exp(-squared_diff / (2 * sigma ** 2))
        
        # Normalize to create a valid probability distribution
        # Sum over the vocab dimension and add small epsilon to avoid division by zero
        normalization = gaussian_probs.sum(dim=-1, keepdim=True) + 1e-10
        target_dist = gaussian_probs / normalization
        
        if torch.isnan(target_dist).any() or torch.isinf(target_dist).any():
            raise ValueError("NaN or Inf detected in Gaussian target distribution")
        
        return target_dist

    def _compute_soft_label_loss(self, logits: torch.Tensor, target: Union[torch.Tensor, SoftTarget], 
                                  hidden_states: Optional[torch.Tensor] = None, is_final_scale: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss: classification loss + ZINB loss (if enabled).
        
        For intermediate scales: uses KL divergence with soft labels
        For final scale: uses classification loss + optional ZINB loss
        
        Args:
            logits (torch.Tensor): Model predictions, shape [B, seq_len, vocab_size]
            target (Union[torch.Tensor, SoftTarget]): Either hard targets or soft target dict
            hidden_states (Optional[torch.Tensor]): Hidden states for ZINB parameter prediction [B, seq_len, embed_dim]
            is_final_scale (bool): Whether this is the final scale
            
        Returns:
            Tuple[torch.Tensor, Dict]: (total_loss, loss_dict with individual components)
        """
        loss_dict = {}
        
        # Handle hard labels (final scale)
        if isinstance(target, torch.Tensor):
            # 1. Classification loss (always computed)
            target_dist = self._create_gaussian_target_distribution(target, logits.device)
            log_probs = F.log_softmax(logits, dim=-1)
            
            if torch.isinf(log_probs).any():
                raise ValueError("Inf detected in log probabilities")
            
            classification_loss = F.kl_div(log_probs, target_dist, reduction='batchmean', log_target=False)
            
            if torch.isnan(classification_loss) or torch.isinf(classification_loss):
                raise ValueError("Invalid classification loss detected")
            
            loss_dict['classification'] = classification_loss.item()
            
            # 2. ZINB loss (only if enabled and is final scale)
            if is_final_scale and self.use_zinb_loss and hidden_states is not None:
                # Extract shared features for ZINB
                zinb_features = self.zinb_feature_extractor(hidden_states)
                
                # Predict ZINB parameters
                mu = self.zinb_mu_head(zinb_features).squeeze(-1)      # [B, seq_len]
                theta = self.zinb_theta_head(zinb_features).squeeze(-1) # [B, seq_len]
                pi = self.zinb_pi_head(zinb_features).squeeze(-1)      # [B, seq_len]
                
                # Compute ZINB loss
                zinb_loss = self.zinb_loss(mu, theta, pi, target.float())
                
                if torch.isnan(zinb_loss) or torch.isinf(zinb_loss):
                    logger.warning(f"Invalid ZINB loss detected! mu range: [{mu.min():.2f}, {mu.max():.2f}]")
                    # Fallback to classification only
                    total_loss = classification_loss
                    loss_dict['zinb'] = float('nan')
                    loss_dict['zinb_fallback'] = True
                else:
                    loss_dict['zinb'] = zinb_loss.item()
                    loss_dict['zinb_fallback'] = False
                    
                    # Hybrid loss: weighted combination
                    total_loss = (
                        (1 - self.zinb_loss_weight) * classification_loss +
                        self.zinb_loss_weight * zinb_loss
                    )
                    
                    # Store ZINB parameter statistics for monitoring
                    loss_dict['mu_mean'] = mu.mean().item()
                    loss_dict['mu_std'] = mu.std().item()
                    loss_dict['theta_mean'] = theta.mean().item()
                    loss_dict['pi_mean'] = pi.mean().item()
            else:
                # Not final scale or ZINB disabled: use classification only
                total_loss = classification_loss
                loss_dict['zinb'] = 0.0
            
            loss_dict['total'] = total_loss.item()
            return total_loss, loss_dict
        
        # Handle soft labels (intermediate scales)
        if not isinstance(target, dict) or 'floor_targets' not in target:
            raise ValueError("Soft target must be a dict containing 'floor_targets', 'ceil_targets', 'weights'")
        
        floor_targets = target['floor_targets']  # [B, seq_len]
        ceil_targets = target['ceil_targets']    # [B, seq_len]
        weights = target['weights']              # [B, seq_len]
        
        B, seq_len, vocab_size = logits.shape
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Construct target probability distribution
        target_dist = torch.zeros_like(log_probs)
        
        # Create indices for scatter operations
        batch_indices = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=logits.device).unsqueeze(0).expand(B, -1)
        
        # Set probabilities for floor targets: P(floor) = 1 - weight
        floor_probs = 1.0 - weights
        target_dist[batch_indices, seq_indices, floor_targets] = floor_probs
        
        # Set probabilities for ceil targets: P(ceil) = weight
        ceil_mask = (ceil_targets != floor_targets)
        if ceil_mask.any():
            ceil_probs = weights * ceil_mask.float()
            target_dist[batch_indices, seq_indices, ceil_targets] = ceil_probs
        
        if torch.isinf(log_probs).any():
            raise ValueError("Inf detected in log probabilities")
        
        # Compute KL divergence
        kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean', log_target=False)
        
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            raise ValueError("Invalid KL loss detected")
        
        loss_dict = {
            'total': kl_loss.item(),
            'classification': kl_loss.item(),
            'zinb': 0.0
        }
        
        return kl_loss, loss_dict

    def forward(
        self,
        histology_features: torch.Tensor,       # [B, 1024]
        spatial_coords: torch.Tensor,           # [B, 2]  normalised [0,1]
        target_genes: Optional[torch.Tensor] = None,          # [B, 200] for training
        cell_type_proportions: Optional[torch.Tensor] = None, # [B, num_cell_types]
        top_k: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for the model.

        Parameters
        ----------
        histology_features : [B, histology_feature_dim]
            Spot patch embeddings from UNI2 / CONCH / ResNet18.
        spatial_coords : [B, 2]
            Normalised (0-1) spot centre coordinates.
        target_genes : [B, num_genes], optional
            Quantised gene expression counts.  Required during training.
        cell_type_proportions : [B, num_cell_types], optional
            Row-normalised cell-type frequency vectors computed from
            ``cell_spatial/{slide_id}_spatial.txt``.  Each entry v[b,c] is
            the fraction of cells within 112 px of spot b that belong to
            cell type c.  When ``None`` or ``use_cell_type_proportion=False``
            the model falls back to histology-only conditioning.
        top_k : int, optional
            Top-k sampling for autoregressive inference.
        """
        # ── Step 1: base condition embedding (histology + spatial) ────────
        histology_embed = self.condition_processor(histology_features, spatial_coords)

        # ── Step 2: enrich with cell-type composition (scale_idx=None at top-level,
        #    per-scale injection happens inside forward_training/inference) ──
        if self.use_cell_type_proportion and cell_type_proportions is not None:
            # Top-level call: use scale_idx=None (neutral scale bias)
            # The per-scale call happens inside forward_training for each scale
            condition_embed = self.cell_proportion_fusion(
                histology_embed, cell_type_proportions, scale_idx=None
            )
        else:
            condition_embed = histology_embed

        # ── Step 3: dispatch to training / inference pass ─────────────────
        # For the contrastive loss we still need spot→cell-type assignments.
        # We derive them from the proportion vector itself (no cell embedding needed).
        if self.training:
            if target_genes is None:
                raise ValueError("target_genes must be provided during training.")
            spot_cell_assignments = self._proportions_to_assignments(cell_type_proportions)
            return self.forward_training(
                condition_embed, target_genes, spot_cell_assignments,
                cell_type_proportions, histology_embed=histology_embed
            )
        else:
            if target_genes is not None:
                spot_cell_assignments = self._proportions_to_assignments(cell_type_proportions)
                return self.forward_training(
                    condition_embed, target_genes, spot_cell_assignments,
                    cell_type_proportions, histology_embed=histology_embed
                )
            else:
                return self.forward_inference(condition_embed, top_k=top_k)

    @staticmethod
    def _proportions_to_assignments(
        cell_type_proportions: Optional[torch.Tensor],
    ) -> Optional[List[List[int]]]:
        """Convert proportion vector to a dummy assignments list for contrastive loss.

        SpotCellTypeContrastModule.type_encoder expects a list-of-lists of cell
        indices plus a ``cell_type_ids`` tensor.  Since we now work with
        aggregate proportion vectors instead of individual cells, we construct
        a synthetic assignment: for each spot b we create one virtual cell per
        cell type whose proportion > 0, using the type index as the cell ID.

        This keeps the existing InfoNCE loss path unchanged while removing the
        dependency on actual cell-level data.
        """
        if cell_type_proportions is None:
            return None
        B, num_types = cell_type_proportions.shape
        assignments = []
        for b in range(B):
            # Virtual cell indices: type c has index c if proportion > 0
            indices = [c for c in range(num_types) if cell_type_proportions[b, c].item() > 0.0]
            assignments.append(indices)
        return assignments
    
    def forward_training(
        self,
        condition_embed: torch.Tensor,                                   # [B, condition_embed_dim]
        target_genes: torch.Tensor,                                      # [B, num_genes]
        spot_cell_assignments: Optional[List[List[int]]] = None,         # pseudo-assignments from proportions
        cell_type_proportions: Optional[torch.Tensor] = None,           # [B, num_cell_types]
        histology_embed: Optional[torch.Tensor] = None,                  # [B, D] pre-fusion, for per-scale re-enrichment
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical training pass with teacher forcing using soft labels.
        
        histology_embed: when provided together with cell_type_proportions,
        each scale re-computes its own condition embedding using the scale-adaptive
        gate biases inside CellTypeProportionFusion.  This allows finer-grained
        scales to inject more cell-type signal than coarser scales.
        """
        B = condition_embed.shape[0]
        device = condition_embed.device
        # Use histology_embed for per-scale re-enrichment; fall back to condition_embed
        _histology_embed = histology_embed if histology_embed is not None else condition_embed
        
        # 1. Create all hierarchical targets with soft labels for intermediate scales
        # This preserves floating-point precision instead of lossy rounding
        hierarchical_targets = self._create_hierarchical_targets(target_genes)
        
        # Initialize storage for scale processing
        scale_embeddings = []  # Store embeddings from each scale for upsampling
        total_loss = 0.0
        final_predictions = None
        final_loss = torch.tensor(0.0, device=device) # Initialize final_loss
        
        # 2. Iteratively train each scale
        for scale_idx, (scale_dim, scale_target) in enumerate(zip(self.scale_dims, hierarchical_targets)):
            # Build cumulative input from all previous scales (not just the last)
            if scale_idx == 0:
                # For the first scale, input is just the start token
                x = self.start_token.expand(B, -1, -1) # [B, 1, D]
            else:
                # For subsequent scales, use ALL previous scale tokens (cumulative input)
                cumulative_input_tokens = []
                
                # Collect tokens from all previous scales
                for prev_idx in range(scale_idx):
                    prev_target = hierarchical_targets[prev_idx]
                    
                    # Extract hard tokens for embedding from soft or hard targets
                    if isinstance(prev_target, dict):
                        # Previous scale used soft labels - extract hard tokens for teacher forcing
                        floor_targets = prev_target['floor_targets']
                        ceil_targets = prev_target['ceil_targets']
                        weights = prev_target['weights']
                        
                        # Sample based on weights: if weight > 0.5, use ceil, otherwise floor
                        prev_scale_tokens = torch.where(weights > 0.5, ceil_targets, floor_targets)
                    else:
                        # Previous scale used hard labels
                        prev_scale_tokens = prev_target
                    
                    cumulative_input_tokens.append(prev_scale_tokens)
                
                # Concatenate all previous scale tokens
                all_prev_tokens = torch.cat(cumulative_input_tokens, dim=1)  # [B, cumulative_length]
                
                # Embed all previous tokens
                input_embed = self.gene_embedding(all_prev_tokens) # [B, cumulative_length, D]
                
                # Prepend start token
                start_token_expanded = self.start_token.expand(B, -1, -1) # [B, 1, D]
                x = torch.cat([start_token_expanded, input_embed], dim=1) # [B, 1 + cumulative_length, D]

            # GenAR-style sequence extension for all scales
            # Extend sequence with target positions for current scale
            # Use intelligent upsampling instead of zero placeholders when possible
            if scale_idx == 0:
                # First scale: still use zeros as we have no previous information
                target_positions = torch.zeros(B, scale_dim, self.embed_dim, device=device)
            elif scale_dim == self.num_genes:
                # Final scale: combine upsampled information with gene identity embeddings
                prev_embeddings = scale_embeddings[-1]  # Get most recent scale embeddings
                upsampled_positions = self.gene_upsampling(
                    prev_embeddings, 
                    source_scale_idx=scale_idx-1, 
                    target_scale_idx=scale_idx
                )
                identity_embeddings = self.gene_identity_embedding.weight.unsqueeze(0).expand(B, -1, -1)
                # Weighted combination: 70% upsampled + 30% identity
                target_positions = 0.7 * upsampled_positions + 0.3 * identity_embeddings
            else:
                # Intermediate scales: use upsampling from previous scale
                prev_embeddings = scale_embeddings[-1]  # Get most recent scale embeddings
                target_positions = self.gene_upsampling(
                    prev_embeddings,
                    source_scale_idx=scale_idx-1,
                    target_scale_idx=scale_idx
                )
            
            x = torch.cat([x, target_positions], dim=1)  # [B, cumulative_length + scale_dim, D]
            
            # Add scale and position embeddings
            current_seq_len = x.shape[1]
            scale_embed = self.scale_embedding(torch.tensor([scale_idx], device=device)).view(1, 1, -1)
            pos_embed = self._get_hierarchical_position_embedding(scale_idx, current_seq_len, device)
            x = x + pos_embed + scale_embed

            # ── Scale-adaptive condition re-enrichment (Improvement A) ────
            if self.use_cell_type_proportion and cell_type_proportions is not None:
                scale_condition_embed = self.cell_proportion_fusion(
                    _histology_embed, cell_type_proportions, scale_idx=scale_idx
                )
            else:
                scale_condition_embed = condition_embed
            
            # Create a causal mask for the current sequence length
            causal_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=device) * float('-inf'), diagonal=1)

            # Pass through transformer blocks with scale-specific condition
            for block in self.transformer_blocks:
                x = block(x, scale_condition_embed, causal_mask)

            # Extract predictions from the appropriate positions
            # For all scales: use the last scale_dim positions (GenAR-style autoregressive prediction)
            x_for_prediction = x[:, -scale_dim:, :]  # [B, scale_dim, D]
            
            # Get logits for the current scale's prediction
            x_for_prediction = self.head_norm(x_for_prediction, scale_condition_embed)
            
            # Apply dynamic gene identity modulation for all scales (conservative approach)
            # Get scale-appropriate gene identity conditions
            scale_conditions = self.gene_identity_pooling.get_scale_conditions(
                scale_idx=scale_idx,
                batch_size=B,
                gene_identity_embedding=self.gene_identity_embedding,
                device=device
            )
            
            if scale_conditions is None:
                raise ValueError(f"Missing gene identity conditions for scale {scale_idx}")

            # Apply FiLM modulation
            x_for_prediction = self.film_layer(x_for_prediction, scale_conditions)

            # ── Improvement E: CellFiLM at final scale ────────────────────
            # At the 200-gene scale, apply a lightweight FiLM directly from the
            # proportion vector. This is a 'last-mile' injection that bypasses
            # any signal dilution through the transformer layers.
            is_final = (scale_dim == self.num_genes)
            if (
                is_final
                and self.use_cell_type_proportion
                and hasattr(self, 'cell_film_head')
                and cell_type_proportions is not None
            ):
                # gamma, beta: [B, D]  (init ≈ 1, 0 so residual starts near identity)
                cell_film_params = self.cell_film_head(cell_type_proportions)  # [B, 2D]
                cell_gamma, cell_beta = cell_film_params.chunk(2, dim=-1)      # each [B, D]
                cell_gamma = cell_gamma.unsqueeze(1)   # [B, 1, D]
                cell_beta  = cell_beta.unsqueeze(1)    # [B, 1, D]
                x_for_prediction = (1.0 + cell_gamma) * x_for_prediction + cell_beta
                logger.debug("Applied CellFiLM at final scale (200 genes)")
            
            logits = self.output_head(x_for_prediction) # Shape: [B, scale_dim, vocab_size]

            logits_for_loss = logits
            
            # Calculate loss for the current scale using hybrid loss (classification + ZINB)
            loss, loss_dict = self._compute_soft_label_loss(
                logits_for_loss, 
                scale_target, 
                hidden_states=x_for_prediction if is_final else None,
                is_final_scale=is_final
            )
            total_loss += loss
            
            if is_final:
                final_loss_dict = loss_dict
            
            predicted_tokens = torch.argmax(logits_for_loss, dim=-1)  # [B, scale_dim]
            current_scale_embeddings = self.gene_embedding(predicted_tokens)
            scale_embeddings.append(current_scale_embeddings)
            
            if is_final:
                if self.use_zinb_for_inference and self.use_zinb_loss:
                    zinb_features = self.zinb_feature_extractor(x_for_prediction)
                    mu_pred = self.zinb_mu_head(zinb_features).squeeze(-1)
                    final_predictions = torch.round(mu_pred).long()
                    final_predictions = torch.clamp(final_predictions, 0, self.vocab_size - 1)
                else:
                    final_predictions = predicted_tokens.float()
                
                final_loss = loss
                
                # ── Soft-positive InfoNCE（仅在最终 scale 计算一次）─────────
                if (
                    self.use_cell_type_contrast
                    and spot_cell_assignments is not None
                    and cell_type_proportions is not None
                ):
                    virtual_cell_type_ids = torch.arange(
                        self.num_cell_types, dtype=torch.long, device=device
                    )
                    loss_contrast = self.cell_type_contrast(
                        spot_features=x_for_prediction,
                        spot_cell_assignments=spot_cell_assignments,
                        cell_type_ids=virtual_cell_type_ids,
                        device=device,
                        cell_type_proportions=cell_type_proportions,  # soft labels
                    )
                    total_loss = total_loss + self.contrast_weight * loss_contrast
                else:
                    loss_contrast = torch.tensor(0.0, device=device)
        
        return {
            'loss': total_loss / self.num_scales,
            'loss_final': final_loss,
            'loss_contrast': loss_contrast if 'loss_contrast' in locals() else torch.tensor(0.0, device=device),
            'predictions': final_predictions.float(),
            'targets': target_genes.float(),
            'loss_dict': final_loss_dict if 'final_loss_dict' in locals() else {}
        }

    def forward_inference(
        self,
        condition_embed: torch.Tensor,      # [B, condition_embed_dim]
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical inference pass.
        Autoregressively generates tokens for each scale, where the output of a coarser
        scale becomes the input for the next, finer scale.
        
        Note: top_p sampling is not currently implemented but kept for future compatibility.
        """
        B = condition_embed.shape[0]
        device = condition_embed.device

        # Store generated tokens from previous scales for cumulative input
        all_generated_scale_tokens = []
        scale_embeddings = []
        generated_tokens = None
        
        if seed is not None:
            torch.manual_seed(seed)
        
        for scale_idx, scale_dim in enumerate(self.scale_dims):
            # Build cumulative input
            if scale_idx == 0:
                x = self.start_token.expand(B, -1, -1)
            else:
                cumulative_tokens = torch.cat(all_generated_scale_tokens, dim=1)
                input_embed = self.gene_embedding(cumulative_tokens)
                start_token_expanded = self.start_token.expand(B, -1, -1)
                x = torch.cat([start_token_expanded, input_embed], dim=1)
            
            # Extend with target positions
            if scale_idx == 0:
                target_positions = torch.zeros(B, scale_dim, self.embed_dim, device=device)
            elif scale_dim == self.num_genes:
                prev_embeddings = scale_embeddings[-1]
                upsampled_positions = self.gene_upsampling(
                    prev_embeddings,
                    source_scale_idx=scale_idx-1,
                    target_scale_idx=scale_idx
                )
                identity_embeddings = self.gene_identity_embedding.weight.unsqueeze(0).expand(B, -1, -1)
                target_positions = 0.7 * upsampled_positions + 0.3 * identity_embeddings
            else:
                prev_embeddings = scale_embeddings[-1]
                target_positions = self.gene_upsampling(
                    prev_embeddings,
                    source_scale_idx=scale_idx-1,
                    target_scale_idx=scale_idx
                )
            
            x = torch.cat([x, target_positions], dim=1)
            
            # Add embeddings
            current_seq_len = x.shape[1]
            scale_embed = self.scale_embedding(torch.tensor([scale_idx], device=device)).view(1, 1, -1)
            pos_embed = self._get_hierarchical_position_embedding(scale_idx, current_seq_len, device)
            x = x + pos_embed + scale_embed
            
            # Causal mask
            causal_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=device) * float('-inf'), diagonal=1)
            
            # Transformer forward
            for block in self.transformer_blocks:
                x = block(x, condition_embed, causal_mask)
            
            # Extract predictions
            x_for_prediction = x[:, -scale_dim:, :]
            
            x_for_prediction = self.head_norm(x_for_prediction, condition_embed)
            
            # Apply gene modulation
            scale_conditions = self.gene_identity_pooling.get_scale_conditions(
                scale_idx=scale_idx,
                batch_size=B,
                gene_identity_embedding=self.gene_identity_embedding,
                device=device
            )
            x_for_prediction = self.film_layer(x_for_prediction, scale_conditions)
            
            logits = self.output_head(x_for_prediction)  # [B, scale_dim, vocab_size]
            
            # Check if this is the final scale
            is_final_scale = (scale_dim == self.num_genes)
            
            if is_final_scale and self.use_zinb_for_inference and self.use_zinb_loss:
                # Optional: Use ZINB predictions for final scale
                # This is more experimental and may be less stable
                zinb_features = self.zinb_feature_extractor(x_for_prediction)
                mu = self.zinb_mu_head(zinb_features).squeeze(-1)  # [B, scale_dim]
                
                # Use mu (mean) as the prediction, rounded to nearest integer
                sampled_tokens = torch.round(mu).long()
                sampled_tokens = torch.clamp(sampled_tokens, 0, self.vocab_size - 1)
            else:
                # Default: Use categorical sampling from classification logits
                # This is the original method and generally more stable
                # For non-final scales: use categorical sampling with temperature and top-k
                # --- START: Top-k Sampling Logic ---
                # Apply temperature scaling
                logits = logits / temperature
                
                # Optional top-k filtering
                if top_k is not None and top_k > 0:
                    # Get top-k values and indices
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    
                    # Create mask for values outside top-k
                    mask = torch.full_like(logits, float('-inf'))
                    mask.scatter_(-1, top_k_indices, top_k_values)
                    logits = mask
                
                # Optional top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)  # [B, scale_dim, vocab_size]
                sampled_tokens = torch.multinomial(
                    probs.view(-1, probs.size(-1)),  # Flatten to [B*scale_dim, vocab_size]
                    num_samples=1
                ).view(B, scale_dim)  # Reshape to [B, scale_dim]
                # --- END: Top-k Sampling Logic ---
            
            # Store current scale tokens for future cumulative input
            all_generated_scale_tokens.append(sampled_tokens)
            
            # NEW: Store current scale embeddings for next scale's upsampling
            current_scale_embeddings = self.gene_embedding(sampled_tokens)  # [B, scale_dim, embed_dim]
            scale_embeddings.append(current_scale_embeddings)
            
            # Keep the last generated tokens for final output
            generated_tokens = sampled_tokens

        return {
            'generated_sequence': generated_tokens.float()
        }

    def inference(
        self,
        histology_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        cell_type_proportions: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode for gene expression prediction.

        Parameters
        ----------
        histology_features : [B, histology_feature_dim]
        spatial_coords : [B, 2]
        cell_type_proportions : [B, num_cell_types], optional
            If provided, enriches the condition embedding with cell-type composition.
        temperature / top_k / top_p / seed : sampling parameters
        """
        self.eval()
        with torch.no_grad():
            histology_embed = self.condition_processor(histology_features, spatial_coords)
            if self.use_cell_type_proportion and cell_type_proportions is not None:
                condition_embed = self.cell_proportion_fusion(histology_embed, cell_type_proportions)
            else:
                condition_embed = histology_embed
            return self.forward_inference(condition_embed, temperature, top_k, top_p, seed)

    def save_checkpoint(self, save_path: str, epoch: Optional[int] = None):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_genes': self.num_genes,
            'scale_dims': self.scale_dims,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'histology_feature_dim': self.histology_feature_dim,
            'spatial_coord_dim': self.spatial_coord_dim,
            'condition_embed_dim': self.condition_embed_dim,
            # NEW keys
            'use_cell_type_proportion': self.use_cell_type_proportion,
            'num_cell_types': self.num_cell_types,
            # legacy key kept for backward compat (stores num_cell_types)
            'cell_embed_dim': self.cell_embed_dim,
            'use_cell_emb': self.use_cell_type_proportion,  # alias
            'adaptive_sigma_alpha': self.adaptive_sigma_alpha,
            'adaptive_sigma_beta': self.adaptive_sigma_beta,
            'use_cell_type_contrast': self.use_cell_type_contrast,
            'contrast_weight': self.contrast_weight,
            'contrast_proj_dim': self.contrast_proj_dim,
            'contrast_num_cell_types': self.contrast_num_cell_types,
            'epoch': epoch
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to: {save_path}")
    
    @classmethod
    def load_checkpoint(cls, ckpt_path: str, device: str = 'cuda') -> 'MultiScaleGenAR':
        """
        Load model from checkpoint
        
        Args:
            ckpt_path: Path to checkpoint
            device: Device to load model on
            
        Returns:
            Loaded MultiScaleGenAR model
        """
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Create model with saved configuration
        # Support both new checkpoints (use_cell_type_proportion) and legacy ones (use_cell_emb)
        use_cell_type_proportion = checkpoint.get(
            'use_cell_type_proportion',
            checkpoint.get('use_cell_emb', True)  # legacy fallback
        )
        # num_cell_types stored as 'num_cell_types' in new ckpts,
        # or fall back to contrast_num_cell_types from old ckpts (they were equal)
        num_cell_types = checkpoint.get(
            'num_cell_types',
            checkpoint.get('contrast_num_cell_types', 5)
        )
        # NOTE: contrast_num_cell_types is no longer a separate __init__ parameter;
        # it is automatically set equal to num_cell_types inside __init__.
        # Do NOT pass it here to avoid TypeError.
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            num_genes=checkpoint['num_genes'],
            scale_dims=checkpoint['scale_dims'],
            embed_dim=checkpoint['embed_dim'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            histology_feature_dim=checkpoint['histology_feature_dim'],
            spatial_coord_dim=checkpoint['spatial_coord_dim'],
            condition_embed_dim=checkpoint['condition_embed_dim'],
            use_cell_type_proportion=use_cell_type_proportion,
            num_cell_types=num_cell_types,
            adaptive_sigma_alpha=checkpoint.get('adaptive_sigma_alpha', 0.1),
            adaptive_sigma_beta=checkpoint.get('adaptive_sigma_beta', 1.0),
            use_cell_type_contrast=checkpoint.get('use_cell_type_contrast', True),
            contrast_weight=checkpoint.get('contrast_weight', 0.1),
            contrast_proj_dim=checkpoint.get('contrast_proj_dim', 256),
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from: {ckpt_path}")
        if 'epoch' in checkpoint:
            logger.info(f"Loaded model from epoch: {checkpoint['epoch']}")
        
        return model
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        condition_params = sum(p.numel() for p in self.condition_processor.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_blocks.parameters())
        embedding_params = (
            self.gene_embedding.weight.numel()
            + sum(p.numel() for p in self.hierarchical_pos_embedding.parameters())
            + self.scale_embedding.weight.numel()
        )
        output_params = (
            sum(p.numel() for p in self.head_norm.parameters())
            + sum(p.numel() for p in self.output_head.parameters())
        )
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'condition_processor_parameters': condition_params,
            'transformer_parameters': transformer_params,
            'embedding_parameters': embedding_params,
            'output_parameters': output_params,
            'num_genes': self.num_genes,
            'scale_dims': self.scale_dims,
            'num_scales': self.num_scales,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'total_sequence_length': self.num_genes + 1,
            'use_cell_type_proportion': self.use_cell_type_proportion,
            'num_cell_types': self.num_cell_types,
        }
        
        if self.use_cell_type_proportion and hasattr(self, 'cell_proportion_fusion'):
            fusion_params = sum(p.numel() for p in self.cell_proportion_fusion.parameters())
            info['cell_proportion_fusion_v2_parameters'] = fusion_params
            info['cell_proportion_fusion_design'] = 'element-wise gate + multi-query + scale-adaptive'

        if self.use_cell_type_proportion and hasattr(self, 'cell_film_head'):
            film_params = sum(p.numel() for p in self.cell_film_head.parameters())
            info['cell_film_head_parameters'] = film_params
        
        if self.use_cell_type_contrast and hasattr(self, 'cell_type_contrast'):
            contrast_params = sum(p.numel() for p in self.cell_type_contrast.parameters())
            info['cell_type_contrast_parameters'] = contrast_params
            info['contrast_proj_dim'] = self.contrast_proj_dim
            info['contrast_weight'] = self.contrast_weight
            info['contrast_loss'] = 'soft-positive InfoNCE'
        
        return info

    def enable_kv_cache(self):
        """Enable KV caching for all transformer blocks during inference"""
        for block in self.transformer_blocks:
            block.enable_kv_cache(True)
    
    def disable_kv_cache(self):
        """Disable KV caching for all transformer blocks during training"""
        for block in self.transformer_blocks:
            block.enable_kv_cache(False)
    
    def enable_multi_scale_gene_modulation(self):
        """Enable multi-scale gene identity modulation"""
        raise RuntimeError("Multi-scale gene identity modulation cannot be toggled in strict mode")

    def disable_multi_scale_gene_modulation(self):
        """Disable multi-scale gene identity modulation (unsupported in strict mode)"""
        raise RuntimeError("Multi-scale gene identity modulation cannot be toggled in strict mode")

    def _compute_weighted_cross_entropy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Distance-aware cross-entropy where nearer tokens incur smaller penalties."""
        vocab_size = logits.shape[-1]
        
        # Distance weights
        token_ids = torch.arange(vocab_size, device=logits.device, dtype=torch.float32)  # [vocab_size]
        target_values = targets.float().unsqueeze(1)  # [total_predictions, 1]

        # Absolute distance between predicted token ids and targets
        distances = torch.abs(token_ids.unsqueeze(0) - target_values)  # [total_predictions, vocab_size]

        # Gaussian weights favouring nearby tokens
        sigma = vocab_size * 0.1
        weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))

        # Weighted log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [total_predictions, vocab_size]

        # Apply weights elementwise
        weighted_log_probs = log_probs * weights  # [total_predictions, vocab_size]

        # Gather weighted log-probabilities for targets
        target_log_probs = weighted_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [total_predictions]

        # Mean loss
        loss = -target_log_probs.mean()
        
        return loss


# Backward compatibility alias
GenARModel = MultiScaleGenAR