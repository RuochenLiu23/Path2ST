import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class GeneIdentityPooling(nn.Module):
    def __init__(
        self,
        num_genes: int = 200,
        scale_dims: Tuple[int, ...] = (1, 4, 8, 40, 100, 200),
        embed_dim: int = 512,
        enable_pooling: bool = True
    ):
        super().__init__()
        self.num_genes = num_genes
        self.scale_dims = scale_dims
        self.embed_dim = embed_dim
        self.enable_pooling = enable_pooling

        if not enable_pooling:
            raise ValueError("Gene Identity Pooling cannot be disabled in strict mode")

        self.scale_poolers = nn.ModuleDict()
        for i, dim in enumerate(scale_dims[:-1]):
            if dim == 1:
                self.scale_poolers[f'scale_{i}'] = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(0.1)
                )
            else:
                self.scale_poolers[f'scale_{i}'] = nn.Sequential(
                    nn.AdaptiveAvgPool1d(dim),
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(0.1)
                )

        self._init_pooling_weights()

        logger.info("Gene Identity Pooling initialised")
        logger.info(f"   - Enable pooling: {enable_pooling}")
        logger.info(f"   - Number of pooling layers: {len(self.scale_poolers)}")
        for i, dim in enumerate(scale_dims[:-1]):
            logger.info(f"   - Scale {i} (dim={dim}): pooling layer created")

    def _init_pooling_weights(self):
        for pooler in self.scale_poolers.values():
            for module in pooler:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def get_scale_identity(self, scale_idx: int, gene_identity_embedding: nn.Embedding) -> torch.Tensor:
        scale_dim = self.scale_dims[scale_idx]
        if scale_dim == self.num_genes:
            return gene_identity_embedding.weight
        pooler_key = f'scale_{scale_idx}'
        if pooler_key not in self.scale_poolers:
            raise ValueError(f"No pooler found for scale {scale_idx}")
        full_identities = gene_identity_embedding.weight
        full_identities_t = full_identities.transpose(0, 1).unsqueeze(0)
        pooler = self.scale_poolers[pooler_key]
        adaptive_pool = pooler[0]
        linear_proj = pooler[1]
        layer_norm = pooler[2]
        dropout = pooler[3]
        pooled_features = adaptive_pool(full_identities_t)
        pooled_features = pooled_features.transpose(1, 2)
        projected = linear_proj(pooled_features)
        normalized = layer_norm(projected)
        final_output = dropout(normalized)
        scale_identities = final_output.squeeze(0)
        return scale_identities

    def get_scale_conditions(
        self,
        scale_idx: int,
        batch_size: int,
        gene_identity_embedding: nn.Embedding,
        device: torch.device
    ) -> torch.Tensor:
        scale_identities = self.get_scale_identity(scale_idx, gene_identity_embedding)
        if scale_identities is None:
            return None
        scale_dim = self.scale_dims[scale_idx]
        scale_conditions = scale_identities.unsqueeze(0).expand(batch_size, scale_dim, self.embed_dim)
        scale_conditions = scale_conditions.to(device)
        return scale_conditions

    def enable(self):
        raise RuntimeError("Gene Identity Pooling cannot be enabled/disabled at runtime in strict mode")

    def disable(self):
        raise RuntimeError("Gene Identity Pooling cannot be enabled/disabled at runtime in strict mode")
