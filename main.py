import os
import sys
import argparse
import logging
import warnings
from copy import deepcopy
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytorch_lightning as pl

from dataset.hest_dataset import STDataset
from model import ModelInterface
from configs import DATASETS, DEFAULT_DATA_ROOT, ENCODER_FEATURE_DIMS
from utils import (
    load_callbacks,
    load_loggers,
    fix_seed
)
from torch.utils.data import DataLoader

# Configure logger
logger = logging.getLogger(__name__)
logging.getLogger('model.path2st').setLevel(logging.WARNING)
logging.getLogger('model.model_utils').setLevel(logging.INFO)

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")


# Path2ST model configuration
PATH2ST_CONFIG = {
        'model_name': 'PATH2ST',
        'num_genes': 200,
        'histology_feature_dim': 1536,  # Depends on encoder choice
        'spatial_coord_dim': 2,
        
        # Multi-scale configuration (progressively refines to 200 genes)
        'gene_patch_nums': (1, 4, 8, 40, 100, 200),
        # vocab_size is derived from max_gene_count (max_gene_count + 1)
        'embed_dim': 512,  # Reduced from 768
        'num_heads': 8,    # Reduced from 12
        'num_layers': 8,   # Reduced from 12
        'mlp_ratio': 3.0,  # Reduced from 4.0

        # Dropout configuration
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,

        # Conditioning configuration
        'condition_embed_dim': 512,  # Matches embed_dim
        'cond_drop_rate': 0.1,

        # Cell-type proportion fusion 
        # Set use_cell_type_proportion=True to enrich spot embeddings with
        # per-spot cell-type frequency vectors derived from CellViT segmentation.
        'use_cell_type_proportion': True,
        'num_cell_types': 5,           # Must match STDataset.num_cell_types
        'cell_type_fusion_heads': 8,
        'cell_type_fusion_dropout': 0.1,

        # ZINB hybrid loss configuration
        # Auxiliary ZINB loss helps model learn count-data distribution
        'use_zinb_loss': True,
        'zinb_loss_weight': 0.3,       # α: weight of ZINB vs classification loss
        'use_zinb_for_inference': False,
        'zinb_ridge_lambda': 1e-5,

        # Cell-type contrastive learning (InfoNCE)
        # Pulls spots with similar cell composition together in embedding space
        'use_cell_type_contrast': True,
        'contrast_weight': 0.1,        # λ: weight of InfoNCE loss
        'contrast_proj_dim': 256,

        # Misc parameters
        'norm_eps': 1e-6,
        'shared_aln': False,
        'attn_l2_norm': True,

        # Adaptive loss hyperparameters
        'adaptive_sigma_alpha': 0.01,
        'adaptive_sigma_beta': 1.0
}

MODEL_CONFIGS = {
    'PATH2ST': PATH2ST_CONFIG,
}

# Default training configuration
DEFAULT_CONFIG = {
    'GENERAL': {
        'seed': 2021,
        'log_path': './logs', 
        'debug': False
    },
    'DATA': {
        'train_dataloader': {
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True,
            'persistent_workers': True
        },
        'val_dataloader': {
            'batch_size': 64,  # Larger validation batch size to speed up evaluation
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': False,
            'persistent_workers': True
        },
        'test_dataloader': {
            'batch_size': 64,  # Match validation batch size for testing
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': False,
            'persistent_workers': True
        }
    },
    'TRAINING': {
        'num_epochs': 200,
        'learning_rate': 1.0e-4,
        'weight_decay': 1.0e-4,
        'mode': 'min',
        'monitor': 'train_loss_final',
        'lr_scheduler': {
            'name': 'reduce_on_plateau',
            'monitor': 'train_loss_final',
            'mode': 'min',
            'patience': 0,  # Disabled by default; opt-in via CLI
            'factor': 0.5
        },
        'gradient_clip_val': 1.0
    },
    'CALLBACKS': {
        'early_stopping': {
            'monitor': 'train_loss_final',
            'patience': 100,  # Large value effectively disables early stopping
            'mode': 'min',
            'min_delta': 0.0
        },
        'model_checkpoint': {
            'monitor': 'train_loss_final',
            'save_top_k': 1,
            'mode': 'min',
            'filename': 'best-epoch={epoch:02d}-loss={train_loss_final:.4f}'
        },
        'learning_rate_monitor': {
            'logging_interval': 'epoch'
        }
    },
    'INFERENCE': {
        'top_k': 1
    },
    'MULTI_GPU': {
        'find_unused_parameters': False,  # No unused params in the new design
        'accumulate_grad_batches': 1
    }
}


def get_parse():
    parser = argparse.ArgumentParser(
        description='Simplified Training for Spatial Transcriptomics Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --dataset PRAD --gpus 4
  
  # With custom parameters
  python main.py --dataset PRAD --encoder uni \\
      --gpus 4 --epochs 200 --batch_size 256 --lr 1e-4
  
  # Single GPU training
  python main.py --dataset her2st --gpus 1
  
  # Test mode with checkpoint
  python src/main.py --dataset her2st --mode test --gpus 1 \\
      --ckpt_path logs/her2st/Path2ST/best-epoch.ckpt
        """
    )
    
    # Core arguments
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()),
                        help='Dataset name (PRAD, her2st, mouse_brain)')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root directory containing dataset folders '
                             '(default: $Path2ST_DATA_ROOT or ./data)')
    parser.add_argument('--model', type=str, default='PATH2ST', choices=list(MODEL_CONFIGS.keys()),
                        help='Model type (default: PATH2ST)')
    parser.add_argument('--encoder', type=str, choices=list(ENCODER_FEATURE_DIMS.keys()),
                        help='Encoder type (uni, conch, resnet18); defaults to the dataset recommendation')

    # Training arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size (default: 256)')
    parser.add_argument('--lr', type=float,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int,
                        help='LR scheduler patience; enables the scheduler only when provided')

    # Multi-GPU arguments
    parser.add_argument('--strategy', type=str, default='auto',
                        choices=['auto', 'ddp', 'ddp_spawn', 'dp'],
                        help='Distributed strategy (default: auto, DDP when applicable)')
    parser.add_argument('--sync-batchnorm', action='store_true',
                        help='Enable synchronized BatchNorm (recommended for multi-GPU)')

    # Gene-count arguments
    parser.add_argument('--max-gene-count', type=int, default=500,
                        help='Upper bound for gene counts (default: 500)')

    # Miscellaneous options
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Execution mode: train or test (default: train)')
    parser.add_argument('--seed', type=int,
                        help='Random seed (default: 2021)')

    # Testing arguments
    parser.add_argument('--ckpt_path', type=str,
                        help='Checkpoint path required when --mode test is used')

    # Multi-scale ablation options
    parser.add_argument('--scale-config', type=str, default='default',
                        choices=['default', 'flat', 'single'],
                        help='Multi-scale layout: default=(1,4,8,40,100,200), flat=(1,20,200), single=(200)')

    # Cell-type proportion fusion options
    parser.add_argument('--use-cell', action='store_true', default=True,
                        help='Enable cell-type proportion fusion (default: True)')
    parser.add_argument('--no-cell', dest='use_cell', action='store_false',
                        help='Disable cell-type proportion fusion (histology only)')
    parser.add_argument('--num-cell-types', type=int, default=5,
                        help='Number of distinct cell types in the dataset (default: 5). '
                             'Must match the number of types in cell_spatial/*.txt files.')

    # Backward-compatibility flag (legacy)
    parser.add_argument('--config', type=str,
                        help='[Deprecated] Use --dataset instead')
    
    return parser


def build_config_from_args(args):
    """Build the runtime configuration from parsed arguments."""
    from addict import Dict

    # Legacy config files are not supported in this strict mode
    if args.config:
        raise ValueError("Legacy --config is not supported; use --dataset and CLI overrides")

    # Required parameters
    if not args.dataset:
        raise ValueError("`--dataset` must be specified")

    if args.dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {args.dataset}; valid options: {list(DATASETS.keys())}")

    if args.mode == 'test' and not args.ckpt_path:
        raise ValueError("`--ckpt_path` is required when running in test mode")

    if args.ckpt_path and not os.path.exists(args.ckpt_path):
        raise ValueError(f"Checkpoint not found: {args.ckpt_path}")

    model_name = (args.model or 'PATH2ST').upper()
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}; choices: {list(MODEL_CONFIGS.keys())}")

    logger.info("Configuration: dataset=%s model=%s mode=%s", args.dataset, model_name, args.mode)

    # Dataset and model metadata
    dataset_info = DATASETS[args.dataset]
    data_root = os.path.abspath(args.data_root)
    dataset_path = os.path.join(data_root, dataset_info['dir_name'])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    model_info = deepcopy(MODEL_CONFIGS[model_name])

    # Determine encoder
    encoder_name = args.encoder or dataset_info['recommended_encoder']

    # Resolve GPU-related arguments
    devices = args.gpus
    strategy = 'ddp' if devices > 1 and args.strategy == 'auto' else args.strategy
    sync_batchnorm = getattr(args, 'sync_batchnorm', False) or (devices > 1)

    # Build configuration object
    config = Dict(DEFAULT_CONFIG)

    # Configure multi-scale settings based on the CLI flag
    SCALE_CONFIGS = {
        'default': (1, 4, 8, 40, 100, 200),
        'flat': (1, 20, 200),
        'single': (200,)
    }

    scale_config_name = getattr(args, 'scale_config', 'default').replace('-', '_')
    selected_scales = SCALE_CONFIGS[scale_config_name]

    # Update log path to include the dataset, model, and scale configuration
    if scale_config_name != 'default':
        config.GENERAL.log_path = f'./logs/{args.dataset}/Path2ST_scale_{scale_config_name}'
    else:
        config.GENERAL.log_path = f'./logs/{args.dataset}/Path2ST'

    logger.debug("Selected multi-scale layout: %s = %s", scale_config_name, selected_scales)

    # Update model configuration
    config.MODEL = Dict(model_info)
    config.MODEL.feature_dim = ENCODER_FEATURE_DIMS[encoder_name]

    config.MODEL.scale_dims = selected_scales
    config.MODEL.gene_patch_nums = selected_scales

    # Adjust gene-count related settings
    max_gene_count = getattr(args, 'max_gene_count', 500)
    # num_genes remains fixed at 200

    # vocab_size mirrors the range 0..max_gene_count
    vocab_size = max_gene_count + 1
    config.MODEL.vocab_size = vocab_size
    config.MODEL.max_gene_count = max_gene_count
    
    # Override training hyperparameters from CLI overrides
    if args.epochs:
        config.TRAINING.num_epochs = args.epochs
    if args.lr:
        config.TRAINING.learning_rate = args.lr
    if args.weight_decay:
        config.TRAINING.weight_decay = args.weight_decay
    batch_size = getattr(args, 'batch_size', None)
    if batch_size:
        config.DATA.train_dataloader.batch_size = batch_size
    if args.patience is not None:
        if args.patience < 0:
            raise ValueError("`--patience` must be >= 0")
        # Enable LR scheduler only when patience is explicitly set
        config.TRAINING.lr_scheduler.patience = args.patience
        config.TRAINING.lr_scheduler.name = 'reduce_on_plateau'
        config.TRAINING.lr_scheduler.monitor = config.TRAINING.monitor
        config.TRAINING.lr_scheduler.mode = config.TRAINING.mode
        # Mirror the patience value into the early stopping callback
        if args.patience == 0:
            # Setting zero keeps early stopping disabled
            config.CALLBACKS.early_stopping.patience = 10000
        else:
            # Otherwise use twice the LR scheduler patience
            config.CALLBACKS.early_stopping.patience = max(10, args.patience * 2)

    # Optional seed override
    if args.seed:
        config.GENERAL.seed = args.seed
    
    # Dataset-related settings
    config.mode = args.mode
    config.expr_name = args.dataset
    config.data_path = dataset_path
    config.slide_val = dataset_info['val_slides']
    config.slide_test = dataset_info['test_slides']
    config.encoder_name = encoder_name
    config.gene_count_mode = 'discrete_tokens'
    config.max_gene_count = getattr(args, 'max_gene_count', 500)
    
    # Optional checkpoint path
    if args.ckpt_path:
        config.ckpt_path = args.ckpt_path
    
    # Multi-GPU configuration
    config.devices = devices
    config.strategy = strategy
    config.sync_batchnorm = sync_batchnorm
    
    # Runtime metadata
    config.GENERAL.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.config = 'built-in'
    
    # Report patience configuration
    lr_patience = config.TRAINING.lr_scheduler.patience
    early_patience = config.CALLBACKS.early_stopping.patience
    patience_status = "disabled" if lr_patience == 0 else f"enabled (LR scheduler: {lr_patience}, early stopping: {early_patience})"
    
    # Align Path2ST model configuration
    config.MODEL.histology_feature_dim = ENCODER_FEATURE_DIMS[encoder_name]
    config.MODEL.gene_count_mode = config.gene_count_mode
    config.MODEL.max_gene_count = config.max_gene_count

    # Cell-type proportion fusion settings
    use_cell = getattr(args, 'use_cell', True)
    num_cell_types = getattr(args, 'num_cell_types', 5)
    config.MODEL.use_cell_type_proportion = use_cell
    config.MODEL.num_cell_types = num_cell_types

    config.MODEL.contrast_num_cell_types = num_cell_types
    # Dataset needs to know these too (picked up in create_dataloaders)
    config.use_cell_features = use_cell
    config.num_cell_types = num_cell_types
    monitor_metric = 'train_loss_final'
    config.TRAINING.monitor = monitor_metric
    config.TRAINING.mode = 'min'
    config.TRAINING.lr_scheduler.monitor = config.TRAINING.monitor
    config.TRAINING.lr_scheduler.mode = config.TRAINING.mode
    config.CALLBACKS.early_stopping.monitor = monitor_metric
    config.CALLBACKS.early_stopping.mode = 'min'
    config.CALLBACKS.model_checkpoint.monitor = monitor_metric
    config.CALLBACKS.model_checkpoint.mode = 'min'
    config.CALLBACKS.model_checkpoint.filename = f'best-epoch={{epoch:02d}}-{monitor_metric}={{{monitor_metric}:.6f}}'
    logger.debug("Path2ST monitor: %s (minimization)", monitor_metric)

    logger.info(
        "Training setup: dataset=%s model=%s encoder=%s gpus=%s epochs=%s batch_size=%s lr=%s gene_vocab=%s use_cell=%s num_cell_types=%s",
        args.dataset,
        model_name,
        encoder_name,
        devices,
        config.TRAINING.num_epochs,
        config.DATA.train_dataloader.batch_size,
        config.TRAINING.learning_rate,
        vocab_size,
        getattr(args, 'use_cell', True),
        getattr(args, 'num_cell_types', 5),
    )

    return config


def create_dataloaders(config):
    """Instantiate train/val/test dataloaders."""
    # Shared dataset parameters
    base_params = {
        'data_path': config.data_path,
        'expr_name': config.expr_name,
        'slide_val': config.slide_val,
        'slide_test': config.slide_test,
        'encoder_name': config.encoder_name,
        'max_gene_count': getattr(config, 'max_gene_count', 500),
        # Cell-type proportion fusion: load cell_spatial files when enabled
        'use_cell_features': getattr(config, 'use_cell_features', False),
        'num_cell_types': getattr(config, 'num_cell_types', 5),
    }
    
    # Build datasets
    train_dataset = STDataset(mode='train', **base_params)
    val_dataset = STDataset(mode='val', **base_params)
    test_dataset = STDataset(mode='test', **base_params)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATA.train_dataloader.batch_size,
        shuffle=config.DATA.train_dataloader.shuffle,
        num_workers=config.DATA.train_dataloader.num_workers,
        pin_memory=config.DATA.train_dataloader.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DATA.val_dataloader.batch_size,
        shuffle=config.DATA.val_dataloader.shuffle,
        num_workers=config.DATA.val_dataloader.num_workers,
        pin_memory=config.DATA.val_dataloader.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.DATA.test_dataloader.batch_size,
        shuffle=config.DATA.test_dataloader.shuffle,
        num_workers=config.DATA.test_dataloader.num_workers,
        pin_memory=config.DATA.test_dataloader.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def main(config):
    """Entry point for training or evaluation."""
    monitor_metric = 'train_loss_final'
    monitor_mode = 'min'

    config.TRAINING.monitor = monitor_metric
    config.TRAINING.mode = monitor_mode
    config.CALLBACKS.early_stopping.monitor = monitor_metric
    config.CALLBACKS.early_stopping.mode = monitor_mode
    config.CALLBACKS.model_checkpoint.monitor = monitor_metric
    config.CALLBACKS.model_checkpoint.mode = monitor_mode
    config.CALLBACKS.model_checkpoint.filename = f'best-epoch={{epoch:02d}}-{monitor_metric}={{{monitor_metric}:.4f}}'
    logger.debug("Path2ST monitor forced to %s (mode: %s)", monitor_metric, monitor_mode)

    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Prepare a copy that is safe to serialize in checkpoints
    clean_config = type(config)(config)
    # Ensure strategy is serializable
    if hasattr(clean_config, 'strategy') and not isinstance(clean_config.strategy, str):
        clean_config.strategy = 'ddp'
    
    model = ModelInterface(clean_config)
    trainer_loggers = load_loggers(config)
    callbacks = load_callbacks(config)

    # Configure training strategy
    strategy_config = config.strategy
    if config.devices > 1 and config.strategy == 'ddp':
        from pytorch_lightning.strategies import DDPStrategy
        strategy_config = DDPStrategy(
            find_unused_parameters=config.MULTI_GPU.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=False
        )
    
    # Gradient accumulation
    accumulate_grad_batches = getattr(config.MULTI_GPU, 'accumulate_grad_batches', 1)
    
    # Trainer configuration
    if config.devices < 1:
        raise ValueError("`--gpus` must be >= 1; CPU training is not supported in strict mode")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; no GPU is available")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config.devices,
        max_epochs=config.TRAINING.num_epochs,
        logger=trainer_loggers,
        callbacks=callbacks,
        precision=32,
        strategy=strategy_config,
        sync_batchnorm=config.sync_batchnorm,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=config.TRAINING.gradient_clip_val,
        deterministic=False,
        enable_model_summary=False,
    )

    # Execute according to the selected mode
    if config.mode == 'train':
        trainer.fit(model, train_loader, val_loader)
    elif config.mode == 'test':
        logger.info("Loading checkpoint: %s", config.ckpt_path)
        trainer.test(model, test_loader, ckpt_path=config.ckpt_path)
        logger.info("Test run finished")

    return model

if __name__ == '__main__':
    parser = get_parse()
    args = parser.parse_args()
    
    # Build configuration and run
    config = build_config_from_args(args)
    fix_seed(config.GENERAL.seed)

    main(config)