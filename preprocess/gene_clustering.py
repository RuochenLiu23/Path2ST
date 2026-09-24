"""
Reorder the selected gene list by clustering genes on their expression
profiles across training spots, so that genes with similar patterns sit
next to each other.
"""

import os
import json
import numpy as np
import shutil
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import logging

from .utils import (
    load_gene_list, save_gene_list, get_train_slides, 
    load_slide_gene_expression
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneClusteringProcessor:
    """Clusters genes per dataset"""
    
    def __init__(
        self,
        scale_dims: Tuple[int, ...] = (1, 4, 8, 40, 100, 200),
        data_root: Optional[str] = None,
        h5ad_root: Optional[str] = None,
    ):

        self.scale_dims = scale_dims
        self.data_root = os.path.abspath(data_root or os.environ.get('Path2ST_DATA_ROOT', './data'))
        self.h5ad_root = h5ad_root or os.environ.get('Path2ST_H5AD_ROOT')
        
        # Validation slides are left out of clustering.
        self.datasets = {
            'PRAD': {
                'dir_name': 'PRAD',
                'val_slides': 'MEND145'
            },
            'her2st': {
                'dir_name': 'her2st',
                'val_slides': 'SPA148'
            },
            'mouse_brain': {
                'dir_name': 'mouse_brain',
                'val_slides': 'NCBI667'
            }
        }
    
    def process_dataset(self, dataset_name: str) -> None:
        """Back up, load training data, cluster, and write results"""

        if dataset_name not in self.datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        dataset_root = os.path.join(self.data_root, dataset_config['dir_name'])
        data_path = os.path.join(dataset_root, 'processed_data')
        val_slide = dataset_config['val_slides']

        if not os.path.exists(dataset_root):
            logger.warning("Dataset root does not exist: %s", dataset_root)
        
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"   data path: {data_path}")
        logger.info(f"   validation slide: {val_slide}")

        self._backup_original_gene_list(data_path)

        train_slides = get_train_slides(data_path, val_slide)

        combined_expr = self._load_training_data(data_path, train_slides)
        logger.info(f"   Training spots: {combined_expr.shape[0]}")

        clustered_order = self._perform_clustering(combined_expr)

        self._save_clustered_gene_list(data_path, clustered_order)

        self._save_clustering_info(data_path, dataset_name, train_slides, 
                                  combined_expr.shape[0], clustered_order)

        logger.info(f"Clustering finished for {dataset_name}")

    def _backup_original_gene_list(self, data_path: str) -> None:
        """Copy the gene list to a backup file."""
        original_file = os.path.join(data_path, 'selected_gene_list.txt')
        backup_file = os.path.join(data_path, 'unclustered_selected_gene_list.txt')
        
        if not os.path.exists(backup_file):
            shutil.copy(original_file, backup_file)
            logger.info(f"Created gene list backup: {backup_file}")
        else:
            logger.info(f"Backup already present: {backup_file}")

    def _load_training_data(self, data_path: str, train_slides: List[str]) -> np.ndarray:

        logger.info(f"Loading {len(train_slides)} training slides")
        
        all_expr_data = []
        
        for slide_id in train_slides:
            try:
                slide_expr = load_slide_gene_expression(
                    data_path,
                    slide_id,
                    h5ad_root=self.h5ad_root,
                )
                logger.info(f"   {slide_id}: {slide_expr.shape[0]} spots, {slide_expr.shape[1]} genes")
                
                # Same 200-gene cutoff as used in training.
                if slide_expr.shape[1] >= 200:
                    slide_expr = slide_expr[:, :200]
                else:
                    logger.warning(f"Slide {slide_id} provides only {slide_expr.shape[1]} genes (<200); skipping")
                    continue
                
                all_expr_data.append(slide_expr)
                
            except Exception as e:
                logger.error(f"Failed to load {slide_id}: {e}")
                continue

        if not all_expr_data:
            raise ValueError("No training data could be loaded")

        combined_expr = np.concatenate(all_expr_data, axis=0)
        logger.info(f"   Combined training matrix: {combined_expr.shape}")
        
        return combined_expr
    
    def _perform_clustering(self, gene_expr_matrix: np.ndarray) -> np.ndarray:
        """Two-level k-means over genes; returns a permutation of gene indices. """
        
        logger.info("Starting gene clustering")

        # Rows are genes, columns are spots.
        gene_features = gene_expr_matrix.T
        logger.info(f"   Gene feature matrix: {gene_features.shape}")

        scaler = StandardScaler()
        gene_features_norm = scaler.fit_transform(gene_features)
        logger.info("   Standardisation complete")

        logger.info("   Stage 1: clustering into four groups")
        kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
        major_clusters = kmeans_4.fit_predict(gene_features_norm)

        cluster_sizes = np.bincount(major_clusters)
        logger.info(f"   Group sizes: {cluster_sizes}")

        logger.info("   Stage 2: refining each group")
        clustered_order = []
        
        for major_group in range(4):
            genes_in_major = np.where(major_clusters == major_group)[0]
            group_features = gene_features_norm[genes_in_major]
            
            if len(genes_in_major) <= 10:
                clustered_order.extend(genes_in_major.tolist())
                logger.info(f"     Group {major_group}: {len(genes_in_major)} genes (no refinement)")
            else:
                n_sub_clusters = max(2, len(genes_in_major) // 12)
                kmeans_sub = KMeans(n_clusters=n_sub_clusters, random_state=42)
                sub_clusters = kmeans_sub.fit_predict(group_features)
                
                for sub_group in range(n_sub_clusters):
                    genes_in_sub = genes_in_major[sub_clusters == sub_group]
                    clustered_order.extend(genes_in_sub.tolist())
                
                logger.info(f"     Group {major_group}: {len(genes_in_major)} genes -> {n_sub_clusters} sub-clusters")

        clustered_order = np.array(clustered_order)
        logger.info(f"   Clustering reordered {len(clustered_order)} genes")
        
        return clustered_order
    
    def _save_clustered_gene_list(self, data_path: str, clustered_order: np.ndarray) -> None:

        backup_file = os.path.join(data_path, 'unclustered_selected_gene_list.txt')
        original_genes = load_gene_list(backup_file)
        
        clustered_genes = [original_genes[i] for i in clustered_order]

        output_file = os.path.join(data_path, 'selected_gene_list.txt')
        save_gene_list(output_file, clustered_genes)

        logger.info(f"Saved clustered gene list to {output_file}")
    
    def _save_clustering_info(self, data_path: str, dataset_name: str,
                            train_slides: List[str], total_spots: int,
                            clustered_order: np.ndarray) -> None:
        """Save the inputs, parameters and resulting order to clustering_info.json."""
        clustering_info = {
            'dataset': dataset_name,
            'train_slides': train_slides,
            'total_spots': total_spots,
            'clustered_order': clustered_order.tolist(),
            'scale_dims': self.scale_dims,
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'kmeans_hierarchical',
            'parameters': {
                'stage1_clusters': 4,
                'normalization': 'zscore',
                'random_state': 42
            }
        }
        
        info_file = os.path.join(data_path, 'clustering_info.json')
        with open(info_file, 'w') as f:
            json.dump(clustering_info, f, indent=2)
        
        logger.info(f"Saved clustering metadata to {info_file}")

    def process_all_datasets(self) -> None:
        """Process every registered dataset, continuing past any that fail."""
        for dataset_name in self.datasets.keys():
            try:
                self.process_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
                continue
