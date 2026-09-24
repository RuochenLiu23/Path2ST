#!/usr/bin/env python3
"""
Run gene clustering preprocessing to preprocess the dataset.
"""

import argparse
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocess.gene_clustering import GeneClusteringProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Gene clustering preprocessing',
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['PRAD', 'her2st', 'mouse_brain'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--all-datasets', 
        action='store_true',
        help='Process all datasets'
    )

    parser.add_argument(
        '--data-root',
        type=str,
        default=os.environ.get('Path2ST_DATA_ROOT', './data'),
        help='Root directory containing dataset folders '
             '(default: $Path2ST_DATA_ROOT or ./data)',
    )

    parser.add_argument(
        '--h5ad-root',
        type=str,
        default=os.environ.get('Path2ST_H5AD_ROOT'),
        help='Root directory containing slide h5ad files '
             '(default: $Path2ST_H5AD_ROOT)',
    )
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all_datasets:
        parser.print_help()
        print("\nError: specify --dataset or --all-datasets")
        return 1
    
    processor = GeneClusteringProcessor(
        data_root=args.data_root,
        h5ad_root=args.h5ad_root,
    )
    
    try:
        # --all-datasets takes precedence if both flags are given.
        if args.all_datasets:
            logger.info("Processing all datasets")
            processor.process_all_datasets()
            logger.info("All datasets processed")
            
        elif args.dataset:
            logger.info(f"Processing dataset: {args.dataset}")
            processor.process_dataset(args.dataset)
            logger.info(f"Dataset processed: {args.dataset}")
            
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1

    print("\nGene clustering preprocessing complete.")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
