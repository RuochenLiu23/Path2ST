"""Extract cell center coordinates and cell types from CellViT segmentation results.

This script processes CellViT segmentation geojson files to extract:
1. Cell center coordinates (centroid of each cell polygon)
2. Cell type (classification name from properties)

For each slide, outputs a text file containing:
- Each line: center_x, center_y, cell_type_id
- Cell type IDs are consistent within each slide (same type = same ID)
- Cells are processed in the same order as in cell_embeddings.py

Output format:
    center_x center_y cell_type_id
    12345.6 8765.4 0
    12346.7 8766.5 1
    ...
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS = {
    "PRAD": {
        "name": "PRAD",
        "dir_name": "PRAD",
        "seg_dir": "PRAD_seg",
        "seg_dir_relative": False,   # seg_dir is relative to data_root
    },
    "mouse_brain": {
        "name": "mouse_brain",
        "dir_name": "mouse_brain",
        "seg_dir": "",
        "seg_dir_relative": True,    # seg_dir = dataset_dir itself, process_slide appends cellvit_seg/
    },
    "her2st": {
        "name": "her2st",
        "dir_name": "her2st",
        "seg_dir": "",
        "seg_dir_relative": True,    # seg_dir = dataset_dir itself, process_slide appends cellvit_seg/
    },
}

OUTPUT_SUBDIR = "processed_data/cell_spatial"
SLIDE_LIST_PATH = "processed_data/all_slide_lst.txt"


# Argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cell center coordinates and types from CellViT segmentation",
    )
    parser.add_argument(
        "--dataset",
        default="her2st",
        choices=list(DATASETS.keys()),
        help="Dataset to process (default: kidney)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/gpfs/work/aac/ruochenliu23/genar-main/src/data"),
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip slides that already have spatial info extracted",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without writing output",
    )
    return parser.parse_args()


def load_slide_ids(dataset_dir: Path) -> List[str]:
    """Load slide IDs from the slide list file."""
    slide_path = dataset_dir / SLIDE_LIST_PATH
    if not slide_path.exists():
        raise FileNotFoundError(f"Slide list not found: {slide_path}")
    with slide_path.open("r", encoding="utf-8") as handle:
        slides = [line.strip() for line in handle if line.strip()]
    if not slides:
        raise RuntimeError(f"No slide IDs discovered in {slide_path}")
    return slides


def ensure_output_dir(dataset_dir: Path) -> Path:
    """Create output directory if it doesn't exist."""
    out_dir = dataset_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# GeoJSON processing
def load_geojson_from_zip(geojson_zip_path: Path) -> dict:
    """Load geojson from a zipped file."""
    if not geojson_zip_path.exists():
        raise FileNotFoundError(f"GeoJSON zip file not found: {geojson_zip_path}")
    
    with zipfile.ZipFile(geojson_zip_path, 'r') as zip_ref:
        # Get the first .geojson file in the zip
        geojson_files = [f for f in zip_ref.namelist() if f.endswith('.geojson')]
        if not geojson_files:
            raise ValueError(f"No .geojson file found in {geojson_zip_path}")
        
        # Read the geojson content
        with zip_ref.open(geojson_files[0]) as f:
            geojson_data = json.load(f)
    
    return geojson_data


def extract_cell_polygons_with_types(geojson_data: dict) -> Tuple[List[List[Tuple[float, float]]], List[str]]:
    """Extract cell polygon coordinates and cell types from CellViT geojson data.
    
    Returns:
        Tuple of (cell_polygons, cell_types) where:
        - cell_polygons: List of cell polygons, each polygon is a list of (x, y) coordinates
        - cell_types: List of cell type names corresponding to each polygon
    """
    cell_polygons = []
    cell_types = []
    
    # Handle list format (actual CellViT format)
    if isinstance(geojson_data, list):
        features = geojson_data
    # Handle FeatureCollection format (backup)
    elif isinstance(geojson_data, dict) and 'features' in geojson_data:
        features = geojson_data['features']
    # Handle single Feature (backup)
    elif isinstance(geojson_data, dict) and geojson_data.get('type') == 'Feature':
        features = [geojson_data]
    else:
        raise ValueError(f"Unexpected GeoJSON structure. Type: {type(geojson_data)}")
    
    # Extract cells from each feature
    for feature in features:
        if not isinstance(feature, dict) or 'geometry' not in feature:
            continue
        
        geometry = feature['geometry']
        
        # CellViT always uses MultiPolygon
        if geometry.get('type') != 'MultiPolygon':
            print(f"Warning: Unexpected geometry type '{geometry.get('type')}', skipping")
            continue
        
        coordinates = geometry.get('coordinates', [])
        if not coordinates:
            continue
        
        # Extract cell type from properties
        properties = feature.get('properties', {})
        classification = properties.get('classification', {})
        cell_type_name = classification.get('name', 'Unknown')
        
        # Each element in coordinates is ONE CELL
        # Structure: coordinates[i] = [outer_ring, hole1, hole2, ...]
        # We only need the outer ring: coordinates[i][0]
        for cell_coords in coordinates:
            if not cell_coords or not cell_coords[0]:
                continue
            
            # Get outer ring coordinates
            outer_ring = cell_coords[0]
            
            # Convert to list of tuples
            polygon = [(float(x), float(y)) for x, y in outer_ring]
            
            # Store polygon and its type
            cell_polygons.append(polygon)
            cell_types.append(cell_type_name)
    
    return cell_polygons, cell_types


def compute_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute the centroid (center of mass) of a polygon.
    
    Args:
        polygon: List of (x, y) coordinates forming the polygon
        
    Returns:
        Tuple of (center_x, center_y)
    """
    if not polygon:
        return (0.0, 0.0)
    
    # Simple centroid: average of all vertices
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    
    return (center_x, center_y)



# Main processing
def process_slide(
    seg_dir: Path,
    slide_id: str,
) -> Tuple[List[Tuple[float, float]], List[int], Dict[str, int]]:
    """Process a single slide and extract cell spatial information.
    
    Returns:
        Tuple of (cell_centers, cell_type_ids, type_mapping) where:
        - cell_centers: List of (x, y) coordinates for each cell center
        - cell_type_ids: List of cell type IDs (integers) for each cell
        - type_mapping: Dictionary mapping cell type names to IDs
    """
    # GeoJSON: e.g., PRAD_seg/cellvit_seg/MEND151_cellvit_seg.geojson.zip
    geojson_path = seg_dir / "cellvit_seg" / f"{slide_id}_cellvit_seg.geojson.zip"
    
    # Load GeoJSON
    geojson_data = load_geojson_from_zip(geojson_path)
    cell_polygons, cell_types = extract_cell_polygons_with_types(geojson_data)
    
    if not cell_polygons:
        raise ValueError(f"No cells found in {slide_id}")
    
    print(f"  Found {len(cell_polygons)} cells with {len(set(cell_types))} unique types")
    
    # Create mapping from cell type names to integer IDs
    unique_types = sorted(set(cell_types))
    type_to_id = {type_name: idx for idx, type_name in enumerate(unique_types)}
    
    # Compute cell centers and convert types to IDs
    cell_centers = []
    cell_type_ids = []
    
    for polygon, cell_type in zip(cell_polygons, cell_types):
        center = compute_centroid(polygon)
        type_id = type_to_id[cell_type]
        
        cell_centers.append(center)
        cell_type_ids.append(type_id)
    
    return cell_centers, cell_type_ids, type_to_id


def save_spatial_info(
    output_path: Path,
    cell_centers: List[Tuple[float, float]],
    cell_type_ids: List[int],
    type_mapping: Dict[str, int]
):
    """Save cell spatial information to a text file.
    
    Format:
        # Cell type mapping: TypeName=ID
        # Epithelial=0
        # Immune=1
        # ...
        center_x center_y cell_type_id
        12345.6 8765.4 0
        12346.7 8766.5 1
        ...
    """
    with output_path.open('w', encoding='utf-8') as f:
        # Write cell type mapping as comments
        f.write("# Cell type mapping:\n")
        for type_name, type_id in sorted(type_mapping.items(), key=lambda x: x[1]):
            f.write(f"# {type_name}={type_id}\n")
        f.write("#\n")
        f.write("# Format: center_x center_y cell_type_id\n")
        
        # Write cell data
        for (center_x, center_y), type_id in zip(cell_centers, cell_type_ids):
            f.write(f"{center_x:.2f} {center_y:.2f} {type_id}\n")


def main() -> None:
    args = parse_args()
    
    # Get dataset configuration
    config = DATASETS[args.dataset]
    
    # Setup paths
    data_root = args.data_root.resolve()
    dataset_dir = data_root / config["dir_name"]
    # seg_dir_relative=True means seg_dir is inside the dataset folder (e.g. ccRCC/ccRCC_seg)
    # seg_dir_relative=False means seg_dir is directly under data_root (e.g. data_root/PRAD_seg)
    if config.get("seg_dir_relative", False):
        seg_dir = dataset_dir / config["seg_dir"] if config["seg_dir"] else dataset_dir
    else:
        seg_dir = data_root / config["seg_dir"]
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_dir}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")
    
    # Load slide IDs
    slides = load_slide_ids(dataset_dir)
    
    # Setup output directory
    out_dir = ensure_output_dir(dataset_dir)
    
    # Process slides
    print(f"\nProcessing dataset: {config['name']}")
    print(f"  Dataset directory: {dataset_dir}")
    print(f"  Segmentation directory: {seg_dir}")
    print(f"  Output directory: {out_dir}")
    print(f"  Number of slides: {len(slides)}")
    
    processed = 0
    total_cells = 0
    
    with tqdm(slides, desc=config['name'], unit="slide") as iterator:
        for slide_id in iterator:
            output_path = out_dir / f"{slide_id}_spatial.txt"
            iterator.set_postfix_str(slide_id)
            
            # Skip if already processed
            if args.skip_existing and output_path.exists():
                continue
            
            # Dry run mode
            if args.dry_run:
                print(f"[DRY-RUN] Would process {slide_id}")
                continue
            
            try:
                # Extract cell spatial information
                cell_centers, cell_type_ids, type_mapping = process_slide(
                    seg_dir=seg_dir,
                    slide_id=slide_id,
                )
                
                num_cells = len(cell_centers)
                
                # Save to file
                save_spatial_info(output_path, cell_centers, cell_type_ids, type_mapping)
                
                processed += 1
                total_cells += num_cells
                
                print(f"  Processed {slide_id}: {num_cells} cells, {len(type_mapping)} types -> {output_path}")
                
            except FileNotFoundError as e:
                print(f"\n  Skipping {slide_id}: {str(e)}")
                continue
            except Exception as e:
                print(f"\n  Error processing {slide_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Print summary
    if args.dry_run:
        print("\nDry run finished — no files were written.")
    else:
        print("\n" + "="*60)
        print("Cell Spatial Information Extraction Summary")
        print("="*60)
        print(f"  Dataset: {config['name']}")
        print(f"  Slides processed: {processed}")
        print(f"  Total cells: {total_cells}")
        print(f"  Average cells per slide: {total_cells/processed if processed > 0 else 0:.1f}")
        print("\nOutput format:")
        print(f"  - Each line: center_x center_y cell_type_id")
        print(f"  - File format: .txt")
        print(f"  - Location: {out_dir}")


if __name__ == "__main__":
    main()