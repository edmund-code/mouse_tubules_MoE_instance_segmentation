# instance_seg/utils/geojson_parser.py

"""
QuPath GeoJSON Annotation Parser for Instance Segmentation.

This module provides utilities to parse QuPath GeoJSON annotation exports
and convert them to instance segmentation masks.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from shapely.geometry import shape, box, Polygon, MultiPolygon
from shapely.validation import make_valid
import cv2
from tqdm import tqdm


class QuPathAnnotationHandler:
    """
    Handler for QuPath GeoJSON annotations.
    
    Parses GeoJSON files exported from QuPath and provides methods to:
    - Filter annotations by class
    - Query annotations by spatial region
    - Generate instance segmentation masks
    
    Parameters
    ----------
    geojson_path : str
        Path to GeoJSON file exported from QuPath.
    class_names : list of str, optional
        List of class names to include. If None, includes all classes.
    """
    
    def __init__(self, geojson_path: str, class_names: Optional[List[str]] = None):
        self.geojson_path = Path(geojson_path)
        self.class_names = class_names
        self.annotations = []
        self.geometries = []
        self.spatial_index = None
        
        self._load_annotations()
        self._build_spatial_index()
    
    def _load_annotations(self) -> None:
        """Load and parse GeoJSON file."""
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)
        
        features = data.get('features', data) if isinstance(data, dict) else data
        
        valid_count = 0
        invalid_count = 0
        skipped_class_count = 0
        
        for idx, feature in enumerate(features):
            if not isinstance(feature, dict):
                continue
            
            # Get properties
            properties = feature.get('properties', {})
            classification = properties.get('classification', {})
            
            # Handle different QuPath export formats
            if isinstance(classification, dict):
                class_name = classification.get('name', 'Unknown')
            else:
                class_name = str(classification) if classification else 'Unknown'
            
            # Filter by class names
            if self.class_names is not None:
                # Case-insensitive matching
                if not any(cn.lower() == class_name.lower() for cn in self.class_names):
                    skipped_class_count += 1
                    continue
            
            # Parse geometry
            geometry_data = feature.get('geometry')
            if geometry_data is None:
                invalid_count += 1
                continue
            
            try:
                geom = shape(geometry_data)
                
                # Validate and fix geometry if needed
                if not geom.is_valid:
                    geom = make_valid(geom)
                
                # Skip empty geometries
                if geom.is_empty:
                    invalid_count += 1
                    continue
                
                # Handle GeometryCollection from make_valid
                if geom.geom_type == 'GeometryCollection':
                    # Extract polygons from collection
                    polygons = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
                    if not polygons:
                        invalid_count += 1
                        continue
                    if len(polygons) == 1:
                        geom = polygons[0]
                    else:
                        geom = MultiPolygon(polygons)
                
                # Ensure we have a valid polygon-type geometry
                if geom.geom_type not in ('Polygon', 'MultiPolygon'):
                    invalid_count += 1
                    continue
                
                self.annotations.append({
                    'id': valid_count,
                    'class_name': class_name,
                    'properties': properties
                })
                self.geometries.append(geom)
                valid_count += 1
                
            except Exception as e:
                invalid_count += 1
                continue
        
        print(f"  Loaded {valid_count} valid annotations")
        if invalid_count > 0:
            print(f"  Skipped {invalid_count} invalid/empty geometries")
        if skipped_class_count > 0:
            print(f"  Skipped {skipped_class_count} annotations (class filter)")
    
    def _build_spatial_index(self) -> None:
        """Build R-tree spatial index for fast region queries."""
        try:
            from rtree import index
            self.spatial_index = index.Index()
            for idx, geom in enumerate(self.geometries):
                self.spatial_index.insert(idx, geom.bounds)
        except ImportError:
            print("  Warning: rtree not installed. Spatial queries will be slower.")
            self.spatial_index = None
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def get_instances_in_region(
        self, 
        bounds: Tuple[float, float, float, float],
        min_overlap: float = 0.1
    ) -> List[Tuple[int, any]]:
        """
        Get all instances that overlap with a region.
        
        Parameters
        ----------
        bounds : tuple
            (min_x, min_y, max_x, max_y) bounding box.
        min_overlap : float
            Minimum overlap ratio (intersection / instance area) to include.
        
        Returns
        -------
        list of tuples
            (instance_id, geometry) pairs for overlapping instances.
        """
        min_x, min_y, max_x, max_y = bounds
        region_box = box(min_x, min_y, max_x, max_y)
        
        # Get candidate indices using spatial index
        if self.spatial_index is not None:
            candidates = list(self.spatial_index.intersection(bounds))
        else:
            # Fall back to checking all geometries
            candidates = range(len(self.geometries))
        
        instances = []
        for idx in candidates:
            geom = self.geometries[idx]
            
            # Safety check - ensure geom is valid
            if geom is None or geom.is_empty:
                continue
            
            try:
                if not region_box.intersects(geom):
                    continue
                
                intersection = region_box.intersection(geom)
                if intersection.is_empty:
                    continue
                
                overlap_ratio = intersection.area / geom.area if geom.area > 0 else 0
                
                if overlap_ratio >= min_overlap:
                    instances.append((self.annotations[idx]['id'], geom))
            except Exception as e:
                # Skip problematic geometries
                continue
        
        return instances
    
    def create_mask_for_region(
        self,
        bounds: Tuple[float, float, float, float],
        mask_shape: Tuple[int, int],
        min_overlap: float = 0.1
    ) -> np.ndarray:
        """
        Create instance segmentation mask for a region.
        
        Parameters
        ----------
        bounds : tuple
            (min_x, min_y, max_x, max_y) bounding box in WSI coordinates.
        mask_shape : tuple
            (height, width) of output mask.
        min_overlap : float
            Minimum overlap ratio to include instance.
        
        Returns
        -------
        np.ndarray
            Instance mask of shape (H, W) with unique integer IDs per instance.
        """
        min_x, min_y, max_x, max_y = bounds
        height, width = mask_shape
        
        # Scale factors
        scale_x = width / (max_x - min_x)
        scale_y = height / (max_y - min_y)
        
        mask = np.zeros((height, width), dtype=np.int32)
        
        instances = self.get_instances_in_region(bounds, min_overlap)
        
        for instance_id, geom in instances:
            try:
                # Clip geometry to region
                region_box = box(min_x, min_y, max_x, max_y)
                clipped_geom = geom.intersection(region_box)
                
                if clipped_geom.is_empty:
                    continue
                
                # Convert to mask coordinates
                instance_mask = self._geometry_to_mask(
                    clipped_geom, 
                    mask_shape,
                    offset=(min_x, min_y),
                    scale=(scale_x, scale_y)
                )
                
                # Add to mask (instance_id + 1 to reserve 0 for background)
                mask[instance_mask > 0] = instance_id + 1
                
            except Exception as e:
                continue
        
        return mask
    
    def _geometry_to_mask(
        self,
        geom,
        mask_shape: Tuple[int, int],
        offset: Tuple[float, float],
        scale: Tuple[float, float]
    ) -> np.ndarray:
        """Convert Shapely geometry to binary mask."""
        height, width = mask_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        def polygon_to_coords(poly):
            """Convert polygon to OpenCV-compatible coordinates."""
            exterior_coords = np.array(poly.exterior.coords)
            # Transform to mask coordinates
            exterior_coords[:, 0] = (exterior_coords[:, 0] - offset[0]) * scale[0]
            exterior_coords[:, 1] = (exterior_coords[:, 1] - offset[1]) * scale[1]
            return exterior_coords.astype(np.int32)
        
        try:
            if geom.geom_type == 'Polygon':
                coords = polygon_to_coords(geom)
                cv2.fillPoly(mask, [coords], 1)
                
                # Handle holes
                for interior in geom.interiors:
                    hole_coords = np.array(interior.coords)
                    hole_coords[:, 0] = (hole_coords[:, 0] - offset[0]) * scale[0]
                    hole_coords[:, 1] = (hole_coords[:, 1] - offset[1]) * scale[1]
                    cv2.fillPoly(mask, [hole_coords.astype(np.int32)], 0)
                    
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = polygon_to_coords(poly)
                    cv2.fillPoly(mask, [coords], 1)
                    
                    for interior in poly.interiors:
                        hole_coords = np.array(interior.coords)
                        hole_coords[:, 0] = (hole_coords[:, 0] - offset[0]) * scale[0]
                        hole_coords[:, 1] = (hole_coords[:, 1] - offset[1]) * scale[1]
                        cv2.fillPoly(mask, [hole_coords.astype(np.int32)], 0)
        except Exception as e:
            pass
        
        return mask


def load_qupath_annotations(
    geojson_path: str,
    class_names: Optional[List[str]] = None
) -> QuPathAnnotationHandler:
    """
    Convenience function to load QuPath annotations.
    
    Parameters
    ----------
    geojson_path : str
        Path to GeoJSON file.
    class_names : list of str, optional
        Class names to filter.
    
    Returns
    -------
    QuPathAnnotationHandler
        Loaded annotation handler.
    """
    return QuPathAnnotationHandler(geojson_path, class_names)