"""
QuPath GeoJSON Parser for Instance Segmentation.

This module provides functions to parse QuPath GeoJSON exports and convert
polygon annotations to instance segmentation masks.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import cv2
from shapely.geometry import shape, box, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely import STRtree


def load_qupath_geojson(geojson_path: str) -> dict:
    """
    Load a QuPath GeoJSON file from disk.
    
    Parameters
    ----------
    geojson_path : str
        Path to the .geojson file exported from QuPath.
    
    Returns
    -------
    dict
        Parsed GeoJSON as a Python dictionary with structure:
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[x1, y1], [x2, y2], ...]]
                    },
                    "properties": {
                        "classification": {"name": "Tubule"},
                        ...
                    }
                },
                ...
            ]
        }
    
    Raises
    ------
    FileNotFoundError
        If geojson_path does not exist.
    json.JSONDecodeError
        If file is not valid JSON.
    """
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    if geojson_data.get("type") != "FeatureCollection":
        raise ValueError(f"Invalid GeoJSON: expected type 'FeatureCollection', got '{geojson_data.get('type')}'")
    
    if "features" not in geojson_data or not isinstance(geojson_data["features"], list):
        raise ValueError("Invalid GeoJSON: 'features' key missing or not a list")
    
    return geojson_data


def filter_annotations_by_class(
    geojson_data: dict,
    class_names: List[str],
    include_unclassified: bool = False
) -> List[dict]:
    """
    Filter GeoJSON features by classification name.
    
    Parameters
    ----------
    geojson_data : dict
        Parsed GeoJSON dictionary from load_qupath_geojson().
    class_names : List[str]
        List of classification names to include (e.g., ["Tubule", "tubule", "Proximal", "Distal"]).
        Matching is case-insensitive.
    include_unclassified : bool, default=False
        If True, include annotations without a classification.
    
    Returns
    -------
    List[dict]
        List of feature dictionaries that match the filter criteria.
        Each feature has "geometry" and "properties" keys.
    """
    class_names_lower = [name.lower() for name in class_names]
    filtered = []
    
    for feature in geojson_data.get("features", []):
        properties = feature.get("properties", {})
        classification = properties.get("classification")
        
        if classification is None:
            if include_unclassified:
                filtered.append(feature)
        else:
            class_name = classification.get("name", "")
            if class_name.lower() in class_names_lower:
                filtered.append(feature)
    
    return filtered


def polygon_to_shapely(geometry: dict) -> Optional[BaseGeometry]:
    """
    Convert GeoJSON geometry to Shapely geometry object.
    
    Parameters
    ----------
    geometry : dict
        GeoJSON geometry dictionary with "type" and "coordinates" keys.
        Supported types: "Polygon", "MultiPolygon"
    
    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon or None
        Shapely geometry object, or None if geometry is invalid/empty.
    """
    try:
        geom = shape(geometry)
        
        if not geom.is_valid:
            geom = geom.buffer(0)
        
        if geom.is_empty:
            return None
            
        return geom
    except Exception:
        return None


def get_annotations_in_region(
    annotations: List[dict],
    region_bounds: Tuple[float, float, float, float],
    min_overlap_ratio: float = 0.1
) -> List[Tuple[dict, BaseGeometry]]:
    """
    Filter annotations that overlap with a specified rectangular region.
    
    Parameters
    ----------
    annotations : List[dict]
        List of GeoJSON feature dictionaries.
    region_bounds : Tuple[float, float, float, float]
        Bounding box as (x_min, y_min, x_max, y_max) in pixel coordinates.
    min_overlap_ratio : float, default=0.1
        Minimum ratio of annotation area that must be within region to be included.
        Value between 0 and 1.
    
    Returns
    -------
    List[Tuple[dict, shapely.geometry]]
        List of tuples containing (original_annotation, clipped_shapely_geometry).
        Geometries are clipped to the region bounds.
    """
    region_box = box(*region_bounds)
    results = []
    
    for annotation in annotations:
        geom = polygon_to_shapely(annotation.get("geometry", {}))
        if geom is None:
            continue
        
        if not region_box.intersects(geom):
            continue
        
        intersection = region_box.intersection(geom)
        if intersection.is_empty:
            continue
        
        overlap_ratio = intersection.area / geom.area
        if overlap_ratio >= min_overlap_ratio:
            results.append((annotation, intersection))
    
    return results


def create_instance_mask(
    geometries: List[BaseGeometry],
    mask_shape: Tuple[int, int],
    offset: Tuple[float, float] = (0, 0)
) -> np.ndarray:
    """
    Create an instance segmentation mask from a list of Shapely geometries.
    
    Parameters
    ----------
    geometries : List[shapely.geometry]
        List of Shapely polygon geometries. Each will be assigned a unique instance ID.
    mask_shape : Tuple[int, int]
        Output mask shape as (height, width).
    offset : Tuple[float, float], default=(0, 0)
        Offset to subtract from geometry coordinates as (x_offset, y_offset).
        Used when creating masks for patches extracted from larger images.
    
    Returns
    -------
    np.ndarray
        Instance mask of shape (height, width) with dtype np.int32.
        Background = 0, instances = 1, 2, 3, ...
    """
    mask = np.zeros(mask_shape, dtype=np.int32)
    x_offset, y_offset = offset
    
    for idx, geom in enumerate(geometries):
        instance_id = idx + 1
        
        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        else:
            continue
        
        for poly in polygons:
            if poly.is_empty:
                continue
            
            exterior_coords = np.array(poly.exterior.coords, dtype=np.float32)
            exterior_coords[:, 0] -= x_offset
            exterior_coords[:, 1] -= y_offset
            exterior_coords = exterior_coords.astype(np.int32)
            
            cv2.fillPoly(mask, [exterior_coords], instance_id)
            
            for interior in poly.interiors:
                interior_coords = np.array(interior.coords, dtype=np.float32)
                interior_coords[:, 0] -= x_offset
                interior_coords[:, 1] -= y_offset
                interior_coords = interior_coords.astype(np.int32)
                cv2.fillPoly(mask, [interior_coords], 0)
    
    return mask


def process_wsi_geojson(
    geojson_path: str,
    class_names: List[str] = ["Tubule"]
) -> List[Tuple[BaseGeometry, dict]]:
    """
    High-level function to load and process a QuPath GeoJSON file for a WSI.
    
    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file.
    class_names : List[str], default=["Tubule"]
        Classification names to include.
    
    Returns
    -------
    List[Tuple[shapely.geometry, dict]]
        List of tuples containing (shapely_geometry, properties_dict) for each annotation.
    """
    geojson_data = load_qupath_geojson(geojson_path)
    filtered_annotations = filter_annotations_by_class(geojson_data, class_names)
    
    results = []
    for annotation in filtered_annotations:
        geom = polygon_to_shapely(annotation.get("geometry", {}))
        if geom is not None:
            properties = annotation.get("properties", {})
            results.append((geom, properties))
    
    return results


class QuPathAnnotationHandler:
    """
    Class to manage QuPath annotations for a single WSI.
    
    This class provides efficient spatial queries for annotations within regions,
    using an R-tree spatial index for performance with large numbers of annotations.
    
    Attributes
    ----------
    geojson_path : str
        Path to the source GeoJSON file.
    annotations : List[Tuple[shapely.geometry, dict]]
        List of (geometry, properties) tuples.
    spatial_index : shapely.STRtree
        Spatial index for efficient region queries.
    """
    
    def __init__(self, geojson_path: str, class_names: List[str] = ["Tubule"]):
        """
        Initialize handler by loading GeoJSON and building spatial index.
        
        Parameters
        ----------
        geojson_path : str
            Path to QuPath GeoJSON file.
        class_names : List[str], default=["Tubule"]
            Classification names to include.
        """
        self.geojson_path = geojson_path
        self.annotations = process_wsi_geojson(geojson_path, class_names)
        
        if len(self.annotations) > 0:
            geometries = [geom for geom, _ in self.annotations]
            self.spatial_index = STRtree(geometries)
            self.geom_to_idx = {id(geom): idx for idx, (geom, _) in enumerate(self.annotations)}
        else:
            self.spatial_index = None
            self.geom_to_idx = {}
    
    def get_instances_in_region(
        self,
        bounds: Tuple[float, float, float, float],
        min_overlap: float = 0.1
    ) -> List[Tuple[int, BaseGeometry]]:
        """
        Get instances overlapping a region using spatial index.
        
        Parameters
        ----------
        bounds : Tuple
            Region as (x_min, y_min, x_max, y_max).
        min_overlap : float
            Minimum overlap ratio to include instance.
        
        Returns
        -------
        List[Tuple[int, shapely.geometry]]
            List of (instance_id, clipped_geometry) tuples.
            instance_id is 1-indexed (0 reserved for background).
        """
        if self.spatial_index is None or len(self.annotations) == 0:
            return []
        
        region_box = box(*bounds)
        candidate_geoms = self.spatial_index.query(region_box)
        
        results = []
        for geom in candidate_geoms:
            if not region_box.intersects(geom):
                continue
            
            intersection = region_box.intersection(geom)
            if intersection.is_empty:
                continue
            
            overlap_ratio = intersection.area / geom.area
            if overlap_ratio >= min_overlap:
                idx = self.geom_to_idx.get(id(geom))
                if idx is not None:
                    instance_id = idx + 1
                    results.append((instance_id, intersection))
        
        return results
    
    def create_mask_for_region(
        self,
        bounds: Tuple[float, float, float, float],
        mask_shape: Tuple[int, int],
        min_overlap: float = 0.1
    ) -> np.ndarray:
        """
        Create instance mask for a region.
        
        Parameters
        ----------
        bounds : Tuple
            Region as (x_min, y_min, x_max, y_max).
        mask_shape : Tuple[int, int]
            Output mask shape as (height, width).
        min_overlap : float
            Minimum overlap ratio to include instance.
        
        Returns
        -------
        np.ndarray
            Instance mask of shape (height, width).
        """
        instances = self.get_instances_in_region(bounds, min_overlap)
        geometries = [geom for _, geom in instances]
        offset = (bounds[0], bounds[1])
        return create_instance_mask(geometries, mask_shape, offset)
    
    def __len__(self) -> int:
        """Return total number of annotations."""
        return len(self.annotations)
