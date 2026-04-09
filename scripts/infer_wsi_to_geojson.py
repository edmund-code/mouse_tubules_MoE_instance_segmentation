#!/usr/bin/env python3
"""
Run trained Semi-MoE instance model on a whole-slide TIFF and export GeoJSON.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tifffile
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

# Add project root for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from instance_seg.dataload.qupath_dataset import get_normalize_transform
from instance_seg.inference.embedding_to_instances import EmbeddingClusterer
from instance_seg.models.instance_expert import create_all_experts
from instance_seg.models.gating_network_4experts import get_gating_network_4experts

# Optional: Shapely for robust geometry validation
try:
    from shapely.geometry import shape, mapping
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def atomic_write_json(payload: Dict, output_path: Path) -> None:
    """Write JSON atomically to avoid truncated/corrupt output files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


def load_wsi_array(tif_path: Path) -> np.ndarray:
    """Load WSI as ndarray (prefers memmap if possible)."""
    try:
        arr = tifffile.memmap(str(tif_path))
    except Exception:
        arr = tifffile.imread(str(tif_path))

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        # Handle CHW
        if arr.shape[0] <= 4 and arr.shape[-1] > 4:
            arr = np.transpose(arr, (1, 2, 0))
        # Handle RGBA/multichannel
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] > 4:
            arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def iter_tile_coords(height: int, width: int, patch_size: int, stride: int):
    """Yield top-left tile coordinates fully inside image bounds."""
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            yield x, y


def simplify_contour(contour: np.ndarray, epsilon: float = 1.5) -> np.ndarray:
    """Simplify contour to reduce points and fix self-intersections."""
    return cv2.approxPolyDP(contour, epsilon, closed=True)


def contour_to_ring(contour: np.ndarray, x_off: int, y_off: int) -> List[List[float]]:
    """Convert OpenCV contour to GeoJSON ring with global offsets."""
    pts = contour.reshape(-1, 2)
    
    # Need at least 3 unique points for a valid polygon
    if len(pts) < 3:
        return []
    
    ring = [[float(x + x_off), float(y + y_off)] for x, y in pts]
    
    # Close the ring (first point == last point)
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    
    # GeoJSON requires at least 4 positions (3 unique + closing)
    if len(ring) < 4:
        return []
    
    return ring


def compute_ring_area(ring: List[List[float]]) -> float:
    """Compute signed area of a polygon ring using shoelace formula."""
    if len(ring) < 4:
        return 0.0
    
    area = 0.0
    for i in range(len(ring) - 1):
        area += ring[i][0] * ring[i + 1][1]
        area -= ring[i + 1][0] * ring[i][1]
    
    return abs(area) / 2.0


def is_valid_polygon(ring: List[List[float]], min_area: float = 1.0) -> bool:
    """Check if polygon ring is valid."""
    if len(ring) < 4:
        return False
    
    # Check for duplicate consecutive points
    for i in range(len(ring) - 1):
        if ring[i] == ring[i + 1]:
            return False
    
    # Check area is non-zero
    return compute_ring_area(ring) >= min_area


def fix_geometry_shapely(geometry: Dict) -> Optional[Dict]:
    """Fix invalid geometry using Shapely."""
    if not SHAPELY_AVAILABLE:
        return geometry
    
    try:
        geom = shape(geometry)
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_empty:
            return None
        # Filter out non-polygon geometries that may result from make_valid
        if geom.geom_type not in ('Polygon', 'MultiPolygon'):
            if geom.geom_type == 'GeometryCollection':
                # Extract polygons from collection
                polygons = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
                if not polygons:
                    return None
                if len(polygons) == 1:
                    geom = polygons[0]
                else:
                    from shapely.geometry import MultiPolygon
                    geom = MultiPolygon(polygons)
            else:
                return None
        return mapping(geom)
    except Exception:
        return None


def instance_mask_to_features(
    instance_mask: np.ndarray,
    x_off: int,
    y_off: int,
    class_name: str,
    min_area: int,
    class_color_rgb: int,
    use_shapely: bool = True
) -> List[Dict]:
    """Convert tile instance mask to GeoJSON polygon features."""
    features: List[Dict] = []
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]

    for inst_id in instance_ids:
        binary = (instance_mask == inst_id).astype(np.uint8)
        area = int(binary.sum())
        if area < min_area:
            continue

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        polygons = []
        hierarchy = hierarchy[0] if hierarchy is not None else None

        for idx, cnt in enumerate(contours):
            # Keep only outer contours (parent == -1)
            if hierarchy is not None and hierarchy[idx][3] != -1:
                continue
            if len(cnt) < 3:
                continue

            # Simplify contour to reduce self-intersections
            cnt = simplify_contour(cnt, epsilon=1.5)
            if len(cnt) < 3:
                continue

            shell = contour_to_ring(cnt, x_off, y_off)
            if not shell or not is_valid_polygon(shell):
                continue

            holes = []
            if hierarchy is not None:
                child = hierarchy[idx][2]
                while child != -1:
                    hole_cnt = contours[child]
                    if len(hole_cnt) >= 3:
                        hole_cnt = simplify_contour(hole_cnt, epsilon=1.5)
                        if len(hole_cnt) >= 3:
                            hole_ring = contour_to_ring(hole_cnt, x_off, y_off)
                            if hole_ring and is_valid_polygon(hole_ring):
                                holes.append(hole_ring)
                    child = hierarchy[child][0]

            polygons.append([shell] + holes)

        if not polygons:
            continue

        if len(polygons) == 1:
            geometry = {"type": "Polygon", "coordinates": polygons[0]}
        else:
            geometry = {"type": "MultiPolygon", "coordinates": polygons}

        # Use Shapely to fix any remaining geometry issues
        if use_shapely and SHAPELY_AVAILABLE:
            geometry = fix_geometry_shapely(geometry)
            if geometry is None:
                continue

        feature_id = f"tile_x{x_off}_y{y_off}_{int(inst_id)}"
        features.append(
            {
                "type": "Feature",
                "id": feature_id,
                "geometry": geometry,
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": class_name, "colorRGB": int(class_color_rgb)},
                    "isLocked": False,
                    "measurements": [
                        {"name": "Area px", "value": float(area)},
                        {"name": "tile_instance_id", "value": float(inst_id)},
                        {"name": "tile_x", "value": float(x_off)},
                        {"name": "tile_y", "value": float(y_off)},
                    ],
                },
            }
        )

    return features


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained experts + gating network from checkpoint."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    ckpt_args = checkpoint.get("args", None)

    if ckpt_args is None:
        raise ValueError("Checkpoint missing training args; cannot infer network dimensions.")

    models = create_all_experts(
        network_name=ckpt_args.network,
        in_channels=3,
        num_classes=ckpt_args.num_classes,
        embedding_dim=ckpt_args.embedding_dim,
    )

    gating_net = get_gating_network_4experts(
        feature_channels=64,
        num_experts=4,
        num_classes=ckpt_args.num_classes,
        embedding_dim=ckpt_args.embedding_dim,
    )

    for k in models:
        models[k].load_state_dict(checkpoint["models"][k], strict=True)
        models[k] = models[k].to(device).eval()

    gating_net.load_state_dict(checkpoint["gating_net"], strict=True)
    gating_net = gating_net.to(device).eval()
    return models, gating_net, ckpt_args


def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check Shapely availability
    if SHAPELY_AVAILABLE:
        print("Shapely available - using robust geometry validation")
    else:
        print("Shapely not available - install with 'pip install shapely' for better geometry handling")

    wsi = load_wsi_array(Path(args.wsi_path))
    h, w, _ = wsi.shape
    print(f"Loaded WSI: {args.wsi_path} with shape {wsi.shape}")

    models, gating_net, ckpt_args = load_model(Path(args.checkpoint), device)
    normalize = get_normalize_transform()

    clusterer = EmbeddingClusterer(
        method=args.cluster_method,
        bandwidth=args.cluster_bandwidth,
        min_instance_area=args.min_instance_area,
        device=str(device),
    )

    coords = list(iter_tile_coords(h, w, args.patch_size, args.stride))
    print(f"Total candidate tiles: {len(coords)}")

    features: List[Dict] = []
    skipped_tiles = 0
    pbar = tqdm(range(0, len(coords), args.batch_size), desc="WSI inference")

    for i in pbar:
        batch_coords = coords[i : i + args.batch_size]
        batch_imgs = []
        kept_coords = []

        for x, y in batch_coords:
            patch = wsi[y : y + args.patch_size, x : x + args.patch_size]
            if patch.shape[:2] != (args.patch_size, args.patch_size):
                continue

            if args.min_tissue_ratio > 0:
                gray = patch.mean(axis=2).astype(np.uint8)
                tissue_ratio = float((gray < 240).sum()) / float(args.patch_size * args.patch_size)
                if tissue_ratio < args.min_tissue_ratio:
                    skipped_tiles += 1
                    continue

            norm = normalize(patch)
            batch_imgs.append(torch.from_numpy(norm).permute(2, 0, 1).float())
            kept_coords.append((x, y))

        if not batch_imgs:
            continue

        images = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(enabled=args.amp):
                seg_feat, _ = models["segment"](images)
                sdf_feat, _ = models["sdf"](images)
                bnd_feat, _ = models["boundary"](images)
                inst_feat, _ = models["instance"](images)
                concat_feat = torch.cat([seg_feat, sdf_feat, bnd_feat, inst_feat], dim=1)
                seg_out, _, _, embed_out = gating_net(concat_feat)

            seg_pred = torch.argmax(seg_out, dim=1)
            inst_masks = clusterer(embed_out, seg_pred).cpu().numpy()

        for inst_mask, (x, y) in zip(inst_masks, kept_coords):
            tile_features = instance_mask_to_features(
                instance_mask=inst_mask.astype(np.int32),
                x_off=x,
                y_off=y,
                class_name=args.class_name,
                min_area=args.min_instance_area,
                class_color_rgb=args.class_color_rgb,
                use_shapely=args.use_shapely,
            )
            features.extend(tile_features)

        pbar.set_postfix({"features": len(features), "skipped": skipped_tiles})

    # GeoJSON FeatureCollection for QuPath import
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    out_path = Path(args.output_geojson)
    atomic_write_json(geojson, out_path)
    print(f"\nSaved {len(features)} features to {out_path}")
    print(f"Skipped {skipped_tiles} low-tissue tiles")


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained model on WSI TIFF and export GeoJSON.")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to input WSI TIFF.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument("--output_geojson", type=str, required=True, help="Output GeoJSON path.")

    parser.add_argument("--patch_size", type=int, default=512, help="Inference tile size.")
    parser.add_argument("--stride", type=int, default=512, help="Tile stride.")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--min_tissue_ratio", type=float, default=0.0, help="Skip low-tissue tiles (< ratio).")

    parser.add_argument("--cluster_method", type=str, default="meanshift", choices=["meanshift", "hdbscan"])
    parser.add_argument("--cluster_bandwidth", type=float, default=1.5, help="Embedding clustering bandwidth.")
    parser.add_argument("--min_instance_area", type=int, default=50, help="Minimum instance area in pixels.")

    parser.add_argument("--class_name", type=str, default="Tubule", help="GeoJSON classification name.")
    parser.add_argument(
        "--class_color_rgb",
        type=int,
        default=-65536,
        help="QuPath signed int colorRGB for class (default: red).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMP inference.",
    )
    parser.add_argument(
        "--use_shapely",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Shapely for geometry validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
