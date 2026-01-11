"""
Mesh Generation
===============
Converts heightmaps into renderable vertex and index buffers.
"""

import numpy as np
from typing import Tuple
from terrain.heightmap import compute_normal


def heightmap_to_mesh(heightmap: np.ndarray, world_scale: float = 1.0,
                      offset_x: float = 0.0, offset_z: float = 0.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a heightmap to a mesh (vertices and indices).
    
    Parameters:
    - heightmap: 2D array of height values, shape (size, size)
    - world_scale: Distance between adjacent vertices in world units
    - offset_x, offset_z: World position offset for this chunk
    
    Returns:
    - vertices: numpy array of shape (n_vertices, 9)
                Each vertex has: x, y, z, nx, ny, nz, r, g, b
    - indices: numpy array of triangle indices
    """
    size = heightmap.shape[0]
    n_vertices = size * size
    n_triangles = (size - 1) * (size - 1) * 2
    
    # Vertex data: position (3) + normal (3) + color (3) = 9 floats
    vertices = np.zeros((n_vertices, 9), dtype=np.float32)
    indices = np.zeros(n_triangles * 3, dtype=np.uint32)
    
    # Fill vertex positions and colors
    for z in range(size):
        for x in range(size):
            idx = z * size + x
            
            # Position
            wx = offset_x + x * world_scale
            wy = heightmap[z, x]
            wz = offset_z + z * world_scale
            
            vertices[idx, 0] = wx
            vertices[idx, 1] = wy
            vertices[idx, 2] = wz
            
            # Normal (computed from heightmap)
            normal = compute_normal(heightmap, x, z, world_scale)
            vertices[idx, 3:6] = normal
            
            # Color based on height and slope
            # Slope from normal (how much it points up)
            slope = normal[1]  # 1 = flat, 0 = vertical
            
            # Height-based coloring
            h = heightmap[z, x]
            max_h = np.max(heightmap) if np.max(heightmap) > 0 else 1.0
            normalized_h = h / max_h
            
            # Color gradient with more saturation for visibility
            if normalized_h < 0.25:
                # Low = dark green (valleys)
                r, g, b = 0.15, 0.35, 0.1
            elif normalized_h < 0.45:
                # Low-mid = grass green
                r, g, b = 0.25, 0.45, 0.15
            elif normalized_h < 0.65:
                # Mid = light green / tan
                if slope > 0.7:
                    r, g, b = 0.35, 0.45, 0.2
                else:
                    r, g, b = 0.45, 0.4, 0.3
            elif normalized_h < 0.8:
                # High = rock gray
                r, g, b = 0.5, 0.48, 0.45
            else:
                # Peaks = light gray/white
                r, g, b = 0.65, 0.63, 0.6
            
            # Slight variation based on position
            variation = ((x * 7 + z * 13) % 100) / 100.0 * 0.08 - 0.04
            vertices[idx, 6] = np.clip(r + variation, 0, 1)
            vertices[idx, 7] = np.clip(g + variation, 0, 1)
            vertices[idx, 8] = np.clip(b + variation, 0, 1)
    
    # Fill indices (two triangles per grid cell)
    idx = 0
    for z in range(size - 1):
        for x in range(size - 1):
            # Vertices of this cell
            v00 = z * size + x
            v10 = z * size + (x + 1)
            v01 = (z + 1) * size + x
            v11 = (z + 1) * size + (x + 1)
            
            # Triangle 1: v00 -> v10 -> v01
            indices[idx] = v00
            indices[idx + 1] = v10
            indices[idx + 2] = v01
            
            # Triangle 2: v10 -> v11 -> v01
            indices[idx + 3] = v10
            indices[idx + 4] = v11
            indices[idx + 5] = v01
            
            idx += 6
    
    return vertices, indices


def create_flat_grid(size: int = 32, world_scale: float = 1.0,
                     offset_x: float = 0.0, offset_z: float = 0.0
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a flat grid mesh (for testing).
    """
    heightmap = np.zeros((size, size), dtype=np.float32)
    return heightmap_to_mesh(heightmap, world_scale, offset_x, offset_z)