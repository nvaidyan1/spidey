"""
Heightmap Generation
====================
Converts noise into usable terrain height data.
"""

import numpy as np
from terrain.noise import fbm, ridge_noise


def generate_heightmap(chunk_x: int, chunk_z: int, size: int = 32,
                       scale: float = 0.01, height_scale: float = 30.0,
                       world_scale: float = 1.0) -> np.ndarray:
    """
    Generate a heightmap for a terrain chunk.
    
    Parameters:
    - chunk_x, chunk_z: Chunk coordinates (not world coordinates)
    - size: Number of vertices per side (size x size grid)
    - scale: Noise scale (smaller = larger features)
    - height_scale: Maximum terrain height
    - world_scale: Size of each grid cell in world units
    
    Returns:
    - 2D numpy array of height values, shape (size, size)
    """
    heightmap = np.zeros((size, size), dtype=np.float32)
    
    # World offset for this chunk
    offset_x = chunk_x * (size - 1) * world_scale
    offset_z = chunk_z * (size - 1) * world_scale
    
    for z in range(size):
        for x in range(size):
            # World position
            wx = offset_x + x * world_scale
            wz = offset_z + z * world_scale
            
            # Sample noise at this position
            nx = wx * scale
            nz = wz * scale
            
            # Simplified noise: just FBM with 4 octaves (faster)
            base = fbm(nx, nz, octaves=4, lacunarity=2.0, persistence=0.5)
            
            # Normalize to [0, 1] then scale
            height = (base + 1.0) / 2.0
            heightmap[z, x] = height * height_scale
    
    return heightmap


def get_height_at(heightmap: np.ndarray, x: float, z: float, 
                  world_scale: float = 1.0) -> float:
    """
    Get interpolated height at any world position within a heightmap.
    Uses bilinear interpolation for smooth results.
    
    Parameters:
    - heightmap: 2D height array
    - x, z: Local coordinates within the chunk (0 to size-1)
    - world_scale: Size of each grid cell
    
    Returns:
    - Interpolated height value
    """
    size = heightmap.shape[0]
    
    # Convert to grid coordinates
    gx = x / world_scale
    gz = z / world_scale
    
    # Get integer and fractional parts
    x0 = int(np.floor(gx))
    z0 = int(np.floor(gz))
    x1 = x0 + 1
    z1 = z0 + 1
    
    # Clamp to valid range
    x0 = max(0, min(size - 1, x0))
    x1 = max(0, min(size - 1, x1))
    z0 = max(0, min(size - 1, z0))
    z1 = max(0, min(size - 1, z1))
    
    # Fractional parts
    fx = gx - np.floor(gx)
    fz = gz - np.floor(gz)
    
    # Bilinear interpolation
    h00 = heightmap[z0, x0]
    h10 = heightmap[z0, x1]
    h01 = heightmap[z1, x0]
    h11 = heightmap[z1, x1]
    
    h0 = h00 * (1 - fx) + h10 * fx
    h1 = h01 * (1 - fx) + h11 * fx
    
    return h0 * (1 - fz) + h1 * fz


def compute_normal(heightmap: np.ndarray, x: int, z: int, 
                   world_scale: float = 1.0) -> np.ndarray:
    """
    Compute the surface normal at a heightmap point using finite differences.
    """
    size = heightmap.shape[0]
    
    # Get neighboring heights (clamped at edges)
    x0 = max(0, x - 1)
    x1 = min(size - 1, x + 1)
    z0 = max(0, z - 1)
    z1 = min(size - 1, z + 1)
    
    # Height differences
    dhdx = (heightmap[z, x1] - heightmap[z, x0]) / (world_scale * (x1 - x0))
    dhdz = (heightmap[z1, x] - heightmap[z0, x]) / (world_scale * (z1 - z0))
    
    # Normal is cross product of tangent vectors
    # Tangent in x: (1, dhdx, 0)
    # Tangent in z: (0, dhdz, 1)
    # Cross: (-dhdx, 1, -dhdz)
    normal = np.array([-dhdx, 1.0, -dhdz], dtype=np.float32)
    
    # Normalize
    length = np.sqrt(np.sum(normal * normal))
    if length > 0:
        normal /= length
    
    return normal