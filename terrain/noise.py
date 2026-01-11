"""
Simplex Noise Implementation
============================
Written from scratch based on Ken Perlin's improved noise and simplex algorithm.

Simplex noise advantages over classic Perlin:
- Fewer directional artifacts
- Scales better to higher dimensions
- Slightly faster (fewer interpolations)

The key insight: instead of interpolating on a square grid (4 corners),
simplex uses a triangular grid (3 corners in 2D). This is more efficient
and produces more organic-looking results.
"""

import numpy as np
from typing import Tuple

# Permutation table - this is the "randomness" of the noise
# We use a fixed table so the same coordinates always give the same value
# (This is what makes it deterministic/seedable)
_PERM = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
], dtype=np.int32)

# Double the permutation table to avoid index wrapping
_PERM = np.concatenate([_PERM, _PERM])

# Gradients for 2D simplex noise
# These are unit vectors pointing in 8 directions
_GRAD2 = np.array([
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1]
], dtype=np.float32)

# Skewing factors for 2D
# These transform the coordinate space from square grid to simplex grid and back
_F2 = 0.5 * (np.sqrt(3.0) - 1.0)  # Skew factor: square -> simplex
_G2 = (3.0 - np.sqrt(3.0)) / 6.0   # Unskew factor: simplex -> square


def _dot2(grad: np.ndarray, x: float, y: float) -> float:
    """Dot product of gradient vector and distance vector."""
    return grad[0] * x + grad[1] * y


def simplex2(x: float, y: float) -> float:
    """
    2D Simplex noise at coordinates (x, y).
    
    Returns a value in approximately [-1, 1] range.
    
    The algorithm:
    1. Skew input space to find which simplex cell we're in
    2. Determine which triangle of the cell we're in
    3. Calculate contribution from each of the 3 corners
    4. Sum and scale the result
    """
    # Skew the input space to determine which simplex cell we're in
    s = (x + y) * _F2
    i = int(np.floor(x + s))
    j = int(np.floor(y + s))
    
    # Unskew back to get the cell origin in original coords
    t = (i + j) * _G2
    X0 = i - t  # Cell origin x
    Y0 = j - t  # Cell origin y
    
    # Distance from cell origin to input point
    x0 = x - X0
    y0 = y - Y0
    
    # Determine which simplex (triangle) we're in
    # In 2D, the simplex is a triangle. The cell is divided into two triangles.
    # We figure out which one based on whether we're above or below the diagonal.
    if x0 > y0:
        # Lower triangle, order: (0,0) -> (1,0) -> (1,1)
        i1, j1 = 1, 0
    else:
        # Upper triangle, order: (0,0) -> (0,1) -> (1,1)
        i1, j1 = 0, 1
    
    # Offsets for the middle corner (in unskewed coords)
    x1 = x0 - i1 + _G2
    y1 = y0 - j1 + _G2
    
    # Offsets for the last corner (always (1,1) from origin, in skewed space)
    x2 = x0 - 1.0 + 2.0 * _G2
    y2 = y0 - 1.0 + 2.0 * _G2
    
    # Hash the corner coordinates to get gradient indices
    ii = i & 255
    jj = j & 255
    
    gi0 = _PERM[ii + _PERM[jj]] % 8
    gi1 = _PERM[ii + i1 + _PERM[jj + j1]] % 8
    gi2 = _PERM[ii + 1 + _PERM[jj + 1]] % 8
    
    # Calculate contribution from each corner
    # Each corner contributes based on distance: (0.5 - distance^2)^4 * gradient·offset
    # If distance > sqrt(0.5), contribution is 0
    
    n0 = n1 = n2 = 0.0
    
    # Corner 0
    t0 = 0.5 - x0*x0 - y0*y0
    if t0 >= 0:
        t0 *= t0
        n0 = t0 * t0 * _dot2(_GRAD2[gi0], x0, y0)
    
    # Corner 1
    t1 = 0.5 - x1*x1 - y1*y1
    if t1 >= 0:
        t1 *= t1
        n1 = t1 * t1 * _dot2(_GRAD2[gi1], x1, y1)
    
    # Corner 2
    t2 = 0.5 - x2*x2 - y2*y2
    if t2 >= 0:
        t2 *= t2
        n2 = t2 * t2 * _dot2(_GRAD2[gi2], x2, y2)
    
    # Scale to [-1, 1] range (70 is an empirical scaling factor)
    return 70.0 * (n0 + n1 + n2)


def simplex2_array(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Vectorized 2D simplex noise for arrays of coordinates.
    xs and ys should be the same shape (or broadcastable).
    Returns noise values in the same shape.
    """
    # For large arrays, we iterate - simplex doesn't vectorize as cleanly as Perlin
    # But this is still fast enough for our purposes
    shape = np.broadcast_shapes(xs.shape, ys.shape)
    xs = np.broadcast_to(xs, shape).flatten()
    ys = np.broadcast_to(ys, shape).flatten()
    
    result = np.array([simplex2(x, y) for x, y in zip(xs, ys)])
    return result.reshape(shape)


def fbm(x: float, y: float, octaves: int = 6, lacunarity: float = 2.0, 
        persistence: float = 0.5) -> float:
    """
    Fractal Brownian Motion - layered noise for natural-looking terrain.
    
    This is the secret sauce for realistic terrain. We layer multiple
    frequencies of noise:
    - Low frequency = large hills and valleys
    - High frequency = small bumps and details
    
    Parameters:
    - octaves: number of noise layers (more = more detail, slower)
    - lacunarity: frequency multiplier per octave (typically 2)
    - persistence: amplitude multiplier per octave (typically 0.5)
    
    With default values:
    - Octave 1: freq=1, amp=1     (big features)
    - Octave 2: freq=2, amp=0.5   (medium features)
    - Octave 3: freq=4, amp=0.25  (small features)
    - etc.
    """
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0  # For normalization
    
    for _ in range(octaves):
        value += amplitude * simplex2(x * frequency, y * frequency)
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    # Normalize to [-1, 1]
    return value / max_value


def fbm_array(xs: np.ndarray, ys: np.ndarray, octaves: int = 6,
              lacunarity: float = 2.0, persistence: float = 0.5) -> np.ndarray:
    """Vectorized FBM for arrays."""
    shape = np.broadcast_shapes(xs.shape, ys.shape)
    xs = np.broadcast_to(xs, shape).flatten()
    ys = np.broadcast_to(ys, shape).flatten()
    
    result = np.array([fbm(x, y, octaves, lacunarity, persistence) 
                       for x, y in zip(xs, ys)])
    return result.reshape(shape)


def ridge_noise(x: float, y: float, octaves: int = 6, lacunarity: float = 2.0,
                persistence: float = 0.5) -> float:
    """
    Ridge noise - creates sharp mountain ridges.
    
    The trick: take absolute value of noise and invert it.
    This turns smooth hills into sharp ridges.
    """
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        # The magic: abs() creates sharp creases, 1-abs() inverts them to ridges
        n = 1.0 - abs(simplex2(x * frequency, y * frequency))
        # Square it to sharpen the ridges further
        n = n * n
        value += amplitude * n
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    return value / max_value


# Convenience function to generate a full heightmap
def generate_noise_map(width: int, height: int, scale: float = 0.01,
                       octaves: int = 6, lacunarity: float = 2.0,
                       persistence: float = 0.5, offset_x: float = 0.0,
                       offset_y: float = 0.0) -> np.ndarray:
    """
    Generate a 2D noise map.
    
    Parameters:
    - width, height: dimensions of output array
    - scale: how "zoomed in" the noise is (smaller = more zoomed in)
    - octaves, lacunarity, persistence: FBM parameters
    - offset_x, offset_y: offset in noise space (for chunk-based generation)
    
    Returns:
    - 2D numpy array of noise values in [0, 1] range
    """
    noise_map = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            nx = (x + offset_x) * scale
            ny = (y + offset_y) * scale
            # FBM returns [-1, 1], normalize to [0, 1]
            noise_map[y, x] = (fbm(nx, ny, octaves, lacunarity, persistence) + 1) / 2
    
    return noise_map


def generate_noise_map_fast(width: int, height: int, scale: float = 0.01,
                            octaves: int = 4, lacunarity: float = 2.0,
                            persistence: float = 0.5, offset_x: float = 0.0,
                            offset_y: float = 0.0) -> np.ndarray:
    """
    Faster noise map generation using reduced octaves and caching.
    Good enough for real-time terrain generation.
    """
    noise_map = np.zeros((height, width), dtype=np.float32)
    
    # Pre-compute coordinates
    for y in range(height):
        for x in range(width):
            nx = (x + offset_x) * scale
            ny = (y + offset_y) * scale
            
            # Simplified FBM with fewer octaves for speed
            value = 0.0
            amplitude = 1.0
            frequency = 1.0
            max_value = 0.0
            
            for _ in range(octaves):
                value += amplitude * simplex2(nx * frequency, ny * frequency)
                max_value += amplitude
                amplitude *= persistence
                frequency *= lacunarity
            
            noise_map[y, x] = (value / max_value + 1) / 2
    
    return noise_map