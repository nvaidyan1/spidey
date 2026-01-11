"""
Math Utilities
==============
Vector and matrix operations for 3D graphics.
We use numpy arrays but implement the graphics math ourselves.
"""

import numpy as np
from typing import Tuple


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    length = np.linalg.norm(v)
    if length > 0:
        return v / length
    return v


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3D vectors."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ], dtype=np.float32)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two vectors."""
    return np.sum(a * b)


# ============================================
# Matrix Construction
# ============================================

def identity() -> np.ndarray:
    """4x4 identity matrix."""
    return np.eye(4, dtype=np.float32)


def translation(x: float, y: float, z: float) -> np.ndarray:
    """4x4 translation matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    """4x4 scale matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotation_x(angle: float) -> np.ndarray:
    """4x4 rotation matrix around X axis. Angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotation_y(angle: float) -> np.ndarray:
    """4x4 rotation matrix around Y axis. Angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotation_z(angle: float) -> np.ndarray:
    """4x4 rotation matrix around Z axis. Angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Create a view matrix looking from eye toward target.
    
    This is the classic "look at" matrix used in 3D graphics.
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    
    # Forward vector (from target to eye, because we're in right-handed coords)
    f = normalize(eye - target)
    # Right vector
    r = normalize(cross(up, f))
    # Recalculate up to ensure orthogonality
    u = cross(f, r)
    
    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = r
    m[1, 0:3] = u
    m[2, 0:3] = f
    
    # Translation component
    m[0, 3] = -dot(r, eye)
    m[1, 3] = -dot(u, eye)
    m[2, 3] = -dot(f, eye)
    
    return m


def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Create a perspective projection matrix.
    
    Parameters:
    - fov: Field of view in radians (vertical)
    - aspect: Aspect ratio (width / height)
    - near: Near clipping plane
    - far: Far clipping plane
    """
    f = 1.0 / np.tan(fov / 2.0)
    
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    
    return m


def ortho(left: float, right: float, bottom: float, top: float, 
          near: float, far: float) -> np.ndarray:
    """Create an orthographic projection matrix."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


# ============================================
# Utility Functions
# ============================================

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamp x to the range [min_val, max_val]."""
    return max(min_val, min(max_val, x))


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """Smooth interpolation (Hermite)."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def remap(value: float, from_min: float, from_max: float, 
          to_min: float, to_max: float) -> float:
    """Remap a value from one range to another."""
    t = (value - from_min) / (from_max - from_min)
    return to_min + t * (to_max - to_min)