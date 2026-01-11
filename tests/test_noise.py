"""
Test script for noise generation.
Outputs PNG images so you can visually verify the noise looks correct.

Run from project root:
    python -m tests.test_noise

Expected output:
- output/noise_simplex.png: Raw simplex noise (should look like smooth clouds)
- output/noise_fbm.png: Fractal brownian motion (should look like terrain from above)
- output/noise_ridge.png: Ridge noise (should have sharp mountain-like features)
"""

import numpy as np
from PIL import Image
import os
import time

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrain.noise import simplex2, fbm, ridge_noise, generate_noise_map


def noise_to_image(noise_array: np.ndarray, filename: str):
    """Convert a [0,1] noise array to a grayscale PNG."""
    # Clamp to [0, 1] just in case
    noise_array = np.clip(noise_array, 0, 1)
    # Convert to 8-bit grayscale
    img_array = (noise_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(filename)
    print(f"  Saved: {filename}")


def test_simplex_noise():
    """Test raw simplex noise."""
    print("\n[1] Testing raw simplex noise...")
    size = 256
    scale = 0.02
    
    noise_map = np.zeros((size, size), dtype=np.float32)
    
    start = time.time()
    for y in range(size):
        for x in range(size):
            # simplex2 returns [-1, 1], normalize to [0, 1]
            noise_map[y, x] = (simplex2(x * scale, y * scale) + 1) / 2
    elapsed = time.time() - start
    
    print(f"  Generated {size}x{size} in {elapsed:.2f}s")
    print(f"  Value range: [{noise_map.min():.3f}, {noise_map.max():.3f}]")
    
    noise_to_image(noise_map, "output/noise_simplex.png")


def test_fbm_noise():
    """Test fractal brownian motion."""
    print("\n[2] Testing FBM noise...")
    size = 256
    scale = 0.01
    
    start = time.time()
    noise_map = generate_noise_map(size, size, scale=scale, octaves=6)
    elapsed = time.time() - start
    
    print(f"  Generated {size}x{size} in {elapsed:.2f}s")
    print(f"  Value range: [{noise_map.min():.3f}, {noise_map.max():.3f}]")
    
    noise_to_image(noise_map, "output/noise_fbm.png")


def test_ridge_noise():
    """Test ridge noise for mountain-like features."""
    print("\n[3] Testing ridge noise...")
    size = 256
    scale = 0.01
    
    noise_map = np.zeros((size, size), dtype=np.float32)
    
    start = time.time()
    for y in range(size):
        for x in range(size):
            noise_map[y, x] = ridge_noise(x * scale, y * scale, octaves=5)
    elapsed = time.time() - start
    
    print(f"  Generated {size}x{size} in {elapsed:.2f}s")
    print(f"  Value range: [{noise_map.min():.3f}, {noise_map.max():.3f}]")
    
    noise_to_image(noise_map, "output/noise_ridge.png")


def test_different_scales():
    """Show how scale affects the noise."""
    print("\n[4] Testing different scales...")
    size = 256
    scales = [0.005, 0.01, 0.02, 0.05]
    
    for scale in scales:
        noise_map = generate_noise_map(size, size, scale=scale, octaves=4)
        filename = f"output/noise_scale_{scale}.png"
        noise_to_image(noise_map, filename)


def test_chunk_continuity():
    """Test that adjacent chunks line up properly."""
    print("\n[5] Testing chunk continuity...")
    chunk_size = 128
    scale = 0.01
    
    # Generate a 2x2 grid of chunks
    full_map = np.zeros((chunk_size * 2, chunk_size * 2), dtype=np.float32)
    
    for cy in range(2):
        for cx in range(2):
            chunk = generate_noise_map(
                chunk_size, chunk_size, 
                scale=scale, octaves=4,
                offset_x=cx * chunk_size,
                offset_y=cy * chunk_size
            )
            full_map[cy*chunk_size:(cy+1)*chunk_size, 
                     cx*chunk_size:(cx+1)*chunk_size] = chunk
    
    noise_to_image(full_map, "output/noise_chunks.png")
    print("  If chunks line up, you should see no seams in the image")


def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    print("=" * 50)
    print("NOISE GENERATION TESTS")
    print("=" * 50)
    
    test_simplex_noise()
    test_fbm_noise()
    test_ridge_noise()
    test_different_scales()
    test_chunk_continuity()
    
    print("\n" + "=" * 50)
    print("All tests complete! Check the output/ folder for images.")
    print("=" * 50)


if __name__ == "__main__":
    main()