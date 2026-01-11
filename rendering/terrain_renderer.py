"""
Terrain Renderer
================
Handles OpenGL buffer management and drawing for terrain.
"""

from OpenGL.GL import *
import numpy as np
from typing import Dict, Tuple, Optional
import os

from terrain.heightmap import generate_heightmap
from terrain.mesh import heightmap_to_mesh
from rendering.shader import create_program, load_program, set_uniform_mat4, set_uniform_vec3, set_uniform_float
from utils.math_utils import identity
from utils.config import get


class ChunkData:
    """OpenGL data for a single terrain chunk."""
    
    def __init__(self, vao: int, vbo: int, ebo: int, index_count: int, heightmap: np.ndarray):
        self.vao = vao
        self.vbo = vbo
        self.ebo = ebo
        self.index_count = index_count
        self.heightmap = heightmap
    
    def destroy(self):
        """Clean up OpenGL resources."""
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])


class TerrainRenderer:
    """
    Manages terrain chunk rendering.
    """
    
    def __init__(self, chunk_size: int = 32, world_scale: float = 2.0,
                 height_scale: float = 40.0, noise_scale: float = 0.008):
        self.chunk_size = chunk_size
        self.world_scale = world_scale
        self.height_scale = height_scale
        self.noise_scale = noise_scale
        
        # Loaded chunks: (cx, cz) -> ChunkData
        self.chunks: Dict[Tuple[int, int], ChunkData] = {}
        
        # View distance in chunks (reduced for performance)
        self.view_distance = get('terrain.view_distance', 3)
        
        # Shader program
        self.shader = None
        
        # Lighting settings
        self.light_dir = np.array(get('lighting.sun_direction', [0.6, 0.8, 0.4]), dtype=np.float32)
        self.fog_color = np.array(get('fog.color', [0.7, 0.8, 0.9]), dtype=np.float32)
        self.fog_density = get('fog.density', 0.004) if get('fog.enabled', True) else 0.0
        self.fog_start = get('fog.start_distance', 150.0)
    
    def init(self):
        """Initialize shader program."""
        # Get shader directory
        shader_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shaders')
        vert_path = os.path.join(shader_dir, 'terrain.vert')
        frag_path = os.path.join(shader_dir, 'terrain.frag')
        
        self.shader = load_program(vert_path, frag_path)
    
    def _create_chunk(self, cx: int, cz: int) -> ChunkData:
        """Generate and upload a terrain chunk to GPU."""
        # World offset for this chunk
        chunk_world_size = (self.chunk_size - 1) * self.world_scale
        offset_x = cx * chunk_world_size
        offset_z = cz * chunk_world_size
        
        # Generate heightmap
        heightmap = generate_heightmap(
            cx, cz,
            size=self.chunk_size,
            scale=self.noise_scale,
            height_scale=self.height_scale,
            world_scale=self.world_scale
        )
        
        # Generate mesh
        vertices, indices = heightmap_to_mesh(
            heightmap,
            world_scale=self.world_scale,
            offset_x=offset_x,
            offset_z=offset_z
        )
        
        # Create VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        
        # Create VBO
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create EBO
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Vertex attributes
        stride = 9 * 4  # 9 floats * 4 bytes
        
        # Position (location 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Normal (location 1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        # Color (location 2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        glBindVertexArray(0)
        
        return ChunkData(vao, vbo, ebo, len(indices), heightmap)
    
    def update(self, camera_pos: np.ndarray):
        """
        Update loaded chunks based on camera position.
        Load nearby chunks, unload distant ones.
        """
        # Calculate which chunk the camera is in
        chunk_world_size = (self.chunk_size - 1) * self.world_scale
        cam_cx = int(np.floor(camera_pos[0] / chunk_world_size))
        cam_cz = int(np.floor(camera_pos[2] / chunk_world_size))
        
        # Determine which chunks should be loaded
        needed = set()
        for dz in range(-self.view_distance, self.view_distance + 1):
            for dx in range(-self.view_distance, self.view_distance + 1):
                needed.add((cam_cx + dx, cam_cz + dz))
        
        # Unload chunks that are too far
        to_remove = []
        for key in self.chunks:
            if key not in needed:
                to_remove.append(key)
        
        for key in to_remove:
            self.chunks[key].destroy()
            del self.chunks[key]
        
        # Load new chunks
        for key in needed:
            if key not in self.chunks:
                self.chunks[key] = self._create_chunk(key[0], key[1])
    
    def get_height_at(self, x: float, z: float) -> float:
        """
        Get terrain height at world position (x, z).
        Returns 0 if chunk not loaded.
        """
        chunk_world_size = (self.chunk_size - 1) * self.world_scale
        
        # Find chunk
        cx = int(np.floor(x / chunk_world_size))
        cz = int(np.floor(z / chunk_world_size))
        
        if (cx, cz) not in self.chunks:
            return 0.0
        
        chunk = self.chunks[(cx, cz)]
        
        # Local position within chunk
        local_x = x - cx * chunk_world_size
        local_z = z - cz * chunk_world_size
        
        # Grid coordinates
        gx = local_x / self.world_scale
        gz = local_z / self.world_scale
        
        # Bilinear interpolation
        x0 = int(np.floor(gx))
        z0 = int(np.floor(gz))
        x1 = x0 + 1
        z1 = z0 + 1
        
        size = self.chunk_size
        x0 = max(0, min(size - 1, x0))
        x1 = max(0, min(size - 1, x1))
        z0 = max(0, min(size - 1, z0))
        z1 = max(0, min(size - 1, z1))
        
        fx = gx - np.floor(gx)
        fz = gz - np.floor(gz)
        
        h00 = chunk.heightmap[z0, x0]
        h10 = chunk.heightmap[z0, x1]
        h01 = chunk.heightmap[z1, x0]
        h11 = chunk.heightmap[z1, x1]
        
        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        
        return h0 * (1 - fz) + h1 * fz
    
    def render(self, view_matrix: np.ndarray, projection_matrix: np.ndarray,
               camera_pos: np.ndarray):
        """Render all loaded terrain chunks."""
        glUseProgram(self.shader)
        
        # Set uniforms
        model = identity()
        set_uniform_mat4(self.shader, "model", model)
        set_uniform_mat4(self.shader, "view", view_matrix)
        set_uniform_mat4(self.shader, "projection", projection_matrix)
        
        # Normalize light direction
        light_dir = self.light_dir / np.linalg.norm(self.light_dir)
        set_uniform_vec3(self.shader, "lightDir", light_dir)
        set_uniform_vec3(self.shader, "viewPos", camera_pos)
        set_uniform_vec3(self.shader, "fogColor", self.fog_color)
        set_uniform_float(self.shader, "fogDensity", self.fog_density)
        set_uniform_float(self.shader, "fogStart", self.fog_start)
        
        # Draw all chunks
        for chunk in self.chunks.values():
            glBindVertexArray(chunk.vao)
            glDrawElements(GL_TRIANGLES, chunk.index_count, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
    
    def destroy(self):
        """Clean up all resources."""
        for chunk in self.chunks.values():
            chunk.destroy()
        self.chunks.clear()
        
        if self.shader:
            glDeleteProgram(self.shader)