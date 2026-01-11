"""
Camera
======
First-person camera with WASD + mouse look controls.
"""

import numpy as np
import glfw
from utils.math_utils import look_at, perspective, normalize, cross
from utils.config import get


class Camera:
    """
    First-person camera with smooth movement.
    """
    
    def __init__(self, position=None, yaw: float = -90.0, pitch: float = -20.0):
        # Position in world space
        self.position = np.array(position if position else [0.0, 50.0, 0.0], dtype=np.float32)
        
        # Euler angles (degrees)
        self.yaw = yaw      # Rotation around Y axis
        self.pitch = pitch  # Rotation around X axis
        
        # Camera vectors (computed from angles)
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Movement settings
        self.move_speed = get('camera.move_speed', 30.0)
        self.sprint_multiplier = get('camera.sprint_multiplier', 3.0)
        self.mouse_sensitivity = get('camera.mouse_sensitivity', 0.1)
        
        # Projection settings
        self.fov = get('camera.fov', 70.0)
        self.near = get('camera.near_clip', 0.1)
        self.far = get('camera.far_clip', 1000.0)
        
        # Update vectors based on initial angles
        self._update_vectors()
    
    def _update_vectors(self):
        """Recalculate front, right, up vectors from yaw and pitch."""
        # Clamp pitch to prevent flipping
        self.pitch = max(-89.0, min(89.0, self.pitch))
        
        # Calculate front vector
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        self.front = np.array([
            np.cos(pitch_rad) * np.cos(yaw_rad),
            np.sin(pitch_rad),
            np.cos(pitch_rad) * np.sin(yaw_rad)
        ], dtype=np.float32)
        self.front = normalize(self.front)
        
        # Recalculate right and up
        self.right = normalize(cross(self.front, self.world_up))
        self.up = normalize(cross(self.right, self.front))
    
    def process_keyboard(self, window, dt: float):
        """
        Handle keyboard input for movement.
        
        Parameters:
        - window: Window instance with input state
        - dt: Delta time in seconds
        """
        speed = self.move_speed * dt
        
        # Sprint
        if window.is_key_pressed(glfw.KEY_LEFT_SHIFT):
            speed *= self.sprint_multiplier
        
        # Movement direction (horizontal plane only for WASD)
        front_flat = normalize(np.array([self.front[0], 0.0, self.front[2]]))
        right_flat = normalize(np.array([self.right[0], 0.0, self.right[2]]))
        
        # WASD movement
        if window.is_key_pressed(glfw.KEY_W):
            self.position += front_flat * speed
        if window.is_key_pressed(glfw.KEY_S):
            self.position -= front_flat * speed
        if window.is_key_pressed(glfw.KEY_A):
            self.position -= right_flat * speed
        if window.is_key_pressed(glfw.KEY_D):
            self.position += right_flat * speed
        
        # Vertical movement (Q/E or Space/Ctrl)
        if window.is_key_pressed(glfw.KEY_SPACE) or window.is_key_pressed(glfw.KEY_E):
            self.position[1] += speed
        if window.is_key_pressed(glfw.KEY_LEFT_CONTROL) or window.is_key_pressed(glfw.KEY_Q):
            self.position[1] -= speed
    
    def process_mouse(self, window, dt: float):
        """
        Handle mouse input for looking around.
        Only processes if mouse is captured.
        """
        if not window.mouse_captured:
            return
        
        dx, dy = window.mouse_delta
        
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity  # Inverted Y
        
        self._update_vectors()
    
    def process_scroll(self, window):
        """Handle scroll wheel for FOV zoom."""
        if window.scroll_delta != 0:
            self.fov -= window.scroll_delta * 2.0
            self.fov = max(30.0, min(110.0, self.fov))
    
    def update(self, window, dt: float):
        """Update camera based on all input."""
        self.process_keyboard(window, dt)
        self.process_mouse(window, dt)
        self.process_scroll(window)
    
    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix for rendering."""
        target = self.position + self.front
        return look_at(self.position, target, self.up)
    
    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Get the projection matrix for rendering."""
        return perspective(
            np.radians(self.fov),
            aspect_ratio,
            self.near,
            self.far
        )