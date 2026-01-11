"""
Window Management
=================
GLFW window creation and input handling.
"""

import glfw
from OpenGL.GL import *
from typing import Callable, Optional, Set
import numpy as np


class Window:
    """
    Manages the GLFW window and input state.
    """
    
    def __init__(self, width: int = 1280, height: int = 720, title: str = "Procedural World"):
        self.width = width
        self.height = height
        self.title = title
        
        # Input state
        self.keys_pressed: Set[int] = set()
        self.keys_just_pressed: Set[int] = set()
        self.mouse_pos = np.array([0.0, 0.0])
        self.mouse_delta = np.array([0.0, 0.0])
        self.last_mouse_pos = np.array([0.0, 0.0])
        self.mouse_captured = False
        self.first_mouse = True
        
        # Scroll state
        self.scroll_delta = 0.0
        
        # Window handle
        self.handle = None
        
    def init(self) -> bool:
        """Initialize GLFW and create window."""
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False
        
        # OpenGL 3.3 Core Profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # Required on Mac
        
        # Create window
        self.handle = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.handle:
            print("Failed to create GLFW window")
            glfw.terminate()
            return False
        
        glfw.make_context_current(self.handle)
        
        # Set callbacks
        glfw.set_key_callback(self.handle, self._key_callback)
        glfw.set_cursor_pos_callback(self.handle, self._mouse_callback)
        glfw.set_scroll_callback(self.handle, self._scroll_callback)
        glfw.set_framebuffer_size_callback(self.handle, self._resize_callback)
        
        # Enable vsync
        glfw.swap_interval(1)
        
        # Initialize mouse position
        mx, my = glfw.get_cursor_pos(self.handle)
        self.mouse_pos = np.array([mx, my])
        self.last_mouse_pos = self.mouse_pos.copy()
        
        return True
    
    def _key_callback(self, window, key, scancode, action, mods):
        """Handle key events."""
        if action == glfw.PRESS:
            self.keys_pressed.add(key)
            self.keys_just_pressed.add(key)
            
            # ESC to release mouse / close window
            if key == glfw.KEY_ESCAPE:
                if self.mouse_captured:
                    self.release_mouse()
                else:
                    glfw.set_window_should_close(window, True)
        elif action == glfw.RELEASE:
            self.keys_pressed.discard(key)
    
    def _mouse_callback(self, window, xpos, ypos):
        """Handle mouse movement."""
        self.mouse_pos = np.array([xpos, ypos])
        
        if self.first_mouse:
            self.last_mouse_pos = self.mouse_pos.copy()
            self.first_mouse = False
    
    def _scroll_callback(self, window, xoffset, yoffset):
        """Handle scroll events."""
        self.scroll_delta += yoffset
    
    def _resize_callback(self, window, width, height):
        """Handle window resize."""
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
    
    def capture_mouse(self):
        """Capture and hide the mouse cursor."""
        glfw.set_input_mode(self.handle, glfw.CURSOR, glfw.CURSOR_DISABLED)
        self.mouse_captured = True
        self.first_mouse = True
    
    def release_mouse(self):
        """Release the mouse cursor."""
        glfw.set_input_mode(self.handle, glfw.CURSOR, glfw.CURSOR_NORMAL)
        self.mouse_captured = False
    
    def update(self):
        """Update input state. Call once per frame."""
        # Calculate mouse delta
        self.mouse_delta = self.mouse_pos - self.last_mouse_pos
        self.last_mouse_pos = self.mouse_pos.copy()
        
        # Clear per-frame state
        self.keys_just_pressed.clear()
        self.scroll_delta = 0.0
        
        # Poll events
        glfw.poll_events()
    
    def is_key_pressed(self, key: int) -> bool:
        """Check if a key is currently held down."""
        return key in self.keys_pressed
    
    def is_key_just_pressed(self, key: int) -> bool:
        """Check if a key was pressed this frame."""
        return key in self.keys_just_pressed
    
    def should_close(self) -> bool:
        """Check if window should close."""
        return glfw.window_should_close(self.handle)
    
    def swap_buffers(self):
        """Swap front and back buffers."""
        glfw.swap_buffers(self.handle)
    
    def get_time(self) -> float:
        """Get time since GLFW init."""
        return glfw.get_time()
    
    def get_aspect_ratio(self) -> float:
        """Get window aspect ratio."""
        if self.height > 0:
            return self.width / self.height
        return 1.0
    
    def destroy(self):
        """Clean up GLFW resources."""
        if self.handle:
            glfw.destroy_window(self.handle)
        glfw.terminate()