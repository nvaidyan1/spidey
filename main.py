"""
Procedural World - Main Entry Point
====================================

Interactive infinite procedural terrain demo.

Controls:
- Click window to capture mouse
- WASD: Move horizontally
- Space/E: Move up
- Ctrl/Q: Move down
- Shift: Sprint
- Mouse: Look around
- Scroll: Zoom (FOV)
- ESC: Release mouse / Exit
"""

from OpenGL.GL import *
import numpy as np

from core.window import Window
from core.camera import Camera
from rendering.terrain_renderer import TerrainRenderer
from utils.config import get


def main():
    print("=" * 50)
    print("PROCEDURAL WORLD")
    print("=" * 50)
    print("\nControls:")
    print("  Click window to capture mouse")
    print("  WASD      - Move")
    print("  Space/E   - Up")
    print("  Ctrl/Q    - Down")
    print("  Shift     - Sprint")
    print("  Mouse     - Look around")
    print("  Scroll    - Zoom")
    print("  ESC       - Release mouse / Exit")
    print("=" * 50)
    
    # Create window
    window = Window(
        get('window.width', 1280),
        get('window.height', 720),
        get('window.title', "Procedural World")
    )

    if not window.init():
        return
    
    # OpenGL settings
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_CULL_FACE)
    # glCullFace(GL_BACK)
    sky = get('sky.color', [0.5, 0.7, 0.9])
    glClearColor(sky[0], sky[1], sky[2], 1.0)
    
    # Create terrain renderer (smaller chunks = faster generation)
    terrain = TerrainRenderer(
        chunk_size=get('terrain.chunk_size', 24),
        world_scale=get('terrain.world_scale', 2.5),
        height_scale=get('terrain.height_scale', 40.0),
        noise_scale=get('terrain.noise_scale', 0.008)
    )
    terrain.init()
    
    # Create camera
    camera = Camera(
        position=get('camera.start_position', [0.0, 80.0, 0.0]),
        yaw=get('camera.start_yaw', -90.0),
        pitch=get('camera.start_pitch', -15.0)
    )
    
    # Initial terrain load
    terrain.update(camera.position)
    
    # Timing
    last_time = window.get_time()
    frame_count = 0
    fps_timer = 0.0
    
    print("\nStarting render loop...")
    print("Click the window to capture mouse and start exploring!")
    
    # Main loop
    while not window.should_close():
        # Calculate delta time
        current_time = window.get_time()
        dt = current_time - last_time
        last_time = current_time
        
        # FPS counter
        frame_count += 1
        fps_timer += dt
        if fps_timer >= 1.0:
            # print(f"FPS: {frame_count}")
            frame_count = 0
            fps_timer = 0.0
        
        # Update input
        window.update()
        
        # Click to capture mouse
        import glfw
        if glfw.get_mouse_button(window.handle, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            if not window.mouse_captured:
                window.capture_mouse()
        
        # Update camera
        camera.update(window, dt)
        
        # Update terrain chunks based on camera position
        terrain.update(camera.position)
        
        # Clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Get matrices
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(window.get_aspect_ratio())
        
        # Render terrain
        terrain.render(view, proj, camera.position)
        
        # Swap buffers
        window.swap_buffers()
    
    # Cleanup
    print("\nShutting down...")
    terrain.destroy()
    window.destroy()
    print("Done!")


if __name__ == "__main__":
    main()