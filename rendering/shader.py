"""
Shader Utilities
================
Loading and compiling OpenGL shaders.
"""

from OpenGL.GL import *
from typing import Optional
import os


def compile_shader(source: str, shader_type: int) -> int:
    """
    Compile a single shader.
    
    Parameters:
    - source: GLSL source code
    - shader_type: GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
    
    Returns:
    - Shader handle
    
    Raises:
    - RuntimeError if compilation fails
    """
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    
    # Check for errors
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        shader_type_name = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
        raise RuntimeError(f"{shader_type_name} shader compilation failed:\n{error}")
    
    return shader


def create_program(vertex_source: str, fragment_source: str) -> int:
    """
    Create and link a shader program.
    
    Parameters:
    - vertex_source: Vertex shader GLSL code
    - fragment_source: Fragment shader GLSL code
    
    Returns:
    - Program handle
    """
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    # Check for linking errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Shader program linking failed:\n{error}")
    
    # Clean up individual shaders (they're now part of the program)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return program


def load_program(vertex_path: str, fragment_path: str) -> int:
    """
    Load and compile shaders from files.
    """
    with open(vertex_path, 'r') as f:
        vertex_source = f.read()
    with open(fragment_path, 'r') as f:
        fragment_source = f.read()
    
    return create_program(vertex_source, fragment_source)


# ============================================
# Uniform Helpers
# ============================================

def set_uniform_mat4(program: int, name: str, matrix):
    """Set a mat4 uniform."""
    import numpy as np
    loc = glGetUniformLocation(program, name)
    if loc >= 0:
        # OpenGL expects column-major, numpy is row-major, so transpose
        glUniformMatrix4fv(loc, 1, GL_TRUE, matrix.astype(np.float32))


def set_uniform_vec3(program: int, name: str, vec):
    """Set a vec3 uniform."""
    loc = glGetUniformLocation(program, name)
    if loc >= 0:
        glUniform3f(loc, vec[0], vec[1], vec[2])


def set_uniform_float(program: int, name: str, value: float):
    """Set a float uniform."""
    loc = glGetUniformLocation(program, name)
    if loc >= 0:
        glUniform1f(loc, value)


def set_uniform_int(program: int, name: str, value: int):
    """Set an int uniform."""
    loc = glGetUniformLocation(program, name)
    if loc >= 0:
        glUniform1i(loc, value)