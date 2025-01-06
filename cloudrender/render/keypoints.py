import numpy as np
from OpenGL import GL as gl
from typing import Optional, Union, List
from dataclasses import dataclass
from .shaders.shader_loader import Shader
from .renderable import Renderable
from .lights import Light, DirectionalLight
from .shadowmap import ShadowMap
from ..camera.models import StandardProjectionCameraModel
from .mesh import Mesh 
import os


class SimpleKeypoint(Mesh):
    """
    A class to render keypoints as points in 3D space.
    Inherits from the Mesh class and supports rendering alongside other mesh objects.
    """

    @dataclass
    class KeypointContainer:
        vertices: np.ndarray  # Keypoint positions (N x 3)
        colors: Optional[np.ndarray] = None  # Keypoint colors (N x 4, RGBA)
        point_size: float = 10.0  # Size of the rendered points

    def __init__(self, *args, draw_shadows: bool = False, generate_shadows: bool = False, **kwargs):
        super().__init__(*args, draw_shadows=draw_shadows, generate_shadows=generate_shadows, **kwargs)
        self.point_size = 10.0  # Default point size

    def _init_shaders(self, camera_model, shader_mode):
        """
        Initialize shaders for rendering keypoints.
        """
        self.shader = shader = Shader()
        dirname = os.path.dirname(os.path.abspath(__file__))

        # Use a simple shader for rendering points
        shader.initShaderFromGLSL(
            [os.path.join(dirname, f"shaders/simple_keypoint/vertex_{camera_model}.glsl")],
            [os.path.join(dirname, "shaders/simple_keypoint/fragment.glsl")]
        )
        self.context.shader_ids.update(self.locate_uniforms(self.shader, ['MVP', 'point_size', 'overlay_color']))

    def _delete_buffers(self):
        """
        Delete OpenGL buffers.
        """
        gl.glDeleteBuffers(2, [self.context.vertexbuffer, self.context.colorbuffer])
        gl.glDeleteVertexArrays(1, [self.context.vao])

    def set_point_size(self, size: float):
        """
        Set the size of the rendered points.
        """
        self.point_size = size

    def _set_buffers(self, keypoints: Union[KeypointContainer, np.ndarray]):
        """
        Set OpenGL buffers for keypoints.
        """
        if isinstance(keypoints, np.ndarray):
            keypoints = self.KeypointContainer(vertices=keypoints)

        glverts = np.copy(keypoints.vertices.astype(np.float32), order='C')
        if keypoints.colors is not None:
            glcolors = np.copy(keypoints.colors.astype(np.float32) / 255.0, order='C')
        else:
            # Default to white if no colors are provided
            glcolors = np.ones((glverts.shape[0], 4), dtype=np.float32)

        self.nglverts = len(glverts)

        # Generate and bind VAO
        self.context.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.context.vao)

        # Vertex buffer
        self.context.vertexbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_DYNAMIC_DRAW)

        # Color buffer
        self.context.colorbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_DYNAMIC_DRAW)

    def _update_buffers(self, keypoints: Union[KeypointContainer, np.ndarray]):
        """
        Update OpenGL buffers for keypoints.
        """
        if isinstance(keypoints, np.ndarray):
            keypoints = self.KeypointContainer(vertices=keypoints)

        glverts = np.copy(keypoints.vertices.astype(np.float32), order='C')
        if keypoints.colors is not None:
            glcolors = np.copy(keypoints.colors.astype(np.float32) / 255.0, order='C')
        else:
            glcolors = np.ones((glverts.shape[0], 4), dtype=np.float32)

        gl.glBindVertexArray(self.context.vao)

        # Update vertex buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_DYNAMIC_DRAW)

        # Update color buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_DYNAMIC_DRAW)

    def _upload_uniforms(self, shader_ids, lights=(), shadowmaps=()):
        """
        Upload uniforms to the shader.
        """
        gl.glUniformMatrix4fv(shader_ids['MVP'], 1, gl.GL_FALSE, self.context.ModelViewProjection)
        gl.glUniform1f(shader_ids['point_size'], self.point_size)
        gl.glUniform4f(shader_ids['overlay_color'], 1.0, 1.0, 1.0, 1.0)  # Default overlay color

    def _draw(self, reset: bool, lights: List[Light], shadowmaps: List[ShadowMap]) -> bool:
        """
        Render the keypoints.
        """
        if not reset:
            return False

        self.shader.begin()
        self.upload_uniforms(self.context.shader_ids, lights, shadowmaps)

        gl.glBindVertexArray(self.context.vao)

        # Enable vertex and color attributes
        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Render as points
        gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)

        # Disable attributes
        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)

        self.shader.end()
        return True