# On some systems, EGL does not start properly if OpenGL was already initialized, that's why it's better
# to keep EGLContext import on top
from cloudrender.libegl import EGLContext
import logging
import numpy as np
import sys
import os
import json
import smplpytorch
from cloudrender.render import SimplePointcloud, DirectionalLight
from cloudrender.render.smpl_legacy import AnimatableSMPLModel
from cloudrender.camera import PerspectiveCameraModel
from cloudrender.camera.trajectory import Trajectory
from cloudrender.scene import Scene
from cloudrender.capturing import AsyncPBOCapture
from videoio import VideoWriter
from OpenGL import GL as gl
from tqdm import tqdm
from cloudrender.utils import trimesh_load_from_zip, load_hps_sequence
from egoallo import training_utils
from pathlib import Path
from typing import Optional
logger = logging.getLogger("main_script")
logger.setLevel(logging.INFO)


training_utils.ipdb_safety_net()

# This example shows how to:
# - render pointcloud
# - render a sequence of frames with moving SMPL mesh
# - smoothly move the camera
# - dump rendered frames to a video
class HPSPathManipulator:
    """Handles path management for HPS dataset and related assets."""
    
    def __init__(self, base_dir: Optional[str] = None, assets_dir: Optional[str] = None):
        # Allow override via environment variables or parameters
        self.base_dir = Path(base_dir or os.getenv(
            'EGOALLO_DATA_DIR',
            '/home/minghao/src/robotflow/egoallo/datasets/HPS'
        ))
        self.assets_dir = Path(assets_dir or os.getenv(
            'EGOALLO_ASSETS_DIR',
            '/home/minghao/src/robotflow/egoallo/assets'
        ))
        
        # Validate paths exist
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.base_dir}")
        if not self.assets_dir.exists():
            raise FileNotFoundError(f"Assets directory not found: {self.assets_dir}")
            
    def get_scan_path(self, scan_name: str) -> Path:
        """Get path to a pointcloud scan file."""
        path = self.base_dir / 'scans' / f"{scan_name}.zip"
        return path

    def get_smpl_path(self, sequence_name: str) -> Path:
        """Get path to SMPL sequence file."""
        path = self.base_dir / 'hps_smpl' / f"{sequence_name}.pkl"
        return path
        
    def get_betas_path(self, subject: str) -> Path:
        """Get path to subject betas file."""
        path = self.base_dir / 'hps_betas' / f"{subject}.json"
        return path
        
    def get_camera_path(self, sequence_name: str) -> Path:
        """Get path to camera localization file."""
        path = self.base_dir / 'head_camera_localizations' / f"{sequence_name}.json"
        return path
        
    def get_smpl_model_path(self, gender: str = "male") -> Path:
        """Get path to SMPL model assets."""
        path = self.assets_dir / "smpl_based_model"
        return path
        
    def load_camera_trajectory(self, sequence_name: str, fps: float) -> tuple[list[dict[str, float]], float, float]:
        """Load and parse camera trajectory file."""
        path = self.get_camera_path(sequence_name)
        path_json = json.load(open(path))
        path_json = {int(k): {**v, "time": float(k)/fps} for k, v in path_json.items() if v is not None}
        first_key, last_key = min(path_json.keys()), max(path_json.keys())
        # return the values, video_start_time, video_end_time
        return list(path_json.values()), path_json[first_key]["time"], path_json[last_key]["time"]

    def validate_paths(self, sequence_name: str, subject: str) -> bool:
        """Validate that all required files exist for a sequence."""
        paths = [
            self.get_smpl_path(sequence_name),
            self.get_betas_path(subject),
            self.get_camera_path(sequence_name)
        ]
        return all(p.exists() for p in paths)

# Initialize path manager
paths = HPSPathManipulator(base_dir="/home/minghao/src/robotflow/egoallo/datasets/HPS",
                           assets_dir="/home/minghao/src/robotflow/egoallo/assets")

# First, let's set the target resolution, framerate, video length and initialize OpenGL context.
# We will use EGL offscreen rendering for that, but you can change it to whatever context you prefer (e.g. OsMesa, X-Server)
resolution = (1280,720)
fps = 30.

logger.info("Initializing EGL and OpenGL")
context = EGLContext()
if not context.initialize(*resolution):
    print("Error during context initialization")
    sys.exit(0)

# Now, let's create and set up OpenGL frame and renderbuffers
_main_cb, _main_db = gl.glGenRenderbuffers(2)
viewport_width, viewport_height = resolution

gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, _main_cb)
gl.glRenderbufferStorage(
    gl.GL_RENDERBUFFER, gl.GL_RGBA,
    viewport_width, viewport_height
)

gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, _main_db)
gl.glRenderbufferStorage(
    gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24,
    viewport_width, viewport_height
)

_main_fb = gl.glGenFramebuffers(1)
gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _main_fb)
gl.glFramebufferRenderbuffer(
    gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
    gl.GL_RENDERBUFFER, _main_cb
)
gl.glFramebufferRenderbuffer(
    gl.GL_DRAW_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
    gl.GL_RENDERBUFFER, _main_db
)

gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _main_fb)
gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0])

# Let's configure OpenGL
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
gl.glClearColor(1.0, 1.0, 1.0, 0)
gl.glViewport(0, 0, *resolution)
gl.glEnable(gl.GL_DEPTH_TEST)
gl.glDepthMask(gl.GL_TRUE)
gl.glDepthFunc(gl.GL_LESS)
gl.glDepthRange(0.0, 1.0)

# Create and set a position of the camera
camera = PerspectiveCameraModel()
camera.init_intrinsics(resolution, fov=75, far=50)
camera.init_extrinsics(np.array([1,np.pi/5,0,0]), np.array([0,-1,2]))

# Create a scene
main_scene = Scene()

# Load pointcloud
logger.info("Loading pointcloud")
renderable_pc = SimplePointcloud(camera=camera)
renderable_pc.generate_shadows = False
renderable_pc.init_context()
pointcloud = trimesh_load_from_zip(str(paths.get_scan_path("MPI_KINO")), "*/pointcloud.ply")
renderable_pc.set_buffers(pointcloud)
main_scene.add_object(renderable_pc)

# Load human motion
logger.info("Loading SMPL animation")
renderable_smpl = AnimatableSMPLModel(
    camera=camera, 
    gender="male",
    smpl_root=str(paths.get_smpl_model_path())
)
renderable_smpl.draw_shadows = False
renderable_smpl.init_context()

sequence_name = "SUB3_MPI_KINO_lecture_multilevel"
subject = "SUB3"
motion_seq = load_hps_sequence(
    str(paths.get_smpl_path(sequence_name)),
    str(paths.get_betas_path(subject))
)
renderable_smpl.set_sequence(motion_seq, default_frame_time=1/fps)
renderable_smpl.set_material(0.3,1,0,0)
main_scene.add_object(renderable_smpl)

# Let's add a directional light with shadows for this scene
light = DirectionalLight(np.array([0., -1., -1.]), np.array([0.8, 0.8, 0.8]))

# We'll create a 4x4x10 meter shadowmap with 1024x1024 texture buffer and center it above the model along the direction
# of the light. We will move the shadomap with the model in the main loop
smpl_model_shadowmap_offset = -light.direction*3
smpl_model_shadowmap = main_scene.add_dirlight_with_shadow(light=light, shadowmap_texsize=(1024,1024),
                                    shadowmap_worldsize=(4.,4.,10.),
                                    shadowmap_center=motion_seq[0]['translation']+smpl_model_shadowmap_offset)

# Set camera trajectory and fill in spaces between keypoints with interpolation
logger.info("Creating camera trajectory")
camera_trajectory = Trajectory()
camera_trajectory_data, video_start_time, video_end_time = paths.load_camera_trajectory(sequence_name, fps)
camera_trajectory.set_trajectory(camera_trajectory_data)
camera_trajectory.refine_trajectory(time_step=1/fps)

### Main drawing loop ###
logger.info("Running the main drawing loop")
# Create a video writer to dump frames to and an async capturing controller
with VideoWriter("test_assets/output_1.mp4", resolution=resolution, fps=fps) as vw, \
        AsyncPBOCapture(resolution, queue_size=50) as capturing:
    for current_time in tqdm(np.arange(video_start_time, video_end_time, 1/fps)):
        # Update dynamic objects
        renderable_smpl.set_time(current_time)
        smpl_model_shadowmap.camera.init_extrinsics(
            pose=renderable_smpl.translation_params.cpu().numpy()-smpl_model_shadowmap_offset)
        # Move the camera along the trajectory
        camera_trajectory.apply(camera, current_time)
        # Clear OpenGL buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Draw the scene
        main_scene.draw()
        # Request color readout; optionally receive previous request
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, _main_fb)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        color = capturing.request_color_async()
        # If received the previous frame, write it to the video
        if color is not None:
            vw.write(color)
    # Flush the remaining frames
    logger.info("Flushing PBO queue")
    color = capturing.get_first_requested_color()
    while color is not None:
        vw.write(color)
        color = capturing.get_first_requested_color()
logger.info("Done")
