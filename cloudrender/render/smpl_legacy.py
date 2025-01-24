import numpy as np
import torch
import smplx

from egoallo import fncsmpl

from .mesh import SimpleMesh
from .renderable import DynamicTimedRenderable
from .utils import MeshNorms
from egoallo.transforms import SE3, SO3
from typing import List
from OpenGL import GL as gl
from .lights import Light
from .shadowmap import ShadowMap

class SMPLModel(SimpleMesh):
    """
    SMPL model with vertex normals.
    NOTE: `_set_buffers` and `_update_buffers` are overridden to use `MeshContainer` with vertex normals.
    """
    def __init__(self, device=None, smpl_root=None, template=None, gender="neutral", model_type="smpl", global_offset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = None
        self.smpl_root = smpl_root
        self.device = torch.device(device if device is not None else "cpu")
        self.pose_params = torch.zeros(72, device=self.device)
        self.translation_params = torch.zeros(3, device=self.device)
        self.template = template
        self.global_offset = global_offset
        self.model_type = model_type
        self.smpl_compatible = False
        if self.smpl_root is None:
            self.smpl_root = "./models"
        if "compat" in self.model_type:
            self.model_type = self.model_type.split("_")[0]
            self.smpl_compatible = True
        self._set_smpl(gender)
        self.nglverts = len(self.get_smpl()[0])
        self.set_uniform_color()

    def _set_smpl(self, gender='neutral', shape_params=None):
        self.model_layer = smplx.create(self.smpl_root, model_type=self.model_type, gender=gender).to(self.device)
        self.model_layer.requires_grad_(False)
        if self.smpl_compatible:
            smpl_model = smplx.create(self.smpl_root, model_type="smpl", gender=gender)
            self.model_layer.shapedirs[:] = smpl_model.shapedirs.detach().to(self.device)
        if self.template is not None:
            self.model_layer.v_template[:] = torch.tensor(self.template, dtype=self.model_layer.v_template.dtype,
                                                          device=self.device)
        if self.global_offset is not None:
            self.model_layer.v_template[:] += torch.tensor(self.global_offset[np.newaxis, :], dtype=self.model_layer.v_template.dtype,
                                                           device=self.device)
        self.normals_layer = MeshNorms(self.model_layer.faces_tensor) #torch.tensor(self.model_layer.faces.astype(int), dtype=torch.long, device=self.device))
        self.gender = gender
        self.shape_params = torch.zeros(10, device=self.device) if shape_params is None else \
            torch.tensor(shape_params, dtype=torch.float32, device=self.device)

    def _preprocess_param(self, param):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float32)
        param = param.to(self.device)
        return param

    def _finalize_init(self):
        super()._finalize_init()
        self.faces_numpy = self.model_layer.faces.astype(int)
        self.faces = self.model_layer.faces_tensor #torch.tensor(self.model_layer.faces.astype(int), dtype=torch.long, device=self.device)
        self.flat_faces = self.faces.view(-1)

    def update_params(self, pose=None, shape=None, translation=None):
        if pose is not None:
            self.pose_params = self._preprocess_param(pose)
        if shape is not None:
            self.shape_params = self._preprocess_param(shape)
        if translation is not None:
            self.translation_params = self._preprocess_param(translation)

    def set_uniform_color(self, color=(200, 200, 200, 255)):
        self.color = color
        self.vertex_colors = np.tile(np.array(color, dtype=np.uint8).reshape(1, 4), (self.nglverts, 1))

    def get_smpl(self, pose_params=None, shape_params=None, translation_params=None):
        """
        NOTE: update internal attrs like pose-params, shape_params, translation_params, and run SMPL forward.
        """
        self.update_params(pose_params, shape_params, translation_params)
        batch_pose_params = self.pose_params.unsqueeze(0) # [1, 72]
        batch_shape_params = self.shape_params.unsqueeze(0) # [1, 10]
        if self.model_type == "smplh":
            batch_pose_params = batch_pose_params[:,:-6] # [1, 66]
        output = self.model_layer(global_orient =batch_pose_params[:, :3],
                                  body_pose=batch_pose_params[:,3:], betas=batch_shape_params)
        verts = output.vertices # [1, 6890, 3]
        # NOTE: normals are inherited from face normals, and arrange them in the order of vertex count. so the `len(normals)` is the same as `len(faces)`.
        normals = self.normals_layer.vertices_norms(verts.squeeze(0))
        return verts.squeeze(0) + self.translation_params.unsqueeze(0), normals
    
    def get_smpl_fncsmpl(self, pose_params=None, shape_params=None, translation_params=None, body_model: fncsmpl.SmplhModel = None):
        assert body_model is not None, "body_model must be provided"
        # TODO: the `fncsmpl.SmplhModel` is not compatible with the smplx.create impl, they maintain a constant offset between each other.
        # TODO: investigate this issues if you have time.
        
        self.update_params(pose_params, shape_params, translation_params)
        batch_pose_params = self.pose_params.unsqueeze(0) # [1, 66]
        batch_shape_params = self.shape_params.unsqueeze(0) # [1, 10]
        batch_translation_params = self.translation_params.unsqueeze(0) # [1, 3]
        if self.model_type == "smplh":
            batch_pose_params = batch_pose_params[:,:-6] # [1, 66]
        T_world_root = SE3.from_rotation_and_translation(
            SO3.exp(batch_pose_params[:, :3]), # [1, 4]
            batch_translation_params[:,:3], # [1, 3]
        ).parameters() # [1, 7]
        shaped: fncsmpl.SmplhShaped = body_model.with_shape(batch_shape_params)
        posed: fncsmpl.SmplhShapedAndPosed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=SO3.exp(batch_pose_params[:, 3:66].reshape(*batch_pose_params.shape[:-1], 21, 3)).wxyz,
            left_hand_quats=None,
            right_hand_quats=None,
        )
        mesh: fncsmpl.SmplMesh = posed.lbs()
        verts = mesh.verts # [1, 6890, 3]
        # NOTE: normals are inherited from face normals, and arrange them in the order of vertex count. so the `len(normals)` is the same as `len(faces)`.
        normals = self.normals_layer.vertices_norms(verts.squeeze(0)) # [1, 6890, 3]
        return verts.squeeze(0), normals

    def get_smpl_mesh(self, pose_params=None, shape_params=None, translation_params=None, **kwargs):
        # verts, normals = self.get_smpl(pose_params, shape_params, translation_params)
        verts, normals = self.get_smpl_fncsmpl(pose_params, shape_params, translation_params, **kwargs)
        mesh = self.MeshContainer(verts.cpu().numpy(), self.faces_numpy, self.vertex_colors, normals.cpu().numpy())
        return mesh

    def _set_buffers(self, pose_params=None, shape_params=None, translation_params=None, **kwargs):
        mesh = self.get_smpl_mesh(pose_params, shape_params, translation_params, **kwargs)
        super()._set_buffers(mesh)

    def _update_buffers(self, pose_params=None, shape_params=None, translation_params=None, **kwargs):
        mesh = self.get_smpl_mesh(pose_params, shape_params, translation_params, **kwargs)
        super()._update_buffers(mesh)

    # disable depth testing and writing to depth buffer
    # Depth writing refers to the process of writing the depth (z-value) of each fragment (pixel) to the depth buffer (also called the z-buffer). The depth buffer is a special buffer in OpenGL that stores the depth of each pixel in the scene. It is used to determine whether a fragment should be drawn or discarded based on its depth relative to other fragments.
    # Depth Buffer: A 2D array (same size as the framebuffer) that stores the depth value of the closest fragment for each pixel.
    # Depth Test: During rendering, OpenGL compares the depth of the current fragment with the value stored in the depth buffer. If the fragment's depth is closer (less than the stored value), it is drawn, and the depth buffer is updated. Otherwise, the fragment is discarded.
    def _draw(self, reset: bool, lights: List[Light], shadowmaps: List[ShadowMap]) -> bool:
        gl.glDepthMask(gl.GL_FALSE)
        res = super()._draw(reset, lights, shadowmaps)
        gl.glDepthMask(gl.GL_TRUE)
        return res

class AnimatableSMPLModel(SMPLModel, DynamicTimedRenderable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _finalize_init(self):
        super()._finalize_init()
        self.set_overlay_color(np.array([200, 200, 200, 128]))

    def _set_sequence(self, params_seq):
        self.params_sequence = params_seq
        self.sequence_len = len(params_seq)

    def _load_current_frame(self, **kwargs):
        params = self.params_sequence[self.current_sequence_frame_ind]
        pose = params['pose'] if 'pose' in params else None
        shape = params['shape'] if 'shape' in params else None
        translation = params['translation'] if 'translation' in params else None
        self.update_buffers(pose, shape, translation, **kwargs)
