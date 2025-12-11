import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
# from engine.utils.gs_data_checker_utils import sigmoid
# from engine.data_structures import GaussianSplattingData
import utils3d


class Gaussian:
    def __init__(
            self, 
            aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).cuda()
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).cuda()

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])
    
    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias
        
    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]
    
    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        
    def from_features(self, features):
        self._features_dc = features
        
    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
        
    ############## OUR ###############    
    # def save_ply(self, path, transform=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]):
    #     xyz = self.get_xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()        
    #     opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        
    #     scale = torch.log(self.get_scaling).detach().cpu().numpy()
    #     rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        
    #     # R_x = np.array([
    #     #     [1, 0, 0],
    #     #     [0, 0, -1],
    #     #     [0, 1, 0]
    #     # ], dtype=np.float32)

    #     # R_y = np.array([
    #     #     [0, 0, 1],
    #     #     [0, 1, 0],
    #     #     [-1, 0, 0]
    #     # ], dtype=np.float32)

    #     # R_y = np.array([
    #     #     [-1, 0, 0],
    #     #     [ 0, 1, 0],
    #     #     [ 0, 0, -1]
    #     # ], dtype=np.float32)

    #     # R_z = np.array([[0, -1, 0],
    #     #                 [1,  0, 0],
    #     #                 [0,  0, 1]], dtype=np.float32)
    #     # # R_x = np.array([[1, 0, 0],
    #     # #                 [0, 0,-1],
    #     # #                 [0, 1, 0]], dtype=np.float32)

    #     R_x = np.array([[1, 0, 0],
    #                [0, 0, 1],
    #                [0,-1, 0]], dtype=np.float32)

    #     # c = np.sqrt(2)/2
    #     # Rx = np.array([
    #     #     [ c, -c, 0],
    #     #     [ c,  c, 0],
    #     #     [ 0,  0, 1]
    #     # ], dtype=np.float32) # z

    #     # c = 0.5
    #     # s = 0.866025
    #     # Rz = np.array([
    #     #     [ c,  s, 0],
    #     #     [-s,  c, 0],
    #     #     [ 0,  0, 1]
    #     # ], dtype=np.float32)

    #     transform = R_x # apply X first, then Y
    #     if transform is not None:
    #         transform = np.array(transform)
    #         xyz = np.matmul(xyz, transform.T)
    #         rotation = utils3d.numpy.quaternion_to_matrix(rotation)
    #         rotation = np.matmul(transform, rotation)
    #         rotation = utils3d.numpy.matrix_to_quaternion(rotation)

    #     # xyz[:, 0] += 0.5 
    
    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)    
       
    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()

        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            rotation = np.matmul(transform, rotation)
            rotation = utils3d.numpy.matrix_to_quaternion(rotation)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_ply_cuda(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Collect tensors directly on GPU ---
        xyz = self.get_xyz.detach().to(device)                       # (N, 3)
        normals = torch.zeros_like(xyz, device=device)               # (N, 3)
        f_dc = (self._features_dc.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .to(device))                                         # (N, F)
        opacities = inverse_sigmoid(self.get_opacity).detach().to(device)  # (N, 1)
        scale = torch.log(self.get_scaling).detach().to(device)            # (N, 3)
        rotation = (self._rotation + self.rots_bias[None, :]).detach().to(device)  # (N, 4)

        # --- Apply transform on GPU ---
        R_x = torch.tensor([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]], dtype=torch.float32, device=device)

        if transform is not None:
            transform = R_x  # Use R_x directly instead of creating a copy
            xyz = xyz @ transform.T

            # convert quaternions <-> matrices all on GPU
            rot_mtx = utils3d.torch.quaternion_to_matrix(rotation)   # (N, 3, 3)
            rot_mtx = transform @ rot_mtx
            rotation = utils3d.torch.matrix_to_quaternion(rot_mtx)   # (N, 4)

        # --- Concatenate all attributes on GPU ---
        data = torch.cat([
            xyz,
            normals,
            f_dc,
            opacities,
            scale,
            rotation
        ], dim=1)

        # --- Transfer ONCE to CPU for saving ---
        data = data.cpu().numpy()

        # --- Structured dtype for PLY ---
        dtype_full = [(attr, 'f4') for attr in self.construct_list_of_attributes()]
        elements = np.zeros(data.shape[0], dtype=dtype_full)

        for i, name in enumerate(elements.dtype.names):
            elements[name] = data[:, i]

        # --- Write binary PLY (fast) ---
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], text=False).write(path)
    
    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform)
            rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            rotation = np.matmul(rotation, transform)
            rotation = utils3d.numpy.matrix_to_quaternion(rotation)
            
        # convert to actual gaussian attributes
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        if self.sh_degree > 0:
            features_extra = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=self.device))
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)
        
        # convert to _hidden attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = self.inverse_scaling_activation(torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)) - self.scale_bias
        self._rotation = rots - self.rots_bias[None, :]
        