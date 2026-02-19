#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import torch.nn
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import math
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import pdb
from utils.mipmap import MipmappedTextureHighPerf, MipmappedTexture3DHighPerf
import open3d as o3d
from scene.NVDIFFREC import create_trainable_env_map_rnd, load_env_map, save_env_map
from utils.graphics_utils import rotation_matrix
import torch


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.mipmap = MipmappedTexture3DHighPerf().cuda()
        self.mlp = nn.Sequential(
            nn.Linear(16*5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        ).cuda()

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.roughness_activation = torch.sigmoid
        self.tint_activation = torch.sigmoid

    def __init__(self, sh_degree : int, brdf_dim : int, brdf_envmap_res: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._roughness = torch.empty(0)
        self._tint = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.k_dim = 4
        self.roughness_bias = 0
        self.default_roughness = 0.0  #TODO: GaussianShader use 0.6
        self.default_tint = 0.0
        self.default_diffuse = 0.0
        self.brdf_dim = brdf_dim
        self.brdf_envmap_res = brdf_envmap_res
        self.use_residual = False
        self.alpha_threshold = 0.0
        self.env_scope_radius = 0.0
        self.env_scope_center = torch.tensor([0.,0.,0.], dtype=torch.float, device="cuda")

        # if (brdf_dim > 0 and sh_degree > 0) or (brdf_dim < 0 and sh_degree < 0) or (brdf_dim == 0 and sh_degree == 0):
        #     raise Exception('Please provide exactly one of either brdf_dim or sh_degree!')

        self.brdf_mlp = create_trainable_env_map_rnd(brdf_envmap_res, scale=0.0, bias=0.8)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._roughness,
            self._tint,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.mipmap.state_dict(),
            self.mlp.state_dict(),
            self.brdf_mlp.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._roughness,
            self._tint,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            mipmap_dict,
            mlp_dict,
            brdf_mlp_dict,
            self.spatial_lr_scale
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.mipmap.load_state_dict(mipmap_dict)
        self.mlp.load_state_dict(mlp_dict)
        self.brdf_mlp.load_state_dict(brdf_mlp_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest[:, :self.k_dim]
        roughness = self.get_roughness
        tint = self.get_tint
        return torch.cat((features_dc, features_rest, roughness, tint), dim=1)

    @property
    def get_k_features(self):
        return self._features_rest[:, :self.k_dim]

    @property
    def get_brdf_features(self):
        return self._features_rest[:, self.k_dim:].reshape(-1, 3, (self.brdf_dim + 1) ** 2)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_diffuse(self):
        return self._features_dc

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness + self.roughness_bias)

    @property
    def get_tint(self):
        return self.tint_activation(self._tint)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_gaussian_minimum_axis(self):
        sorted_idx = torch.argsort(self.get_scaling, descending=False, dim=-1)
        R = build_rotation(self.get_rotation)
        R_sorted = torch.gather(R, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
        x_axis = R_sorted[:,0,:] # normalized by defaut
        return x_axis

    @property
    def get_surfel_minimum_axis(self):
        scales = self.get_scaling
        rotations = self.get_rotation
        R = build_rotation(rotations)
        # Find index of minimum scale for each surfel
        sorted_idx = torch.argsort(scales, descending=False, dim=-1)  # Sort scales per surfel
        R_xy = R[:, :, :2]  # Take the first two columns (corresponding to 2D surfel axes)
        R_sorted = torch.gather(R_xy, dim=2, index=sorted_idx[:, None, :].expand(-1, 3, -1))
        min_axis = R_sorted[:, :, 0]  # First column corresponds to the shortest axis
        return min_axis

    @property
    def get_mask(self):
        if self.env_scope_radius > 0:
            mask = torch.sum((self.get_xyz - self.env_scope_center[None])**2, dim=-1) < (self.env_scope_radius ** 2)
        else:
            mask = torch.ones(len(self.get_xyz), device=self.get_xyz.device).bool()
        return mask[..., None]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def activate_residual(self):
        self.use_residual = True

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float,
                        default_roughness : float = 0.0, default_tint : float = 0.0, default_diffuse : float = 0.0):
        self.spatial_lr_scale = spatial_lr_scale
        self.default_roughness = default_roughness
        self.default_tint = default_tint
        self.default_diffuse = default_diffuse
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        brdf_feat_len = 3 * (self.brdf_dim + 1) ** 2 if self.brdf_dim > 0 else 0
        features = torch.zeros((fused_color.shape[0], 3 + self.k_dim + brdf_feat_len)).float().cuda()  # [N, 3*(D+1)^2+4] = [N, 3+L+K]
        features[:, :3] = fused_color  # diffuse color [N, 3]
        features[:, 3:] = 0.0  # features k and sh [N, K + L]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        roughnesses = torch.full((fused_point_cloud.shape[0], 1), self.default_roughness, dtype=torch.float, device="cuda")
        tint = torch.full((fused_point_cloud.shape[0], 3), self.default_tint, dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:3].contiguous().requires_grad_(True) - self.default_diffuse)
        self._features_rest = nn.Parameter(features[:,3:].contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._roughness = nn.Parameter(roughnesses.requires_grad_(True))
        self._tint = nn.Parameter(tint.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},  # diffuse color
            {'params': [self._features_rest], 'lr': training_args.feature_lr, "name": "f_rest"}, # feature vector
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"}, # roughness
            {'params': [self._tint], 'lr': training_args.tint_lr, "name": "tint"},  # tint from GS
            {'params': self.mipmap.parameters(), 'lr': training_args.feature_lr, "name": "mipmap"},
            {'params': self.mlp.parameters(), 'lr': 1e-4, "name": "mlp"},
            {'params': self.brdf_mlp.parameters(), 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.brdf_mlp_lr_init,
                                        lr_final=training_args.brdf_mlp_lr_final,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult,
                                        max_steps=training_args.brdf_mlp_lr_max_steps)

    def set_residual_mode(self, training_args):
        self.activate_residual()
        l = [
            {'params': [self._features_rest], 'lr': training_args.feature_lr, "name": "f_rest"}, # feature vector
            {'params': self.mipmap.parameters(), 'lr': training_args.feature_lr, "name": "mipmap"},
            {'params': self.mlp.parameters(), 'lr': 1e-4, "name": "mlp"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if not self.fix_brdf_lr and param_group["name"] == "brdf_mlp":
                lr = self.brdf_mlp_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self, viewer_fmt=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('roughness')
        for i in range(self._tint.shape[1]):
            l.append('tint{}'.format(i))
        l.append('radii2D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        tint = self._tint.detach().cpu().numpy()
        radii = self.max_radii2D.detach().cpu().numpy()[..., np.newaxis]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scale, rotation, roughness, tint, radii), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz)
        pcd_o3d.colors = o3d.utility.Vector3dVector(f_dc)

        return pcd_o3d

    def reset_opacity(self):
        opacity_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacity_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        raddii = np.asarray(plydata.elements[0]["radii2D"])
        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        brdf_feat_len = 3 * (self.brdf_dim + 1) ** 2 if self.brdf_dim > 0 else 0
        features_extra = np.zeros((xyz.shape[0], self.k_dim + brdf_feat_len))
        if len(extra_f_names) == self.k_dim + brdf_feat_len:
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            print(f"NO INITIAL SH FEATURES FOUND!!! USE ZERO SH AS INITIALIZE.")
            features_extra = features_extra.reshape((features_extra.shape[0], self.k_dim + brdf_feat_len))

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

        tint_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("tint")]
        tint = np.zeros((xyz.shape[0], len(tint_names)))
        for idx, attr_name in enumerate(tint_names):
            tint[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacity, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._tint = nn.Parameter(torch.tensor(tint, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.tensor(raddii, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def save_mlp_checkpoints(self, path):
        torch.save({'mlp': self.mlp.state_dict(), 'mipmap': self.mipmap.state_dict()}, os.path.join(path, "mlp.pth"))
        save_env_map(os.path.join(path, "brdf_mlp.hdr"), self.brdf_mlp)  # save in HDR and PNG formats
        save_env_map(os.path.join(path, "brdf_mlp2.hdr"), self.brdf_mlp, rotated=True)  # save in HDR and PNG formats with rotated configuration
        torch.save(self.brdf_mlp.state_dict(), os.path.join(path, "brdf_mlp.pth"))  # save state in pth format

    def load_mlp_checkpoints(self, path, hdr_mode=False):
        chkpt = torch.load(os.path.join(path, "mlp.pth"))
        self.mlp.load_state_dict(chkpt['mlp'])
        self.mipmap.load_state_dict(chkpt['mipmap'])
        if hdr_mode:
            self.load_env_map(path)
        else:
            self.brdf_mlp.load_state_dict(torch.load(os.path.join(path, "brdf_mlp.pth"))); self.brdf_mlp.build_mips()

    def load_env_map(self, path, tonemap=lambda x: x):
        if not any([path.lower().endswith(x) for x in [".hdr", ".hdri", ".exr", ".tiff", ".pfm", ".jpg", ".png"]]):
            path = os.path.join(path, "brdf_mlp.hdr")
        self.brdf_mlp = load_env_map(path, scale=1.0, res=[self.brdf_envmap_res]*2, tonemap=tonemap)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group["name"] or 'mip' in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._roughness = optimizable_tensors["roughness"]
        self._tint = optimizable_tensors["tint"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group["name"] or 'mip' in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_roughness, new_tint):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "roughness" : new_roughness,
            "tint" : new_tint,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._roughness = optimizable_tensors["roughness"]
        self._tint = optimizable_tensors["tint"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_tint = self._tint[selected_pts_mask].repeat(N,1)

        self.densification_postfix(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacity=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
            new_roughness=new_roughness,
            new_tint=new_tint,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if torch.sum(selected_pts_mask) == 0:
            return

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tint = self._tint[selected_pts_mask]

        self.densification_postfix(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacity=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
            new_roughness=new_roughness,
            new_tint=new_tint,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

