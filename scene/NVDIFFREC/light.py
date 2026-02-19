# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru
from utils import tonemap

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    )
                                    # indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.mtx = None
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)  # (6, 64, 64, 3)
        self.register_parameter('base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)

    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]  # [(6,64,64,3)]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]  # [[6,64,64,3], [6,32,32,3], [6,16,16,3]]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])  # (6,16,16,3)

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def shade(self, gb_pos, gb_normal, kd, ks, kr, view_pos):
        # (H, W, N, C)
        wo = util.safe_normalize(view_pos - gb_pos)

        diffuse_raw = kd
        roughness = kr
        spec_col  = ks
        diff_col  = 1.0 - ks

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(1, reflvec.shape[0] * reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(1, nrmvec.shape[0] * nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        ambient = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        # specular_linear = ambient * diff_col

        # Lookup FG term from lookup texture
        NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
        fg_uv = torch.cat((NdotV, roughness), dim=-1)
        if not hasattr(self, '_FG_LUT'):
            self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
        fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

        # Roughness adjusted specular env lookup
        miplevel = self.get_mip(roughness)
        spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

        # Compute aggregate lighting
        reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
        # specular_linear += spec * reflectance
        specular_linear = ambient * diff_col + spec * reflectance
        extras = {"specular": specular_linear}

        # compute diffuse component
        diffuse_linear = torch.sigmoid(diffuse_raw - np.log(3.0))
        extras["diffuse"] = diffuse_linear

        rgb = specular_linear + diffuse_linear

        return rgb, extras

    def shade2(self, gb_pos, gb_normal, kd, ks, kr, view_pos):
        wo = util.safe_normalize(view_pos - gb_pos)

        roughness = kr
        metallic  = ks.mean(dim=-1, keepdim=True)
        spec_col  = (1.0 - metallic) * 0.04 + kd * metallic
        diff_col  = kd * (1.0 - metallic)

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        diffuse_col = diffuse * diff_col

        # Lookup FG term from lookup texture
        NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
        fg_uv = torch.cat((NdotV, roughness), dim=-1)
        if not hasattr(self, '_FG_LUT'):
            self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
        fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

        # Roughness adjusted specular env lookup
        miplevel = self.get_mip(roughness)
        spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

        # Compute aggregate lighting
        reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
        specular_col = spec * reflectance

        # combine diffuse and specular colors
        shaded_col = diffuse_col + specular_col

        return shaded_col, {"diffuse": diffuse_col, "specular": specular_col}

######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_map_hdr(fn, scale=1.0, res=[64, 64], tonemap=lambda x: x):
    # read envmap from HDR file
    latlong_img = util.load_image(fn)

    # apply tonemapping
    latlong_img = tonemap(latlong_img)
    # torch.clamp_(latlong_img, 0.0, 1.0)
    # latlong_img = tonemap.LinearTonemap()(latlong_img)
    # latlong_img = tonemap.FilmicTonemap()(latlong_img)
    # latlong_img = tonemap.gamma_tonemap(latlong_img)

    # convert to tensor
    latlong_img = scale * torch.tensor(latlong_img, dtype=torch.float32, device='cuda').contiguous()

    # convert to cubemap-based light representation
    # cubemap = util.latlong_to_cubemap(latlong_img, res)
    cubemap = util.latlong_to_cubemap_orig(latlong_img, res)
    l = EnvironmentLight(cubemap)

    # generate cube mip maps
    l.build_mips()
    return l

def load_env_map(fn, scale=1.0, res=[64, 64], tonemap=lambda x: x):
    if any([fn.lower().endswith(x) for x in [".hdr", ".hdri", ".exr", ".tiff", ".pfm"]]):
        return _load_env_map_hdr(fn, scale, res, tonemap)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light, rotated=False):
    color = extract_env_map(light, [512, 1024], rotated)
    util.save_image_raw(fn, color.detach().cpu().numpy())
    util.save_image(fn.replace('hdr', 'png'), color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_map_rnd(base_res, scale=0.5, bias=0.25):
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base)

def extract_env_map(light, resolution=[512, 1024], rotated=False):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if rotated:
        color = util.cubemap_to_latlong_orig(light.base, resolution)
    else:
        color = util.cubemap_to_latlong(light.base, resolution)
    return color
