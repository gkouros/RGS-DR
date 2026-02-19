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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.NVDIFFREC import extract_env_map
from utils.point_utils import depth_to_normal


def get_pixel_world_coordinates(viewpoint_camera):
    # get camera image dimensions
    W, H = viewpoint_camera.image_width, viewpoint_camera.image_height

    # Compute pixel coordinates in NDC space
    x = torch.linspace(0, W-1, W, device="cuda")
    y = torch.linspace(0, H-1, H, device="cuda")
    i, j = torch.meshgrid(x, y, indexing='xy')

    # Convert to NDC coordinates
    ndc_x = (i + 0.5) / W * 2 - 1
    ndc_y = (j + 0.5) / H * 2 - 1

    # Create homogeneous coordinates
    ndc_coords = torch.stack([ndc_x, ndc_y, torch.ones_like(ndc_x), torch.ones_like(ndc_x)], dim=-1)

    # Convert NDC coordinates to world coordinates
    world_coords = ((viewpoint_camera.projection_matrix.inverse().T @ ndc_coords.reshape(-1, 4).T)).T
    world_coords = world_coords / world_coords[:, 3:]
    world_coords = (viewpoint_camera.world_view_transform.inverse().T @ world_coords.T).T
    world_coords = world_coords[:, :3] / world_coords[:, 3:]
    return world_coords


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # get image size
    W, H = viewpoint_camera.image_width, viewpoint_camera.image_height

    # setup resterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,  # pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # get gaussian attributes
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    k_features = pc.get_k_features

    # color residual based on gaussian shader
    if pc.use_residual and pc.brdf_dim > 0:
        # get brdf features corresponding to SH coefficients
        shs_view = pc.get_brdf_features.view(-1, 3, (pc.brdf_dim+1)**2)
        # eval SH coefficients to get residual color per gaussian for the given viewing direction
        view_dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
        view_dir_pp = view_dir_pp / view_dir_pp.norm(dim=1, keepdim=True) # (N, 3)
        color_delta = eval_sh(pc.brdf_dim, shs_view, view_dir_pp)
        k_features = torch.cat((color_delta, torch.zeros_like(opacity)), dim=1)

    # collect features for rasterization
    features = torch.concatenate((pc.get_diffuse, k_features, pc.get_roughness, pc.get_tint, pc.get_mask), axis=1)  # (N, 12))

    # render the material properties and features
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=features,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # get rendered maps bg=[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    diffuse_map = rendered_image[0:3]  # diffuse map I_d
    render_K_map = rendered_image[3:7]  # feature map K
    M_map = rendered_image[7:8]  # roughness map M
    tint_map = rendered_image[8:11]  # specular tint map T
    mask_map = rendered_image[11:12].detach()  # env scope mask

    # get alpha for masking
    render_alpha = allmap[1:2]
    # get normal map and transform from view space to world space
    render_normal = allmap[2:5].permute(1,2,0)
    # render_normal = render_normal / torch.norm(render_normal, dim=-1, keepdim=True)
    render_normal = (render_normal @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)  # (3, H, W)

    # get world coordinates of pixels
    world_coords = get_pixel_world_coordinates(viewpoint_camera)

    # mask out the low alpha pixels
    alpha_threshold = pc.alpha_threshold
    try:
        mask = render_alpha.squeeze().detach() > alpha_threshold
        while not mask.any():
            alpha_threshold /= 2.0
            mask = render_alpha.squeeze().detach() > alpha_threshold
    except:
        mask = torch.ones_like(render_alpha.squeeze().detach()).bool()

    if pc.use_residual and not pc.brdf_dim > 0:
        # Compute viewing direction
        viewing_direction = world_coords.reshape(H, W, 3) - viewpoint_camera.camera_center
        viewing_direction = viewing_direction / torch.norm(viewing_direction, dim=-1, keepdim=True)

        # Compute reflected direction using normal map and viewing direction
        render_normal_flat = render_normal.permute(1,2,0)
        render_normal_flat = render_normal_flat.reshape(-1, 3)  # (H*W, 3)
        viewing_direction_flat = viewing_direction.reshape(-1, 3) # (H*W, 3)
        dot_product = torch.sum(viewing_direction_flat * render_normal_flat, dim=-1, keepdim=True)  # NdotV -> (H*W, 1)
        reflected_direction = viewing_direction_flat - 2 * dot_product * render_normal_flat  # R = V - 2*(V·N)*N -> (H*W, 3)
        reflected_direction = reflected_direction.reshape(H, W, 3).permute(2,0,1)  # Reshape back to image dims (3, H, W)

        # Convert reflected direction to spherical coordinates
        rho = torch.norm(reflected_direction, dim=0, keepdim=True)  # r = sqrt(x^2 + y^2 + z^2)
        phi = torch.acos(reflected_direction[2:3] / rho)  # theta = arccos(z/r) [polar angle]
        theta = torch.atan2(reflected_direction[1:2], reflected_direction[0:1])  # phi = arctan2(y, x) [azimuthal angle]

        # Convert theta and phi to UV coordinates (normalized to [0,1])
        u = phi / math.pi  # Map phi [0,pi] to u [0,1]
        v = (theta + math.pi) / (2 * math.pi)  # Map theta [-pi,pi] to v [0,1]
        uv = torch.cat([u, v], dim=0)  # (2, H, W)
        uv = uv.permute(1, 2, 0)  # Permute to (H, W, 2) for grid_sample

        # Reshape inputs to (N, 2) and (N, 1)
        uv_flat = uv.reshape(-1, 2)  # (H*W, 2)
        M_map_flat = M_map.reshape(-1)  # (H*W, 1)
        K_map = render_K_map.permute(1, 2, 0).reshape(H*W, -1)

        # Sample from mipmap using UV coordinates and M_map
        S_map = pc.mipmap(uv_flat, M_map_flat)

        # Compute outer product between K_map and S_map
        # K_map: (N, 4), S_map: (N, 16) -> outer product: (N, 4*16)
        outer_product = torch.bmm(K_map.unsqueeze(2), S_map.unsqueeze(1))  # (N, 4, 16)
        outer_product = outer_product.reshape(-1, 4*16)  # Flatten last two dimensions

        # Concatenate S_map and outer_product along the feature dimension
        combined_features = torch.cat([S_map, outer_product], dim=-1)  # (N, 16 + 4*16)
        combined_features = combined_features.reshape(H, W, -1)  # (H, W, 16 + 4*16)

        # Pass combined_features through MLP to get specular residual color
        specular_residual = pc.mlp(combined_features)  # (N, 3)
        # Reshape to image dimensions (H, W, 3) and permute to (3, H, W)
        specular_residual = specular_residual.reshape(H, W, 3)
        # permute to (3, H, W)
        specular_residual = specular_residual.permute(2, 0, 1)  # (3, H, W)
    elif pc.use_residual and pc.brdf_dim > 0:
        specular_residual = render_K_map[:3]  # (3, H, W)
    else:
        specular_residual = torch.zeros_like(diffuse_map)  # (3, H, W)

    # Switch shading functions
    if True:
        # Simplified shading model of Gaussian Shader
        _, brdf_pkg = pc.brdf_mlp.shade(
            world_coords.reshape(H,W,3)[mask][None, :, None],
            render_normal.permute(1,2,0)[mask][None, :, None],
            diffuse_map.permute(1,2,0)[mask][None, :, None],
            tint_map.permute(1,2,0)[mask][None, :, None],
            M_map.permute(1,2,0)[mask][None, :, None],
            viewpoint_camera.camera_center[None, None, None],
        )
        # masked shading
        diffuse_color = diffuse_map  # to backpropagate gradients to the background
        if False:
            diffuse_color[:, mask] = brdf_pkg['diffuse'].squeeze().permute(1, 0)
        specular_color = torch.zeros_like(diffuse_color)
        specular_color[:, mask] = brdf_pkg['specular'].squeeze().permute(1, 0)
    else:
        # Metallic-based shading model
        _, brdf_pkg = pc.brdf_mlp.shade2(
            world_coords.reshape(H,W,3)[None],
            render_normal.permute(1,2,0)[None],
            diffuse_map.permute(1,2,0)[None],
            tint_map.permute(1,2,0)[None],  # treat tint as metallic
            M_map.permute(1,2,0)[None],
            viewpoint_camera.camera_center[None, None, None],
        )  # ouputs in (1, H, W, 3)

        # masked shading
        if False:
            diffuse_color = diffuse_map
        else:
            diffuse_color = brdf_pkg['diffuse'].squeeze().permute(2, 0, 1)  # (3, H, W)
            diffuse_color = diffuse_color * render_alpha.detach() + (1 - render_alpha.detach())
        specular_color = brdf_pkg['specular'].squeeze().permute(2, 0, 1)  # (3, H, W)

        # filter out specularity of low opacity pixels
        # diffuse_color[:, ~mask] = 1
        specular_color[:, ~mask] = 0

    # compute final rendered color
    rendered_rgb = diffuse_color + specular_color

    if pc.env_scope_radius > 0.0 and pc.use_residual and pc.brdf_dim > 0:
        rendered_rgb = rendered_rgb * mask_map + specular_residual * (1 - mask_map)
    elif pc.use_residual:  # apply residual color
        # specular_residual[:, ~mask] = 0  # filter out points outside mask
        rendered_rgb = rendered_rgb + specular_residual

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * render_alpha.detach()  # remember to multiply with accum_alpha since render_normal is unnormalized.

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_rgb,
        "rend_diffuse": diffuse_color,
        "rend_specular_color": specular_color,
        "rend_specular_residual": specular_residual,
        "rend_tint": tint_map,
        "rend_roughness": M_map,
        "rend_k": render_K_map[:3],
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "rend_alpha": render_alpha,
        "rend_dist": render_dist,
        "surf_depth": surf_depth,
        "rend_normal": render_normal,
        "surf_normal": surf_normal,
        "mask_map": mask_map,
    }

    return rets