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
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from copy import deepcopy
import open3d as o3d
from functools import partial
from scene.NVDIFFREC import util
from scene.NVDIFFREC.light import load_env_map
from utils.tonemap import estimate_tonemap, apply_tonemapping, save_tonemap_params, load_tonemap_params, gamma_tonemap

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--skip_misc", action="store_true")
    parser.add_argument("--disable_residual", action="store_true")
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--num_render", default=-1, type=int,  help="How many samples to render")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--gt_envmap_path", default="", help="The original envmap that was used to generate the scene")
    parser.add_argument("--relight_envmap_path", default="", help="The envmap to use to relight the scene")
    parser.add_argument("--relight_gt_path", default="", help="The relighted dataset to compare against")
    parser.add_argument("--rescale_relighted", action="store_true")
    parser.add_argument("--diffuse_mult", '--list', default='1.0,1.0,1.0', type=str, help='Multiplier for diffuse attributes of gaussians for scene editing')
    parser.add_argument("--roughness_mult", default=1.0, type=float, help='Multiplier for diffuse attributes of gaussians for scene editing')
    parser.add_argument("--tint_mult", default=1.0, type=float, help='Multiplier for diffuse attributes of gaussians for scene editing')
    parser.add_argument("--intensity_low_thresh", default=0.0, type=float, help='Multiplier for diffuse attributes of gaussians for scene editing')
    parser.add_argument("--intensity_high_thresh", default=1.0, type=float, help='Multiplier for diffuse attributes of gaussians for scene editing')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, dataset.brdf_envmap_res)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    if dataset.use_residual and not args.disable_residual:
        gaussians.activate_residual()
    bg_color = [1,1,1,0,0,0,0,0,0,0,0] if dataset.white_background else [0,0,0,0,0,0,0,0,0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color, num_render=args.num_render)

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir, skip_misc=args.skip_misc)

    if (not args.skip_test) or (len(scene.getTestCameras()) == 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir, skip_misc=args.skip_misc)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))

    if args.render_path and not args.relight_envmap_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_frames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)
        gaussExtractor.reconstruction(cam_traj, traj=True)
        gaussExtractor.export_image(traj_dir, skip_misc=False, traj=True)

    if args.relight_envmap_path:
        # deactivate residual since it's specifc to the original ground truth map
        gaussians.use_residual = False
        relight_envmap_name = "".join(args.relight_envmap_path.split("/")[-1])
        print(f"Relighting using {relight_envmap_name}")
        dirname = "relight" if not args.render_path else "traj_relight"
        relight_dir = os.path.join(args.model_path, dirname, relight_envmap_name, "ours_{}".format(scene.loaded_iter))
        os.makedirs(relight_dir, exist_ok=True)
        os.makedirs(os.path.join(relight_dir, 'envmaps'), exist_ok=True)

        # estimate tonemapping between gt envmap and the relight envmap
        if os.path.exists(args.gt_envmap_path):
            point_cloud_path = os.path.join(args.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter))

            # load gt envmap and save
            gt_envmap = util.load_image_raw(args.gt_envmap_path)
            util.save_image_raw(os.path.join(relight_dir, 'envmaps', 'gt_envmap.hdr'), gt_envmap)
            util.save_image(os.path.join(relight_dir, 'envmaps', 'gt_envmap.png'), gt_envmap)

            # load estimated envmap and save
            est_envmap = util.load_image_raw(os.path.join(point_cloud_path, "brdf_mlp.hdr"))
            util.save_image_raw(os.path.join(relight_dir, 'envmaps', 'estimated_envmap.hdr'), est_envmap)
            util.save_image(os.path.join(relight_dir, 'envmaps', 'estimated_envmap.png'), est_envmap)

            # estimate tonemapping params, save them and create tonemapping function
            try:
                est_tonemap_params, gt_envmap_tonemapped = estimate_tonemap(est_envmap, gt_envmap)
                save_tonemap_params(est_tonemap_params, point_cloud_path)  # save tonemapping params
                est_tonemap = partial(apply_tonemapping, params=est_tonemap_params)  # create tonemapping funcction
                print("Estimated tonemap params: ", est_tonemap_params)
            except RuntimeError as err:
                print(f'Failed to estimate tonemapping params with error {err}. Using simple gamma tonemapping.')
                est_tonemap = gamma_tonemap
                gt_envmap_tonemapped = est_tonemap(gt_envmap)

            # tonemap the original envmap and save
            util.save_image_raw(os.path.join(relight_dir, 'envmaps', "gt_envmap_tonemapped.hdr"), gt_envmap_tonemapped)
            util.save_image(os.path.join(relight_dir, 'envmaps', "gt_envmap_tonemapped.png"), gt_envmap_tonemapped)
        else:
            est_tonemap = gamma_tonemap

        # load relight envmap and save
        relight_envmap = util.load_image_raw(args.relight_envmap_path)
        util.save_image_raw(os.path.join(relight_dir, 'envmaps', 'relight_envmap.hdr'), relight_envmap)
        util.save_image(os.path.join(relight_dir, 'envmaps', 'relight_envmap.png'), relight_envmap)

        # tonemap the relight envmap and save
        relight_envmap_tonemapped = est_tonemap(relight_envmap)
        util.save_image_raw(os.path.join(relight_dir, 'envmaps', "relight_envmap_tonemapped.hdr"), relight_envmap_tonemapped)
        util.save_image(os.path.join(relight_dir, 'envmaps', "relight_envmap_tonemapped.png"), relight_envmap_tonemapped)

        # adapt scene for relighted dataset
        if args.relight_gt_path and not args.render_path:
            relight_dataset = deepcopy(dataset)
            relight_dataset.source_path = args.relight_gt_path
            relight_scene = Scene(relight_dataset, gaussians, load_iteration=iteration, shuffle=False, relight=True)
            del scene, dataset
        else:
            relight_scene = scene

        if args.render_path:
            n_frames = 240
            cam_traj = generate_path(relight_scene.getTrainCameras(), n_frames=n_frames)
        else:
            cam_traj = relight_scene.getTestCameras()

        rescale_relighted = args.rescale_relighted and not args.render_path
        gaussians.load_env_map(args.relight_envmap_path, lambda x: np.roll(est_tonemap(x), x.shape[1]//4, axis=1))
        gaussExtractor.reconstruction(cam_traj, relight=True, rescale=rescale_relighted, traj=args.render_path)
        gaussExtractor.export_image(relight_dir, relight=True, skip_misc=True, traj=args.render_path)

    # apply scene editing if the multipliers differ from 1.0
    if args.edit:
        # edit attributes
        gaussians.use_residual = False
        diffuse_mult = [float(item) for item in args.diffuse_mult.split(',')]
        with torch.no_grad():
            diffuse_mask = torch.logical_and(gaussians._features_dc.mean(axis=1) > args.intensity_low_thresh,
                                             gaussians._features_dc.mean(axis=1) < args.intensity_high_thresh)
            gaussians._features_dc[diffuse_mask] = gaussians._features_dc[diffuse_mask] * torch.tensor(diffuse_mult).to(gaussians._features_dc.device)
            gaussians._roughness = gaussians._roughness * torch.tensor(args.roughness_mult).to(gaussians._roughness.device)
            gaussians._tint = gaussians._tint * torch.tensor(args.tint_mult).to(gaussians._tint.device)

        # render
        edit_dir = f'edit/dm{diffuse_mult}_rm{args.roughness_mult}_tm{args.tint_mult}'
        test_dir = os.path.join(args.model_path, edit_dir, "ours_{}".format(scene.loaded_iter))
        print(f"export edited images with settings {edit_dir} ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir, skip_misc=True)
