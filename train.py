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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, validate_state_dicts
import uuid
from tqdm.auto import tqdm
from utils.image_utils import psnr, render_net_image
from utils.tonemap import gamma_tonemap
from utils.loss_utils import binary_cross_entropy
from utils.general_utils import colormap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.NVDIFFREC.light import extract_env_map
import pdb
from scene.cameras import Camera
import numpy as np
import wandb
import torch.nn.functional as F


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, server=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, dataset.brdf_envmap_res)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=True)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        first_iter = scene.loaded_iter if scene.loaded_iter else first_iter

    # bounding volume filtering for real scenes based on 3DGS-DR
    if opt.use_env_scope:
        env_center = gaussians.env_scope_center
        env_radius = gaussians.env_scope_radius

    # bg_color = [1,1,1,0,0,0,0,0] if dataset.white_background else [0,0,0,0,0,0,0,0]
    bg_color = [1,1,1,0,0,0,0,0,0,0,0,0] if dataset.white_background else [0,0,0,0,0,0,0,0,0,0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_mask_for_log = 0.0
    ema_env_scope_for_log = 0.0

    first_iter += 1
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    progress_bar = tqdm(initial=first_iter, total=opt.iterations, desc="Training progress")
    for iteration in range(first_iter, opt.iterations+1):

        if dataset.use_residual and iteration > opt.residual_from_iter and not gaussians.use_residual:
            print('Activating residual mode. Everything is frozen and only residual is trained...')
            if opt.residual_from_iter == 0:
                gaussians.activate_residual()  # enables residual optimization
            else:
                gaussians.set_residual_mode(opt)  # disables all other gradients and only optimizes residual
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # build mipmap of envmap
        gaussians.brdf_mlp.build_mips()

        # render the scene and scene attributes
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)

        if opt.clip_output:
            render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

        # extract renderings
        image = render_pkg["render"] if iteration > opt.warmup_until_iter else render_pkg["rend_specular_residual"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        diffuse_map = render_pkg['rend_diffuse']
        specular_color = render_pkg['rend_specular_color']
        specular_residual = render_pkg['rend_specular_residual']
        alpha_map = render_pkg["rend_alpha"]
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        env_scope_mask = render_pkg['mask_map']

        gt_image = viewpoint_cam.original_image[:3, :,:].cuda()

        # compute losses
        Ll1 = l1_loss(image, gt_image)
        # Loss = (1-λ) * Lrec + λ * Lssim + λα * Lα
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # distortion loss
        lambda_dist = opt.lambda_dist if iteration > opt.dist_loss_from_iter else 0.0
        dist_loss = lambda_dist * rend_dist.mean()

        # normals regularization
        lambda_normal = opt.lambda_normal if iteration > opt.normal_reg_from_iter else 0.0
        normal_loss = lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0))[None].mean()

        # compute loss for alpha
        if viewpoint_cam.gt_alpha_mask is not None:
            mask = viewpoint_cam.gt_alpha_mask.cuda()
            mask[mask > 0] = 1.0
        else:
            mask = torch.ones_like(alpha_map)
        activate_mask_loss = iteration >= opt.mask_loss_from_iter and iteration < opt.mask_loss_until_iter
        lambda_alpha = opt.lambda_alpha if activate_mask_loss else opt.lambda_alpha_drop_factor * opt.lambda_alpha
        if not opt.dilate_mask:
            alpha_loss = lambda_alpha * l1_loss(mask, alpha_map) * binary_cross_entropy(mask, alpha_map)
        else:
            dilated_mask = F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3)
            dilated_mask = torch.logical_or(dilated_mask == 0, dilated_mask > 0.9)
            alpha_loss = lambda_alpha * l1_loss(mask[dilated_mask], alpha_map[dilated_mask]) \
                * binary_cross_entropy(mask[dilated_mask] * env_scope_mask[dilated_mask],
                                       alpha_map[dilated_mask] * env_scope_mask[dilated_mask])

        # filter points outside of bounding volume for real scenes
        if opt.use_env_scope:
            if gaussians.use_residual:
                env_scope_Ll1 = l1_loss(specular_residual * (1-env_scope_mask), gt_image * (1-env_scope_mask))
                env_scope_SSIM = 1.0 - ssim(specular_residual * (1-env_scope_mask), gt_image * (1-env_scope_mask))
                env_scope_loss = opt.lambda_env_scope * ((1.0 - opt.lambda_dssim) * env_scope_Ll1 + opt.lambda_dssim * env_scope_SSIM)
            else:
                outside_mask = torch.sum((gaussians.get_xyz - env_center[None])**2, dim=-1) > env_radius ** 2
                roughness = gaussians.get_roughness
                if opt.env_scope_attribute == "roughness":
                    env_scope_loss = opt.lambda_env_scope * roughness[outside_mask].mean()
                elif opt.env_scope_attribute == "glossiness":
                    glossiness = torch.ones_like(roughness) - roughness
                    env_scope_loss = opt.lambda_env_scope * glossiness[outside_mask].mean()
        else:
            env_scope_loss = torch.tensor(0.0, device=loss.device)

        # Total loss computation
        if iteration < opt.warmup_until_iter:
            total_loss = loss
        else:
            total_loss = loss + dist_loss + normal_loss + alpha_loss + env_scope_loss

        # backpropagate loss
        total_loss.backward()

        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_mask_for_log = 0.4 * alpha_loss.item() + 0.6 * ema_mask_for_log
            ema_env_scope_for_log = 0.4 * env_scope_loss.item() + 0.6 * ema_env_scope_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "envscope": f"{ema_env_scope_for_log:.{5}f}",
                    "mask": f"{ema_mask_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            elapsed = iter_start.elapsed_time(iter_end)

            # Log and save
            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/loss', ema_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/recon_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/mask_loss', ema_mask_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/env_scope_loss', ema_env_scope_for_log, iteration)
                tb_writer.add_scalar('iter_time', elapsed, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            if wandb.run:
                wandb.log({
                    "train/loss": ema_loss_for_log,
                    "train/rec_loss": Ll1.item(),
                    "train/dist_loss": ema_dist_for_log,
                    "train/normal_loss": ema_normal_for_log,
                    "train/mask_loss": ema_mask_for_log,
                    "train/env_scope_loss": ema_env_scope_for_log,
                    "train/iter_time": elapsed,
                    "train/#points": gaussians._xyz.size(0),
                })

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, render, (pipe, background))
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # clip cube mipmap
            if opt.clip_envmap:
                gaussians.brdf_mlp.clamp_(min=0.0, max=1.0)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        #viser!!
        if server is None or server["client"] is None:
            continue

        render_type = network_gui.on_gui_change()
        with torch.no_grad():
            client = server["client"]
            RT_w2v = viser.transforms.SE3(wxyz_xyz=np.concatenate([client.camera.wxyz, client.camera.position], axis=-1)).inverse()
            R = torch.tensor(RT_w2v.rotation().as_matrix().astype(np.float32)).numpy()
            T = torch.tensor(RT_w2v.translation().astype(np.float32)).numpy()
            FoVx = viewpoint_cam.FoVx
            FoVy = viewpoint_cam.FoVy

            camera = Camera(
                colmap_id=None,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=gt_image,
                gt_alpha_mask = None,
                image_name="",
                uid=None,
            )

            render_pkg = render(camera, gaussians, pipe, background)

            image = render_pkg["render"] if iteration > opt.warmup_until_iter else render_pkg["rend_specular_residual"]
            viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            rend_dist = render_pkg["rend_dist"]
            alpha_map = render_pkg["rend_alpha"]
            rend_normal  = render_pkg['rend_normal']
            rend_depth = render_pkg['surf_depth']
            surf_normal = render_pkg['surf_normal']
            diffuse_map = render_pkg['rend_diffuse']
            M_map =  render_pkg['rend_roughness']
            specular_color = render_pkg['rend_specular_color']
            specular_residual = render_pkg['rend_specular_residual']
            specular_tint = render_pkg['rend_tint']
            mask_map = render_pkg['mask_map']

            output = None
            if render_type == "Rendered":
                image = torch.clamp(image, 0.0, 1.0)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "gt image":
                image = torch.clamp(gt_image, 0.0, 1.0)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "render normal":
                rendered_image = (rend_normal.detach().cpu().permute(1, 2, 0) + 1)/2
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "surf normal":
                rendered_image = (surf_normal.detach().cpu().permute(1, 2, 0) + 1)/2
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "surf depth":
                max_depth = rend_depth.max()
                max_depth = 10.0
                rend_depth = torch.clamp(rend_depth, 0.0, max_depth)
                rendered_image = colormap((rend_depth / max_depth).cpu().numpy()[0], cmap='turbo', bar=False)
                rendered_image = rendered_image.permute(1, 2, 0)
                rendered_image = (rendered_image * 255).byte().numpy()
                output = rendered_image
            elif render_type == "diffuse color":
                image = torch.clamp(diffuse_map, 0.0, 1.0)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "roughness":
                image = torch.clamp(M_map, 0.0, 1.0).repeat(3,1,1)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "specular color":
                image = torch.clamp(specular_color, 0.0, 1.0)
                # image += 1 - alpha_map
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "specular residual":
                image = torch.clamp(specular_residual, 0.0, 1.0)
                # image += 1 - alpha_map
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "specular tint":
                image = torch.clamp(specular_tint, 0.0, 1.0)
                # image += 1 - alpha_map
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "feature map":
                image = gaussians.mipmap.visualization()
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "envmap":
                image = extract_env_map(gaussians.brdf_mlp)
                rendered_image = image.detach().cpu()
                rendered_image = gamma_tonemap(rendered_image) * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "envmap2":
                image = extract_env_map(gaussians.brdf_mlp, rotated=True)
                rendered_image = image.detach().cpu()
                rendered_image = gamma_tonemap(rendered_image) * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "alpha map":
                image = alpha_map.repeat(3,1,1)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "bounding volume map":
                image = mask_map.repeat(3,1,1)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            else:
                print(f"Unsupported render type: {render_type}")

            client.scene.set_background_image(
                output,
                format="jpeg"
            )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if wandb.run:
        with open(os.path.join(args.model_path, "wandb_run_id.txt"), 'w') as wandb_id_f:
            wandb_id_f.write(wandb.run.id)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        envmap = gamma_tonemap(extract_env_map(scene.gaussians.brdf_mlp).detach().permute(2, 0, 1).cpu())

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in tqdm(enumerate(config['cameras']), total=len(config['cameras']), desc=f"Evaluating {config['name']} at iter {iteration}"):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    rend_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    error_map = np.linalg.norm((rend_image - gt_image).abs().permute(1,2,0).cpu().numpy(), axis=2)
                    error_cmap = colormap(error_map, cmap='turbo')

                    # log images in tensorboard and wandb
                    # if idx % (len(config['cameras'] // 5)):
                    if idx == 0 and (tb_writer or wandb.run):
                        rend_alpha = render_pkg['rend_alpha']
                        rend_diffuse = render_pkg["rend_diffuse"]
                        rend_roughness = render_pkg["rend_roughness"]
                        rend_specular = render_pkg["rend_specular_color"]
                        rend_residual = render_pkg['rend_specular_residual']
                        rend_tint = render_pkg["rend_tint"]
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                        rend_depth = render_pkg["surf_depth"]
                        depth = colormap((rend_depth / rend_depth.max()).cpu().numpy()[0], cmap='turbo')
                        rend_dist = colormap(render_pkg['rend_dist'].cpu().numpy()[0])
                        rend_k = render_pkg["rend_k"]

                        if tb_writer:
                            try:
                                if iteration == testing_iterations[0]:
                                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                                tb_writer.add_images(config['name'] + "_view_{}/error".format(viewpoint.image_name), error_cmap[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), rend_image[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/diffuse".format(viewpoint.image_name), rend_diffuse[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/specular".format(viewpoint.image_name), rend_specular[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/residual".format(viewpoint.image_name), rend_residual[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/roughness".format(viewpoint.image_name), rend_roughness[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/tint".format(viewpoint.image_name), rend_tint[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/rend_feats".format(viewpoint.image_name), rend_k[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/envmap".format(viewpoint.image_name), envmap[None], global_step=iteration)
                            except:
                                pass

                        # log images in wandb
                        if wandb.run:
                            wandb.log({f"{config['name']}/images": [
                                wandb.Image(torch.clamp(gt_image, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="GT"),
                                wandb.Image(torch.clamp(rend_image, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Rendering"),
                                wandb.Image(torch.clamp(error_cmap, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Error map"),
                                wandb.Image(torch.clamp(rend_alpha, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Alpha"),
                                wandb.Image(torch.clamp(rend_diffuse, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Diffuse"),
                                wandb.Image(torch.clamp(rend_specular, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Specular"),
                                wandb.Image(torch.clamp(rend_residual, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Residual"),
                                wandb.Image(torch.clamp(rend_roughness, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Roughness"),
                                wandb.Image(torch.clamp(rend_tint, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Tint"),
                                wandb.Image(torch.clamp(rend_normal, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Normal"),
                                wandb.Image(torch.clamp(surf_normal, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Depth Normal"),
                                wandb.Image(torch.clamp(depth, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Depth"),
                                wandb.Image(torch.clamp(rend_dist, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Rend Dist"),
                                wandb.Image(torch.clamp(rend_k, 0, 1).permute(1, 2, 0).cpu().numpy(), caption="Rend Features"),
                            ]}, step=iteration)

                    l1_test += l1_loss(rend_image, gt_image).mean().double()
                    psnr_test += psnr(rend_image, gt_image).mean().double()

                l1_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if wandb.run:
                    wandb.log({f"{config['name']}/mse": l1_test, f"{config['name']}/psnr": psnr_test}, step=iteration)

        if wandb.run:
            wandb.log({"test/envmap": wandb.Image(envmap.permute(1, 2, 0).cpu().numpy(), caption="envmap")}, step=iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.save_iterations.append(args.residual_from_iter)
    for iterations in range(args.test_iterations[-1] + 5000, args.iterations, 5000):
        args.test_iterations.append(iterations)
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.gui:
        import viser
        server = network_gui.init()
    else:
        server = None

    if args.wandb:
        for id_name in ["wandb_run_id" , "wandb_id", "run_id"]:
            run_id_path = os.path.join(args.model_path, f"{id_name}.txt")
            if os.path.exists(run_id_path):
                with open(run_id_path, 'r') as f:
                    args.run_id = f.read().strip()
                print(f"File {id_name}.txt found. Logging metrics in session with id {args.run_id}.")
                break
        else:
            print(f"No {id_name}.txt file found. Logging metrics in session with id {args.run_id}.")

        exp_name, dataset_name, scene_name = args.model_path.split('/')[-3:]
        wandb.init(
            project="RGS-DR",  # set the wandb project where this run will be logged
            config=vars(args),  # track hyperparameters and run metadata
            group=dataset_name,
            name=f"{scene_name}.{exp_name}",
            job_type=exp_name.split('.')[0],
            id=args.run_id,
            resume="allow",
        )

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, server)

    # All done
    print("\nTraining complete.")