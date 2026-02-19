import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from gaussian_renderer import network_gui
from utils.image_utils import render_net_image
import torch
import viser
import numpy as np
from scene.cameras import Camera
from scene.NVDIFFREC.light import extract_env_map

# init gui
server = network_gui.init(initial_value="envmap")

def view(dataset, pipe, iteration, relight_envmap_path):

    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, dataset.brdf_envmap_res)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, no_cameras=True)
    if relight_envmap_path:
        gaussians.load_env_map(relight_envmap_path, tonemap=lambda x: np.roll(x, shift=x.shape[1]//4, axis=1))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    while True:

        if server is None or server["client"] is None:
            continue

        render_type = network_gui.on_gui_change()

        with torch.no_grad():
            client = server["client"]
            RT_w2v = viser.transforms.SE3(wxyz_xyz=np.concatenate([client.camera.wxyz, client.camera.position], axis=-1)).inverse()
            R = torch.tensor(RT_w2v.rotation().as_matrix().astype(np.float32)).numpy()
            T = torch.tensor(RT_w2v.translation().astype(np.float32)).numpy()
            # FoVx = viewpoint_cam.FoVx # TODO: client fov
            # FoVy = viewpoint_cam.FoVy
            FoVx = FoVy = 0.5

            camera = Camera(
                colmap_id=None,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=torch.zeros((3, 800, 800)).cuda(),
                gt_alpha_mask = None,
                image_name="",
                uid=None,
            )

            render_pkg = render(camera, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            rend_dist = render_pkg["rend_dist"]
            alpha_map = render_pkg["rend_alpha"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            diffuse_map = render_pkg['rend_diffuse']
            M_map =  render_pkg['rend_roughness']
            specular_color = render_pkg['rend_specular_color']
            specular_residual = render_pkg['rend_specular_residual']
            specular_tint = render_pkg['rend_tint']

            output = None
            if render_type == "Rendered":
                image = torch.clamp(image, 0.0, 1.0)
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
                image = torch.clamp(specular_color, 0.0, 1.0).repeat(3,1,1)
                image += 1 - alpha_map
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "specular residual":
                image = torch.clamp(specular_residual, 0.0, 1.0).repeat(3,1,1)
                image += 1 - alpha_map
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "specular tint":
                image = torch.clamp(specular_tint, 0.0, 1.0).repeat(3,1,1)
                image += 1 - alpha_map
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
                rendered_image = torch.clamp(rendered_image, 0, 1) * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "envmap2":
                image = extract_env_map(gaussians.brdf_mlp, rotated=True)
                rendered_image = image.detach().cpu()
                rendered_image = torch.clamp(rendered_image, 0, 1) * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif render_type == "alpha map":
                image = alpha_map.repeat(3,1,1)
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


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("-e", "--relight_envmap_path", default="", help="Envmap path to relight with")
    args = get_combined_args(parser)
    print(args)
    # args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)
    view(lp.extract(args), pp.extract(args), args.iteration, args.relight_envmap_path)
    print("\nViewing complete.")