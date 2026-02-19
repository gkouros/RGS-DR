# RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting
Georgios Kouros, Minye Wu, Tinne Tuytelaars

| [Project page](https://gkouros.github.io/projects/RGS-DR/) | [Full paper](https://arxiv.org/abs/2504.18468) |

**This repository contains the official implementation of the paper "RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting" that will appear at 3DV 2026.**


## Abstract
In this work, we address specular appearance in inverse rendering using 2D Gaussian splatting with deferred shading and argue for a refinement stage to improve specular detail, thereby bridging the gap with reconstruction-only methods. Our pipeline estimates editable material properties and environment illumination while employing a directional residual pass that captures leftover view-dependent effects for further refining novel view synthesis. In contrast to per-Gaussian shading with shortest-axis normals and normal residuals, which tends to result in more noisy geometry and specular appearance, a pixel-deferred surfel formulation with specular residuals yields sharper highlights, cleaner materials, and improved editability. We evaluate our approach on rendering and reconstruction quality on three popular datasets featuring glossy objects, and also demonstrate high-quality relighting and material editing.

## Pipeline
![Pipeline figure.](assets/pipeline.png)
Our rendering pipeline consists of three passes. The geometry pass produces screen-space diffuse color Id, specular tint
S, roughness M, normals N, and low-dimensional features K, which feed into the subsequent passes. The lighting pass employs a cube
mipmap to model environmental light for shading, as in [GaussianShader](https://github.com/Asparagus15/GaussianShader). Meanwhile, the residual pass uses a spherical-mip-based directional
encoding (inspired by [Ref-GS](https://github.com/YoujiaZhang/Ref-GS?tab=readme-ov-file)) along with a shallow MLP fres to predict view-dependent effects not captured by the lighting pass.

## Installation Instructions
```bash
conda env create -f environment.yml
conda activate rgsdr

pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
```

## Datasets
1) Download the datasets [Shiny Synthetic](https://storage.googleapis.com/gresearch/refraw360/ref.zip), [Shiny Real](https://storage.googleapis.com/gresearch/refraw360/ref_real.zip), and [Glossy Synthetic](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EvNz_o6SuE1MsXeVyB0VoQ0B9zL8NZXjQQg0KknIh6RKjQ?e=MaonKe).

2) Convert the glossy dataset to blender format:
```bash
python scripts/nero2blender.py --path ./data/GlossySynthetic
```

3) Arrange the datasets as follows:
```bash
./data/
├── glossy_synthetic/
│   ├── angel/
│   │   ├── rgb/
│   │   ├── test_transforms.json
│   │   └── train_transforms.json
.   .
.   .
├── ref_shiny/
│   ├── ball/
│   │   ├── train/
│   │   ├── test/
│   │   ├── test_transforms.json
│   │   └── train_transforms.json
.   .
.   .
└── ref_real/
    ├── garenspheres/
    │   ├── images/
    │   ├── test_transforms.json
    │   └── train_transforms.json
    .
    .

```

## Evaluation
To evaluate our method run the following commands:
```bash
python scripts/glossy_eval.py --scene=all <exp_name> # choose from {all, angel, bell, cat, horse, luyu, potion, tbell, teapot}

python scripts/shiny_eval.py --scene=all <exp_name> # choose from {all, ball, car, coffee, helmet, teapot, toaster}

python scripts/real_eval.py --scene=all <exp_name> # choose from {all, gardenspheres, sedan, toycar}
```

## Visualization
To visualize a scene, first run the web server with the following command:
```bash
python view.py -m "/path/to/experiment"
```
Then on a web browser go to the following address: https://localhost:8080

## Acknowledgement
We gratefully acknowledge the following works that were instrumental in the development of our method:
- [3DGS-DR](https://github.com/gapszju/3DGS-DR)
- [GaussianShader](https://github.com/Asparagus15/GaussianShader)
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting)

## BibTeX
```bibtex
@misc{kouros2025rgsdr,
      title={RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting},
      author={Georgios Kouros and Minye Wu and Tinne Tuytelaars},
      year={2025},
      eprint={2504.18468},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.18468},
}
```