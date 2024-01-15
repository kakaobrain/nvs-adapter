import argparse
import os.path
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
import pyrootutils
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sgm.util import instantiate_from_config
from sgm.geometry import make_intrinsic_matrix, get_rays
from sgm.data.single_image import SingleImageDataset
from sgm.models.nvsadapter import NVSAdapterDiffusionEngine


@torch.cuda.amp.autocast(enabled=False)
@torch.no_grad()
def process_batch(batch, model, prompt):
    support_rgbs = batch["support_rgbs"].cuda()
    support_latents = model.encode_first_stage(support_rgbs)

    h_latents, w_latents = support_latents.shape[-2:]
    h_rgbs, w_rgbs = support_rgbs.shape[-2:]

    assert h_rgbs / h_latents == w_rgbs / w_latents, "The ratio of height and width should be the same."
    stride = int(h_rgbs / h_latents)

    support_rays_offset, support_rays_direction = get_rays(
        intrinsics=batch["support_intrinsics"],
        c2ws=batch["support_c2ws"],
        image_size=(h_rgbs, w_rgbs),
        stride=stride,
    )
    query_rays_offset, query_rays_direction = get_rays(
        intrinsics=batch["query_intrinsics"],
        c2ws=batch["query_c2ws"],
        image_size=(h_rgbs, w_rgbs),
        stride=stride,
    )

    batch = {
        "support_latents": support_latents,
        "support_rgbs": support_rgbs.flatten(0, 1),
        "support_rgbs_cond": support_rgbs,
        "txt": prompt,
        "support_rays_offset": support_rays_offset,
        "support_rays_direction": support_rays_direction,
        "query_rays_offset": query_rays_offset,
        "query_intrinsics": batch["query_intrinsics"],
        "query_c2ws": batch["query_c2ws"],
        "query_rays_direction": query_rays_direction,
    }
    return batch


def novel_view_sampling(args):
    device = torch.device("cuda")

    pl.seed_everything(args.seed)

    assert len(args.elevations) == len(args.azimuths), f"{len(args.elevations)=}, {len(args.azimuths)=} are different"
    assert len(args.elevations) % args.num_query == 0, f"{len(args.elevations)=} % {args.num_query=} == 0"

    expname = os.path.splitext(os.path.basename(args.config_path))[0]

    with open(args.config_path) as fp:
        config = OmegaConf.load(fp)

    for cfg_path in args.additional_configs:
        with open(cfg_path) as fp:
            config = OmegaConf.merge(config, OmegaConf.load(fp))

    model_config = config.model
    model_config.params.use_ema = args.use_ema
    model_config.params.sd_ckpt_path = None
    model_config.params.ckpt_path = args.ckpt_path

    if args.cfg_scale is not None:
        model_config.params.sampler_config.params.guider_config.params.scale = args.cfg_scale
        cfg_scale = args.cfg_scale
    else:
        cfg_scale = model_config.params.sampler_config.params.guider_config.params.scale

    litmodule: NVSAdapterDiffusionEngine = instantiate_from_config(model_config)
    litmodule.eval().to(device)

    image_path = args.input_image_path
    elevations = args.elevations
    azimuths = args.azimuths
    resolution = args.resolution
    num_query = args.num_query

    dists = [1.5] * len(args.elevations)

    dataset = SingleImageDataset(image_path, support_elevation=0, support_azimuth=0, support_dist=1.5,
                                 elevations=elevations, azimuths=azimuths, dists=dists,
                                 resolution=resolution, num_query=num_query, use_relative=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)

    # prepare batch
    image_name = os.path.basename(os.path.splitext(image_path)[0])

    save_dir = Path(args.logdir, expname)
    save_dir = save_dir.joinpath(f"{args.name}_{image_name}_cfg_scale_{cfg_scale}_use_ema_{args.use_ema}_seed_{args.seed}")
    save_dir.mkdir(exist_ok=True, parents=True)
    save_dir = save_dir.as_posix()

    for batch_idx, batch in enumerate(dataloader):
        batch = process_batch(batch, litmodule, prompt=[args.prompt])
        pred_images = litmodule.novel_view_sample(batch, num_query=args.num_query)
        pred_images = pred_images.flatten(0, 1)
        pred_images = pred_images.permute(0, 2, 3, 1).cpu().numpy()

        for idx, pred_image in enumerate(pred_images):
            pred_image = (pred_image * 255).astype(np.uint8)
            image_idx = batch_idx * num_query + idx
            output_image_path = os.path.join(save_dir, f"{image_idx:03d}.png")
            Image.fromarray(pred_image).save(output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default=None, help="name of the visualization")
    parser.add_argument("--config_path", type=str, default=None, help="path to config of pre-trained model", required=True)
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to checkpoint of pre-trained model", required=True)
    parser.add_argument("--logdir", type=str, default="./logs_viz", help="path to save the visualization")
    parser.add_argument("--use_ema", action="store_true", default=False, help="whether to use EMA model")
    parser.add_argument("--cfg_scale", type=float, default=None, help="scale for classifier free guidance")
    parser.add_argument("--input_image_path", type=str, default=False, help="path of input image", required=True)
    parser.add_argument("--num_query", type=int, default=1, help="number of query per input")
    parser.add_argument("--elevations", nargs="+", type=float, help="elevation list of camera in degree", required=True)
    parser.add_argument("--azimuths", nargs="+", type=float, help="azimuth list of camera in degree", required=True)
    parser.add_argument("--resolution", type=int, default=256, help="resolution of output image")
    parser.add_argument("--prompt", type=str, default="", help="prompt for generation")
    parser.add_argument("--seed", type=int, default=0, help="seed for random number generator")
    parser.add_argument("--use_controlnet", action="store_true", default=False, help="whether to use controlnet")
    parser.add_argument("-c", "--additional_configs", nargs="*", default=list())
    args = parser.parse_args()

    print('=' * 100)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 100)

    novel_view_sampling(args)
