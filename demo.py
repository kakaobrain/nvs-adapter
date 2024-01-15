import argparse
import sys
from functools import partial

import numpy as np
import torch
import einops
from torchvision.transforms import ToTensor, ToPILImage, Compose, Lambda, InterpolationMode, Resize
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image 
import gradio as gr
import fire

from sgm.util import instantiate_from_config
from sgm.models.nvsadapter import NVSAdapterDiffusionEngine
from sgm.geometry import make_view_matrix, make_intrinsic_matrix, get_rays
from sgm.data.single_image import decode_image

sys.path.append("thirdparty/carvekit")
from carvekit.api.high import HiInterface

_GPU_INDEX = 0
_CHECKPOINT = "checkpoints/demo.ckpt"
_TITLE = "NVS-Adapter: Plug-and-play Novel View Synthesis from a Single Image"
_DESCRIPTION = '''
This demo allows you to test our model with an arbitrary input image. You can set arbitrary azimuth and elevation for each view. Check out our [project webpage](https://postech-cvlab.github.io/nvsadapter/) and [paper](https://arxiv.org/abs/2312.07315) 
'''

_ARTICLE = 'See uses.md'

def create_carvekit_interface():
    interface = HiInterface(
        object_type="object",
        batch_size_seg=5,
        batch_size_matting=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seg_mask_size=640,
        matting_mask_size=2048,
        trimap_prob_threshold=231,
        trimap_dilation=30,
        trimap_erosion_iters=5,
        fp16=False
    )

    return interface


def decode_image(image: Image, color = [255, 255, 255, 255]):
    image = np.array(image, dtype=np.float32)
    if image.shape[-1] == 4:
        image[image[:, :, -1] == 0.0] = color
    return Image.fromarray(np.uint8(image[:, :, :3]))

def load_and_preprocess(carvekit_model, image: Image, use_carvekit=True):

    if use_carvekit:
        image = image.convert('RGB')
        image_wo_bkgd = np.array(carvekit_model([image])[0])
        est_seg = image_wo_bkgd > 127
        image = np.array(image)
        foreground = est_seg[:, : , -1].astype(np.bool_)
        image[~foreground] = [255., 255., 255.]
        image = Image.fromarray(np.array(image))

    else: 
        image = decode_image(image)

    image_transform = Compose([
        ToTensor(),
        Resize(
            (256, 256),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ),
        Lambda(lambda x: x * 2.0 - 1.0),
    ])

    return image_transform(image).clamp(-1, 1)


def prepare_batch(intrinsic, poses, input_tensor, model):
    
    support_rgbs = einops.rearrange(input_tensor, "(b n c) h w -> b n c h w", b=1, n=1)
    support_c2w = make_view_matrix(np.deg2rad(0), np.deg2rad(0), 1.5)
    support_intrinsics = einops.rearrange(intrinsic, "(b n i) j -> b n i j", b=1, n=1)
    support_c2ws = einops.rearrange(support_c2w, "(b n i) j -> b n i j", b=1, n=1)
    support_c2ws[..., :3, :3] *= -1
    query_intrinsics = einops.repeat(support_intrinsics, "b n i j -> b (n repeat) i j", repeat=4)
    query_c2ws = einops.rearrange(poses, "(b n) i j -> b n i j", b=1)
    query_c2ws[..., :3, :3] *= -1

    inverse_support_c2ws = torch.inverse(support_c2ws)
    support_c2ws = inverse_support_c2ws @ support_c2ws
    query_c2ws = inverse_support_c2ws @ query_c2ws

    support_latents = model.encode_first_stage(support_rgbs)

    h_latents, w_latents = support_latents.shape[-2:]
    h_rgbs, w_rgbs = support_rgbs.shape[-2:]

    assert h_rgbs / h_latents == w_rgbs / w_latents, "The ratio of height and width should be the same."
    stride = int(h_rgbs / h_latents)

    support_rays_offset, support_rays_direction = get_rays(
        intrinsics=support_intrinsics,
        c2ws=support_c2ws,
        image_size=(h_rgbs, w_rgbs),
        stride=stride,
    )
    query_rays_offset, query_rays_direction = get_rays(
        intrinsics=query_intrinsics,
        c2ws=query_c2ws,
        image_size=(h_rgbs, w_rgbs),
        stride=stride,
    )
    batch = {
        "support_latents": support_latents,
        "support_rgbs": support_rgbs.flatten(0, 1),
        "support_rgbs_cond": support_rgbs,
        "txt": [""],
        "support_rays_offset": support_rays_offset,
        "support_rays_direction": support_rays_direction,
        "query_rays_offset": query_rays_offset,
        "query_rays_direction": query_rays_direction,
    }

    return batch


def main_run(
    model, 
    device,
    input_image,
    cfg_scale,
    num_steps,
    seed,
    rm_bkgd,
    azimuth_1,
    azimuth_2, 
    azimuth_3,
    azimuth_4,
    elevation_1,
    elevation_2, 
    elevation_3, 
    elevation_4,
):
    seed_everything(seed)

    if input_image is None:
        return None, None, None, None, None

    model.sampler.num_steps = num_steps
    model.sampler.guider.scale_schedule = lambda sigma: cfg_scale

    if rm_bkgd:
        carvekit_model = create_carvekit_interface()
    else:
        carvekit_model = None
    rm_bkgd_img = load_and_preprocess(carvekit_model, input_image, rm_bkgd)

    input_tensor = rm_bkgd_img.to(device)

    pose1 = make_view_matrix(np.deg2rad(azimuth_1), np.deg2rad(elevation_1), 1.5)
    pose2 = make_view_matrix(np.deg2rad(azimuth_2), np.deg2rad(elevation_2), 1.5)
    pose3 = make_view_matrix(np.deg2rad(azimuth_3), np.deg2rad(elevation_3), 1.5)
    pose4 = make_view_matrix(np.deg2rad(azimuth_4), np.deg2rad(elevation_4), 1.5)

    poses = torch.stack([pose1, pose2, pose3, pose4])

    intrinsic = make_intrinsic_matrix(np.deg2rad(49.1), 256, 256)

    batch = prepare_batch(intrinsic, poses, input_tensor, model)
    pred_images = model.novel_view_sample(batch, 4)
    rm_bkgd_img = (rm_bkgd_img + 1) / 2.

    to_pil = ToPILImage()
    out_image1 = to_pil(pred_images[0, 0].clamp(0, 1))
    out_image2 = to_pil(pred_images[0, 1].clamp(0, 1))
    out_image3 = to_pil(pred_images[0, 2].clamp(0, 1))
    out_image4 = to_pil(pred_images[0, 3].clamp(0, 1))
    out_rm_bkgd_img = to_pil(rm_bkgd_img)

    return out_rm_bkgd_img, out_image1, out_image2, out_image3, out_image4
    

def run_demo(
    device,
    config,
    ckpt_path,
    server_name,
    server_port,
):
    
    demo = gr.Blocks(title=_TITLE)
    device = device if torch.cuda.is_available() else "cpu"

    with open(config) as fp:
        config = OmegaConf.load(fp)

    model_config = config.model    
    model_config.params.use_ema = True
    model_config.params.sd_ckpt_path = None
    model_config.params.ckpt_path = ckpt_path

    model: NVSAdapterDiffusionEngine = instantiate_from_config(model_config)
    model.eval().to(device)

    with demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(variant='panel'):

                image_block = gr.Image(type='pil', image_mode='RGBA', label='Input image of single object', value="sample/kunkun.png")

                with gr.Accordion('Advanced options', open=False):
                    scale_slider = gr.Slider(0, 30, value=11, step=1, label='Diffusion guidance scale')
                    steps_slider = gr.Slider(5, 200, value=50, step=5, label='Number of diffusion inference steps')
                    seed_slider = gr.Number(value=777, label="Seed")
                    rm_bkgd = gr.Checkbox(True, label="Use carvekit to remove background.")

                run_btn = gr.Button('Run Generation', variant='primary')
                bkgd_rm_output = gr.Image(label="Background removed image", type="pil")

            with gr.Row(variant='panel'):
                with gr.Column(variant="panel"):
                    with gr.Column(variant="panel"):
                        gen_output_1 = gr.Image(label="view1", type="pil")
                        with gr.Accordion():
                            azimuth_1 = gr.Slider(0, 360, value=72, step=5, label="Azimuth")
                            elevation_1 = gr.Slider(-90, 90, value=0, step=5, label="Elevation")                        
                    
                    with gr.Column(variant="panel"):
                        gen_output_2 = gr.Image(label="view2", type="pil")
                        with gr.Accordion():
                            azimuth_2 = gr.Slider(0, 360, value=216, step=5, label="Azimuth")
                            elevation_2 = gr.Slider(-90, 90, value=0, step=5, label="Elevation")

                with gr.Column(variant="panel"):
                    with gr.Column(variant="panel"):
                        gen_output_3 = gr.Image(label="view3", type="pil")
                        with gr.Accordion():
                            azimuth_3 = gr.Slider(0, 360, value=144, step=5, label="Azimuth")
                            elevation_3 = gr.Slider(-90, 90, value=0, step=5, label="Elevation")                        
                    
                    with gr.Column(variant="panel"):
                        gen_output_4 = gr.Image(label="view4", type="pil")
                        with gr.Accordion():
                            azimuth_4 = gr.Slider(0, 360, value=288, step=5, label="Azimuth")
                            elevation_4 = gr.Slider(-90, 90, value=0, step=5, label="Elevation")        

        run_btn.click(
            fn=partial(main_run, model, device), 
            inputs=[
                image_block, 
                scale_slider, 
                steps_slider, 
                seed_slider,
                rm_bkgd,
                azimuth_1, 
                azimuth_2, 
                azimuth_3, 
                azimuth_4, 
                elevation_1, 
                elevation_2, 
                elevation_3, 
                elevation_4
            ],
            outputs=[bkgd_rm_output, gen_output_1, gen_output_2, gen_output_3, gen_output_4]
        )

        demo.launch(server_name=server_name, server_port=server_port, share=True)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, help="GPU index")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--ckpt_path", type=str, help="path to checkpoint")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server host name")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    
    if args.device is None:
        args.device = _GPU_INDEX
    
    if args.ckpt_path is None: 
        args.ckpt_path = _CHECKPOINT

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    demo_run_fn = partial(
        run_demo, 
        device=args.device, 
        config=args.config,
        ckpt_path=args.ckpt_path,
        server_name=args.server_name,
        server_port=args.server_port,
    )

    fire.Fire(demo_run_fn)
