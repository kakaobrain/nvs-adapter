import sys
import importlib
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
from torchvision import transforms
from PIL import Image

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *

sys.path.append(os.path.abspath("../"))

from sgm.geometry import get_rays, make_view_matrix, make_intrinsic_matrix
from sgm.models.nvsadapter import NVSAdapterDiffusionEngine
from sgm.modules.nvsadapter.wrappers import NVSAdapterWrapper
from sgm.util import append_dims, instantiate_from_config


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# load model
def load_model_from_config(config, ckpt, device, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    config.model.params.sd_ckpt_path = None

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if verbose:
        print("[INFO] missing keys: ", len(m), m)
        print("[INFO] unexpected keys: ", len(u), u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


def decode_image(path: str, color: List, has_alpha: bool = True) -> np.array:
    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    if has_alpha:
        img[img[:, :, -1] == 0.0] = color
    return Image.fromarray(np.uint8(img[:, :, :3]))


@threestudio.register("nvsadapter-guidance")
class NvsAdapterGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "load/nvsadapter/last.ckpt"
        pretrained_config: str = "load/nvsadapter/default_single_query.yaml"

        cond_image_path: str = "load/images/hamburger_rgba.png"
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2
        cond_fovy_deg: float = 49.1

        guidance_scale: float = 5.0

        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 8

        num_query: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading NVS Adapter ...")

        self.config = OmegaConf.load(self.cfg.pretrained_config)
        # TODO: seems it cannot load into fp16...
        self.weights_dtype = torch.float32
        self.model: NVSAdapterDiffusionEngine = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            verbose=True,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.unet: NVSAdapterWrapper = self.model.model

        model_config = self.config.model
        denoiser_config = model_config.params.denoiser_config

        self.loss_fn = self.model.loss_fn
        self.sampler = self.model.sampler
        self.denoiser = self.model.denoiser

        self.alphas: Float[Tensor, "..."] = torch.tensor(
            self.sampler.discretization.alphas_cumprod, dtype=torch.float32, device=self.device)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = denoiser_config.params.num_idx
        self.set_min_max_steps()  # set to default value

        self.conditioner = self.model.conditioner

        self.grad_clip_val = None

        self.prepare_embeddings(self.cfg.cond_image_path)

        self.num_query = self.cfg.num_query

        self.intrinsic = make_intrinsic_matrix(fov_rad=np.deg2rad(self.cfg.cond_fovy_deg), h=256, w=256)

        support_c2w = make_view_matrix(azimuth=np.deg2rad(self.cfg.cond_azimuth_deg),
                                       elevation=np.deg2rad(self.cfg.cond_elevation_deg),
                                       dist=self.cfg.cond_camera_distance)
        self.support_c2w = support_c2w.to(device=self.device, dtype=self.weights_dtype)
        self.support_w2c = torch.inverse(self.support_c2w)

        threestudio.info(f"Loaded NVS Adapter!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def prepare_embeddings(self, image_path: str) -> None:
        # load cond image for zero123
        assert os.path.exists(image_path)

        color_background = [255.0, 255.0, 255.0, 255.0]

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (256, 256),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
        ])

        rgb_BCHW_256 = image_transform(decode_image(image_path, color_background))
        rgb_BCHW_256 = rgb_BCHW_256.to(self.device).unsqueeze(0)

        self.rgb_BCHW_256 = rgb_BCHW_256 * 2.0 - 1.0
        self.img_latents = self.encode_images(rgb_BCHW_256).detach()

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.encode_images(imgs.to(self.weights_dtype))
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond(
        self,
        azimuth: Float[Tensor, "B"],
        elevation: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        in_batch_size = elevation.shape[0]

        assert in_batch_size % self.num_query == 0, f"{in_batch_size=} % {self.num_query=} == 0"

        out_batch_size = math.ceil(in_batch_size / self.num_query)

        h_latents, w_latents = self.img_latents.shape[-2:]
        h_rgbs, w_rgbs = self.rgb_BCHW_256.shape[-2:]

        assert h_rgbs / h_latents == w_rgbs / w_latents, "The ratio of height and width should be the same."
        stride = int(h_rgbs / h_latents)

        support_intrinsics = torch.stack([self.intrinsic] * out_batch_size).to(self.device)
        support_intrinsics = rearrange(support_intrinsics, "(b n) r c -> b n r c", b=out_batch_size, n=1)

        query_intrinsics = torch.stack([self.intrinsic] * in_batch_size).to(self.device)
        query_intrinsics = rearrange(query_intrinsics,"(b n) r c -> b n r c", b=out_batch_size, n=self.num_query)

        support_c2ws = torch.stack([self.support_c2w] * out_batch_size).to(self.device)
        support_c2ws = rearrange(support_c2ws,"(b n) r c -> b n r c", b=out_batch_size, n=1)

        query_c2ws = []
        for azi, ele, dist in zip(azimuth, elevation, camera_distances):
            c2w = make_view_matrix(azimuth=torch.deg2rad(azi),
                                   elevation=torch.deg2rad(ele),
                                   dist=dist).to(self.device)
            query_c2ws.append(c2w)
        query_c2ws = rearrange(query_c2ws,"(b n) r c -> b n r c", b=out_batch_size, n=self.num_query)

        support_c2ws = self.support_w2c @ support_c2ws
        query_c2ws = self.support_w2c @ query_c2ws

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

        support_latents = torch.cat([self.img_latents] * out_batch_size)
        support_latents = rearrange(support_latents,"(b n) c h w -> b n c h w", b=out_batch_size, n=1)
        support_rgbs = torch.cat([self.rgb_BCHW_256] * out_batch_size)

        batch = {
            "support_latents": support_latents,
            "support_rgbs": support_rgbs,
            "txt": [""] * out_batch_size,
            "support_rays_offset": support_rays_offset,
            "support_rays_direction": support_rays_direction,
            "query_rays_offset": query_rays_offset,
            "query_rays_direction": query_rays_direction,
        }

        c, uc = self.conditioner.get_unconditional_conditioning(batch)
        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k].to(self.device), (c, uc))

        cond = dict()
        for k in c:
            if k in ["image/crossattn", "txt/crossattn", "ray/concat", "support_latents/concat"]:
                cond[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                cond[k] = c[k]

        ucg_mask = torch.cat([torch.ones(out_batch_size),
                              torch.zeros(out_batch_size)]).to(device=self.device, dtype=torch.bool)

        return cond, ucg_mask

    @torch.inference_mode()
    def apply_model(
        self,
        x_in: Float[Tensor, "..."],
        s_in: Float[Tensor, "..."],
        cond: dict,
        ucg_mask: Bool[Tensor, "..."],
        x0_to_epsilon: bool = True,
    ) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:

        out = self.denoiser(self.unet, x_in, s_in, cond, ucg_mask=ucg_mask)

        if x0_to_epsilon:
            x0_pred = out
            noise_pred = (x0_pred - x_in) / append_dims(-s_in, x_in.ndim)

            # debug
            x0_pred_uncond, x0_pred_cond = x0_pred.chunk(2)
            x0_pred = x0_pred_uncond + self.cfg.guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )
            x0_pred = x0_pred.flatten(0, 1)

            return noise_pred, x0_pred
        else:
            return out


    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        assert batch_size % self.num_query == 0, f"{batch_size=} % {self.num_query=} == 0"
        internal_batch_size = math.ceil(batch_size / self.num_query)

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = (
                F.interpolate(rgb_BCHW, (32, 32), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        cond, ucg_mask = self.get_cond(azimuth, elevation, camera_distances)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        internal_t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [internal_batch_size],
            dtype=torch.long,
            device=self.device,
        )
        internal_sigmas = self.model.denoiser.idx_to_sigma(internal_t)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator

            internal_latents = rearrange(latents, "(b n) c h w -> b n c h w", b=internal_batch_size, n=self.num_query)
            internal_noise = rearrange(noise, "(b n) c h w -> b n c h w", b=internal_batch_size, n=self.num_query)
            internal_latents_noisy = self.loss_fn.add_noise(internal_latents, internal_noise, internal_sigmas)
            # pred noise
            x_in = torch.cat([internal_latents_noisy] * 2)
            s_in = torch.cat([internal_sigmas] * 2)
            noise_pred, x0_pred = self.apply_model(x_in, s_in, cond, ucg_mask=ucg_mask)

        t = internal_t.repeat_interleave(self.num_query)
        latents_noisy = rearrange(internal_latents_noisy, "b n c h w -> (b n) c h w")

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        noise_pred = rearrange(noise_pred, "b n c h w -> (b n) c h w")

        w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straightforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }
        if guidance_eval:
            guidance_eval_utils = {
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": noise_pred,
                "x0_pred": x0_pred,
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(self, t_orig, latents_noisy, noise_pred, x0_pred):
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size

        fracs = (t_orig[:bs] / self.num_train_timesteps).cpu().numpy()
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(x0_pred[:bs]).permute(0, 2, 3, 1)
        imgs_1step = imgs_1orig
        imgs_final = imgs_1orig

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
