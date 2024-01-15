from typing import Any, Dict, List, Union, Tuple
from omegaconf import ListConfig, OmegaConf
import json

import math
import torch
from einops import rearrange
from safetensors.torch import load_file as load_safetensors
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import LambdaLR
from piqa.ssim import SSIM
from piqa.lpips import LPIPS
from piqa.psnr import PSNR

from sgm.geometry import get_rays
from sgm.models.diffusion import DiffusionEngine
from sgm.modules.ema import LitEma
from sgm.util import instantiate_from_config


class NVSAdapterDiffusionEngine(DiffusionEngine):
    def __init__(
        self, 
        network_config, 
        denoiser_config, 
        first_stage_config, 
        conditioner_config: Dict | ListConfig | OmegaConf | None = None, 
        sampler_config: Dict | ListConfig | OmegaConf | None = None, 
        non_cfg_sampler_config: Dict | ListConfig | OmegaConf | None = None,
        optimizer_config: Dict | ListConfig | OmegaConf | None = None, 
        scheduler_config: Dict | ListConfig | OmegaConf | None = None, 
        loss_fn_config: Dict | ListConfig | OmegaConf | None = None, 
        network_wrapper: str | None = None, 
        ckpt_path: str | None = None, 
        sd_ckpt_path: str | None = None, # path to the pre-trained SD model.
        controlnet_ckpt_path: str | None = None, # path to the pre-trained controlnet model.
        use_ema: bool = False, 
        ema_decay_rate: float = 0.9999, 
        scale_factor: float = 1, 
        disable_first_stage_autocast=False, 
        input_key: str = "query_rgbs", 
        log_keys: List | None = None, 
        no_cond_log: bool = False,
        compile_model: bool = False,
        lr_mult_for_pretrained: float = 0. # mult factor for the learning rate of the pre-trained weights.
    ):
        super(NVSAdapterDiffusionEngine, self).__init__(
            network_config=network_config, 
            denoiser_config=denoiser_config, 
            first_stage_config=first_stage_config, 
            conditioner_config=conditioner_config, 
            sampler_config=sampler_config, 
            optimizer_config=optimizer_config, 
            scheduler_config=scheduler_config, 
            loss_fn_config=loss_fn_config, 
            network_wrapper=network_wrapper, 
            ckpt_path=None, # do not load from the checkpoint for Diffusion Engine. 
            use_ema=False, # do not create the EMA model for Diffusion Engine. 
            ema_decay_rate=0., # no need to pass the ema decay rate (avoiding potential bugs) 
            scale_factor=scale_factor, 
            disable_first_stage_autocast=disable_first_stage_autocast, 
            input_key=input_key, 
            log_keys=log_keys, 
            no_cond_log=no_cond_log, 
            compile_model=compile_model,
        )
        # for full fine-tuning
        self.lr_mult_for_pretrained = lr_mult_for_pretrained
        self.test_step_outputs = dict(psnr=[], ssim=[], lpips=[])

        pretrained_params = list(self.model.diffusion_model.pretrained_parameters())
        if self.lr_mult_for_pretrained == 0.: # no fine-tuning
            for param in pretrained_params:
                param.requires_grad_(False)

        if sd_ckpt_path is not None:
            print(f"Loaded SD from {sd_ckpt_path}")
            self.init_from_pretrained_sd(sd_ckpt_path, use_ema)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # for checking non-CFG sampling
        non_cfg_sampler_config = sampler_config.copy()
        non_cfg_sampler_config.params.guider_config.params.scale = 1.0
        self.non_cfg_sampler = instantiate_from_config(non_cfg_sampler_config)

        if ckpt_path is not None:
            print(f"Loaded checkpoint from {ckpt_path}")
            self.init_from_ckpt(ckpt_path)

        if controlnet_ckpt_path is not None:
            print(f"Loaded controlnet ckpt from {controlnet_ckpt_path}")
            self.load_controlnet_ckpt(controlnet_ckpt_path)

    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
            if param.requires_grad:
                yield name, param

    def configure_optimizers(self):
        lr = self.learning_rate
        scratch_params = list(self.model.diffusion_model.scratch_parameters())
        pretrained_params = list(self.model.diffusion_model.pretrained_parameters())
        grouped_params = [
            {"params": scratch_params, "lr": lr},
        ]

        if self.lr_mult_for_pretrained != 0.0: # full fine-tuning
            grouped_params.append(
            {"params": pretrained_params, "lr": lr * self.lr_mult_for_pretrained})

        embedder_params = []
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                embedder_params.extend(list(embedder.parameters()))
        
        if embedder_params != []:
            grouped_params.append({"params": embedder_params, "lr": lr})

        opt = self.instantiate_optimizer_from_config(grouped_params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt


    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # prepare ray directions and offsets for both support and query images.
        # also latents should be ready before forwarding the main model. 
        support_rgbs = batch["support_rgbs"]
        support_latents = self.encode_first_stage(support_rgbs)
        batch["support_latents"] = support_latents

        assert support_rgbs.shape[1] == 1, "only single support image is supported for now."
        batch["support_rgbs"] = support_rgbs.flatten(0, 1) # CLIP emb
        
        batch["support_rgbs_cond"] = support_rgbs # for depth cond
        batch["query_rgbs_cond"] = batch["query_rgbs"] # for depth cond

        # pass the empty text (additional condition can be added here)
        batch_size = support_rgbs.shape[0]
        batch["txt"] = [""] * batch_size
        
        # passing the offset and direction of rays of both support and query images.
        H_latents, W_latents = support_latents.shape[-2:]
        H_rgbs, W_rgbs = support_rgbs.shape[-2:]

        assert H_rgbs / H_latents == W_rgbs / W_latents, "The ratio of H and W should be the same."
        stride = int(H_rgbs / H_latents)

        support_rays_offset, support_rays_direction = get_rays(
            intrinsics=batch["support_intrinsics"],
            c2ws=batch["support_c2ws"],
            image_size=(H_rgbs, W_rgbs),
            stride=stride,
        )
        query_rays_offset, query_rays_direction = get_rays(
            intrinsics=batch["query_intrinsics"],
            c2ws=batch["query_c2ws"],
            image_size=(H_rgbs, W_rgbs),
            stride=stride,
        )

        batch["support_rays_offset"] = support_rays_offset
        batch["support_rays_direction"] = support_rays_direction
        batch["query_rays_offset"] = query_rays_offset
        batch["query_rays_direction"] = query_rays_direction

        return batch

    def validation_step(self, batch, *args: Any, **kwargs: Any):
        batch_size = batch["support_rgbs"].shape[0]
        loss, loss_dict = self.shared_step(batch)
        # To separate the logging from training.
        loss_dict = {"val/" + k: v for k,v in loss_dict.items()}
        self.log_dict(
            loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size
        )
        return loss
    
    def select_query_idx(self, batch, indices, mask, query_idx):
        ret_batch = dict()
        ret_batch.update(batch)

        num_invalid = len(mask[query_idx]) - mask[query_idx].sum()
        indices_batch = indices[query_idx]
        max_index = indices[mask].max()
        if num_invalid != 0:
            indices_batch[-num_invalid:] = torch.randint(0, max_index, (num_invalid,))

        for key in batch.keys():
            if "query" in key:
                ret_batch[key] = batch[key][:, indices_batch]
        return ret_batch
    
    def on_test_start(self) -> None:
        self.psnr = PSNR(reduction="none").to(self.device)
        self.ssim = SSIM(reduction="none").to(self.device)
        self.lpips = LPIPS(reduction="none", network="vgg").to(self.device)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        num_total_views = batch["query_rgbs"].shape[1]
        num_query_views = self.model.diffusion_model.num_query
        
        num_batch = math.ceil(float(num_total_views) / num_query_views)
        mask = torch.zeros(num_batch * num_query_views, dtype=torch.bool)
        mask[:num_total_views] = True
        mask = mask.reshape(num_batch, num_query_views)
        indices = torch.arange(num_batch * num_query_views).reshape(num_batch, num_query_views)
        
        for query_idx in range(num_batch):
            curr_batch = self.select_query_idx(batch, indices, mask, query_idx)
            x = self.get_input(curr_batch)
            x_latent = self.encode_first_stage(x)
            c, uc = self.conditioner.get_unconditional_conditioning(curr_batch)
            with self.ema_scope("Plotting"):
                samples = self.sample(c, uc, batch_size=x_latent.shape[0], shape=x_latent.shape[1:])
            
            samples = ((self.decode_first_stage(samples) + 1) / 2.).clamp(0, 1)
            support_rgbs = ((curr_batch["support_rgbs"] + 1) / 2.).clamp(0, 1)
            support_rgbs = support_rgbs[:, None].repeat(1, samples.shape[1], 1, 1, 1)
            query_rgbs = ((x + 1) / 2.).clamp(0, 1)

            samples = samples[:, mask[query_idx]]
            support_rgbs = support_rgbs[:, mask[query_idx]]
            query_rgbs = query_rgbs[:, mask[query_idx]]
            
            support_rgbs_viz = rearrange(support_rgbs, "b n c h w -> c (b n h) w")
            query_rgbs_viz = rearrange(query_rgbs, "b n c h w -> c (b n h) w")
            samples_viz = rearrange(samples, "b n c h w -> c (b n h) w")

            tensor_grid = torch.cat([support_rgbs_viz, query_rgbs_viz, samples_viz], dim=-1)

            viz = ToPILImage()(tensor_grid)
            viz.save(self.save_dir.joinpath(f"batch_{batch_idx}_query_{query_idx}_rank_{self.global_rank}.png"))

            samples_flatten = rearrange(samples, "b n c h w -> (b n) c h w")
            query_rgbs_flatten = rearrange(query_rgbs, "b n c h w -> (b n) c h w")

            self.test_step_outputs["psnr"].append(self.psnr(samples_flatten, query_rgbs_flatten))
            self.test_step_outputs["ssim"].append(self.ssim(samples_flatten, query_rgbs_flatten))
            self.test_step_outputs["lpips"].append(self.lpips(samples_flatten, query_rgbs_flatten))

    def on_test_epoch_end(self) -> None:

        psnr_cat = torch.cat(self.test_step_outputs["psnr"])
        ssim_cat = torch.cat(self.test_step_outputs["ssim"])
        lpips_cat = torch.cat(self.test_step_outputs["lpips"])

        psnr = psnr_cat.mean()
        ssim = ssim_cat.mean()
        lpips = lpips_cat.mean()

        with open(self.save_dir.joinpath("test_metrics.txt"), "w") as f:
            f.write(f"PSNR: {psnr}\n")
            f.write(f"SSIM: {ssim}\n")
            f.write(f"LPIPS: {lpips}\n")

        with open(self.save_dir.joinpath("psnr.json"), "w") as f:
            json.dump(psnr_cat.tolist(), f)

        with open(self.save_dir.joinpath("ssim.json"), "w") as f:
            json.dump(ssim_cat.tolist(), f)
        
        with open(self.save_dir.joinpath("lpips.json"), "w") as f:
            json.dump(lpips_cat.tolist(), f)

        self.test_step_outputs["psnr"].clear()
        self.test_step_outputs["ssim"].clear()
        self.test_step_outputs["lpips"].clear()
        return super().on_test_epoch_end()

    def init_from_pretrained_sd(self, sd_path: str, use_ema: bool) -> None:
        # similar to the init_from_ckpt but load pre-trained SD model only (not whole NVS model).
        if sd_path.endswith("ckpt"):
            sd = torch.load(sd_path, map_location="cpu")["state_dict"]
        elif sd_path.endswith("safetensors"):
            sd = load_safetensors(sd_path)
        else:
            raise NotImplementedError

        for key in list(sd.keys()):
            if key.startswith("model.diffusion_model"):
                sd[key.replace("model.diffusion_model", "model.diffusion_model.unet")] = sd.pop(key)
                
        missing, unexpected = self.load_state_dict(sd, strict=False)

        # # For checking whether the loaded weights are correct.
        # missing = [k for k in missing if not "image_attn_module" in k and not "transformer_blocks" in k]
        # missing = [k for k in missing if not "cross_view_attn_module" in k and not "transformer_blocks" in k]
        # missing = [k for k in missing if not "pe_blocks" in k]
        # missing = [k for k in missing if not k.startswith("conditioner")]

        # # cond stage model not required (they are using pre-trained weights)
        # unexpected = [k for k in unexpected if not k.startswith("cond_stage_model.")] 

    def load_controlnet_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path)
        sd_prefix = "control_model."
        model_prefix = "model.diffusion_model.control_model."
        controlnet_sd = {k.replace(sd_prefix, model_prefix): v for k, v in sd.items() if k.startswith("control_model.")}
        missing, unexpected = self.load_state_dict(controlnet_sd, strict=False)
    
    @torch.inference_mode()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        with_cfg: bool = True,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)
        # forwarding ucg_mask is necessary to be compatible with the original implementation.
        denoiser = lambda input, sigma, c, ucg_mask: self.denoiser(
            self.model, input, sigma, c, ucg_mask=ucg_mask, **kwargs
        )
        samples = (self.sampler if with_cfg else self.non_cfg_sampler)(denoiser, randn, cond, uc=uc)
        return samples

    @torch.inference_mode()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        # in the case of nvs adapter, no force-zero embedding option is used, since 
        # the unconditional conditioning is done inside the model (not in the conditioner).
        # See the implementation `sgm.modules.nvsadapter.threedim.ThreeDiMCondDrop` for more details.
        c, uc = self.conditioner.get_unconditional_conditioning(batch)

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        z = self.encode_first_stage(x)

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        support_rgbs = batch["support_rgbs"][:N, None].repeat(1, x.shape[1], 1, 1, 1)

        support_rgbs_viz = rearrange(support_rgbs, "b n c h w -> c (b n h) w")
        query_rgbs_viz = rearrange(x, "b n c h w -> c (b n h) w")

        if sample:
            with self.ema_scope("Plotting"):
                cfg_samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, with_cfg=True, **sampling_kwargs
                )
            cfg_samples = self.decode_first_stage(cfg_samples)
            with self.ema_scope("Plotting"):
                non_cfg_samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, with_cfg=False, **sampling_kwargs
                )
            non_cfg_samples = self.decode_first_stage(non_cfg_samples)

            cfg_samples_viz = rearrange(cfg_samples, "b n c h w -> c (b n h) w")
            non_cfg_samples_viz = rearrange(non_cfg_samples, "b n c h w -> c (b n h) w")

            log["viz"] = torch.cat([support_rgbs_viz, query_rgbs_viz, cfg_samples_viz, non_cfg_samples_viz], dim=-1)
        else:
            log["viz"] = torch.cat([support_rgbs_viz, query_rgbs_viz], dim=-1)

        return log

    @torch.inference_mode()
    def novel_view_sample(self, batch, num_query: int):
        c, uc = self.conditioner.get_unconditional_conditioning(batch)
        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k].to(self.device), (c, uc))

        support_latents = batch["support_latents"]
        batch_size, num_support, ch_latent, h_latent, w_latent = support_latents.shape

        shape = (num_query, ch_latent, h_latent, w_latent)

        sampling_kwargs = {}

        with self.ema_scope("Plotting"):
            samples = self.sample(c, shape=shape, uc=uc, batch_size=batch_size, with_cfg=True, **sampling_kwargs)

        samples = self.decode_first_stage(samples)
        samples = (samples * 0.5 + 0.5).clamp(0, 1)

        return samples

    @torch.no_grad()
    def encode_first_stage(self, x):
        if len(x.shape) == 5:
            bsz, num_image = x.shape[:2]
            x = rearrange(x, "b n c h w -> (b n) c h w")
            out = super().encode_first_stage(x)
            out = rearrange(out, "(b n) c h w -> b n c h w", b=bsz, n=num_image)
        else:
            out = super().encode_first_stage(x)
        return out
    
    @torch.no_grad()
    def decode_first_stage(self, z):
        if len(z.shape) == 5:
            bsz, num_image = z.shape[:2]
            z = rearrange(z, "b n c h w -> (b n) c h w")
            out = super().decode_first_stage(z)
            out = rearrange(out, "(b n) c h w -> b n c h w", b=bsz, n=num_image)
        else:
            out = super().decode_first_stage(z)
        return out

    # for score distillation sampling
    def encode_first_stage_with_grad(self, x):
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def encode_images(self, x):
        if len(x.shape) == 5:
            bsz, num_image = x.shape[:2]
            x = rearrange(x, "b n c h w -> (b n) c h w")
            out = self.encode_first_stage_with_grad(x)
            out = rearrange(out, "(b n) c h w -> b n c h w", b=bsz, n=num_image)
        else:
            out = self.encode_first_stage_with_grad(x)
        return out
