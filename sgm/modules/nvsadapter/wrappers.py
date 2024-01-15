import torch

from sgm.modules.diffusionmodules.denoiser import DiscreteDenoiser, append_dims
from sgm.modules.diffusionmodules.sampling import EulerEDMSampler
from sgm.modules.diffusionmodules.wrappers import IdentityWrapper
from sgm.modules.diffusionmodules.guiders import VanillaCFG


class NVSAdapterDiscreteDenoiser(DiscreteDenoiser):
    def __call__(self, network, input, sigma, cond, ucg_mask=None):
        # follows the original implementation but adding ucg_mask for forwarding
        # for training, ucg_mask is None since CondDrop will cover the unconditional cases
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return network(input * c_in, c_noise, cond, ucg_mask=ucg_mask) * c_out + input * c_skip
    

class NVSAdapterEulerEDMSampler(EulerEDMSampler):
    def denoise(self, x, denoiser, sigma, cond, uc):
        batch_size = x.shape[0]
        # first #bsz elements are conditional forward
        # last #bsz elements are unconditional forward
        ucg_mask = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).to(device=x.device, dtype=torch.bool)
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), ucg_mask=ucg_mask)
        denoised = self.guider(denoised, sigma)
        return denoised


class NVSAdapterWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        txt_context = {k[len("txt/"):]: v for k, v in c.items() if k.startswith("txt/")}
        image_context = {k[len("image/"):]: v for k, v in c.items() if k.startswith("image/")}
        rays_context = {k[len("ray/"):]: v for k, v in c.items() if k.startswith("ray/")}
        support_latents = {k[len("support_latents/"):]: v for k, v in c.items() if k.startswith("support_latents/")}
        control_context = {k[len("control/"):]: v for k, v in c.items() if k.startswith("control/")}

        assert rays_context is not None, "ray conditions are required"
        assert support_latents is not None, "support latents are required"

        image_emb = image_context.get("crossattn", None)
        txt_emb = txt_context.get("crossattn", None)

        rays_emb = rays_context.get("concat", None)
        support_latents = support_latents.get("concat", None) 
        control_context = control_context.get("concat", None)

        return self.diffusion_model(
            x,
            timesteps=t,
            control_context=control_context,
            txt_context=txt_emb,
            image_context=image_emb,
            rays_context=rays_emb,
            support_latents=support_latents,
            **kwargs,
        )


class NVSAdapterCFG(VanillaCFG):
    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["image/crossattn", "txt/crossattn", "ray/concat", "support_latents/concat", "control/concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]

        return torch.cat([x] * 2), torch.cat([s] * 2), c_out
