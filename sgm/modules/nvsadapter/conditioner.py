from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from open_clip.transformer import ResidualAttentionBlock

from sgm.util import instantiate_from_config
from sgm.modules.encoders.modules import AbstractEmbModel, FrozenOpenCLIPImageEmbedder
from sgm.modules.nvsadapter.midas.api import MiDaSInference


class MultipleGeneralConditioners(nn.Module):
    """
    Original GeneralConditioner concatenates the embeddings so that we cannot use
    them separately. This class allows us to use conditions separately.
    """

    def __init__(self, conditioners_config, ucg_rate=0.0):
        
        super(MultipleGeneralConditioners, self).__init__()
        conditioners, names = [], []

        for conditioner_config in conditioners_config:
            name = getattr(conditioner_config, "name", "")
            conditioner = instantiate_from_config(conditioner_config)
            conditioners.append(conditioner)
            names.append(name)

        self.conditioners = nn.ModuleList(conditioners)
        self.names = names

        embedders = []
        for conditioner in self.conditioners:
            for embedder in conditioner.embedders:
                embedders.append(embedder)
                
        self.embedders = embedders # to be compatible with GeneralConditioner
        self.ucg_rate = ucg_rate


    def __call__(self, batch: Dict):        
        conditions = dict()
        for name, conditioner in zip(self.names, self.conditioners):
            conditions.update({name + "/" + k : v for k, v in conditioner(batch).items()})

        return conditions

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        c_dict, uc_dict = dict(), dict()
        for name, conditioner in zip(self.names, self.conditioners):
            c, uc = conditioner.get_unconditional_conditioning(
                batch_c, batch_uc, force_uc_zero_embeddings
            )
            c_dict.update({name + "/" + k : v for k, v in c.items()})
            uc_dict.update({name + "/" + k : v for k, v in uc.items()})
        return c_dict, uc_dict


class ImageEmbedAttentionProjector(FrozenOpenCLIPImageEmbedder):
    """
    Uses the OpenCLIP vision transformer encoder with an additional attention layer (trainable)
    """
    def __init__(self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        proj_d_model=1280,
        proj_n_head=16,
    ):
        super(ImageEmbedAttentionProjector, self).__init__(
            arch=arch,
            version=version,
            device=device, 
            max_length=max_length,
            freeze=freeze,
            antialias=antialias,
            ucg_rate=ucg_rate,
            unsqueeze_dim=unsqueeze_dim,
            repeat_to_max_len=repeat_to_max_len,
            num_image_crops=num_image_crops,
            output_tokens=output_tokens,
        )

        assert not output_tokens, "This model does not output tokens"
        assert unsqueeze_dim, "This model requires unsqueeze_dim=True to work"

        # Residual Attention layer to project the image embedding
        self.projection_layer = ResidualAttentionBlock(
            d_model=proj_d_model,
            n_head=proj_n_head,
        )
        self.model.visual.transformer = nn.Sequential(
            self.model.visual.transformer,
            self.projection_layer,
        )


class ImageEmbedLinearProjector(FrozenOpenCLIPImageEmbedder):
    """
    Uses the OpenCLIP vision transformer encoder with an additional linear projection layer
    """
    def __init__(self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        out_proj=1024,
    ):
        super(ImageEmbedLinearProjector, self).__init__(
            arch=arch,
            version=version,
            device=device, 
            max_length=max_length,
            freeze=freeze,
            antialias=antialias,
            ucg_rate=ucg_rate,
            unsqueeze_dim=unsqueeze_dim,
            repeat_to_max_len=repeat_to_max_len,
            num_image_crops=num_image_crops,
            output_tokens=output_tokens,
        )

        assert not output_tokens, "This model does not output tokens"
        assert unsqueeze_dim, "This model requires unsqueeze_dim=True to work"

        # Residual Attention layer to project the image embedding
        self.projection_layer = nn.Linear(self.model.visual.output_dim, out_proj)
        nn.init.eye_(list(self.projection_layer.parameters())[0])
        nn.init.zeros_(list(self.projection_layer.parameters())[1])

    def forward(self, image, no_dropout=False):
        feat = super().forward(image, no_dropout)
        return self.projection_layer(feat)


class RayPosConditionEmbedder(AbstractEmbModel):
    def __init__(self, offset_deg, direction_deg, use_plucker):
        """
        config
            offset_deg (List[int]): the min and max degree of ray offset vectors
            direction_deg (List[int]): the min and max degree of ray direction vectors
            use_plucker (bool): whether to use plucker coordinate
        """

        super(RayPosConditionEmbedder, self).__init__()
        self.offset_deg = offset_deg
        self.direction_deg = direction_deg
        self.use_plucker = use_plucker

        self.posemb_out_dim = (
            self.offset_deg[1] - self.offset_deg[0] + self.direction_deg[1] - self.direction_deg[0]
        ) * 6 + 6

    def rays_to_plucker(self, offset, direction):
        """
        args:
            rays_offset (torch.Tensor, [bsz, N_rays, 3]): the offset vectors of rays.
            rays_direction (torch.Tensor, [bsz, N_rays, 3]): the direction vectors of rays.
        returns:
            cross_od (torch.Tensor, [bsz, N_rays, 3]): the cross vector of the offset vectors and
                normalized direction vectors.
            normalized_d (torch.Tensor, [bsz, N_rays, 3]): the normalized direction vectors.
        """
        normalized_d = F.normalize(direction, dim=-1)
        cross_od = torch.cross(offset, normalized_d, dim=-1)
        return cross_od, normalized_d

    def sinusoidal_emb(self, x, degree):
        """
        args:
            x (torch.Tensor, [bsz, H, W, 3]): the input tensor
            degree (Tuple[int]): the tuple of min and max degree for positional encoding
        returns:
            pos_emb (torch.Tensor, [bsz, H, W, (max_deg - min_deg) * 6 + 3]): positional embedding
        """
        device = x.device
        min_deg, max_deg = degree
        scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=device)

        scales = scales[None, None, None, None, :]
        xb = (x.unsqueeze(-1) * scales).flatten(-2, -1)
        emb = torch.sin(torch.cat([xb, xb + np.pi / 2], dim=-1))
        return torch.cat([x, emb], dim=-1)

    def forward_each(self, rays_offset, rays_direction):
        if self.use_plucker:
            # changes (o, d) -> (d/|d|, o x d)
            rays_offset, rays_direction = self.rays_to_plucker(rays_offset, rays_direction)

        # get sinusoidal embeddings for (offset/direction) & vectors.
        rays_offset_pe = self.sinusoidal_emb(rays_offset, self.offset_deg)
        rays_direction_pe = self.sinusoidal_emb(rays_direction, self.direction_deg)

        # by concatenating the PEs, get positional embs for rays.
        rays_pe = torch.cat([rays_offset_pe, rays_direction_pe], dim=-1)
        rays_pe = rearrange(rays_pe, "b n h w c -> b n c h w").contiguous()

        return rays_pe

    def forward(self, support_rays_offset, support_rays_direction, query_rays_offset, query_rays_direction):
        support_rays_pe = self.forward_each(support_rays_offset, support_rays_direction)
        query_rays_pe = self.forward_each(query_rays_offset, query_rays_direction)
        rays_pe = torch.cat([support_rays_pe, query_rays_pe], dim=1)
        return rays_pe


class ExtrinsicEmbedder(AbstractEmbModel):

    def __init__(self, deg):
        super(ExtrinsicEmbedder, self).__init__()
        self.deg = deg 

    def sinusoidal_emb(self, x, degree):
        """
        args:
            x (torch.Tensor, [bsz, H, W, 3]): the input tensor
            degree (Tuple[int]): the tuple of min and max degree for positional encoding
        returns:
            pos_emb (torch.Tensor, [bsz, H, W, (max_deg - min_deg) * 6 + 3]): positional embedding
        """
        device = x.device
        min_deg, max_deg = degree
        scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=device)

        scales = scales[None, None, None, :]
        xb = (x.unsqueeze(-1) * scales).flatten(-2, -1)
        emb = torch.sin(torch.cat([xb, xb + np.pi / 2], dim=-1))
        return torch.cat([x, emb], dim=-1)

    def forward(self, support_c2ws, query_c2ws, support_latents):

        h, w = support_latents.shape[-2:]
        support_rays_pe = repeat(
            self.sinusoidal_emb(
                rearrange(
                    support_c2ws, 
                    "b n i j -> b n (i j)"
                ), self.deg
            ), "b n c -> b n c h w", h=h, w=w
        )
        query_rays_pe = repeat(
            self.sinusoidal_emb(
                rearrange(
                    query_c2ws, 
                    "b n i j -> b n (i j)"
                ), self.deg
            ), "b n c -> b n c h w", h=h, w=w
        )
        rays_pe = torch.cat([support_rays_pe, query_rays_pe], dim=1)
        return rays_pe
    

class MiDASDepthConditioner(AbstractEmbModel):

    def __init__(self, model_type):
        super(MiDASDepthConditioner, self).__init__()
        self.midas_model = MiDaSInference(model_type)

    def zero_to_one(self, x):
        return (x + 1) / 2.
    
    def minus_one_to_one(self, x):
        return x * 2. - 1.

    def forward_each(self, x):
        bsz = x.shape[0]
        # midas uses [0, 1] images for inference
        x = self.zero_to_one(x)
        x = rearrange(x, "b n c h w -> (b n) c h w")
        # for batchfied inference
        midas_output = self.midas_model(x)
        midas_output = midas_output - midas_output.amin(dim=[1, 2, 3], keepdim=True)
        midas_output = midas_output / midas_output.amax(dim=[1, 2, 3], keepdim=True)
        # roll back the shape
        midas_output = rearrange(midas_output, "(b n) c h w -> b n c h w", b=bsz)
        return midas_output
    
    def HWC3(self, x):
        return torch.cat([x, x, x], axis=2)
    
    def forward(self, support_rgbs_cond, query_rgbs_cond):
        support_midas_output = self.forward_each(support_rgbs_cond)
        query_midas_output = self.forward_each(query_rgbs_cond)
        midas_output = torch.cat([support_midas_output, query_midas_output], dim=1)
        return self.HWC3(midas_output)
