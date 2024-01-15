import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange

from sgm.modules.attention import SpatialTransformer, BasicTransformerBlock
from sgm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, TimestepBlock
from sgm.modules.diffusionmodules.util import timestep_embedding, checkpoint
from sgm.util import instantiate_from_config


class ThreeDiMCondDrop(nn.Module):
    def __init__(self, ucg_rate):
        super(ThreeDiMCondDrop, self).__init__()
        self.ucg_rate = ucg_rate
        assert ucg_rate < 1.0, "ucg_rate must be less than 1.0"

    def forward(self, text_context, image_context, rays_context, support_latents, ucg_mask):

        bsz = rays_context.shape[0]

        if self.training:
            assert ucg_mask is None, "ucg_mask must be None for training"
            ucg_mask = torch.rand(bsz) < self.ucg_rate
            while ucg_mask.all(): 
                # occasionally all the samples are selected as unconditional due to the randomness
                # such cases cause error since some operations are skipped. (gradients are not computed for the skipped modules)
                # to avoid this, we re-sample the ucg_mask until at least one sample is selected as conditional
                ucg_mask = torch.rand(bsz) < self.ucg_rate
        else:
            if ucg_mask is None:
                ucg_mask = torch.zeros(bsz, dtype=torch.bool)

        # zero out the rays_context
        rays_context[ucg_mask] = 0.
        
        if support_latents is not None:
            # switch to the random latents with gaussian distribution
            support_latents[ucg_mask] = torch.randn_like(support_latents[ucg_mask])

        if image_context is not None:
            image_context[ucg_mask] = 0.

        # in the case of text context, the dataloader itself returns the unconditional samples only.

        return text_context, image_context, rays_context, support_latents, ucg_mask


class ThreeDiMTimestepEmbedSequential(TimestepEmbedSequential):

    def forward(
        self,
        x,
        emb,
        text_context=None,
        image_context=None,
        posemb=None,
        num_support=None,
        num_query=None,
        **kwargs,
    ):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ThreeDiMSpatialTransformer):
                x = layer(x, text_context, image_context, posemb, num_support, num_query)
            elif isinstance(layer, SpatialTransformer):
                assert False, "SpatialTransformer is not supported in this module."
            else:
                x = layer(x)
        return x


class ThreeDiMSpatialTransformer(SpatialTransformer):

    def __init__(
        # from the super class
        self, 
        in_channels,
        n_heads,
        d_head,
        # for additional layers (view-attn, image-attn)
        image_context_dim: int, 
        image_attn_mode: str | None,
        query_composer_mode: str | None,
        query_emb_scale: float,
        imgemb_to_text: bool,
        view_attn_mode: str,
        attn_out_zero_init: bool,
        feat_resolution: int,
        # from the super class
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        sdp_backend=None,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            context_dim=context_dim,
            disable_self_attn=disable_self_attn,
            use_linear=use_linear,
            attn_type=attn_type,
            use_checkpoint=use_checkpoint,
            sdp_backend=sdp_backend,
        )

        inner_dim = n_heads * d_head
        # override the transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                ThreeDiMBasicTransformerBlock(
                    dim=inner_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    image_context_dim=image_context_dim,
                    image_attn_mode=image_attn_mode,
                    imgemb_to_text=imgemb_to_text,
                    view_attn_mode=view_attn_mode,
                    query_composer_mode=query_composer_mode,
                    query_emb_scale=query_emb_scale,
                    attn_out_zero_init=attn_out_zero_init,
                    feat_resolution=feat_resolution,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    checkpoint=use_checkpoint,
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )

    def forward(self, x, text_context=None, image_context=None, posemb=None, num_support=None, num_query=None):
        
        assert posemb is not None, "posemb must be provided"
        
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        posemb = rearrange(posemb, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, text_context=text_context, image_context=image_context, posemb=posemb,
                      num_support=num_support, num_query=num_query)

        if self.use_linear:
            x = self.proj_out(x)
        
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        
        if not self.use_linear:
            x = self.proj_out(x)
        
        return x + x_in


class ThreeDiMBasicTransformerBlock(BasicTransformerBlock):
    
    def __init__(
        self,
        # from the super class
        dim,
        n_heads,
        d_head,
        # for additional layers (view-attn, image-attn)
        image_context_dim: int, 
        image_attn_mode: str | None,
        imgemb_to_text: bool,
        view_attn_mode: str,
        query_composer_mode: str | None,
        query_emb_scale: float,
        attn_out_zero_init: bool,
        feat_resolution: int,
        # from the super class
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
    ):
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
            gated_ff=gated_ff,
            checkpoint=checkpoint,
            disable_self_attn=disable_self_attn,
            attn_mode=attn_mode,
            sdp_backend=sdp_backend,
        )

        self.image_attn_mode = image_attn_mode
        self.imgemb_to_text = imgemb_to_text
        self.view_attn_mode = view_attn_mode
        self.query_composer_mode = query_composer_mode

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.cross_view_attn_module = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=dim,
            backend=sdp_backend,
        )
        if attn_out_zero_init:
            # initializing the output layer to zero
            init.zeros_(self.cross_view_attn_module.to_out[0].weight)
            init.zeros_(self.cross_view_attn_module.to_out[0].bias)

        if self.image_attn_mode is not None:
            self.image_attn_module = attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=image_context_dim,
                backend=sdp_backend,
            )
            if attn_out_zero_init:
                init.zeros_(self.image_attn_module.to_out[0].weight)
                init.zeros_(self.image_attn_module.to_out[0].bias)

        if self.query_composer_mode == "learnable_emb":
            self.support_query_attn = attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=dim,
                backend=sdp_backend,
            )
            self.query_out_attn = attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=dim,
                backend=sdp_backend,
            )
            if attn_out_zero_init:
                init.zeros_(self.query_out_attn.to_out[0].weight)
                init.zeros_(self.query_out_attn.to_out[0].bias)

            emb = torch.randn((1, int(feat_resolution * feat_resolution * query_emb_scale), dim))
            self.register_parameter("support_emb", nn.Parameter(emb, requires_grad=True))

        elif self.query_composer_mode == "all_to_all":
            self.query_attn = attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=dim,
                backend=sdp_backend,
            )
            if attn_out_zero_init:
                init.zeros_(self.query_attn.to_out[0].weight)
                init.zeros_(self.query_attn.to_out[0].bias)
        
        else: 
            raise NameError(f"Unknown query composer mode: {self.query_composer_mode}")


    def compose_query_views(self, x, num_support, num_query):
        num_total_views = x.shape[0]
        bsz = num_total_views // (num_support + num_query)
        assert num_total_views % (num_support + num_query) == 0, "num_total_views must be divisible by num_support + num_query"
        support_x, query_x = torch.split(x, [bsz*num_support, bsz*num_query], dim=0)
        support_x = support_x.unflatten(0, (bsz, num_support))
        query_x = query_x.unflatten(0, (bsz, num_query))
        
        if self.query_composer_mode == "learnable_emb":
            sq_attn_query = self.support_emb.repeat((bsz, 1, 1))

            context_query_updater = (
                self.support_query_attn(
                    sq_attn_query,
                    context=query_x.flatten(1, 2)
                ).unflatten(1, (num_support, -1))
            )

            out = self.query_out_attn(
                query_x.flatten(1, 2),
                context=context_query_updater.flatten(1, 2)
            ).unflatten(1, (num_query, -1))
        
        elif self.query_composer_mode == "all_to_all":
            out = self.query_attn(
                query_x.flatten(1, 2),
                context=query_x.flatten(1, 2)
            ).unflatten(1, (num_query, -1))

        else:
            raise NameError(f"Unknown query composer mode: {self.query_composer_mode}")

        support_x_attn = torch.zeros_like(support_x.flatten(0, 1))
        query_x_attn = out.flatten(0, 1)

        return torch.cat([support_x_attn, query_x_attn], dim=0)

    def forward(
        self, x, text_context, image_context, posemb, num_support, num_query, additional_tokens=None
    ):
        return checkpoint(
            self._forward,
            (x, text_context, image_context, posemb, num_support, num_query),
            self.parameters(), self.checkpoint
        )

    def _forward(
        self, x, text_context, image_context, posemb, num_support, num_query,
        additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        if self.imgemb_to_text:
            text_context = image_context

        # attn1
        x = (
            self.attn1(
                self.norm1(x),
                context=text_context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )

        if self.query_composer_mode == "learnable_emb":
            x = self.compose_query_views(x + posemb, num_support=num_support, num_query=num_query) + x

        # view-cross-attn
        x = (
            self.cross_view_attn(self.cross_view_attn_module, x + posemb,
            num_support=num_support, num_query=num_query) + x
        )

        # image-attn
        # no image attn needed when self.imgemb_to_text is True
        if not self.imgemb_to_text and self.image_attn_mode is not None:
            x = (
                self.image_attn(self.image_attn_module, x,
                                context=image_context, num_support=num_support, num_query=num_query)
                + x
            )
            
        # attn2
        x = (
            self.attn2(
                self.norm2(x), context=text_context, additional_tokens=additional_tokens
            )
            + x
        )

        x = self.ff(self.norm3(x)) + x
        return x

    def image_attn(self, module, x, context, num_support, num_query):
        num_total_views = x.shape[0]
        bsz = num_total_views // (num_support + num_query)
        assert num_total_views % (num_support + num_query) == 0, "num_total_views must be divisible by num_support + num_query"
        support_x, query_x = torch.split(x, [bsz*num_support, bsz*num_query], dim=0)
        support_context, query_context = torch.split(context, [bsz*num_support, bsz*num_query], dim=0)

        if self.image_attn_mode == "query_only":
            support_x_attn = support_x
            query_x_attn = module(query_x, context=query_context)
        elif self.image_attn_mode == "support_query":
            support_x_attn = module(support_x, context=support_context)
            query_x_attn = module(query_x, context=query_context)
        else:
            raise NameError(f"Unknown image attention mode: {self.image_attn_mode}")
        
        return torch.cat([support_x_attn, query_x_attn], dim=0)
    
    def cross_view_attn(self, module, x, num_support, num_query):
        num_total_views = x.shape[0]
        bsz = num_total_views // (num_support + num_query)
        assert num_total_views % (num_support + num_query) == 0, "num_total_views must be divisible by num_support + num_query"
        support_x, query_x = torch.split(x, [bsz*num_support, bsz*num_query], dim=0)

        if self.view_attn_mode == "support_skip":
            support_x_attn = torch.zeros_like(support_x)
            query_x_attn = module(query_x, context=support_x)
        elif self.view_attn_mode == "support_sattn":
            support_x_attn = module(support_x, context=support_x)
            query_x_attn = module(query_x, context=support_x)
        elif self.view_attn_mode == "support_sattn_mq":
            support_x_attn = module(support_x, context=support_x)
            support_x_repeat = support_x[:, None].repeat((1, num_query, 1, 1)).flatten(0, 1)
            query_x_attn = module(query_x, context=support_x_repeat)
        elif self.view_attn_mode == "support_xattn":
            support_x_attn = module(support_x, context=query_x)
            query_x_attn = module(query_x, context=support_x)
        else:
            raise NameError(f"Unknown view attention mode: {self.view_attn_mode}")
        
        return torch.cat([support_x_attn, query_x_attn], dim=0)


class ThreeDiMAdapter(nn.Module):
    def __init__(
        self, 
        sd_config, 
        cond_drop_config, 
        image_attn_mode: str | None,
        imgemb_to_text: bool, # if true, use the image context as the text context
        view_attn_mode: str,
        num_support: int,
        num_query: int,
        attn_out_zero_init: bool,
        image_context_dim: int, 
        posemb_dim: int,
        max_timesteps: int = 1000,
        control_model_config  = None,
        query_composer_mode: str | None = None,
        query_emb_scale: float = 1.0,
        use_checkpoint: bool = False,
        latent_resolution: int = 32,
    ):
        super(ThreeDiMAdapter, self).__init__()

        self.unet = instantiate_from_config(sd_config)

        if control_model_config is not None:
            self.control_model = instantiate_from_config(control_model_config)

        # store pre-trained model parameters
        self.unet_param_name = [name for name, parameter in self.named_parameters()]

        self._set_unet_info()
        self._wrap_unet(
            image_context_dim=image_context_dim, 
            image_attn_mode=image_attn_mode, 
            imgemb_to_text=imgemb_to_text,
            view_attn_mode=view_attn_mode, 
            query_composer_mode=query_composer_mode,
            query_emb_scale=query_emb_scale,
            attn_out_zero_init=attn_out_zero_init,
            latent_resolution=latent_resolution,
            use_checkpoint=use_checkpoint,
        )
        self._set_posemb(posemb_dim)
        self.cond_drop = instantiate_from_config(cond_drop_config)

        self.max_timesteps = max_timesteps
        self.num_support = num_support
        self.num_query = num_query
    
    def scratch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not name in self.unet_param_name:
                params.append(param)
        return params

    def pretrained_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if name in self.unet_param_name:
                params.append(param)
        return params

    def _set_unet_info(self):
        # get channel dimension for each module.
        num_res_blocks = self.unet.num_res_blocks
        curr_stride, strides = 0, []
        for num_resblock in num_res_blocks:
            for _ in range(num_resblock):
                strides.append(curr_stride)
            strides.append(curr_stride)
            curr_stride += 1

        self.num_updown_blocks = len(num_res_blocks)
        self.num_unet_modules = len(self.unet.input_blocks)
        self.strides = strides

    def _get_spatial_transformer_kwargs(self, block):
        # no need to get the `use_checkpoint` from the original block since the block 
        return dict(
            in_channels=block.in_channels,
            n_heads=block.n_heads,
            d_head=block.d_head,
            depth=block.depth,
            dropout=block.dropout,
            context_dim=block.context_dim,
            disable_self_attn=block.disable_self_attn,
            use_linear=block.use_linear,
            attn_type=block.attn_type,
            sdp_backend=block.sdp_backend,
        )

    def _wrap_unet(
        self, 
        image_context_dim, 
        image_attn_mode, 
        imgemb_to_text,
        view_attn_mode, 
        query_composer_mode,
        query_emb_scale,
        attn_out_zero_init,
        latent_resolution,
        use_checkpoint,
    ):
        
        spatial_transformer_kwargs = dict(
            image_context_dim=image_context_dim,
            image_attn_mode=image_attn_mode,
            imgemb_to_text=imgemb_to_text,
            attn_out_zero_init=attn_out_zero_init,
            view_attn_mode=view_attn_mode,
            query_composer_mode=query_composer_mode,
            query_emb_scale=query_emb_scale,
            use_checkpoint=use_checkpoint,
        )

        # input blocks
        for module_list_idx in range(self.num_unet_modules):
            module_list = self.unet.input_blocks[module_list_idx]
            for module_idx in range(len(module_list)):
                input_block = self.unet.input_blocks[module_list_idx][module_idx]
                if isinstance(input_block, SpatialTransformer):
                    self.unet.input_blocks[module_list_idx][module_idx] = ThreeDiMSpatialTransformer(
                        **self._get_spatial_transformer_kwargs(input_block),
                        **spatial_transformer_kwargs,
                        feat_resolution=int(latent_resolution / 2 ** self.strides[module_list_idx]),
                    )

            if isinstance(module_list, TimestepEmbedSequential):
                self.unet.input_blocks[module_list_idx] = ThreeDiMTimestepEmbedSequential(*module_list)
            else:
                raise TypeError(f"input block must be TimestepEmbedSequential, but got {type(module_list)}")

        # output blocks
        for module_list_idx in range(self.num_unet_modules):
            module_list = self.unet.output_blocks[module_list_idx]
            for module_idx in range(len(module_list)):
                output_block = self.unet.output_blocks[module_list_idx][module_idx]
                if isinstance(output_block, SpatialTransformer):
                    self.unet.output_blocks[module_list_idx][module_idx] = ThreeDiMSpatialTransformer(
                        **self._get_spatial_transformer_kwargs(output_block),
                        **spatial_transformer_kwargs,
                        feat_resolution=int(latent_resolution / 2 ** self.strides[::-1][module_list_idx]),
                    )

            if isinstance(module_list, TimestepEmbedSequential):
                self.unet.output_blocks[module_list_idx] = ThreeDiMTimestepEmbedSequential(*module_list)
            else:
                raise TypeError(f"output block must be TimestepEmbedSequential, but got {type(module_list)}")

        # middle block
        if isinstance(self.unet.middle_block, TimestepEmbedSequential):
            for module_idx in range(len(self.unet.middle_block)):
                middle_block = self.unet.middle_block[module_idx]
                if isinstance(middle_block, SpatialTransformer):
                    self.unet.middle_block[module_idx] = ThreeDiMSpatialTransformer(
                        **self._get_spatial_transformer_kwargs(middle_block),
                        **spatial_transformer_kwargs,
                        feat_resolution=int(latent_resolution / 2 ** self.strides[-1]),
                    )
            self.unet.middle_block = ThreeDiMTimestepEmbedSequential(*self.unet.middle_block)
        else:
            raise TypeError(f"middle block must be TimestepEmbedSequential, but got {type(self.unet.middle_block)}")

    def _set_posemb(self, posemb_dim):
        channel_mult, model_channels = self.unet.channel_mult, self.unet.model_channels
        pe_blocks = []
        strides = list(range(self.num_updown_blocks))
        for idx in range(self.num_updown_blocks):
            stride = 2 ** strides[idx]
            pe_blocks.append(
                nn.Conv2d(
                    in_channels=posemb_dim,
                    out_channels=channel_mult[idx] * model_channels,
                    kernel_size=(stride, stride),
                    stride=stride, 
                )
            )
        self.pe_blocks = nn.ModuleList(pe_blocks)

    def forward(
        self,
        inputs,  # query latents
        timesteps,
        control_context,
        txt_context,
        image_context,
        rays_context,
        support_latents,
        ucg_mask = None,
    ):
        txt_context, image_context, rays_context, support_latents, ucg_mask = self.cond_drop(
            txt_context, image_context, rays_context, support_latents, ucg_mask
        )

        batch_size = timesteps.shape[0]
        num_supports = 1 # only one support image is supported
        assert support_latents.shape[1] == 1, "only one support image is supported"
        num_queries = inputs.shape[1]

        # concatenate timesteps
        query_timesteps = timesteps
        support_timesteps = torch.zeros_like(query_timesteps)
        support_timesteps[ucg_mask] = self.max_timesteps - 1
        timesteps = torch.cat(
            [
                support_timesteps.repeat_interleave(num_supports, dim=0),
                query_timesteps.repeat_interleave(num_queries, dim=0),
            ]
        )

        # concatenate latents 
        if support_latents is not None:
            latents = torch.cat([support_latents.flatten(0, 1), inputs.flatten(0, 1)], dim=0)

        # the same for support and query
        if txt_context is not None:
            support_txt_context = txt_context[:, None].repeat((1, num_supports, 1, 1)).flatten(0, 1)
            query_txt_context = txt_context[:, None].repeat((1, num_queries, 1, 1)).flatten(0, 1)
            txt_context = torch.cat([support_txt_context, query_txt_context], dim=0)

        if image_context is not None:
            support_image_context = image_context[:, None].repeat((1, num_supports, 1, 1)).flatten(0, 1)
            query_image_context = image_context[:, None].repeat((1, num_queries, 1, 1)).flatten(0, 1)
            image_context = torch.cat([support_image_context, query_image_context], dim=0)

        if rays_context is not None:
            support_rays_context = rays_context[:, :num_supports].flatten(0, 1)
            query_rays_context = rays_context[:, num_supports:].flatten(0, 1)
            rays_context = torch.cat([support_rays_context, query_rays_context], dim=0)
    
        if control_context is not None:
            support_control_context = control_context[:, :num_supports].flatten(0, 1)
            query_control_context = control_context[:, num_supports:].flatten(0, 1)
            control_context = torch.cat([support_control_context, query_control_context], dim=0)

        model_output = self.xunet_forward(
            latents=latents,
            timesteps=timesteps,
            control_context=control_context,
            txt_context=txt_context,
            image_context=image_context,
            rays_context=rays_context,
            num_support=num_supports,
            num_query=num_queries,
        )

        return model_output[support_latents.shape[0]:].unflatten(0, (batch_size, num_queries))

    def xunet_forward(
        self, 
        latents,
        timesteps,
        control_context,
        txt_context,
        image_context,
        rays_context,
        num_support,
        num_query,
    ):

        hs = []
        t_emb = timestep_embedding(timesteps, self.unet.model_channels, repeat_only=False)
        emb = self.unet.time_embed(t_emb)

        if hasattr(self, "control_model"):
            controls = self.control_model(latents, control_context, timesteps, txt_context)
        else:
            controls = None

        h = latents
        for block_idx in range(self.num_unet_modules):
            stride_lev = self.strides[block_idx]
            posemb = self.pe_blocks[stride_lev](rays_context)
            h = self.unet.input_blocks[block_idx](h, emb, txt_context, image_context, posemb, num_support, num_query)
            hs.append(h)

        h = self.unet.middle_block(h, emb, txt_context, image_context, posemb, num_support, num_query)
        if controls is not None:
            h = h + controls.pop()

        for block_idx in range(self.num_unet_modules):
            skip_connect = hs.pop()
            if controls is not None:
                skip_connect = skip_connect + controls.pop()
            h = torch.cat([h, skip_connect], dim=1)
            stride_lev = self.strides[::-1][block_idx]
            posemb = self.pe_blocks[stride_lev](rays_context)
            h = self.unet.output_blocks[block_idx](h, emb, txt_context, image_context, posemb, num_support, num_query)
            
        h = h.type_as(latents)
        return self.unet.out(h)
