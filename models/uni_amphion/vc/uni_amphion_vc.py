import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange

from models.uni_amphion.base.transformer import ReferenceEncoder, DiffTransformer
from models.uni_amphion.base.wavenet import DiffWaveNet
from models.uni_amphion.base.diffusion import Diffusion


class UniAmphionVC(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        cfg = cfg.model
        self.cfg = cfg

        self.reference_encoder = ReferenceEncoder(cfg=cfg.reference_encoder)
        if cfg.diffusion.diff_model_type == "Transformer":
            self.diffusion = Diffusion(
                cfg=cfg.diffusion,
                diff_model=DiffTransformer(cfg=cfg.diffusion.diff_transformer),
            )
        elif cfg.diffusion.diff_model_type == "WaveNet":
            self.diffusion = Diffusion(
                cfg=cfg.diffusion,
                diff_model=DiffWaveNet(cfg=cfg.diffusion.diff_wavenet),
            )
        else:
            raise NotImplementedError()

        self.content_f0_enc = nn.Sequential(
            nn.LayerNorm(cfg.vc_feature.content_feature_dim + 1),
            Rearrange("b t d -> b d t"),
            nn.Conv1d(
                cfg.vc_feature.content_feature_dim + 1,
                cfg.vc_feature.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            Rearrange("b d t -> b t d"),
        )

        self.reset_parameters()

    def forward(
        self,
        x=None,
        content_feature=None,
        pitch=None,
        x_ref=None,
        x_mask=None,
        x_ref_mask=None,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        condition_embedding = torch.cat([content_feature, pitch[:, :, None]], dim=-1)
        condition_embedding = self.content_f0_enc(condition_embedding)

        diff_out = self.diffusion(
            x=x,
            condition_embedding=condition_embedding,
            x_mask=x_mask,
            reference_embedding=reference_embedding,
        )

        return diff_out

    @torch.no_grad()
    def inference(
        self,
        content_feature=None,
        pitch=None,
        x_ref=None,
        x_ref_mask=None,
        inference_steps=1000,
        sigma=1.2,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        condition_embedding = torch.cat([content_feature, pitch[:, :, None]], dim=-1)
        condition_embedding = self.content_f0_enc(condition_embedding)

        bsz, l, _ = condition_embedding.shape
        if self.cfg.diffusion.diff_model_type == "Transformer":
            z = (
                torch.randn(bsz, l, self.cfg.diffusion.diff_transformer.in_dim).to(
                    condition_embedding.device
                )
                / sigma
            )
        elif self.cfg.diffusion.diff_model_type == "WaveNet":
            z = (
                torch.randn(bsz, l, self.cfg.diffusion.diff_wavenet.input_size).to(
                    condition_embedding.device
                )
                / sigma
            )

        x0 = self.diffusion.reverse_diffusion(
            z=z,
            condition_embedding=condition_embedding,
            x_mask=None,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
        )

        return x0

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)
