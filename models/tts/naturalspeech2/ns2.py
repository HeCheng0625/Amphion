# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.tts.naturalspeech2.diffusion import Diffusion
from models.tts.naturalspeech2.diffusion_flow import DiffusionFlow
from models.tts.naturalspeech2.wavenet import DiffWaveNet
from models.tts.naturalspeech2.prior_encoder import PriorEncoder
from modules.naturalpseech2.transformers import TransformerEncoder, DiffTransformer
from encodec import EncodecModel
from einops import rearrange, repeat

import os
import json


class NaturalSpeech2Old(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.latent_dim = cfg.latent_dim
        self.query_emb_num = cfg.query_emb.query_token_num

        self.prior_encoder = PriorEncoder(cfg.prior_encoder)
        if cfg.diffusion.diffusion_type == "diffusion":
            self.diffusion = Diffusion(cfg.diffusion)
        elif cfg.diffusion.diffusion_type == "flow":
            self.diffusion = DiffusionFlow(cfg.diffusion)

        self.prompt_encoder = TransformerEncoder(cfg=cfg.prompt_encoder)
        if self.latent_dim != cfg.prompt_encoder.encoder_hidden:
            self.prompt_lin = nn.Linear(
                self.latent_dim, cfg.prompt_encoder.encoder_hidden
            )
            self.prompt_lin.weight.data.normal_(0.0, 0.02)
        else:
            self.prompt_lin = None

        self.query_emb = nn.Embedding(self.query_emb_num, cfg.query_emb.hidden_size)
        self.query_attn = nn.MultiheadAttention(
            cfg.query_emb.hidden_size, cfg.query_emb.head_num, batch_first=True
        )

        codec_model = EncodecModel.encodec_model_24khz()
        codec_model.set_target_bandwidth(12.0)
        codec_model.requires_grad_(False)
        self.quantizer = codec_model.quantizer

    @torch.no_grad()
    def code_to_latent(self, code):
        latent = self.quantizer.decode(code.transpose(0, 1))
        return latent

    def latent_to_code(self, latent, nq=16):
        residual = latent
        all_indices = []
        all_dist = []
        for i in range(nq):
            layer = self.quantizer.vq.layers[i]
            x = rearrange(residual, "b d n -> b n d")
            x = layer.project_in(x)
            shape = x.shape
            x = layer._codebook.preprocess(x)
            embed = layer._codebook.embed.t()
            dist = -(
                x.pow(2).sum(1, keepdim=True)
                - 2 * x @ embed
                + embed.pow(2).sum(0, keepdim=True)
            )
            indices = dist.max(dim=-1).indices
            indices = layer._codebook.postprocess_emb(indices, shape)
            dist = dist.reshape(*shape[:-1], dist.shape[-1])
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
            all_dist.append(dist)

        out_indices = torch.stack(all_indices)
        out_dist = torch.stack(all_dist)

        return out_indices, out_dist  # (nq, B, T); (nq, B, T, 1024)

    @torch.no_grad()
    def latent_to_latent(self, latent, nq=16):
        codes, _ = self.latent_to_code(latent, nq)
        latent = self.quantizer.vq.decode(codes)
        return latent

    def forward(
        self,
        code=None,
        pitch=None,
        duration=None,
        phone_id=None,
        phone_id_frame=None,
        frame_nums=None,
        ref_code=None,
        ref_frame_nums=None,
        phone_mask=None,
        mask=None,
        ref_mask=None,
    ):
        ref_latent = self.code_to_latent(ref_code)
        latent = self.code_to_latent(code)

        if self.latent_dim is not None:
            ref_latent = self.prompt_lin(ref_latent.transpose(1, 2))

        ref_latent = self.prompt_encoder(ref_latent, ref_mask, condition=None)
        spk_emb = ref_latent.transpose(1, 2)  # (B, d, T')

        spk_query_emb = self.query_emb(
            torch.arange(self.query_emb_num).to(latent.device)
        ).repeat(
            latent.shape[0], 1, 1
        )  # (B, query_emb_num, d)
        spk_query_emb, _ = self.query_attn(
            spk_query_emb,
            spk_emb.transpose(1, 2),
            spk_emb.transpose(1, 2),
            key_padding_mask=~(ref_mask.bool()),
        )  # (B, query_emb_num, d)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=mask,
            ref_emb=spk_emb,
            ref_mask=ref_mask,
            is_inference=False,
        )
        prior_condition = prior_out["prior_out"]  # (B, T, d)

        diff_out = self.diffusion(latent, mask, prior_condition, spk_query_emb)

        return diff_out, prior_out

    @torch.no_grad()
    def inference(
        self, ref_code=None, phone_id=None, ref_mask=None, inference_steps=1000
    ):
        ref_latent = self.code_to_latent(ref_code)

        if self.latent_dim is not None:
            ref_latent = self.prompt_lin(ref_latent.transpose(1, 2))

        ref_latent = self.prompt_encoder(ref_latent, ref_mask, condition=None)
        spk_emb = ref_latent.transpose(1, 2)  # (B, d, T')

        spk_query_emb = self.query_emb(
            torch.arange(self.query_emb_num).to(ref_latent.device)
        ).repeat(
            ref_latent.shape[0], 1, 1
        )  # (B, query_emb_num, d)
        spk_query_emb, _ = self.query_attn(
            spk_query_emb,
            spk_emb.transpose(1, 2),
            spk_emb.transpose(1, 2),
            key_padding_mask=~(ref_mask.bool()),
        )  # (B, query_emb_num, d)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=None,
            pitch=None,
            phone_mask=None,
            mask=None,
            ref_emb=spk_emb,
            ref_mask=ref_mask,
            is_inference=True,
        )
        prior_condition = prior_out["prior_out"]  # (B, T, d)

        z = torch.randn(
            prior_condition.shape[0], self.latent_dim, prior_condition.shape[1]
        ).to(ref_latent.device) / (1.20)
        x0 = self.diffusion.reverse_diffusion(
            z, None, prior_condition, inference_steps, spk_query_emb
        )

        return x0, prior_out

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self,
        code=None,
        pitch=None,
        duration=None,
        phone_id=None,
        ref_code=None,
        phone_mask=None,
        mask=None,
        ref_mask=None,
        n_timesteps=None,
        t=None,
    ):
        # o Only for debug

        ref_latent = self.code_to_latent(ref_code)
        latent = self.code_to_latent(code)

        if self.latent_dim is not None:
            ref_latent = self.prompt_lin(ref_latent.transpose(1, 2))

        ref_latent = self.prompt_encoder(ref_latent, ref_mask, condition=None)
        spk_emb = ref_latent.transpose(1, 2)  # (B, d, T')

        spk_query_emb = self.query_emb(
            torch.arange(self.query_emb_num).to(latent.device)
        ).repeat(
            latent.shape[0], 1, 1
        )  # (B, query_emb_num, d)
        spk_query_emb, _ = self.query_attn(
            spk_query_emb,
            spk_emb.transpose(1, 2),
            spk_emb.transpose(1, 2),
            key_padding_mask=~(ref_mask.bool()),
        )  # (B, query_emb_num, d)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=mask,
            ref_emb=spk_emb,
            ref_mask=ref_mask,
            is_inference=False,
        )
        prior_condition = prior_out["prior_out"]  # (B, T, d)

        diffusion_step = (
            torch.ones(
                latent.shape[0],
                dtype=latent.dtype,
                device=latent.device,
                requires_grad=False,
            )
            * t
        )
        diffusion_step = torch.clamp(diffusion_step, 1e-5, 1.0 - 1e-5)
        xt, _ = self.diffusion.forward_diffusion(
            x0=latent, diffusion_step=diffusion_step
        )
        # print(torch.abs(xt-latent).max(), torch.abs(xt-latent).mean(), torch.abs(xt-latent).std())

        x0 = self.diffusion.reverse_diffusion_from_t(
            xt, mask, prior_condition, n_timesteps, spk_query_emb, t_start=t
        )

        return x0, prior_out, xt


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer=None,
        encoder_hidden=None,
        encoder_head=None,
        conv_filter_size=None,
        conv_kernel_size=None,
        encoder_dropout=None,
        use_skip_connection=None,
        use_new_ffn=None,
        ref_in_dim=None,
        ref_out_dim=None,
        use_query_emb=None,
        num_query_emb=None,
        cfg=None,
    ):
        super().__init__()

        self.encoder_layer = (
            encoder_layer if encoder_layer is not None else cfg.encoder_layer
        )
        self.encoder_hidden = (
            encoder_hidden if encoder_hidden is not None else cfg.encoder_hidden
        )
        self.encoder_head = (
            encoder_head if encoder_head is not None else cfg.encoder_head
        )
        self.conv_filter_size = (
            conv_filter_size if conv_filter_size is not None else cfg.conv_filter_size
        )
        self.conv_kernel_size = (
            conv_kernel_size if conv_kernel_size is not None else cfg.conv_kernel_size
        )
        self.encoder_dropout = (
            encoder_dropout if encoder_dropout is not None else cfg.encoder_dropout
        )
        self.use_skip_connection = (
            use_skip_connection
            if use_skip_connection is not None
            else cfg.use_skip_connection
        )
        self.use_new_ffn = use_new_ffn if use_new_ffn is not None else cfg.use_new_ffn
        self.in_dim = ref_in_dim if ref_in_dim is not None else cfg.ref_in_dim
        self.out_dim = ref_out_dim if ref_out_dim is not None else cfg.ref_out_dim
        self.use_query_emb = (
            use_query_emb if use_query_emb is not None else cfg.use_query_emb
        )
        self.num_query_emb = (
            num_query_emb if num_query_emb is not None else cfg.num_query_emb
        )

        if self.in_dim != self.encoder_hidden:
            self.in_linear = nn.Linear(self.in_dim, self.encoder_hidden)
            self.in_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.in_dim = None

        if self.out_dim != self.encoder_hidden:
            self.out_linear = nn.Linear(self.encoder_hidden, self.out_dim)
            self.out_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.out_linear = None

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            encoder_hidden=self.encoder_hidden,
            encoder_head=self.encoder_head,
            conv_kernel_size=self.conv_kernel_size,
            conv_filter_size=self.conv_filter_size,
            encoder_dropout=self.encoder_dropout,
            use_new_ffn=self.use_new_ffn,
            use_cln=False,
            use_skip_connection=False,
            add_diff_step=False,
        )

        if self.use_query_emb:
            self.query_embs = nn.Embedding(self.num_query_emb, self.encoder_hidden)
            self.query_attn = nn.MultiheadAttention(
                self.encoder_hidden, self.encoder_hidden // 64, batch_first=True
            )

    def forward(self, x_ref, key_padding_mask=None):
        # x_ref: (B, T, d_ref)
        # key_padding_mask: (B, T)
        # return speaker embedding: x_spk
        # if self.use_query_embs: shape is (B, N_query, d_out)
        # else: shape is (B, 1, d_out)

        if self.in_linear != None:
            x = self.in_linear(x_ref)

        x = self.transformer_encoder(
            x, key_padding_mask=key_padding_mask, condition=None, diffusion_step=None
        )

        if self.use_query_emb:
            spk_query_emb = self.query_embs(
                torch.arange(self.num_query_emb).to(x.device)
            ).repeat(x.shape[0], 1, 1)
            spk_embs, _ = self.query_attn(
                query=spk_query_emb,
                key=x,
                value=x,
                key_padding_mask=(
                    ~(key_padding_mask.bool()) if key_padding_mask is not None else None
                ),
            )

            if self.out_linear != None:
                spk_embs = self.out_linear(spk_embs)

        else:
            spk_query_emb = None

        return spk_embs, x


class NaturalSpeech2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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

        self.prior_encoder = PriorEncoder(cfg=cfg.prior_encoder)

        self.reset_parameters()

    def forward(
        self,
        x=None,
        pitch=None,
        duration=None,
        phone_id=None,
        x_ref=None,
        phone_mask=None,
        x_mask=None,
        x_ref_mask=None,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=x_mask,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=False,
        )

        condition_embedding = prior_out["prior_out"]

        diff_out = self.diffusion(
            x=x,
            condition_embedding=condition_embedding,
            x_mask=x_mask,
            reference_embedding=reference_embedding,
        )

        return diff_out, prior_out

    @torch.no_grad()
    def inference(
        self,
        phone_id=None,
        x_ref=None,
        x_ref_mask=None,
        inference_steps=1000,
        sigma=1.2,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=None,
            pitch=None,
            phone_mask=None,
            mask=None,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=True,
        )

        condition_embedding = prior_out["prior_out"]

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

        return x0, prior_out

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self,
        x,
        pitch=None,
        duration=None,
        phone_id=None,
        x_ref=None,
        phone_mask=None,
        x_mask=None,
        x_ref_mask=None,
        inference_steps=None,
        t=None,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        diffusion_step = (
            torch.ones(
                x.shape[0],
                dtype=x.dtype,
                device=x.device,
                requires_grad=False,
            )
            * t
        )
        diffusion_step = torch.clamp(diffusion_step, 1e-5, 1.0 - 1e-5)
        xt, _ = self.diffusion.forward_diffusion(x0=x, diffusion_step=diffusion_step)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=x_mask,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=True,
        )

        condition_embedding = prior_out["prior_out"]

        x0 = self.diffusion.reverse_diffusion_from_t(
            z=xt,
            condition_embedding=condition_embedding,
            x_mask=x_mask,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
            t_start=t,
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
