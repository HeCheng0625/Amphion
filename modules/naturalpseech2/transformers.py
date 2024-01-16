# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Linear2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim * 2)
        self.linear_2 = nn.Linear(dim * 2, dim)
        self.linear_1.weight.data.normal_(0.0, 0.02)
        self.linear_2.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.in_dim = normalized_shape
        self.norm = nn.LayerNorm(self.in_dim, eps=eps, elementwise_affine=False)
        self.style = nn.Linear(self.in_dim, self.in_dim * 2)
        self.style.bias.data[: self.in_dim] = 1
        self.style.bias.data[self.in_dim :] = 0

    def forward(self, x, condition):
        # x: (B, T, d); condition: (B, T, d)

        style = self.style(torch.mean(condition, dim=1, keepdim=True))

        gamma, beta = style.chunk(2, -1)

        out = self.norm(x)

        out = gamma * out + beta
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

        self.dropout = dropout
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return F.dropout(x, self.dropout, training=self.training)


class TransformerFFNLayer(nn.Module):
    def __init__(
        self, encoder_hidden, conv_filter_size, conv_kernel_size, encoder_dropout
    ):
        super().__init__()

        self.encoder_hidden = encoder_hidden
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout

        self.ffn_1 = nn.Conv1d(
            self.encoder_hidden,
            self.conv_filter_size,
            self.conv_kernel_size,
            padding=self.conv_kernel_size // 2,
        )
        self.ffn_1.weight.data.normal_(0.0, 0.02)
        self.ffn_2 = nn.Linear(self.conv_filter_size, self.encoder_hidden)
        self.ffn_2.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        # x: (B, T, d)
        x = self.ffn_1(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (B, T, d) -> (B, d, T) -> (B, T, d)
        x = F.silu(x)
        x = F.dropout(x, self.encoder_dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class TransformerFFNLayerOld(nn.Module):
    def __init__(
        self, encoder_hidden, conv_filter_size, conv_kernel_size, encoder_dropout
    ):
        super().__init__()

        self.encoder_hidden = encoder_hidden
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout

        self.ffn_1 = nn.Linear(self.encoder_hidden, self.conv_filter_size)
        self.ffn_1.weight.data.normal_(0.0, 0.02)
        self.ffn_2 = nn.Linear(self.conv_filter_size, self.encoder_hidden)
        self.ffn_2.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        x = self.ffn_1(x)
        x = F.silu(x)
        x = F.dropout(x, self.encoder_dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        encoder_hidden,
        encoder_head,
        conv_filter_size,
        conv_kernel_size,
        encoder_dropout,
        use_cln,
        use_skip_connection,
        use_new_ffn,
        add_diff_step,
    ):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.encoder_head = encoder_head
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout
        self.use_cln = use_cln
        self.use_skip_connection = use_skip_connection
        self.use_new_ffn = use_new_ffn
        self.add_diff_step = add_diff_step

        if not self.use_cln:
            self.ln_1 = nn.LayerNorm(self.encoder_hidden)
            self.ln_2 = nn.LayerNorm(self.encoder_hidden)
        else:
            self.ln_1 = StyleAdaptiveLayerNorm(self.encoder_hidden)
            self.ln_2 = StyleAdaptiveLayerNorm(self.encoder_hidden)

        self.self_attn = nn.MultiheadAttention(
            self.encoder_hidden, self.encoder_head, batch_first=True
        )

        if self.use_new_ffn:
            self.ffn = TransformerFFNLayer(
                self.encoder_hidden,
                self.conv_filter_size,
                self.conv_kernel_size,
                self.encoder_dropout,
            )
        else:
            self.ffn = TransformerFFNLayerOld(
                self.encoder_hidden,
                self.conv_filter_size,
                self.conv_kernel_size,
                self.encoder_dropout,
            )

        if self.use_skip_connection:
            self.skip_linear = nn.Linear(self.encoder_hidden * 2, self.encoder_hidden)
            self.skip_linear.weight.data.normal_(0.0, 0.02)
            self.skip_layernorm = nn.LayerNorm(self.encoder_hidden)

        if self.add_diff_step:
            self.diff_step_emb = SinusoidalPosEmb(dim=self.encoder_hidden)
            # self.diff_step_projection = nn.linear(self.encoder_hidden, self.encoder_hidden)
            # self.encoder_hidden.weight.data.normal_(0.0, 0.02)
            self.diff_step_projection = Linear2(self.encoder_hidden)

    def forward(
        self, x, key_padding_mask, conditon=None, skip_res=None, diffusion_step=None
    ):
        # x: (B, T, d); key_padding_mask: (B, T), mask is 0; condition: (B, T, d); skip_res: (B, T, d); diffusion_step: (B,)

        if self.use_skip_connection and skip_res != None:
            x = torch.cat([x, skip_res], dim=-1)  # (B, T, 2*d)
            x = self.skip_linear(x)
            x = self.skip_layernorm(x)

        if self.add_diff_step and diffusion_step != None:
            diff_step_embedding = self.diff_step_emb(diffusion_step)
            diff_step_embedding = self.diff_step_projection(diff_step_embedding)
            x = x + diff_step_embedding.unsqueeze(1)

        residual = x

        # pre norm
        if self.use_cln:
            x = self.ln_1(x, conditon)
        else:
            x = self.ln_1(x)

        # self attention
        if key_padding_mask != None:
            key_padding_mask_input = ~(key_padding_mask.bool())
        else:
            key_padding_mask_input = None
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask_input
        )
        x = F.dropout(x, self.encoder_dropout, training=self.training)

        x = residual + x

        # pre norm
        residual = x
        if self.use_cln:
            x = self.ln_2(x, conditon)
        else:
            x = self.ln_2(x)

        # ffn
        x = self.ffn(x)

        x = residual + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        enc_emb_tokens=None,
        encoder_layer=None,
        encoder_hidden=None,
        encoder_head=None,
        conv_filter_size=None,
        conv_kernel_size=None,
        encoder_dropout=None,
        use_cln=None,
        use_skip_connection=None,
        use_new_ffn=None,
        add_diff_step=None,
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
        self.use_cln = use_cln if use_cln is not None else cfg.use_cln
        self.use_skip_connection = (
            use_skip_connection
            if use_skip_connection is not None
            else cfg.use_skip_connection
        )
        self.add_diff_step = (
            add_diff_step if add_diff_step is not None else cfg.add_diff_step
        )
        self.use_new_ffn = use_new_ffn if use_new_ffn is not None else cfg.use_new_ffn

        if enc_emb_tokens != None:
            self.use_enc_emb = True
            self.enc_emb_tokens = enc_emb_tokens
        else:
            self.use_enc_emb = False

        self.position_emb = PositionalEncoding(
            self.encoder_hidden, self.encoder_dropout
        )

        self.layers = nn.ModuleList([])
        if self.use_skip_connection:
            self.layers.extend(
                [
                    TransformerEncoderLayer(
                        self.encoder_hidden,
                        self.encoder_head,
                        self.conv_filter_size,
                        self.conv_kernel_size,
                        self.encoder_dropout,
                        self.use_cln,
                        use_skip_connection=False,
                        use_new_ffn=self.use_new_ffn,
                        add_diff_step=self.add_diff_step,
                    )
                    for i in range(
                        (self.encoder_layer + 1) // 2
                    )  # for example: 12 -> 6; 13 -> 7
                ]
            )
            self.layers.extend(
                [
                    TransformerEncoderLayer(
                        self.encoder_hidden,
                        self.encoder_head,
                        self.conv_filter_size,
                        self.conv_kernel_size,
                        self.encoder_dropout,
                        self.use_cln,
                        use_skip_connection=True,
                        use_new_ffn=self.use_new_ffn,
                        add_diff_step=self.add_diff_step,
                    )
                    for i in range(
                        self.encoder_layer - (self.encoder_layer + 1) // 2
                    )  # 12 -> 6;  13 -> 6
                ]
            )
        else:
            self.layers.extend(
                [
                    TransformerEncoderLayer(
                        self.encoder_hidden,
                        self.encoder_head,
                        self.conv_filter_size,
                        self.conv_kernel_size,
                        self.encoder_dropout,
                        self.use_cln,
                        use_new_ffn=self.use_new_ffn,
                        add_diff_step=self.add_diff_step,
                        use_skip_connection=False,
                    )
                    for i in range(self.encoder_layer)
                ]
            )

        if self.use_cln:
            self.last_ln = StyleAdaptiveLayerNorm(self.encoder_hidden)
        else:
            self.last_ln = nn.LayerNorm(self.encoder_hidden)

        if self.add_diff_step:
            self.diff_step_emb = SinusoidalPosEmb(dim=self.encoder_hidden)
            # self.diff_step_projection = nn.linear(self.encoder_hidden, self.encoder_hidden)
            # self.encoder_hidden.weight.data.normal_(0.0, 0.02)
            self.diff_step_projection = Linear2(self.encoder_hidden)

    def forward(self, x, key_padding_mask, condition=None, diffusion_step=None):
        if len(x.shape) == 2 and self.use_enc_emb:
            x = self.enc_emb_tokens(x)
            x = self.position_emb(x)
        else:
            x = self.position_emb(x)  # (B, T, d)

        if self.add_diff_step and diffusion_step != None:
            diff_step_embedding = self.diff_step_emb(diffusion_step)
            diff_step_embedding = self.diff_step_projection(diff_step_embedding)
            x = x + diff_step_embedding.unsqueeze(1)

        if self.use_skip_connection:
            skip_res_list = []
            # down
            for layer in self.layers[: self.encoder_layer // 2]:
                x = layer(x, key_padding_mask, condition)
                res = x
                skip_res_list.append(res)
            # middle
            for layer in self.layers[
                self.encoder_layer // 2 : (self.encoder_layer + 1) // 2
            ]:
                x = layer(x, key_padding_mask, condition)
            # up
            for layer in self.layers[(self.encoder_layer + 1) // 2 :]:
                skip_res = skip_res_list.pop()
                x = layer(x, key_padding_mask, condition, skip_res)
        else:
            for layer in self.layers:
                x = layer(x, key_padding_mask, condition)

        if self.use_cln:
            x = self.last_ln(x, condition)
        else:
            x = self.last_ln(x)

        return x


class DiffTransformer(nn.Module):
    def __init__(
        self,
        encoder_layer=None,
        encoder_hidden=None,
        encoder_head=None,
        conv_filter_size=None,
        conv_kernel_size=None,
        encoder_dropout=None,
        use_cln=None,
        use_skip_connection=None,
        use_new_ffn=None,
        add_diff_step=None,
        cat_diff_step=None,
        in_dim=None,
        out_dim=None,
        cond_dim=None,
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
        self.use_cln = use_cln if use_cln is not None else cfg.use_cln
        self.use_skip_connection = (
            use_skip_connection
            if use_skip_connection is not None
            else cfg.use_skip_connection
        )
        self.use_new_ffn = use_new_ffn if use_new_ffn is not None else cfg.use_new_ffn
        self.add_diff_step = (
            add_diff_step if add_diff_step is not None else cfg.add_diff_step
        )
        self.cat_diff_step = (
            cat_diff_step if cat_diff_step is not None else cfg.cat_diff_step
        )
        self.in_dim = in_dim if in_dim is not None else cfg.in_dim
        self.out_dim = out_dim if out_dim is not None else cfg.out_dim
        self.cond_dim = cond_dim if cond_dim is not None else cfg.cond_dim

        if self.in_dim != self.encoder_hidden:
            self.in_linear = nn.Linear(self.in_dim, self.encoder_hidden)
            self.in_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.in_dim = None

        if self.out_dim != self.encoder_hidden:
            self.out_linear = nn.Linear(self.encoder_hidden, self.out_dim)
            self.out_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.out_dim = None

        assert not ((self.cat_diff_step == True) and (self.add_diff_step == True))

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            encoder_hidden=self.encoder_hidden,
            encoder_head=self.encoder_head,
            conv_kernel_size=self.conv_kernel_size,
            conv_filter_size=self.conv_filter_size,
            encoder_dropout=self.encoder_dropout,
            use_cln=self.use_cln,
            use_skip_connection=self.use_skip_connection,
            use_new_ffn=self.use_new_ffn,
            add_diff_step=self.add_diff_step,
        )

        self.cond_project = nn.Linear(self.cond_dim, self.encoder_hidden)
        self.cond_project.weight.data.normal_(0.0, 0.02)
        self.cat_linear = nn.Linear(self.encoder_hidden * 2, self.encoder_hidden)
        self.cat_linear.weight.data.normal_(0.0, 0.02)

        if self.cat_diff_step:
            self.diff_step_emb = SinusoidalPosEmb(dim=self.encoder_hidden)
            self.diff_step_projection = Linear2(self.encoder_hidden)

    def forward(
        self,
        x,
        condition_embedding,
        key_padding_mask=None,
        reference_embedding=None,
        diffusion_step=None,
    ):
        # x: shape is (B, T, d_x)
        # key_padding_mask: shape is (B, T),  mask is 0
        # condition_embedding: from condition adapter, shape is (B, T, d_c)
        # reference_embedding: from reference encoder, shape is (B, N, d_r), or (B, 1, d_r), or (B, d_r)

        # TODO: How to add condition emedding? concatenate then linear (or FilM?) or cross attention?
        # concatenate then linear
        # TODO: How to add reference embedding to the model? use style adaptive norm or cross attention?
        # use style adaptive norm
        # TODO: How to add diffusion step embedding? add a timestep token? add timestep embedding in each layers? use style adaptive norm?
        # choose: (add_diff_step) add timestep embedding in each layers followed by a linear layer / (cat_diff_step) cat a timestep token before the first tokens

        if self.in_linear != None:
            x = self.in_linear(x)
        condition_embedding = self.cond_project(condition_embedding)

        x = torch.cat([x, condition_embedding], dim=-1)
        x = self.cat_linear(x)

        if self.cat_diff_step and diffusion_step != None:
            diff_step_embedding = self.diff_step_emb(diffusion_step)
            diff_step_embedding = self.diff_step_projection(
                diff_step_embedding
            ).unsqueeze(
                1
            )  # (B, 1, d)
            x = torch.cat([diff_step_embedding, x], dim=1)
            if key_padding_mask != None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.ones(key_padding_mask.shape[0], 1).to(
                            key_padding_mask.device
                        ),
                    ],
                    dim=1,
                )

        x = self.transformer_encoder(
            x,
            key_padding_mask=key_padding_mask,
            condition=reference_embedding,
            diffusion_step=diffusion_step,
        )

        if self.cat_diff_step and diffusion_step != None:
            x = x[:, 1:, :]

        if self.out_linear != None:
            x = self.out_linear(x)

        return x


class DurationPredictor(nn.Module):
    def __init__(
        self,
        input_size=None,
        filter_size=None,
        kernel_size=None,
        conv_layers=None,
        cross_attn_per_layer=None,
        attn_head=None,
        drop_out=None,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size if input_size is not None else cfg.input_size
        self.filter_size = filter_size if filter_size is not None else cfg.filter_size
        self.kernel_size = kernel_size if kernel_size is not None else cfg.kernel_size
        self.conv_layers = conv_layers if conv_layers is not None else cfg.conv_layers
        self.cross_attn_per_layer = (
            cross_attn_per_layer
            if cross_attn_per_layer is not None
            else cfg.cross_attn_per_layer
        )
        self.attn_head = attn_head if attn_head is not None else cfg.attn_head
        self.drop_out = drop_out if drop_out is not None else cfg.drop_out

        self.conv = nn.ModuleList()
        self.cattn = nn.ModuleList()

        for idx in range(self.conv_layers):
            in_dim = self.input_size if idx == 0 else self.filter_size
            self.conv += [
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        self.filter_size,
                        self.kernel_size,
                        padding=self.kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(self.filter_size),
                    nn.Dropout(self.drop_out),
                )
            ]
            if idx % self.cross_attn_per_layer == 0:
                self.cattn.append(
                    torch.nn.Sequential(
                        nn.MultiheadAttention(
                            self.filter_size,
                            self.attn_head,
                            batch_first=True,
                            kdim=self.filter_size,
                            vdim=self.filter_size,
                        ),
                        nn.LayerNorm(self.filter_size),
                        nn.Dropout(0.2),
                    )
                )

        self.linear = nn.Linear(self.filter_size, 1)
        self.linear.weight.data.normal_(0.0, 0.02)

    def forward(self, x, mask, ref_emb, ref_mask):
        """
        input:
        x: (B, N, d)
        mask: (B, N), mask is 0
        ref_emb: (B, d, T')
        ref_mask: (B, T'), mask is 0

        output:
        dur_pred: (B, N)
        dur_pred_log: (B, N)
        dur_pred_round: (B, N)
        """

        input_ref_mask = ~(ref_mask.bool())  # (B, T')

        x = x.transpose(1, -1)

        for idx, (conv, act, ln, dropout) in enumerate(self.conv):
            res = x

            if idx % self.cross_attn_per_layer == 0:
                attn_idx = idx // self.cross_attn_per_layer
                attn, attn_ln, attn_drop = self.cattn[attn_idx]

                attn_res = y_ = x.transpose(1, 2)  # (B, d, N) -> (B, N, d)

                y_ = attn_ln(y_)

                y_, _ = attn(
                    y_,
                    ref_emb.transpose(1, 2),
                    ref_emb.transpose(1, 2),
                    key_padding_mask=input_ref_mask,
                )

                y_ = attn_drop(y_)
                y_ = (y_ + attn_res) / math.sqrt(2.0)

                x = y_.transpose(1, 2)

            x = conv(x)

            x = act(x)
            x = ln(x.transpose(1, 2))

            x = x.transpose(1, 2)

            x = dropout(x)

            if idx != 0:
                x += res

            if mask is not None:
                x = x * mask.to(x.dtype)[:, None, :]

        x = self.linear(x.transpose(1, 2))
        x = torch.squeeze(x, -1)

        dur_pred = x.exp() - 1
        dur_pred_round = torch.clamp(torch.round(x.exp() - 1), min=0).long()

        return {
            "dur_pred_log": x,
            "dur_pred": dur_pred,
            "dur_pred_round": dur_pred_round,
        }


class PitchPredictor(nn.Module):
    def __init__(
        self,
        input_size=None,
        filter_size=None,
        kernel_size=None,
        conv_layers=None,
        cross_attn_per_layer=None,
        attn_head=None,
        drop_out=None,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size if input_size is not None else cfg.input_size
        self.filter_size = filter_size if filter_size is not None else cfg.filter_size
        self.kernel_size = kernel_size if kernel_size is not None else cfg.kernel_size
        self.conv_layers = conv_layers if conv_layers is not None else cfg.conv_layers
        self.cross_attn_per_layer = (
            cross_attn_per_layer
            if cross_attn_per_layer is not None
            else cfg.cross_attn_per_layer
        )
        self.attn_head = attn_head if attn_head is not None else cfg.attn_head
        self.drop_out = drop_out if drop_out is not None else cfg.drop_out

        self.conv = nn.ModuleList()
        self.cattn = nn.ModuleList()

        for idx in range(self.conv_layers):
            in_dim = self.input_size if idx == 0 else self.filter_size
            self.conv += [
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        self.filter_size,
                        self.kernel_size,
                        padding=self.kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(self.filter_size),
                    nn.Dropout(self.drop_out),
                )
            ]
            if idx % self.cross_attn_per_layer == 0:
                self.cattn.append(
                    torch.nn.Sequential(
                        nn.MultiheadAttention(
                            self.filter_size,
                            self.attn_head,
                            batch_first=True,
                            kdim=self.filter_size,
                            vdim=self.filter_size,
                        ),
                        nn.LayerNorm(self.filter_size),
                        nn.Dropout(0.2),
                    )
                )

        self.linear = nn.Linear(self.filter_size, 1)
        self.linear.weight.data.normal_(0.0, 0.02)

    def forward(self, x, mask, ref_emb, ref_mask):
        """
        input:
        x: (B, N, d)
        mask: (B, N), mask is 0
        ref_emb: (B, d, T')
        ref_mask: (B, T'), mask is 0

        output:
        pitch_pred: (B, T)
        """

        input_ref_mask = ~(ref_mask.bool())  # (B, T')

        x = x.transpose(1, -1)  # (B, N, d) -> (B, d, N)

        for idx, (conv, act, ln, dropout) in enumerate(self.conv):
            res = x
            if idx % self.cross_attn_per_layer == 0:
                attn_idx = idx // self.cross_attn_per_layer
                attn, attn_ln, attn_drop = self.cattn[attn_idx]

                attn_res = y_ = x.transpose(1, 2)  # (B, d, N) -> (B, N, d)

                y_ = attn_ln(y_)
                y_, _ = attn(
                    y_,
                    ref_emb.transpose(1, 2),
                    ref_emb.transpose(1, 2),
                    key_padding_mask=input_ref_mask,
                )
                # y_, _ = attn(y_, ref_emb.transpose(1, 2), ref_emb.transpose(1, 2))
                y_ = attn_drop(y_)
                y_ = (y_ + attn_res) / math.sqrt(2.0)

                x = y_.transpose(1, 2)

            x = conv(x)
            x = act(x)
            x = ln(x.transpose(1, 2))
            x = x.transpose(1, 2)

            x = dropout(x)

            if idx != 0:
                x += res

        x = self.linear(x.transpose(1, 2))
        x = torch.squeeze(x, -1)

        return x


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        device = x.device
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
