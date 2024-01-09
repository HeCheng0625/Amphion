import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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
