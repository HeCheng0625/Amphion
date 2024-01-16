# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.naturalpseech2.transformers import (
    TransformerEncoder,
    DurationPredictor,
    PitchPredictor,
    LengthRegulator,
)


class PriorEncoder(nn.Module):
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
        vocab_size=None,
        cond_dim=None,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg

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
        self.vocab_size = vocab_size if vocab_size is not None else cfg.vocab_size
        self.cond_dim = cond_dim if cond_dim is not None else cfg.cond_dim

        self.enc_emb_tokens = nn.Embedding(
            self.vocab_size, self.encoder_hidden, padding_idx=0
        )
        self.enc_emb_tokens.weight.data.normal_(mean=0.0, std=1e-5)

        self.encoder = TransformerEncoder(
            enc_emb_tokens=self.enc_emb_tokens,
            encoder_layer=self.encoder_layer,
            encoder_hidden=self.encoder_hidden,
            encoder_head=self.encoder_head,
            conv_filter_size=self.conv_filter_size,
            conv_kernel_size=self.conv_kernel_size,
            encoder_dropout=self.encoder_dropout,
            use_new_ffn=self.use_new_ffn,
            use_cln=True,
            use_skip_connection=False,
            add_diff_step=False,
        )

        self.cond_project = nn.Linear(self.cond_dim, self.encoder_hidden)
        self.cond_project.weight.data.normal_(0.0, 0.02)

        # TODO: add params (not cfg) for DurationPredictor and PitchPredictor
        self.duration_predictor = DurationPredictor(cfg=cfg.duration_predictor)
        self.pitch_predictor = PitchPredictor(cfg=cfg.pitch_predictor)
        self.length_regulator = LengthRegulator()

        self.pitch_min = cfg.pitch_min
        self.pitch_max = cfg.pitch_max
        self.pitch_bins_num = cfg.pitch_bins_num

        pitch_bins = torch.exp(
            torch.linspace(
                np.log(self.pitch_min), np.log(self.pitch_max), self.pitch_bins_num - 1
            )
        )
        self.register_buffer("pitch_bins", pitch_bins)

        self.pitch_embedding = nn.Embedding(self.pitch_bins_num, self.encoder_hidden)
        self.pitch_embedding.weight.data.normal_(mean=0.0, std=1e-5)

    def forward(
        self,
        phone_id,
        duration=None,
        pitch=None,
        phone_mask=None,
        mask=None,
        ref_emb=None,
        ref_mask=None,
        is_inference=False,
    ):
        ref_emb = self.cond_project(ref_emb)

        x = self.encoder(phone_id, phone_mask, ref_emb)

        dur_pred_out = self.duration_predictor(
            x, phone_mask, ref_emb.transpose(1, 2), ref_mask
        )

        if is_inference or duration is None:
            x, mel_len = self.length_regulator(
                x,
                dur_pred_out["dur_pred_round"],
                max_len=torch.max(torch.sum(dur_pred_out["dur_pred_round"], dim=1)),
            )
        else:
            x, mel_len = self.length_regulator(x, duration, max_len=pitch.shape[1])

        pitch_pred_log = self.pitch_predictor(
            x, mask, ref_emb.transpose(1, 2), ref_mask
        )

        if is_inference or pitch is None:
            # pitch_tokens = torch.bucketize(pitch_pred_log.exp(), self.pitch_bins)
            pitch_tokens = torch.bucketize(pitch_pred_log.exp() - 1, self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)
        else:
            pitch_tokens = torch.bucketize(pitch, self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)

        x = x + pitch_embedding

        if (not is_inference) and (mask is not None):
            x = x * mask.to(x.dtype)[:, :, None]

        prior_out = {
            "dur_pred_round": dur_pred_out["dur_pred_round"],
            "dur_pred_log": dur_pred_out["dur_pred_log"],
            "dur_pred": dur_pred_out["dur_pred"],
            "pitch_pred_log": pitch_pred_log,
            "pitch_token": pitch_tokens,
            "mel_len": mel_len,
            "prior_out": x,
        }

        return prior_out
