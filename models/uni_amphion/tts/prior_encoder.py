import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.uni_amphion.base.transformer import TransformerEncoder


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
