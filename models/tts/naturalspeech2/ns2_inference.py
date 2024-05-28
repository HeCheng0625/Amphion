# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
import soundfile as sf
import librosa
import numpy as np

from models.tts.gpt_tts.g2p_old_en import process, PHPONE2ID
from g2p_en import G2p
from models.tts.naturalspeech2.ns2 import NaturalSpeech2
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from encodec.utils import convert_audio
from utils.util import load_config


class NS2Inference:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args

        self.model = self.build_model()
        self.codec_enc, self.codec_dec = self.build_codec()

        self.g2p = G2p()

    def build_model(self):
        model = NaturalSpeech2(cfg=self.cfg.model)
        model.load_state_dict(
            torch.load(
                os.path.join(self.args.checkpoint_path, "pytorch_model.bin"),
                map_location="cpu",
            )
        )
        model.eval()
        model.requires_grad_(False)
        model = model.to(self.args.device)
        return model

    def build_codec(self):
        codec_enc = CodecEncoder(cfg=self.cfg.model.codec.encoder)
        codec_dec = CodecDecoder(cfg=self.cfg.model.codec.decoder)

        codec_enc.load_state_dict(
            torch.load(
                os.path.join(self.args.checkpoint_path, "codec_enc.bin"),
                map_location="cpu",
            )
        )
        codec_dec.load_state_dict(
            torch.load(
                os.path.join(self.args.checkpoint_path, "codec_dec.bin"),
                map_location="cpu",
            )
        )
        codec_enc.eval()
        codec_dec.eval()

        codec_enc.requires_grad_(False)
        codec_dec.requires_grad_(False)
        codec_enc = codec_enc.to(device=self.args.device)
        codec_dec = codec_dec.to(device=self.args.device)
        return codec_enc, codec_dec

    def get_ref_code(self):
        ref_wav_path = self.args.ref_audio
        ref_wav, sr = librosa.load(ref_wav_path, sr=16000)
        ref_wav = torch.from_numpy(ref_wav).float().to(device=self.args.device)
        ref_wav = ref_wav.unsqueeze(0)
        ref_latent = self.codec_enc(ref_wav.unsqueeze(1))

        ref_latent = ref_latent.transpose(1, 2)
        ref_mask = torch.ones(ref_latent.size(0), ref_latent.size(1)).to(device=self.args.device)
        # print(ref_latent.shape, ref_mask.shape)
        return ref_latent, ref_mask

    def inference(self):
        ref_latent, ref_mask = self.get_ref_code()

        txt_struct, txt = process(self.args.text, self.g2p)
        phone_seq = [p for w in txt_struct for p in w[1]][1:-1]
        phone_ids = [PHPONE2ID[p] for p in phone_seq]

        phone_ids = torch.tensor(phone_ids).unsqueeze(0).to(device=self.args.device)
        x0, prior_out = self.model.inference(
            phone_id=phone_ids,
            x_ref=ref_latent,
            x_ref_mask=ref_mask,
            inference_steps=200,
            sigma=1.2,
        )
        print(x0.shape)

        recon_wav = self.codec_dec(x0.transpose(1, 2), vq=False)
        recon_ref_wav = self.codec_dec(ref_latent.transpose(1, 2), vq=False)

        os.makedirs(self.args.output_dir, exist_ok=True)

        sf.write(
            "{}/{}.wav".format(
                self.args.output_dir, self.args.text.replace(" ", "_", 100)
            ),
            recon_wav.squeeze().squeeze().cpu().numpy(),
            samplerate=16000,
        )

    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--ref_audio",
            type=str,
            default="",
            help="Reference audio path",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )
        parser.add_argument(
            "--inference_step",
            type=int,
            default=200,
            help="Total inference steps for the diffusion model",
        )
