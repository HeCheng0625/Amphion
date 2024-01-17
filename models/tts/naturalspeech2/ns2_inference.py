# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
import soundfile as sf
import numpy as np
import json
import librosa
import torchaudio

from models.tts.naturalspeech2.ns2 import NaturalSpeech2
from models.tts.naturalspeech2.vocoder import BigVGAN as Generator
from models.tts.naturalspeech2.get_feature import mel_spectrogram
from encodec import EncodecModel
from encodec.utils import convert_audio
from utils.util import load_config

from text import text_to_sequence
from text.cmudict import valid_symbols
from text.g2p import preprocess_english, read_lexicon
from text.g2p_extend import process as preprocess_english_extend, PHPONE2ID
from g2p_en import G2p


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class NS2InferenceOld:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args

        self.model = self.build_model()
        self.codec = self.build_codec()

        self.symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
        self.phone2id = {s: i for i, s in enumerate(self.symbols)}
        self.id2phone = {i: s for s, i in self.phone2id.items()}

    def build_model(self):
        model = NaturalSpeech2(self.cfg.model)
        model.load_state_dict(
            torch.load(
                os.path.join(self.args.checkpoint_path, "pytorch_model.bin"),
                map_location="cpu",
            )
        )
        model = model.to(self.args.device)
        return model

    def build_codec(self):
        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model = encodec_model.to(device=self.args.device)
        encodec_model.set_target_bandwidth(12.0)
        return encodec_model

    def get_ref_code(self):
        ref_wav_path = self.args.ref_audio
        ref_wav, sr = torchaudio.load(ref_wav_path)
        ref_wav = convert_audio(
            ref_wav, sr, self.codec.sample_rate, self.codec.channels
        )
        ref_wav = ref_wav.unsqueeze(0).to(device=self.args.device)

        with torch.no_grad():
            encoded_frames = self.codec.encode(ref_wav)
            ref_code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        # print(ref_code.shape)

        ref_mask = torch.ones(ref_code.shape[0], ref_code.shape[-1]).to(ref_code.device)
        # print(ref_mask.shape)

        return ref_code, ref_mask

    def inference(self):
        ref_code, ref_mask = self.get_ref_code()

        lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)
        phone_seq = preprocess_english(self.args.text, lexicon)
        print(phone_seq)

        phone_id = np.array(
            [
                *map(
                    self.phone2id.get,
                    phone_seq.replace("{", "").replace("}", "").split(),
                )
            ]
        )
        phone_id = torch.from_numpy(phone_id).unsqueeze(0).to(device=self.args.device)
        print(phone_id)

        x0, prior_out = self.model.inference(
            ref_code, phone_id, ref_mask, self.args.inference_step
        )
        print(prior_out["dur_pred"])
        print(prior_out["dur_pred_round"])
        print(torch.sum(prior_out["dur_pred_round"]))

        latent_ref = self.codec.quantizer.vq.decode(ref_code.transpose(0, 1))

        rec_wav = self.codec.decoder(x0)
        # ref_wav = self.codec.decoder(latent_ref)

        os.makedirs(self.args.output_dir, exist_ok=True)

        sf.write(
            "{}/{}.wav".format(
                self.args.output_dir, self.args.text.replace(" ", "_", 100)
            ),
            rec_wav[0, 0].detach().cpu().numpy(),
            samplerate=24000,
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


class NS2Inference:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args

        self.model = self.build_model()
        self.vocoder = self.build_vocoder()
        self.g2p = G2p()

    def build_model(self):
        model = NaturalSpeech2(self.cfg.model)
        model.load_state_dict(
            torch.load(
                os.path.join(self.args.checkpoint_path, "pytorch_model.bin"),
                map_location="cpu",
            )
        )
        model = model.to(self.args.device)
        return model

    def build_vocoder(self):
        config_file = os.path.join(self.args.vocoder_config_path)
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        vocoder = Generator(h).to(self.args.device)
        checkpoint_dict = torch.load(
            self.args.vocoder_path, map_location=self.args.device
        )
        vocoder.load_state_dict(checkpoint_dict["generator"])

        return vocoder

    def get_ref_mel(self):
        ref_wav_path = self.args.ref_audio
        ref_wav, sr = librosa.load(ref_wav_path, sr=self.cfg.preprocess.sampling_rate)
        ref_wav = torch.from_numpy(ref_wav).to(self.args.device)
        ref_wav = ref_wav[None, :]
        ref_mel = mel_spectrogram(
            ref_wav,
            n_fft=self.cfg.preprocess.n_fft,
            num_mels=self.cfg.preprocess.num_mels,
            sampling_rate=self.cfg.preprocess.sampling_rate,
            hop_size=self.cfg.preprocess.hop_size,
            win_size=self.cfg.preprocess.win_size,
            fmin=self.cfg.preprocess.fmin,
            fmax=self.cfg.preprocess.fmax,
        )
        ref_mel = ref_mel.transpose(1, 2).to(self.args.device)
        ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(ref_mel.device)

        return ref_mel, ref_mask

    @torch.no_grad()
    def inference(self):
        ref_mel, ref_mask = self.get_ref_mel()

        txt_struct, txt = preprocess_english_extend(self.args.text, self.g2p)
        phone_seq = [p for w in txt_struct for p in w[1]]
        print(phone_seq)
        phone_id = [PHPONE2ID[p] for p in phone_seq]
        phone_id = torch.LongTensor(phone_id).unsqueeze(0).to(self.args.device)

        x0, prior_out = self.model.inference(
            phone_id=phone_id,
            x_ref=ref_mel,
            x_ref_mask=ref_mask,
            inference_steps=self.args.inference_step,
            sigma=1.2,
        )

        os.makedirs(self.args.output_dir, exist_ok=True)

        x0 = x0.transpose(1, 2)

        self.vocoder.eval()
        self.vocoder.remove_weight_norm()

        rec_wav = self.vocoder(x0)
        rec_wav = rec_wav.squeeze()
        rec_wav = rec_wav * 32768.0
        rec_wav = rec_wav.cpu().numpy().astype("int16")

        sf.write(
            "{}/{}.wav".format(
                self.args.output_dir, self.args.text.replace(" ", "_", 100)
            ),
            rec_wav,
            samplerate=self.cfg.preprocess.sampling_rate,
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
        parser.add_argument(
            "--vocoder_config_path",
            type=str,
            default="",
            help="Vocoder config path",
        )
        parser.add_argument(
            "--vocoder_path",
            type=str,
            default="",
            help="Vocoder checkpoint path",
        )
