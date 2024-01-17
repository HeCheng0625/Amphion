# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gradio as gr
import os

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


def build_model(cfg, device):
    model = NaturalSpeech2(cfg.model)
    model.load_state_dict(
        torch.load(
            "ckpts/ns2/pytorch_model.bin",
            map_location="cpu",
        )
    )
    model = model.to(device=device)
    return model


def build_vocoder(cfg, device):
    config_file = "ckpts/ns2/bigvgan/config.json"
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Generator(h).to(device)
    checkpoint_dict = torch.load("ckpts/ns2/bigvgan/g_00490000", map_location=device)
    vocoder.load_state_dict(checkpoint_dict["generator"])

    return vocoder


def get_ref_mel(prmopt_audio_path, cfg, device):
    ref_wav_path = prmopt_audio_path
    ref_wav, sr = librosa.load(ref_wav_path, sr=cfg.preprocess.sampling_rate)
    ref_wav = torch.from_numpy(ref_wav).to(device)
    ref_wav = ref_wav[None, :]
    ref_mel = mel_spectrogram(
        ref_wav,
        n_fft=cfg.preprocess.n_fft,
        num_mels=cfg.preprocess.num_mels,
        sampling_rate=cfg.preprocess.sampling_rate,
        hop_size=cfg.preprocess.hop_size,
        win_size=cfg.preprocess.win_size,
        fmin=cfg.preprocess.fmin,
        fmax=cfg.preprocess.fmax,
    )
    ref_mel = ref_mel.transpose(1, 2).to(device)
    ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(ref_mel.device)

    return ref_mel, ref_mask


@torch.no_grad()
def ns2_inference(
    prmopt_audio_path,
    text,
    diffusion_steps=200,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["WORK_DIR"] = "./"
    cfg = load_config("egs/tts/NaturalSpeech2/exp_config.json")

    model = build_model(cfg, device)
    vocoder = build_vocoder(cfg, device)
    g2p_model = G2p()

    ref_mel, ref_mask = get_ref_mel(prmopt_audio_path, cfg, device)
    txt_struct, txt = preprocess_english_extend(text, g2p_model)
    phone_seq = [p for w in txt_struct for p in w[1]]
    print(phone_seq)
    phone_id = [PHPONE2ID[p] for p in phone_seq]
    phone_id = torch.LongTensor(phone_id).unsqueeze(0).to(device)

    with torch.no_grad():
        x0, prior_out = model.inference(
            phone_id=phone_id,
            x_ref=ref_mel,
            x_ref_mask=ref_mask,
            inference_steps=diffusion_steps,
            sigma=1.2,
        )

    x0 = x0.transpose(1, 2)

    vocoder.eval()
    vocoder.remove_weight_norm()

    rec_wav = vocoder(x0)
    rec_wav = rec_wav.squeeze()
    rec_wav = rec_wav * 32768.0
    rec_wav = rec_wav.cpu().numpy().astype("int16")

    os.makedirs("result", exist_ok=True)

    result_file = "result/{}.wav".format(
        prmopt_audio_path.split("/")[-1][:-4] + "_zero_shot_result"
    )

    sf.write(
        result_file,
        rec_wav,
        samplerate=cfg.preprocess.sampling_rate,
    )

    return result_file


demo_inputs = [
    gr.Audio(
        sources=["upload", "microphone"],
        label="Upload a reference speech you want to clone timbre",
        type="filepath",
    ),
    gr.Textbox(
        value="Amphion is a toolkit that can speak, make sounds, and sing.",
        label="Text you want to generate",
        type="text",
    ),
    gr.Slider(
        10,
        1000,
        value=200,
        step=1,
        label="Diffusion Inference Steps",
        info="As the step number increases, the synthesis quality will be better while the inference speed will be lower",
    ),
]
demo_outputs = gr.Audio(label="")

demo = gr.Interface(
    fn=ns2_inference,
    inputs=demo_inputs,
    outputs=demo_outputs,
    title="Amphion Zero-Shot TTS NaturalSpeech2",
    description="The Model is trained on LibriLight dataset.",
)

if __name__ == "__main__":
    demo.launch()
