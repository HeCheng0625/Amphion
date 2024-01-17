import argparse
import os
import torch
import soundfile as sf
import numpy as np

import json

from models.tts.naturalspeech2.ns2_trainer import NS2Trainer
from models.tts.naturalspeech2.ns2_dataset import NS2Dataset
from models.tts.naturalspeech2.ns2 import NaturalSpeech2
from encodec import EncodecModel
from encodec.utils import convert_audio
from utils.util import load_config

from text import text_to_sequence
from text.cmudict import valid_symbols
from text.g2p import preprocess_english, read_lexicon

import torchaudio

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def build_trainer(args, cfg):
    supported_trainer = {
        "NaturalSpeech2": NS2Trainer,
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        # action="store_true",
        help="The model name to restore",
    )
    parser.add_argument(
        "--log_level", default="info", help="logging level (info, debug, warning)"
    )
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.exp_name = args.exp_name

    # Model saving dir
    args.log_dir = os.path.join(cfg.log_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=True)

    if not cfg.train.ddp:
        args.local_rank = torch.device("cuda:1")

    model = NaturalSpeech2(cfg.model)
    model.load_state_dict(
        torch.load(
            "/mnt/data2/wangyuancheng/ns2_ckpts/ns2_mel_debug/550k/pytorch_model.bin",
            map_location="cpu",
        )
    )

    print(model)

    num_param = sum(param.numel() for param in model.parameters())
    print("Number of parameters: %f M" % (num_param / 1e6))

    from models.tts.naturalspeech2.inference_utils.vocoder import BigVGAN as Generator
    config_file = "/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/config.json"
    with open(config_file) as f:
        data = f.read()
    json_file = json.loads(data)
    h = AttrDict(json_file)
    vocoder = Generator(h)
    state_dict_g = torch.load("/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000",
                              map_location="cpu")
    vocoder.load_state_dict(state_dict_g['generator'])
    print(vocoder)
    num_param = sum(param.numel() for param in vocoder.parameters())
    print("Number of parameters: %f M" % (num_param / 1e6))


if __name__ == "__main__":
    main()
