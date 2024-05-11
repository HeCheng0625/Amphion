# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import BigVGAN as Generator
import librosa
import soundfile as sf


h = None
device = None
torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(
        x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax
    )


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ""
    return sorted(cp_list)[-1]


def inference(a, h):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        test_json = "/blob/v-zeqianju/dataset/tts/librispeech/test/ref_dur_3_test_merge_1pspk_with_punc_refmeta_normwav_fix_refuid_new_diffprompt.json"

        with open(test_json, "r") as f:
            test_data = json.load(f)
            print(test_data.keys())

        test_wav_output = (
            "/blob/v-shenkai/checkpoints/tts/vocoder/bigvgan/v2/wav_22wstep"
        )

        os.makedirs(test_wav_output, exist_ok=True)

        for case in test_data["test_cases"]:
            print(case.keys())

            wav_path = case["wav_path"]

            wav = librosa.load(wav_path, sr=h.sampling_rate, mono=True)[0]
            wav = torch.FloatTensor(wav).to(device)
            # compute mel spectrogram from the ground truth audio
            x = get_mel(wav.unsqueeze(0))

            y_g_hat = generator(x)

            audio = y_g_hat.squeeze().detach().cpu()
            # audio = audio * MAX_WAV_VALUE
            # audio = audio.cpu().numpy().astype('int16')

            file_output_path = os.path.join(test_wav_output, os.path.basename(wav_path))

            sf.write(file_output_path, audio, h.sampling_rate)
            case["synthesized_wav_path"] = file_output_path
        with open(os.path.join(test_wav_output, "test.json"), "w") as f:
            json.dump(test_data, f, indent=4)

        # for i, filname in enumerate(filelist):
        #     # load the ground truth audio and resample if necessary
        #     wav, sr = librosa.load(os.path.join(a.input_wavs_dir, filname), h.sampling_rate, mono=True)
        #     wav = torch.FloatTensor(wav).to(device)
        #     # compute mel spectrogram from the ground truth audio
        #     x = get_mel(wav.unsqueeze(0))

        #     y_g_hat = generator(x)

        #     audio = y_g_hat.squeeze()
        #     audio = audio * MAX_WAV_VALUE
        #     audio = audio.cpu().numpy().astype('int16')

        #     output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
        #     write(output_file, h.sampling_rate, audio)
        #     print(output_file)


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_file",
        default="/blob/v-shenkai/checkpoints/tts/vocoder/bigvgan/v2/g_00220000",
        type=str,
    )

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # device = torch.device('cpu')
    inference(a, h)


if __name__ == "__main__":
    main()
