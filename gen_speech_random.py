import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from IPython.display import Audio
import matplotlib.pyplot as plt
import soundfile as sf
import json

from models.tts.gpt_tts.gpt_tts import GPTTTS
from models.tts.gpt_tts.g2p_old_en import process, PHPONE2ID
from g2p_en import G2p
from models.codec.codec_latent.codec_latent import (
    LatentCodecEncoder,
    LatentCodecDecoderWithTimbre,
)
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from utils.util import load_config


def set_model():

    cfg = load_config("egs/tts/LaCoTTS/exp_config_base.json")

    wav_codec_enc = CodecEncoder(cfg=cfg.model.wav_codec.encoder)

    wav_codec_dec = CodecDecoder(cfg=cfg.model.wav_codec.decoder)

    latent_codec_enc = LatentCodecEncoder(cfg=cfg.model.latent_codec.encoder)

    latent_codec_dec = LatentCodecDecoderWithTimbre(cfg=cfg.model.latent_codec.decoder)

    wav_codec_enc.load_state_dict(torch.load("ckpts/wav_codec/wav_codec_enc.bin"))
    wav_codec_dec.load_state_dict(torch.load("ckpts/wav_codec/wav_codec_dec.bin"))
    latent_codec_enc.load_state_dict(
        torch.load("ckpts/latent_codec/latent_codec_enc.bin")
    )
    latent_codec_dec.load_state_dict(
        torch.load("ckpts/latent_codec/latent_codec_dec.bin")
    )

    wav_codec_enc.eval()
    wav_codec_dec.eval()
    latent_codec_enc.eval()
    latent_codec_dec.eval()

    wav_codec_enc.cuda()
    wav_codec_dec.cuda()
    latent_codec_enc.cuda()
    latent_codec_dec.cuda()

    # requires_grad false
    wav_codec_enc.requires_grad_(False)
    wav_codec_dec.requires_grad_(False)
    latent_codec_enc.requires_grad_(False)
    latent_codec_dec.requires_grad_(False)

    gpt_tts = GPTTTS(cfg=cfg.model.gpt_tts)
    gpt_tts.load_state_dict(
        torch.load("ckpts/gpt_tts/latent_codec_gpt_tts.bin", map_location="cpu")
    )

    gpt_tts.eval()
    gpt_tts.cuda()
    gpt_tts.requires_grad_(False)

    return wav_codec_enc, wav_codec_dec, latent_codec_enc, latent_codec_dec, gpt_tts


def prepare_prompt_json():
    speech_json_file = "temp_meta_info/libritts_train_clean_100.json"
    # the data you saved
    with open(speech_json_file, "r") as f:
        speech_data = json.load(f)
    return speech_data


def get_random_prompt(speech_data):
    random_index = np.random.randint(0, len(speech_data))
    text = speech_data[random_index]["text"]
    wav_path = speech_data[random_index]["path"]
    base_dir_path = "/blob/v-yuancwang/LibriTTS"
    wav, sr = librosa.load(os.path.join(base_dir_path, wav_path), sr=16000)
    return wav, text


def gen_speech(
    prompt_wav,
    prompt_text,
    target_text,
    g2p,
    wav_codec_enc,
    wav_codec_dec,
    latent_codec_enc,
    latent_codec_dec,
    gpt_tts,
):
    text = prompt_text + " " + target_text
    txt_struct, txt = process(text, g2p)
    phone_seq = [p for w in txt_struct for p in w[1]]
    phone_id = [PHPONE2ID[p] for p in phone_seq]
    phone_id = torch.LongTensor(phone_id).unsqueeze(0).cuda()
    # speech tokenize
    prompt_wav = torch.FloatTensor(prompt_wav).unsqueeze(0).to("cuda")
    # wav to latent
    vq_emb = wav_codec_enc(prompt_wav.unsqueeze(1))
    vq_emb = latent_codec_enc(vq_emb)
    # latent to token
    (
        _,
        vq_indices,
        _,
        _,
        _,
        speaker_embedding,
    ) = latent_codec_dec(vq_emb, vq=True, eval_vq=False, return_spk_embs=True)
    prompt_id = vq_indices[0, :, :]
    gen_tokens = gpt_tts.sample_hf(
        phone_id,
        prompt_id,
        max_length=3600,
        temperature=0.9,
        top_k=8192,
        top_p=0.85,
        repeat_penalty=1.0,
        classifer_free_guidance=1.0,
    )
    vq_post_emb = latent_codec_dec.vq2emb(gen_tokens.unsqueeze(0))
    recovered_latent = latent_codec_dec(
        vq_post_emb, vq=False, speaker_embedding=speaker_embedding
    )
    recovered_audio = wav_codec_dec(recovered_latent, vq=False)
    return recovered_audio.squeeze().cpu().numpy(), prompt_wav


if __name__ == "__main__":

    # set model
    wav_codec_enc, wav_codec_dec, latent_codec_enc, latent_codec_dec, gpt_tts = (
        set_model()
    )
    
    g2p = G2p()
    
    speech_data = prepare_prompt_json()
    target_text = "What is the UV index like today?"

    for i in range(10):
        wav, text = get_random_prompt(speech_data)
        gen_wav, _ = gen_speech(wav, text, target_text, g2p, wav_codec_enc, wav_codec_dec, latent_codec_enc, latent_codec_dec, gpt_tts)
        target_path = "/home/t-zeqianju/yuancwang/Amphion/temp_wavs/recon/{}.wav".format(str(i))
        sf.write(target_path, gen_wav, 16000)