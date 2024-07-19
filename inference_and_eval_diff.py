import torch, time, logging
import torch.nn.functional as F
import numpy as np
import librosa
import os
import soundfile as sf
import math
import accelerate

from models.codec.kmeans.kmeans_model import KMeans, KMeansEMA
from models.codec.kmeans.repcodec_model import RepCodec
from models.tts.soundstorm.soundstorm_model import SoundStorm
from models.tts.difft2s.difft2s_model import DiffT2S
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from transformers import Wav2Vec2BertModel
import safetensors
from utils.util import load_config
from tqdm import tqdm

from transformers import SeamlessM4TFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration 
processor = SeamlessM4TFeatureExtractor.from_pretrained("./ckpt/w2v-bert-2")
from models.tts.difft2s.ar_dur_model import DurPredictorAR
from models.tts.difft2s.fm_dur_model import DurPredictorFM

from models.tts.text2semantic.t2s_model import T2SLlama
from utils.g2p_liwei.g2p_liwei import liwei_g2p
from models.tts.difft2s.aligner.aligner import ASRCNN

from funasr import AutoModel
import zhconv
import scipy
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string

from speaker_verification import init_model
import torchaudio

from accelerate import Accelerator
from accelerate.utils import gather_object

class WarningFilter(logging.Filter):
    def filter(self, record):
        # 只过滤 phonemizer 中的 WARNING 级别日志
        if record.name == "phonemizer" and record.levelno == logging.WARNING:
            return False
        if record.name == "jieba" and record.levelno == logging.DEBUG:
            return False
        return True

filter = WarningFilter()
logging.getLogger("phonemizer").addFilter(filter)
logging.getLogger("jieba").addFilter(filter)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def liwei_g2p_(text, language):
    return liwei_g2p(text, language)

def build_t2s_model(cfg, device):
    t2s_model = DiffT2S(cfg=cfg.model.difft2s)
    t2s_model.eval()
    t2s_model.to(device)
    return t2s_model

def build_soundstorm(cfg, device):
    soundstorm_model = SoundStorm(cfg=cfg.model.soundstorm)
    soundstorm_model.eval()
    soundstorm_model.to(device)
    return soundstorm_model

def build_kmeans_model(cfg, device):
    if cfg.model.kmeans.type == "kmeans":
        kmeans_model = KMeans(cfg=cfg.model.kmeans.kmeans)
    elif cfg.model.kmeans.type == "kmeans_ema":
        kmeans_model = KMeansEMA(cfg=cfg.model.kmeans.kmeans)
    elif cfg.model.kmeans.type == "repcodec":
        kmeans_model = RepCodec(cfg=cfg.model.kmeans.repcodec)
    kmeans_model.eval()
    pretrained_path =cfg.model.kmeans.pretrained_path
    if ".bin" in pretrained_path:
        kmeans_model.load_state_dict(torch.load(pretrained_path))
    elif ".safetensors" in pretrained_path:
        safetensors.torch.load_model(kmeans_model, pretrained_path)
    kmeans_model.to(device)
    return kmeans_model

def build_semantic_model(cfg, device):
    semantic_model = Wav2Vec2BertModel.from_pretrained("./ckpt/w2v-bert-2")
    semantic_model.eval()
    semantic_model.to(device)
    # layer_idx = 15
    # if layer_idx == 23:
    #     output_idx = 0
    # else:
    #     output_idx = layer_idx + 2
    layer_idx = 15
    output_idx = 17
    stat_mean_var = torch.load(cfg.model.kmeans.stat_mean_var_path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    # print(
    #     "semantic mean: ", semantic_mean, "semantic std: ", semantic_std
    # )
    return semantic_model, semantic_mean, semantic_std

def build_codec_model(cfg, device):
    codec_encoder = CodecEncoder(cfg=cfg.model.codec.encoder)
    codec_decoder = CodecDecoder(cfg=cfg.model.codec.decoder)
    if ".bin" in cfg.model.codec.encoder.pretrained_path:
        codec_encoder.load_state_dict(
            torch.load(cfg.model.codec.encoder.pretrained_path)
        )
        codec_decoder.load_state_dict(
            torch.load(cfg.model.codec.decoder.pretrained_path)
        )
    else:
        accelerate.load_checkpoint_and_dispatch(codec_encoder, cfg.model.codec.encoder.pretrained_path)
        accelerate.load_checkpoint_and_dispatch(codec_decoder, cfg.model.codec.decoder.pretrained_path)
    codec_encoder.eval()
    codec_decoder.eval()
    codec_encoder.to(device)
    codec_decoder.to(device)
    return codec_encoder, codec_decoder

@torch.no_grad()
def extract_acoustic_code(speech):
    vq_emb = codec_encoder(speech.unsqueeze(1))
    _, vq, _, _, _ = codec_decoder.quantizer(vq_emb)
    acoustic_code = vq.permute(
        1, 2, 0
    )  # (num_quantizer, T, C) -> (T, C, num_quantizer)
    return acoustic_code

@torch.no_grad()
def extract_semantic_code(semantic_mean, semantic_std, input_features, attention_mask):
    vq_emb = semantic_model(
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = vq_emb.hidden_states[17]  # (B, T, C)
    feat = (feat - semantic_mean.to(feat)) / semantic_std.to(feat)

    semantic_code, rec_feat = kmeans_model.quantize(feat)  # (B, T)
    return semantic_code, rec_feat

@torch.no_grad()
def extract_features(speech, processor):
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"][0]
    attention_mask = inputs["attention_mask"][0]
    return input_features, attention_mask

@torch.no_grad()
def text2semantic(prompt_speech, prompt_text, prompt_language, target_text, target_language, target_len=800, use_prompt_text=True, n_timesteps=50):
    if use_prompt_text:
        prompt_phone_id = liwei_g2p_(prompt_text, prompt_language)[1]
        prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(device)

        target_phone_id = liwei_g2p_(target_text, target_language)[1]
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(device)  

        # phone_id = torch.cat([prompt_phone_id, torch.LongTensor([4]).to(device), target_phone_id])
        phone_id = torch.cat([prompt_phone_id, target_phone_id]) 
    else:
        target_phone_id = liwei_g2p_(target_text, target_language)[1]
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(device)
        phone_id = target_phone_id

    input_fetures, attention_mask = extract_features(prompt_speech, processor)
    input_fetures = input_fetures.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    semantic_code, _ = extract_semantic_code(semantic_mean, semantic_std, input_fetures, attention_mask)

    predict_semantic = t2s_model.reverse_diffusion(semantic_code[:, :], target_len, phone_id.unsqueeze(0), n_timesteps=n_timesteps, cfg=2.5, rescale_cfg=0.75)

    print("predict semantic shape", predict_semantic.shape)

    combine_semantic_code = torch.cat([semantic_code[:,:], predict_semantic], dim=-1)
    prompt_semantic_code = semantic_code

    return combine_semantic_code, prompt_semantic_code

@torch.no_grad()
def semantic2acoustic(combine_semantic_code, acoustic_code):

    semantic_code = combine_semantic_code

    if soundstorm_1layer.cond_code_layers == 1:
        cond = soundstorm_1layer.cond_emb(semantic_code)
    else:
        cond = soundstorm_1layer.cond_emb[0](semantic_code[0,:,:])
        for i in range(1, soundstorm_1layer.cond_code_layers):
            cond += soundstorm_1layer.cond_emb[i](semantic_code[i,:,:])
        cond  = cond / math.sqrt(soundstorm_1layer.cond_code_layers)

    prompt = acoustic_code[:,:,:]
    # predict_1layer = soundstorm_1layer.reverse_diffusion(cond=cond, prompt=prompt, temp=1.5, filter_thres=0.98, n_timesteps=[40], cfg=1.0, rescale_cfg=1.0)
    predict_1layer = soundstorm_1layer.reverse_diffusion(cond=cond, prompt=prompt, temp=1.5, filter_thres=0.98, n_timesteps=[40], cfg=2.5, rescale_cfg=0.75)

    if soundstorm_full.cond_code_layers == 1:
        cond = soundstorm_full.cond_emb(semantic_code)
    else:
        cond = soundstorm_full.cond_emb[0](semantic_code[0,:,:])
        for i in range(1, soundstorm_full.cond_code_layers):
            cond += soundstorm_full.cond_emb[i](semantic_code[i,:,:])
        cond  = cond / math.sqrt(soundstorm_full.cond_code_layers)

    prompt = acoustic_code[:,:,:]
    # predict_full = soundstorm_full.reverse_diffusion(cond=cond, prompt=prompt, temp=1.5, filter_thres=0.98, n_timesteps=[40,16,10,10,10,10,10,10,10,10,10,10], cfg=1.0, rescale_cfg=1.0, gt_code=predict_1layer)
    predict_full = soundstorm_full.reverse_diffusion(cond=cond, prompt=prompt, temp=1.5, filter_thres=0.98, n_timesteps=[40,16,1,1,1,1,1,1,1,1,1,1], cfg=2.5, rescale_cfg=0.75, gt_code=predict_1layer)
    vq_emb = codec_decoder.vq2emb(predict_full.permute(2,0,1), n_quantizers=12)
    recovered_audio = codec_decoder(vq_emb)
    prompt_vq_emb = codec_decoder.vq2emb(prompt.permute(2,0,1), n_quantizers=12)
    recovered_prompt_audio = codec_decoder(prompt_vq_emb)
    recovered_prompt_audio = recovered_prompt_audio[0][0].cpu().numpy()
    recovered_audio = recovered_audio[0][0].cpu().numpy()
    combine_audio = np.concatenate([recovered_prompt_audio, recovered_audio])

    return combine_audio, recovered_audio

def build_aligner_model(device):
    aligner = ASRCNN(input_dim=1024, hidden_dim=256, n_token=1024, token_embedding_dim=256)
    aligner.to(device)
    aligner_ckpt = safetensors.torch.load_file("./ckpt/aligner/aligner.safetensors")
    aligner.load_state_dict(aligner_ckpt)
    return aligner

def build_dur_predictor(cfg, device):
    t2s_model = DurPredictorFM(cfg=cfg.model.dur_predictor)
    t2s_model.eval()
    t2s_model.to(device)
    return t2s_model

@torch.no_grad()
def get_frame_phone_id(phone_id, phone_mask, feat, feat_mask):
    frame_phone_id, phone_duration = aligner.inference(phone_id, phone_mask, feat, feat_mask)
    return frame_phone_id, phone_duration

@torch.no_grad()
def phone2dur(prompt_speech, prompt_text, target_text, prompt_language, target_language):
    prompt_phone_id = liwei_g2p_(prompt_text, prompt_language)[1]
    prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(device)
    prompt_phone_mask = torch.ones_like(prompt_phone_id)
    target_phone_id = liwei_g2p_(target_text, target_language)[1]
    target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(device)
    # phone_id = torch.cat([prompt_phone_id, torch.LongTensor([4]).to(device), target_phone_id])
    phone_id = torch.cat([prompt_phone_id, target_phone_id])

    input_fetures, attention_mask = extract_features(prompt_speech, processor)
    input_fetures = input_fetures.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    semantic_code, rec_feat = extract_semantic_code(semantic_mean, semantic_std, input_fetures, attention_mask)

    _, prompt_phone_dur = get_frame_phone_id(prompt_phone_id.unsqueeze(0), prompt_phone_mask.unsqueeze(0), rec_feat, attention_mask)
    prompt_phone_dur = prompt_phone_dur.unsqueeze(-1)
    prompt_phone_dur = torch.log(prompt_phone_dur + 1)
    # print(prompt_phone_dur)

    phone_id = phone_id.unsqueeze(0)

    cond = dur_predictor.cond_emb(phone_id)
    target_phone_dur = dur_predictor.reverse_diffusion(cond, prompt_phone_dur, n_timesteps=4, cfg=0, rescale_cfg=1)
    target_phone_dur = target_phone_dur.squeeze(-1)
    # print(target_phone_dur)
    target_phone_dur = torch.exp(target_phone_dur) - 1

    prompt_phone_dur = prompt_phone_dur.squeeze(-1)
    prompt_phone_dur = torch.exp(prompt_phone_dur) - 1

    phone_dur = torch.cat([prompt_phone_dur, target_phone_dur], dim=1).long()
    phone_dur = phone_dur[:,:phone_id.shape[1]]
    if phone_dur.shape[1] < phone_id.shape[1]:
        phone_dur = F.pad(phone_dur, (phone_id.shape[1]-phone_dur.shape[1],0,0,0), 'constant', 3)
    
    repeat = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(phone_id, phone_dur)][0].unsqueeze(0)

    target_len = repeat.shape[1] - input_fetures.shape[1]

    return repeat, target_len

    # return repeat, target_len, target_len_2

def make_test_list(testset_floder, prompt_type, language_type):
    test_list = []
    language = ""
    if language_type == "en":
        language = "en"
    elif language_type == "zh" or language_type == "zh_hard":
        language = "zh"
    else:
        raise ValueError("language_type must be 'en' or 'zh' or 'zh_hard'")
    filename = "meta" if language_type != "zh_hard" else "hardcase"
    if prompt_type == "all":
        test_path = os.path.join(testset_floder, "{}/{}.lst".format(language, filename))
        with open(test_path, "r", encoding='utf-8') as f:
            test_list = f.readlines()
    elif prompt_type.startswith("within"):
        limit = float(prompt_type.split(" ")[-1])
        all_test_list = os.path.join(testset_floder, "{}/{}.lst".format(language, filename))
        with open(all_test_list, "r", encoding='utf-8') as f:
            test_info = f.readlines()
        for info in tqdm(test_info):
            _, _, prompt_wav, _ = info.strip().split("|")
            prompt_wav = os.path.join(testset_floder, os.path.join(language, prompt_wav))
            y, sr = librosa.load(prompt_wav, sr = None)
            duration = librosa.get_duration(y = y, sr = sr)
            if duration <= limit:
                test_list.append(info.strip())
        test_path = os.path.join(testset_floder, "{}/{}_within_{}.lst".format(language, filename, limit))
        with open(test_path, "w", encoding='utf-8') as f:
            for line in test_list:
                f.write(line + "\n")
    else:
        raise ValueError("prompt_type must be 'all' or 'within n(s)'")
    return test_list, language

def load_asr_model(asr_en_model, asr_zh_model, language, device):
    if language == "en":
        processor = WhisperProcessor.from_pretrained(asr_en_model)
        model = WhisperForConditionalGeneration.from_pretrained(asr_en_model).to(device)
    elif language == "zh":
        model = AutoModel(model=asr_zh_model, device=accelerator.process_index)
        processor = None
    return processor, model

def process_asr_one(hypo, truth, language):
    raw_truth = truth
    raw_hypo = hypo
    punctuation_all = punctuation + string.punctuation
    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')
    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if language == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif language == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)

def extract_wavlm_similarity(target_wav, reference_wav, speaker_encoder):
    
    emb1 = speaker_encoder(target_wav)  # emb.shape = (batch_size, embedding_dim)
    emb1 = emb1.cpu() 

    emb2 = speaker_encoder(reference_wav)  # emb.shape = (batch_size, embedding_dim)
    emb2 = emb2.cpu()
    
    sim = F.cosine_similarity(emb1, emb2)
    cos_sim_score = sim[0].item()
    return cos_sim_score


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
accelerator = Accelerator()
device = accelerator.device if torch.cuda.is_available() else "cpu"
cfg_soundstorm_1layer = load_config("./egs/tts/SoundStorm/exp_config_16k_emilia_llama_new_semantic_repcodec_8192_1q_1layer_24k.json")
cfg_soundstorm_full = load_config("./egs/tts/SoundStorm/exp_config_16k_emilia_llama_new_semantic_repcodec_8192_1q_24k.json")
soundstorm_1layer = build_soundstorm(cfg_soundstorm_1layer, device)
soundstorm_full = build_soundstorm(cfg_soundstorm_full, device)
semantic_model, semantic_mean, semantic_std = build_semantic_model(cfg_soundstorm_full, device)
kmeans_model = build_kmeans_model(cfg_soundstorm_full, device)
codec_encoder, codec_decoder = build_codec_model(cfg_soundstorm_full, device)

soundstorm_1layer_path = "./ckpt/soundstorm/soundstorm_16k_emilia_50k_llama_new_semantic_repcodec_8192_86kstep_1layer_24k/561ksteps/model.safetensors"
soundstorm_full_path = "./ckpt/soundstorm/soundstorm_16k_emilia_50k_llama_new_semantic_repcodec_8192_86kstep_full_24k/519ksteps/model.safetensors"
safetensors.torch.load_model(soundstorm_1layer, soundstorm_1layer_path)
safetensors.torch.load_model(soundstorm_full, soundstorm_full_path)

t2s_cfg = load_config("./egs/tts/DiffT2S/exp_config_difft2s_prefix_phone_large.json")
t2s_model = build_t2s_model(t2s_cfg, device)
safetensors.torch.load_model(t2s_model, "./ckpt/difft2s/712ksteps/model.safetensors")
# print(t2s_model.bos_target_id, t2s_model.eos_target_id, t2s_model.bos_phone_id, t2s_model.eos_phone_id, t2s_model.pad_token_id)
dur_cfg = load_config("./egs/tts/DiffT2S/exp_config_fm_dur.json")
dur_predictor = build_dur_predictor(dur_cfg, device)
safetensors.torch.load_model(dur_predictor, "./ckpt/dur_predictor/204ksteps/model.safetensors")
# https://cuhko365-my.sharepoint.com/:f:/g/personal/119010319_link_cuhk_edu_cn/EiRWbFcN5SJClnzRc1PjN8kB4gGqiOC6oGgPr0f6hfQOsQ?e=oR0mle

aligner = build_aligner_model(device)

if __name__ == "__main__":
    
    # Sample rate set
    t2s_sr = 16000 # AR is 16k
    soundstorm_sr = cfg_soundstorm_1layer.preprocess.sample_rate # NAR is 24k

    # AR setting
    n_timesteps = [10]# [10, 25, 50, 75]
    
    # Prompt type setting
    prompt_type = "all" # "all" or "within 5"(within n(s))...

    # Rerank setting
    rerank_times = 2

    # Metric setting
    asr_en_model = "openai/whisper-large-v3" # huggingface model id
    asr_zh_model = "paraformer-zh" # hugingface model id
    sim_model = "./wavlm_large_finetune.pth" # local model path
    cal_wer = True
    cal_sim = True

    # Language setting
    language_type = "en" # "en" or "zh" or "zh_hard"

    # Folder setting
    save_folder = "./seedtts_result_diff"
    testset_floder = "./seedtts_testset" # The folder of SeedTTS testset
    print("Make list...")
    test_list, language = make_test_list(testset_floder, prompt_type, language_type)
    prompt_type = prompt_type.replace(" ", "_")
   
    print("Start inference...")
    for n_timestep in n_timesteps:
        for i in range(rerank_times):
            print("Rerank {}/{}: SeedTTS {} rerank in {} with n_timesteps={}".format(i + 1, rerank_times, language_type, prompt_type, n_timestep))
            save_path = os.path.join(save_folder, "seedtts_{}_{}/rerank_{}".format(language_type, n_timestep, i + 1))
            os.makedirs(save_path, exist_ok=True)
            accelerator.wait_for_everyone()
            start=time.time()

            # divide the prompt list onto the available GPUs 
            with accelerator.split_between_processes(test_list) as test_data:
            
                # have each GPU do inference, prompt by prompt
                for info in tqdm(test_data):
                    target_name, prompt_text, prompt_wav, target_text = info.strip().split("|")
                    if os.path.exists(os.path.join(save_path, "{}.wav".format(target_name))):
                        continue
                    prompt_wav = os.path.join(testset_floder, os.path.join(language, prompt_wav))
                    speech_16k = librosa.load(prompt_wav, sr=16000)[0]
                    speech = librosa.load(prompt_wav, sr=cfg_soundstorm_1layer.preprocess.sample_rate)[0]
            
                    if_success = False
                    while if_success == False:
                        try:
                            frame_phone_id, target_len = phone2dur(speech_16k, prompt_text, target_text, language, language)
                            # 获取帧数
                            print("target_len: ", target_len)
                            combine_semantic_code, _ = text2semantic(speech_16k, prompt_text, language, target_text, language, target_len, n_timesteps=n_timestep)
                            acoustic_code = extract_acoustic_code(torch.tensor(speech).unsqueeze(0).to(device))
                            print(acoustic_code.shape)
                            _, recovered_audio = semantic2acoustic(combine_semantic_code, acoustic_code)
                            if_success = True
                        except:
                            pass
                    sf.write(os.path.join(save_path, "{}.wav".format(target_name)), recovered_audio, samplerate=soundstorm_sr)
            
            if accelerator.is_main_process:
                timediff = time.time()-start
                print("Inference in {} s".format(timediff))
    
        print("Inference done!")

        accelerator.wait_for_everyone()
        if cal_wer == True:
            print("Start calculate WER...")
            asr_processor, model = load_asr_model(asr_en_model, asr_zh_model, language, device)
            wer_scores = []
            for i in range(rerank_times):
                print("Rerank {}/{}: Cal WER {} rerank in {} with n_timesteps={}".format(i + 1, rerank_times, language_type, prompt_type, n_timestep))
                save_path = os.path.join(save_folder, "seedtts_{}_{}/rerank_{}".format(language_type, n_timestep, i + 1))
                res_path = os.path.join(save_path, "wav_res_ref_text_{}.wer".format(prompt_type))
                all_res_path = os.path.join(save_path, "wav_res_ref_text_all.wer")
                tmp_wer = []
                if os.path.exists(res_path):
                    if accelerator.is_main_process:
                        with open(res_path, "r", encoding='utf-8') as f:
                            lines = f.readlines()[1:-1]
                            assert len(lines) == len(test_list)
                            for line in lines:
                                tmp_wer.append(float(line.strip().split("\t")[1]))
                elif prompt_type != "all" and os.path.exists(all_res_path):
                    if accelerator.is_main_process:
                        filter_list = []
                        for info in tqdm(test_list):
                            target_name, _, _, _ = info.strip().split("|")
                            output_wav_path = os.path.join(save_path, target_name + ".wav")
                            filter_list.append(output_wav_path)
                        fout = open(res_path, "w", encoding='utf-8')
                        fout.write("wav_res" + '\t' + 'res_wer' + '\t' + 'text_ref' + '\t' + 'text_res' + '\t' + 'res_wer_ins' + '\t' + 'res_wer_del' + '\t' + 'res_wer_sub' + '\n')
                        with open(all_res_path, "r", encoding='utf-8') as f:
                            lines = f.readlines()[1:-1]
                            for line in lines:
                                output_wav_path, wer, raw_truth, raw_hypo, inse, dele, subs = line.strip().split("\t")
                                if output_wav_path in filter_list:
                                    tmp_wer.append(float(wer))
                                    fout.write(f"{output_wav_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
                                    fout.flush()
                        fout.write(f"WER: {round(np.mean(tmp_wer)*100, 4)}%\n")
                        fout.close()  
                else:
                    # divide the prompt list onto the available GPUs 
                    with accelerator.split_between_processes(test_list) as test_data:
                        results = dict(outputs = [], tmp_wer = [])
                        for info in tqdm(test_data):
                            target_name, prompt_text, prompt_wav, target_text = info.strip().split("|")
                            output_wav_path = os.path.join(save_path, "{}.wav".format(target_name))
                            if language == "en":
                                wav, sr = sf.read(output_wav_path)
                                if sr != 16000:
                                    wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
                                input_features = asr_processor(wav, sampling_rate=16000, return_tensors="pt").input_features
                                input_features = input_features.to(device)
                                forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="english", task="transcribe")
                                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                                transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                            elif language == "zh":
                                res = model.generate(input=output_wav_path, batch_size_s=300)
                                transcription = res[0]["text"]
                                transcription = zhconv.convert(transcription, 'zh-cn')
                            raw_truth, raw_hypo, wer, subs, dele, inse = process_asr_one(transcription, target_text, language)
                            
                            results["outputs"].append(f"{output_wav_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
                            results["tmp_wer"].append(wer)
                        results = [results]
                    # collect results from all the GPUs
                    results_gathered = gather_object(results)
                    if accelerator.is_main_process:
                        fout = open(res_path, "w", encoding='utf-8')
                        fout.write("wav_res" + '\t' + 'res_wer' + '\t' + 'text_ref' + '\t' + 'text_res' + '\t' + 'res_wer_ins' + '\t' + 'res_wer_del' + '\t' + 'res_wer_sub' + '\n')
                        for r in results_gathered:
                            for w in r["tmp_wer"]:
                                tmp_wer.append(w)
                            for out in r["outputs"]:
                                fout.write(str(out))
                        fout.flush()
                        fout.write(f"WER: {round(np.mean(tmp_wer)*100, 4)}%\n")
                        fout.close()
                if accelerator.is_main_process:
                    # print(f"WER: {np.mean(tmp_wer)}")                         
                    wer_scores.append(tmp_wer)
            if accelerator.is_main_process:
                print(f"File number: {len(np.min(wer_scores, axis=0))}")
                print(f"All WER: {round(np.mean(np.min(wer_scores, axis=0))*100, 4)}%")

        if cal_sim == True:
            print("Start calculate SIM...")
            speaker_encoder = init_model(checkpoint = sim_model).to(device).eval()
            similarity_scores = []
            for i in range(rerank_times):
                print("Rerank {}/{}: Cal SIM {} rerank in {} with n_timesteps={}".format(i + 1, rerank_times, language_type, prompt_type, n_timestep))
                save_path = os.path.join(save_folder, "seedtts_{}_{}/rerank_{}".format(language_type, n_timestep, i + 1))
                res_path = os.path.join(save_path, "wav_res_ref_text_{}.sim".format(prompt_type))
                all_res_path = os.path.join(save_path, "wav_res_ref_text_all.sim")
                tmp_sim = []
                if os.path.exists(res_path):
                    if accelerator.is_main_process:
                        with open(res_path, "r", encoding='utf-8') as f:
                            lines = f.readlines()[1:-1]
                            for line in lines:
                                tmp_sim.append(float(line.strip().split("\t")[-1]))
                elif prompt_type != "all" and os.path.exists(all_res_path):
                    if accelerator.is_main_process:
                        filter_list = []
                        for info in tqdm(test_list):
                            target_name, _, _, _ = info.strip().split("|")
                            output_wav_path = os.path.join(save_path, target_name + ".wav")
                            filter_list.append(output_wav_path)
                        fout = open(res_path, "w", encoding='utf-8')
                        fout.write("utt" + '\t' + "wav_res" + '\n')
                        with open(all_res_path, "r", encoding='utf-8') as f:
                            lines = f.readlines()[1:-1]
                            for line in lines:
                                output_wav_path, sim_o = line.strip().split("\t")
                                if output_wav_path in filter_list:
                                    tmp_sim.append(float(sim_o))
                                    fout.write(f"{output_wav_path}\t{sim_o}\n")
                                    fout.flush() 
                        fout.write(f"SIM-O: {round(np.mean(tmp_sim)*100, 4)}%\n")
                        fout.close() 
                else:
                    # divide the prompt list onto the available GPUs 
                    with accelerator.split_between_processes(test_list) as test_data:
                        results = dict(outputs = [], tmp_sim = [])
                        for info in tqdm(test_data):
    
                            target_name, prompt_text, prompt_wav, target_text = info.strip().split("|")    
                            original_wav_path = os.path.join(testset_floder, os.path.join(language, prompt_wav))
                            reference_wav, sr = librosa.load(original_wav_path)
                            reference_wav = torch.tensor(reference_wav).to(device)
                            if sr != 16000:
                                reference_wav = torchaudio.functional.resample(reference_wav, orig_freq=sr, new_freq=16000)
                            output_wav_path = os.path.join(save_path, target_name + ".wav")
                            output_wav, sr = librosa.load(output_wav_path)
                            output_wav = torch.tensor(output_wav).to(device)
                            if sr != 16000:
                                output_wav = torchaudio.functional.resample(output_wav, orig_freq=sr, new_freq=16000)
                            sim_o = extract_wavlm_similarity(output_wav.unsqueeze(0), reference_wav.unsqueeze(0), speaker_encoder)
                            results["outputs"].append(f"{output_wav_path}\t{sim_o}\n")
                            results["tmp_sim"].append(sim_o)
                        results = [results]

                    # collect results from all the GPUs
                    results_gathered = gather_object(results)
                    if accelerator.is_main_process:
                        fout = open(res_path, "w", encoding='utf-8')
                        fout.write("utt" + '\t' + "wav_res" + '\n')
                        for r in results_gathered:
                            for s in r["tmp_sim"]:
                                tmp_sim.append(s)
                            for out in r["outputs"]:
                                fout.write(str(out))
                        fout.flush()
                        fout.write(f"SIM-O: {round(np.mean(tmp_sim)*100, 4)}%\n")
                        fout.close()
                if accelerator.is_main_process:
                    # print(f"SIM-O: {np.mean(tmp_sim)}")
                    similarity_scores.append(tmp_sim)
            if accelerator.is_main_process:
                print(f"File number: {len(np.min(similarity_scores, axis=0))}")                 
                print(f"All SIM-O: {round(np.mean(np.max(similarity_scores, axis=0))*100, 4)}%")
