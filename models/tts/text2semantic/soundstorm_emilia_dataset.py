# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import oss2  # pip install oss2
import io
import librosa
import torch
import json
import tqdm
import numpy as np
import logging
import pickle
import os
import time
from torch.utils.data import Dataset
from multiprocessing import Pool
import concurrent.futures
from pathlib import Path
from transformers import SeamlessM4TFeatureExtractor
from utils.g2p.g2p import phonemizer_g2p

os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1'
os.environ['PHONEMIZER_ESPEAK_PATH'] = '/usr/bin/espeak-ng'

LANG2CODE = {
    'zh': 349,
    'en': 350,
    'ja': 351,
    'ko': 352,
    'fr': 353,
    'de': 354,
}

class PhonemizerWarningFilter(logging.Filter):
    def filter(self, record):
        # 只过滤 phonemizer 中的 WARNING 级别日志
        if record.name == "phonemizer" and record.levelno == logging.WARNING:
            return False
        return True


logger = logging.getLogger("phonemizer")
filter = PhonemizerWarningFilter()
logger.addFilter(filter)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AK = "xxx"
SK = "xxx"
bucket_name = "pjlab-3090-openmmlabpartner"
MOUNT_PATH = "/mnt/data/oss_beijing/"
data_json_path = "Emilia/Emilia-zh+en/Emilia-1k.json.gz"
data_json_path = "/mnt/bn/yuacnwang-speech/dataset/Emilia/emilia_json/Emilia-50k.json.gz"
duration_setting = {'min': 3, 'max': 20}


class SoundStormDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        access_key_id=AK,
        access_key_secret=SK,
        bucket_name=bucket_name,
        cache_type="path",
        cfg=None,
    ):  # 'path' or 'meta'
        self.cache_type = cache_type
        self.cfg = cfg
        # Initialize OSS client
        self.init_client(access_key_id, access_key_secret, bucket_name)
        self.json_paths = []
        self.wav_paths = []
        self.language_list = ["zh", "en"]  # Data language list
        self.wav_path_index2duration = []
        self.wav_path_index2phonelen = []
        self.index2num_frames = []

        self.json_path2meta = {}
        self.json2filtered_idx = {}

        # self.cache_folder = "cache/{}_cache".format(cache_type)
        self.cache_folder = '/mnt/petrelfs/hehaorui/jiaqi/vc-dev/cache/path_cache'
        # self.cache_folder = "/mnt/bn/yuacnwang-speech/dataset/Emilia/cache/emilia_50k"
        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)

        self.wav_paths_cache = os.path.join(self.cache_folder, "wav_paths_cache.pkl")
        self.json_paths_cache = os.path.join(self.cache_folder, "json_paths_cache.pkl")
        self.duration_cache = os.path.join(self.cache_folder, "duration_cache.pkl")
        self.phone_count_cache = os.path.join(
            self.cache_folder, "phone_count_cache.pkl"
        )
        self.json_path2meta_cache = os.path.join(
            self.cache_folder, "json_path2meta.pkl"
        )

        if cache_type == "path":
            if (
                os.path.exists(self.wav_paths_cache)
                and os.path.exists(self.json_paths_cache)
                and os.path.exists(self.duration_cache)
                and os.path.exists(self.phone_count_cache)
            ):
                self.load_cached_paths()
            else:
                logger.info("No cache exists")
                self.get_all_paths_from_json(data_json_path)
                self.save_cached_paths()
        elif cache_type == "meta":
            if os.path.exists(self.wav_paths_cache) and os.path.exists(
                self.json_paths_cache
            ):
                self.load_cached_paths()
            else:
                logger.info("No cache exists")
                self.get_all_paths_from_json(data_json_path)
                self.save_cached_paths()
        else:
            logger.info("Incorrect cache loading way")
            exit()

        if cache_type == "meta":
            if os.path.exists(self.json_path2meta_cache):
                self.load_path2meta()
            else:
                self.get_jsoncache_multiprocess(pool_size=8)

        self.num_frame_indices = np.array(
            sorted(
                range(len(self.index2num_frames)),
                key=lambda k: self.index2num_frames[k],
            )
        )

        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

    def init_client(self, access_key_id, access_key_secret, bucket_name):

        logger.info("Start to initialize OSS client")
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(
            self.auth, "https://oss-cn-beijing.aliyuncs.com", bucket_name
        )
        logger.info("OSS client initialized successfully")

    def load_cached_paths(self):
        logger.info("Loaded paths from cache files")
        with open(self.wav_paths_cache, "rb") as f:
            self.wav_paths = pickle.load(f)
        with open(self.json_paths_cache, "rb") as f:
            self.json_paths = pickle.load(f)
        if self.cache_type == "path":
            with open(self.duration_cache, "rb") as f:
                self.wav_path_index2duration = pickle.load(f)
            with open(self.phone_count_cache, "rb") as f:
                self.wav_path_index2phonelen = pickle.load(f)
            # for duration, phone_count in zip(self.wav_path_index2duration, self.wav_path_index2phonelen):
            #     self.index2num_frames.append(duration * num_token_per_second + phone_count)
            for duration in self.wav_path_index2duration:
                self.index2num_frames.append(duration * self.cfg.preprocess.sample_rate)
        logger.info("All paths got successfully")
        logger.info(
            "Number of wavs: %d, Number of jsons: %d"
            % (len(self.wav_paths), len(self.json_paths))
        )

    def save_cached_paths(self):
        with open(self.wav_paths_cache, "wb") as f:
            pickle.dump(self.wav_paths, f)
        with open(self.json_paths_cache, "wb") as f:
            pickle.dump(self.json_paths, f)
        if self.cache_type == "path":
            with open(self.duration_cache, "wb") as f:
                pickle.dump(self.wav_path_index2duration, f)
            with open(self.phone_count_cache, "wb") as f:
                pickle.dump(self.wav_path_index2phonelen, f)
        logger.info("Saved paths to cache files")

    # Load JSON data from a compressed GZIP file
    def load_compressed_json(self, filename):
        import gzip

        with gzip.open(filename, "rt", encoding="utf-8") as f:
            return json.load(f)
    def g2p(self, text, language):
        return phonemizer_g2p(text, language)

    def get_path_from_json(self, data): 
        if data['language'][0] not in self.language_list:
            return 
        self.json_paths.append(data['json_path'])
        is_exists = True
        try:
            if not self.bucket.object_exists(data['wav_path'][0]):
                is_exists = False
        except oss2.api.Exception as e:
            is_exists = False
        remove_idx = []
        for wav, duration, phone_count in zip(data['wav_path'], data['duration'], data['phone_count']):
            if duration < duration_setting['min'] or duration > duration_setting['max']:
                idx = wav.split("_")[-1].split(".")[0]
                remove_idx.append(idx)
                continue
            if is_exists:
                self.wav_paths.append(wav)
            else:
                if '.mp3' in wav:
                    wav = wav.replace('.mp3', '.wav')
                    self.wav_paths.append(wav)
                else:
                    wav = wav.replace('.wav', '.mp3')
                    self.wav_paths.append(wav)
            self.wav_path_index2duration.append(duration)
            self.wav_path_index2phonelen.append(phone_count)
            self.index2num_frames.append(duration * self.cfg.preprocess.sample_rate)
        
        self.json2filtered_idx[data['json_path']] = [int(i) for i in data['filtered_idx'].split(',') if i not in remove_idx]
        if not self.json2filtered_idx[data['json_path']]:
            self.json_paths.pop()


    def get_all_paths_from_json(self, json_path):

        data_list = self.load_compressed_json(json_path)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.get_path_from_json, data)
                for data in tqdm.tqdm(data_list)
            ]
            data = [future.result() for future in tqdm.tqdm(futures)]

    # Only 'meta' cache type use
    def get_phone_count_and_duration(self, meta, idx_list):
        new_meta = {}
        if meta[0]["language"] not in self.language_list:
            new_meta["0"] = meta[0]
            return new_meta
        text_list = []
        for i in idx_list:
            text_list.append(meta[i]["text"])
        token_id = self.g2p(text_list, meta[0]["language"])[1]
        for i, token in zip(idx_list, token_id):
            nm = {}
            nm["language"] = meta[i]["language"]
            nm["phone_id"] = token
            nm["phone_count"] = len(token)
            nm["duration"] = meta[i]["end"] - meta[i]["start"]
            new_meta[str(i)] = nm
        del meta
        return new_meta

    # Only 'meta' cache type use
    def process_json_cache(self, json_path):
        default_meta = [{"text": "-1", "language": "others"}]
        try:
            file_bytes = self.bucket.get_object(json_path)
            buffer = io.BytesIO(file_bytes.read())
            json_cache = json.load(buffer)
            del buffer, file_bytes
            if json_cache is None:
                logger.info("json is none")
            elif isinstance(json_cache, (dict, list)) and not json_cache:
                logger.info("json is none")
            else:
                return json_cache
        except oss2.exceptions.NoSuchKey as e:
            logger.info(
                "Not found: http_status={0}, request_id={1}".format(
                    e.status, e.request_id
                )
            )
        except Exception as e:
            logger.info("Error json: {} error: {}".format(json_path, e))
        return default_meta

    # Only 'meta' cache type use
    def get_jsoncache_multiprocess(self, pool_size):
        logger.info("Start to build json pool")
        logger.info("Start to get json cache")
        json2meta = []
        json_data = []
        tmp_json_cache = os.path.join(self.cache_folder, "json_cache.pkl")
        if os.path.exists(tmp_json_cache):
            with open(tmp_json_cache, "rb") as f:
                json_data = pickle.load(f)
            logging.info("Load json_cache.pkl")
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=pool_size
            ) as executor:
                futures = [
                    executor.submit(self.process_json_cache, path)
                    for path in self.json_paths
                ]
                json_data = [future.result() for future in tqdm.tqdm(futures)]
            with open(tmp_json_cache, "wb") as f:
                pickle.dump(json_data, f)
            logging.info("Save json_cache.pkl")
        logging.info("Get meta from cache")
        for json, path in tqdm.tqdm(
            zip(json_data, self.json_paths), total=len(json_data)
        ):
            # print(json)
            json2meta.append(
                self.get_phone_count_and_duration(json, self.json2filtered_idx[path])
            )
        error_json_path_list = []
        for i in range(len(json2meta)):
            if (
                json2meta[i][next(iter(json2meta[i]))]["language"]
                not in self.language_list
            ):
                error_json_path_list.append(self.json_paths[i])
            else:
                self.json_path2meta[self.json_paths[i]] = json2meta[i]
        logger.info("Remove error json path {}".format(error_json_path_list))
        error_wav_path_list = []
        for error in tqdm.tqdm(error_json_path_list):
            self.json_paths.remove(error)
            error = error.split(".json")[0]
            for wav in self.wav_paths:
                if error in wav:
                    error_wav_path_list.append(wav)
        logger.info("Remove error wav path {}".format(error_wav_path_list))
        for error in tqdm.tqdm(error_wav_path_list):
            self.wav_paths.remove(error)
        logger.info("Update cache")
        with open(self.wav_paths_cache, "wb") as f:
            pickle.dump(self.wav_paths, f)
        with open(self.json_paths_cache, "wb") as f:
            pickle.dump(self.json_paths, f)
        with open(self.json_path2meta_cache, "wb") as f:
            pickle.dump(self.json_path2meta, f)
        logger.info("Json cache write to json_path2meta.pkl successfully")
        del json2meta, error_wav_path_list, error_json_path_list

    # Only 'meta' cache type use
    def load_path2meta(self):
        logger.info("Loaded meta from cache files")
        self.json_path2meta = pickle.load(open(self.json_path2meta_cache, "rb"))
        for path in self.wav_paths:
            duration = self.get_meta_from_wav_path(path)["duration"]
            phone_count = self.get_meta_from_wav_path(path)["phone_count"]
            self.wav_path_index2duration.append(duration)
            self.wav_path_index2phonelen.append(phone_count)
            # self.index2num_frames.append(duration * num_token_per_second + phone_count)
            self.index2num_frames.append(duration * self.cfg.preprocess.sample_rate)

    def get_meta_from_wav_path(self, wav_path):
        index = int(wav_path.split("_")[-1].split(".")[0])
        audio_name = "_".join(wav_path.split("/")[-1].split("_")[:-1])
        dir_name = "/".join(wav_path.split("/")[:-1])
        json_name = audio_name + ".json"
        json_path = dir_name + "/" + json_name
        meta = None
        if self.cache_type == "meta":
            meta = self.json_path2meta[json_path][str(index)]
            return meta
        elif self.cache_type == "path":
            try:
                file_bytes = self.bucket.get_object(json_path)
                buffer = io.BytesIO(file_bytes.read())
                meta = json.load(buffer)[index]
            except oss2.exceptions.NoSuchKey as e:
                logger.info(
                    "Not found: http_status={0}, request_id={1}".format(
                        e.status, e.request_id
                    )
                )
            except Exception as e:
                logger.info("Error json: {} error: {}".format(json_path, e))
        del index, audio_name, dir_name, json_name, json_path
        return meta

    def __len__(self):
        return self.wav_paths.__len__()

    def get_num_frames(self, index):
        # return self.wav_path_index2duration[index] * num_token_per_second + self.wav_path_index2phonelen[index]
        return self.wav_path_index2duration[index] * self.cfg.preprocess.sample_rate

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        file_bytes = None
        try:
            # for i in range(3):
            #     try:
            #         file_bytes = self.bucket.get_object(wav_path.replace("_new", ""))
            #         break
            #     except Exception as e:
            #         print(f"[Filter meta func] Error is {e}")
            #         time.sleep(i)
            #         print("retry")
            file_bytes = self.bucket.get_object(wav_path.replace("_new", ""))
        except:
            logger.info("Get data from oss failed. Get another.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            del position
            return self.__getitem__(random_index)

        meta = self.get_meta_from_wav_path(wav_path)
        if file_bytes is not None and meta is not None:
            buffer = io.BytesIO(file_bytes.read())

            try:
                speech, sr = librosa.load(buffer, sr=self.cfg.preprocess.sample_rate)
                if len(speech) > duration_setting["max"] * self.cfg.preprocess.sample_rate:
                    position = np.where(self.num_frame_indices == idx)[0][0]
                    random_index = np.random.choice(self.num_frame_indices[:position])
                    del position
                    return self.__getitem__(random_index)
            except:
                logger.info("Failed to load file. Get another.")
                position = np.where(self.num_frame_indices == idx)[0][0]
                random_index = np.random.choice(self.num_frame_indices[:position])
                del position
                return self.__getitem__(random_index)

            single_feature = dict()

            # pad the speech to the multiple of hop_size
            speech = np.pad(
                speech,
                (
                    0,
                    self.cfg.preprocess.hop_size
                    - len(speech) % self.cfg.preprocess.hop_size,
                ),
                mode="constant",
            )
            # resample the speech to 16k for feature extraction
            if self.cfg.preprocess.sample_rate != 16000:
                speech_16k = librosa.resample(
                    speech, orig_sr=self.cfg.preprocess.sample_rate, target_sr=16000
                )
            else:
                speech_16k = speech
                
            phone_id = self.g2p(meta['text'], meta['language'])[1] if self.cache_type == 'path' else meta['phone_id']
            phone_id = torch.tensor(phone_id, dtype=torch.long)
            phone_id = torch.cat([torch.tensor(LANG2CODE[meta['language']], dtype=torch.long).reshape(1), phone_id]) # add language token
                
                
            inputs = self.processor(speech_16k, sampling_rate=16000)
            input_features = inputs["input_features"][0]
            attention_mask = inputs["attention_mask"][0]
            # get speech mask
            speech_frames = len(speech) // self.cfg.preprocess.hop_size
            mask = np.ones(speech_frames)
            
            phone_mask = np.ones(len(phone_id))

            single_feature.update(
                {
                    "input_features": input_features,
                    "attention_mask": attention_mask,
                    # "speech": speech,
                    "mask": mask,
                    "phone_id": phone_id,
                    "phone_mask": phone_mask,
                }
            )

            return single_feature

        else:
            logger.info("Failed to get file after retries.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            del position
            return self.__getitem__(random_index)
