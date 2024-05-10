import oss2 #pip install oss2
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
from utils.g2p import PhonemeBpeTokenizer
logging.basicConfig(level=logging.INFO)  # Configure logging level to INFO
logger = logging.getLogger(__name__)

LANG2CODE = {
    'en': 655,
    'zh': 654,
}
lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
    "fr": "[FR]",
    "kr": "[KR]",
    "de": "[DE]",
}
# 示例用法

class EmiliaDataset(Dataset):
    def __init__(self, access_key_id=AK, access_key_secret=SK, bucket_name=bucket_name, resample_to_16k=False):
        # Initialize OSS client
        self.init_client(access_key_id, access_key_secret, bucket_name) # 建立连接
        self.json_paths = []
        self.wav_paths = []
        self.json_path2meta = {}
        # Load paths from cache files if available, otherwise get them from OSS
        if os.path.exists("wav_paths_cache.pkl") and os.path.exists("json_paths_cache.pkl"):
            self.load_cached_paths()
        else:
            self.get_all_paths(language='zh', duration_limit=200) # 200 hours
            self.save_cached_paths()
        if os.path.exists("json_path2meta.pkl"):
            self.json_path2meta = pickle.load(open("json_path2meta.pkl", "rb"))
        else:
            self.get_jsoncache()
        # self.json_path2meta
        self.filter_by_meta()
        
        self.num_frame_indices = np.array(sorted(range(len(self.index2num_frames)), key=lambda k: self.index2num_frames[k]))

        self.text_tokenizer = PhonemeBpeTokenizer()

    def load_cached_paths(self):
        with open("wav_paths_cache.pkl", "rb") as f:
            self.wav_paths = pickle.load(f)
        with open("json_paths_cache.pkl", "rb") as f:
            self.json_paths = pickle.load(f)
        logger.info("Loaded paths from cache files")
        logger.info("All paths got successfully")
        # num wavs, num jsons
        logger.info(
            "Number of wavs: %d, Number of jsons: %d"
            % (len(self.wav_paths), len(self.json_paths))
        )

    def save_cached_paths(self):
        with open("wav_paths_cache.pkl", "wb") as f:
            pickle.dump(self.wav_paths, f)
        with open("json_paths_cache.pkl", "wb") as f:
            pickle.dump(self.json_paths, f)
        logger.info("Saved paths to cache files") 

    def init_client(self, access_key_id, access_key_secret, bucket_name):
        # 初始化 OSS 客户端
        logger.info("Start to initialize OSS client")
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(
            self.auth, "https://oss-cn-beijing.aliyuncs.com", bucket_name
        )
        logger.info("OSS client initialized successfully")


    def get_paths(self, path, num_wav):
        json_path = MOUNT_PATH + bucket_name + '/' + path.split(bucket_name)[1]
        self.json_paths.append(json_path)
        for i in range(num_wav):
            wav_path = json_path.split(".json")[0] + "_" + i + ".wav" # maybe '.mp3' in the furture
            self.wav_paths.append(wav_path)

    def get_all_paths(self, language, duration_limit):
        logger.info("Start to get all paths")
        from mysql.connector.pooling import MySQLConnection
        from AudioDataCollection.database.db_manager import DatabaseManager

        db_manager = DatabaseManager()
        try:
            conn: MySQLConnection = db_manager.get_connection()
            # 开启事务
            conn.start_transaction()

            cursor = conn.cursor()

            tmp_duration = 0
            page = 0
            num_per_page = 10
            duration_limit = duration_limit * 3600
            while tmp_duration <= duration_limit:

                cursor.execute(
                    f"""
                    SELECT result_path, valid_duration, valid_segement
                    FROM video
                    WHERE 'language' = {language} AND 'status' = 4
                    LIMIT {page * num_per_page}, {num_per_page}
                """
                )

                record = cursor.fetchall()

                if record is None:
                    conn.rollback()
                    continue

                # result_path, valid_duration, valid_segement = record

                for path, duration, num_wav in record:

                    self.get_paths(path, num_wav)
                    tmp_duration += duration

                page += 1
                
                logger.info("All paths got successfully")
                # num wavs, num jsons
                logger.info(
                    "Number of wavs: %d, Number of jsons: %d"
                    % (len(self.wav_paths), len(self.json_paths))
                )

        except Exception as e:
            # 发生错误时回滚事务
            conn.rollback()
            # raise e
            logger.info("Gat path error {}".format(e))
        finally:
            if conn:
                conn.close()


    def get_meta_from_wav_path(self, wav_path):
        index = int(wav_path.split("_")[-1].split(".")[0])  # 0
        audio_name = "_".join(wav_path.split("/")[-1].split("_")[:-1])
        dir_name = "/".join(wav_path.split("/")[:-1])
        json_name = audio_name + ".json"  # xmly00000_10028458_47798215.json
        json_path = dir_name + "/" + json_name
        meta = self.json_path2meta[json_path][index]
        return meta 

    def get_bounds(self, meta):
        avg_durations = []
        for m in meta:
            phone_count = len(m['phone_id'].split())
            duration = m['end'] - m['start']
            try:
                avg_durations.append(duration / phone_count)
            except:
                pass
        q1 = np.percentile(avg_durations, 25)
        q3 = np.percentile(avg_durations, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # 写进meta
        for m in meta:
            m['lower_bound'] = lower_bound
            m['upper_bound'] = upper_bound
        return meta

    def get_jsoncache(self):
        logger.info("Start to get json cache")
        for json_path in tqdm.tqdm(self.json_paths):
            try:
                file_bytes = self.bucket.get_object(json_path)
                buffer = io.BytesIO(file_bytes.read())
                json_cache = json.load(buffer)
                del buffer
                if json_cache is None:
                    logger.info("json is none")
                    continue
                elif isinstance(json_cache, (dict, list)) and not json_cache:
                    logger.info("json is none")
                    continue
                else:
                    logger.info(json_path)
                    json_cache = self.get_bounds(json_cache)
                    self.json_path2meta[json_path] = json_cache
            except oss2.exceptions.NoSuchKey as e:
                print(
                    "{0} not found: http_status={1}, request_id={2}".format(
                        self.file_key, e.status, e.request_id
                    )
                )
                return None
        # write to cache file
        with open("json_path2meta.pkl", "wb") as f:
            pickle.dump(self.json_path2meta, f)
        logger.info("Json cache write to json_path2meta.pkl successfully")
 
    def filter_by_meta(self):
        # Filter out the data with 'remove' field in meta set to True
        # 从数据集中删除 remove 字段设置为 True 的数据
        logger.info("Start to filter data by meta")
        wav_paths_filtered = []
        self.wav_path2duration = {}
        self.wav_path_index2duration = {}
        self.wav_path_index2phonelen = {}
        self.index2num_frames = []
        valid_duration = 0.0
        all_duration = 0.0
        for wav_path in tqdm.tqdm(self.wav_paths):
            try:
                meta = self.get_meta_from_wav_path(wav_path) #这里的meta是个字典
            except:
                continue
            duration = meta["end"] - meta["start"]
            all_duration += duration
            # filter by dns mos
            if meta["mos"]["dnsmos"] < 3:
                continue
            if duration < 3 or duration > 30:
                continue
            # filter by phone duration
            phone_count = len(meta['phone_id'].split())
            if phone_count <= 5:
                continue
            avg_duration = duration / phone_count
            if avg_duration < meta['lower_bound'] or avg_duration > meta['upper_bound']:
                continue
            self.wav_path2duration[wav_path] = duration
            wav_paths_filtered.append(wav_path)
            self.wav_path_index2duration[len(wav_paths_filtered)-1] = duration
            self.wav_path_index2phonelen[len(wav_paths_filtered)-1] = phone_count
            valid_duration += duration
            self.index2num_frames.append(duration * 75 + phone_count)
        # 保存过滤后的所有的wavpath
        self.wav_paths = wav_paths_filtered # 这个也可以考虑保存到文件中，下次直接读取
        # valid duration in hours
        logger.info("Valid duration: %.2f hours" % (all_duration / 3600)) #700
        logger.info("Filtered duration: %.2f hours" % (valid_duration / 3600)) #500
        logger.info("Data filtered successfully")
        logger.info("Number of wavs after filtering: %d" % len(self.wav_paths))

    def g2p(self, text, language):
        
        text = text.replace("\n", "").strip(" ")

        lang_token = lang2token[language]

        text = lang_token + text + lang_token

        return self.text_tokenizer.tokenize(text=f"{text}".strip(), language=language)
    
    def get_num_frames(self, index):
        return self.wav_path_index2duration[index] * 80 + self.wav_path_index2phonelen[index]

    def __len__(self):
        # 返回数据集的长度
        return self.wav_paths.__len__()

    def __getitem__(self, idx):

        # 根据索引 idx 返回数据
        wav_path = self.wav_paths[
            idx
        ]  # qianyi/raw/xima_processed/xmly00000_10028458_47798215_0.wav"
        try:
            for i in range(3):
                try:
                    file_bytes = self.bucket.get_object(wav_path)
                    break
                except Exception as e:
                    print(f"[Filter meta func] Error is {e}")
                    time.sleep(0.05 * i)
                    print("retry")
        except:
            logger.info("Get data from oss failed. Get another.")
            return self.__getitem__(np.random.randint(0, len(self) - 1))
        buffer = io.BytesIO(file_bytes.read())
        speech, _ = librosa.load(buffer, sr=16000) # 24000是采样率，这个可以根据实际情况调整
        # # resample to 16k
        # speech = librosa.resample(speech, orig_sr=24000, target_sr=16000)

        shape = speech.shape
        pad_shape = ((shape[0] // 200) + 1) * 200 - shape[0]
        speech = np.pad(speech, (0, pad_shape), mode='constant')

        del buffer, pad_shape, shape
        speech_tensor = torch.tensor(speech, dtype=torch.float32)
        meta = self.get_meta_from_wav_path(wav_path) # 获取对应的meta信息
        phone_id = self.g2p(meta['text'], meta['language'])[1]
        phone_id = torch.tensor([int(i) for i in phone_id], dtype=torch.long)
        phone_id = torch.cat([torch.tensor(LANG2CODE[meta['language']], dtype=torch.long).reshape(1), phone_id]) # add language token
        return dict(
            speech=speech_tensor,
            phone_id=phone_id,
        )


if __name__ == '__main__':
    # 创建数据集实例
    dataset = EmiliaDataset(AK, SK, bucket_name)
    dataset.bucket.get_object("qianyi/raw/xima_processed/xmly00000_259276_4534001/xmly00000_259276_4534001_27.wav")
    # Test loading a specific item from the dataset
    # for idx in tqdm.tqdm(range(len(dataset))):
    #     batch = dataset[idx]
    #     print(dataset.wav_paths[idx])