import oss2 #pip install oss2
from mysql.connector.pooling import MySQLConnection # pip install mysql-connector-python
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
from multiprocessing import Pool
logging.basicConfig(level=logging.INFO)  # Configure logging level to INFO
logger = logging.getLogger(__name__)

LANG2CODE = {
    'zh': 654,
    'en': 655,
    'ja': 656,
    'kr': 657,
    'fr': 658,
    'de': 659,
}
lang2token = {
    'zh': "[ZH]",
    "en": "[EN]",
    'ja': "[JA]",
    "kr": "[KR]",
    "fr": "[FR]",
    "de": "[DE]",
}
# 示例用法

class EmiliaDataset(Dataset):
    def __init__(self, 
                access_key_id=AK, 
                access_key_secret=SK, 
                bucket_name=bucket_name, 
                resample_to_16k=False, 
                from_database=True):

        # Initialize OSS client
        self.init_client(access_key_id, access_key_secret, bucket_name) # 建立连接
        self.text_tokenizer = PhonemeBpeTokenizer()
        self.json_paths = []
        self.wav_paths = []
        self.json_path2meta = {}
        self.path_is_filtered = False
        self.detabase_connection = None
        # Load paths from cache files if available, otherwise get them from OSS
        wav_paths_cache = "wav_paths_cache.pkl"
        filtered_wav_paths_cache = "filtered_wav_paths_cache.pkl"
        json_paths_cache = "json_paths_cache.pkl"
        json_path2meta = "json_path2meta.pkl"
        if os.path.exists(filtered_wav_paths_cache) and os.path.exists(json_paths_cache):
            self.load_cached_paths(filtered_wav_paths_cache, json_paths_cache)
            self.path_is_filtered = True
        elif os.path.exists(wav_paths_cache) and os.path.exists(json_paths_cache):
            self.load_cached_paths(wav_paths_cache, json_paths_cache)
        else:
            logger.info("No cache exists")
            if from_database:
                self.detabase_connection = self.init_database()
                if self.detabase_connection:
                    self.detabase_connection.start_transaction()
                    # Load path from database
                    self.get_all_paths_from_database(language="zh", duration_limit=12000, filtered=False) # hours
                    self.get_all_paths_from_database(language="en", duration_limit=12000, filtered=False) # hours
                    # If need other language, change language and call it again
                    self.detabase_connection.close()
            else:
                # Load path from oss one by one
                self.get_all_paths_form_oss(folder_path="qianyi/raw/xima_processed/", num_limit=100000) 
                # If need other folder_path, change folder_path and call it again
            self.save_cached_paths()
                
        self.wav_path_index2duration = []
        self.wav_path_index2phonelen = []
        self.index2num_frames = []

        self.cache_limit = 10000 # josn2meta cache
        if os.path.exists(json_path2meta):
            self.load_path2meta(json_path2meta)
        else:
            self.get_jsoncache_multiprocess()
        
        if not self.path_is_filtered:
            self.filter_by_meta()
        
        self.num_frame_indices = np.array(sorted(range(len(self.index2num_frames)), key=lambda k: self.index2num_frames[k]))


    def init_client(self, access_key_id, access_key_secret, bucket_name):
        # 初始化 OSS 客户端
        logger.info("Start to initialize OSS client")
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(
            self.auth, "https://oss-cn-beijing.aliyuncs.com", bucket_name
        )
        logger.info("OSS client initialized successfully")
    
    def load_cached_paths(self, wav_paths_cache, json_paths_cache):
        with open(wav_paths_cache, "rb") as f:
            self.wav_paths = pickle.load(f)
        with open(json_paths_cache, "rb") as f:
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

    def init_database(self):
        try:
            from AudioDataCollection.database.db_manager import DatabaseManager
            db_manager = DatabaseManager()
            conn: MySQLConnection = db_manager.get_connection()
            return conn
        except Exception as e:
            # raise e
            logger.info("Gat database error {}".format(e))
            return None
        
    def get_path_from_database(self, path, num_wav, idx=None):
        json_path = path.split(bucket_name)[1][1:]
        # logger.info("Get json from {}".format(path))
        self.json_paths.append(json_path)
        audio_name = json_path.split(".json")[0]
        is_mp3 = True if 'mp3' in json_path else False
        if idx is None:
            for i in range(num_wav):
                wav_path = audio_name + "_" + str(i) + ".wav"
                mp3_path = audio_name + "_" + str(i) + ".mp3"
                try:
                    if not is_mp3: # or self.bucket.object_exists(wav_path):
                        self.wav_paths.append(wav_path)
                    else:
                        self.wav_paths.append(mp3_path)
                except oss2.exceptions.NoSuchKey:
                    continue
                except oss2.exceptions.OssError as e:
                    continue
        else:
            idx = idx.split(',')
            for x in idx:
                try:
                    wav_path = audio_name + "_" + str(x) + ".wav"
                    mp3_path = audio_name + "_" + str(x) + ".mp3"
                    if is_mp3 or self.bucket.object_exists(mp3_path):
                        self.wav_paths.append(mp3_path)
                    else:
                        self.wav_paths.append(wav_path)
                except oss2.exceptions.NoSuchKey:
                    continue

    def get_all_paths_from_database(self, language, duration_limit, filtered=False):
        logger.info("Start to get all {} paths".format(language))
        try:
            cursor = self.detabase_connection.cursor()

            tmp_duration = 0
            page = 0
            num_per_page = 100
            duration_limit = duration_limit * 3600
 
            logger.info("Get paths from database")
            is_enough = False
            while not is_enough:

                if not filtered:
                    cursor.execute(
                        f"""
                        SELECT result_path, valid_duration, valid_segement
                        FROM video
                        WHERE language = '{language}' AND status = 4
                        LIMIT {page * num_per_page}, {num_per_page}
                        """
                    )

                    record = cursor.fetchall()
                    if record is None:
                        self.detabase_connection.rollback()
                        continue
                    
                    for single_record in record:
                        path, duration, num_wav = single_record
                        self.get_path_from_database(path, num_wav)
                        if tmp_duration < duration_limit:
                            tmp_duration += duration
                        else:
                            is_enough = True
                            break
                else:
                    cursor.execute(
                        f"""
                            SELECT result_path, filtered_duration, filtered_segement, filtered_idx
                            FROM video
                            WHERE language = '{language}' AND status = 4
                            LIMIT {page * num_per_page}, {num_per_page}
                        """
                        )

                    record = cursor.fetchall()
                    if record is None:
                        self.detabase_connection.rollback()
                        continue

                    for single_record in record:
                        path, duration, num_wav, idx = single_record
                        self.get_path_from_database(path, num_wav, idx)
                        tmp_duration += duration
                logger.info("Get %.2f hours" % (tmp_duration / 3600))
                page += 1
                
            logger.info("All paths got successfully")
            # num wavs, num jsons
            logger.info(
                "Number of wavs: %d, Number of jsons: %d"
                % (len(self.wav_paths), len(self.json_paths))
            )
        except Exception as e:
            # 发生错误时回滚事务
            self.detabase_connection.rollback()
            # raise e
            logger.info("Gat path error {}".format(e))

    def get_all_paths_form_oss(self, folder_path, num_limit):
        logger.info("Start to get all paths")
        # 这个每次重启需要重新遍历一遍，要花一些时间，可以考虑把把所有wav和json的路径保存到一个文件中，下次直接读取
        counter = 0
        logger.info("Folder path: {}".format(folder_path))
        for obj in oss2.ObjectIterator(self.bucket, prefix=folder_path):
            if obj.key.endswith(".json"):
                self.json_paths.append(obj.key)
                counter += 1
            elif obj.key.endswith( ".wav"):
                self.wav_paths.append(obj.key)
                counter += 1
            elif obj.key.endswith( ".mp3"):
                self.wav_paths.append(obj.key)
                counter += 1
            if counter > num_limit:  # this is just for testing
                logger.info("More than {}".format(counter))
                break
        logger.info("All paths got successfully")
        logger.info(
            "Number of wavs: %d, Number of jsons: %d"
            % (len(self.wav_paths), len(self.json_paths))
        )

    def get_bounds(self, meta, json_path):
        avg_durations = []
        for m in meta:
            phone_id = [-1] # error value
            if m['language'] in ['zh', 'en']:
                phone_id = self.g2p(m['text'], m['language'])[1]
            m['phone_id'] = " ".join(map(str, phone_id))
            phone_count = len(phone_id)
            m['phone_count'] = phone_count
            duration = m['end'] - m['start']
            m['duration'] = duration
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

    def process_json_cache(self, json_path):
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
                print(json_path)
                json_cache = self.get_bounds(json_cache, json_path)
                self.json_path2meta[json_path] = json_cache
                self.process_num += 1
                print("Process num: {}, total num: {}".format(self.process_num, len(self.json_paths)))
        except oss2.exceptions.NoSuchKey as e:
            logger.info(
                "Not found: http_status={1}, request_id={2}".format(
                    e.status, e.request_id
                )
            )
            self.error_path.append(json_path) 
        except Exception as e:
            logger.info("Error json: {}".format(json_path))
            self.error_path.append(json_path) 

    def get_jsoncache_multiprocess(self):
        logger.info("Start to get json cache, build pool")
        pool = Pool(processes=48, maxtasksperchild=500)  # 创建进程池
        self.error_path = []
        self.process_num = 0
        args = [(self.json_paths[i]) for i in range(len(self.json_paths))]  # 构造参数列表
        logger.info("Start to get json cache")
        seqs_tmp = pool.map(self.process_json_cache, args)  # 多线程调用process_audio_wrapper
        pool.close()  # 关闭进程池
        pool.join()  # 等待所有进程完成
        logger.info("Remove error path")
        for error in self.error_path:
            self.json_paths.remove(error)
        # write to cache file
        with open("json_path2meta.pkl", "wb") as f:
            pickle.dump(self.json_path2meta, f)
        logger.info("Json cache write to json_path2meta.pkl successfully")

    def get_jsoncache(self):
        logger.info("Start to get json cache")
        error_path = []  # 用于存储需要删除的元素
        for json_path in tqdm.tqdm(self.json_paths):
            try:
                file_bytes = self.bucket.get_object(json_path)
                buffer = io.BytesIO(file_bytes.read())
                json_cache = json.load(buffer)
                del buffer, file_bytes
                if json_cache is None:
                    logger.info("json is none")
                    continue
                elif isinstance(json_cache, (dict, list)) and not json_cache:
                    logger.info("json is none")
                    continue
                else:
                    json_cache = self.get_bounds(json_cache)
                    self.json_path2meta[json_path] = json_cache
            except oss2.exceptions.NoSuchKey as e:
                logger.info(
                    "Not found: http_status={1}, request_id={2}".format(
                        e.status, e.request_id
                    )
                )
                error_path.append(json_path)
                continue
        logger.info("Remove error path")
        for error in error_path:
            self.json_paths.remove(error)
        # write to cache file
        with open("json_path2meta.pkl", "wb") as f:
            pickle.dump(self.json_path2meta, f)
        logger.info("Json cache write to json_path2meta.pkl successfully")

    def get_meta_from_wav_path(self, wav_path):
        index = int(wav_path.split("_")[-1].split(".")[0])  # 0
        audio_name = "_".join(wav_path.split("/")[-1].split("_")[:-1])
        dir_name = "/".join(wav_path.split("/")[:-1])
        json_name = audio_name + ".json"  # xmly00000_10028458_47798215.json
        json_path = dir_name + "/" + json_name
        meta = self.json_path2meta[json_path][index]
        return meta 

    def load_path2meta(self, path2meta):
        self.json_path2meta = pickle.load(open(path2meta, "rb"))
        for path in self.wav_paths:
            duration = self.get_meta_from_wav_path(path)['duration']
            phone_count = self.get_meta_from_wav_path(path)['phone_count']
            
            self.wav_path_index2duration.append(duration)
            self.wav_path_index2phonelen.append(phone_count)
            self.index2num_frames.append(duration * 80 + phone_count)

    def filter_by_meta(self):
        # Filter out the data with 'remove' field in meta set to True
        logger.info("Start to filter data by meta")
        wav_paths_filtered = []
        valid_duration = 0.0
        all_duration = 0.0
        for wav_path in tqdm.tqdm(self.wav_paths):
            try:
                meta = self.get_meta_from_wav_path(wav_path) #这里的meta是个字典
            except:
                continue
            duration = meta['duration']
            all_duration += duration
            phone_count = meta['phone_count']
            if meta['phone_id'] == '-1':
                continue
            # filter by dns mos
            if 'remove' in meta:
                if meta['remove'] == 'true':
                    continue
            if meta['mos']['dnsmos'] < 3:
                continue
            if duration < 3 or duration > 30:
                continue
            # filter by phone duration
            if phone_count <= 5:
                continue
            avg_duration = duration / phone_count
            if avg_duration < meta['lower_bound'] or avg_duration > meta['upper_bound']:
                continue
            wav_paths_filtered.append(wav_path)
            self.wav_path_index2duration.append(duration)
            self.wav_path_index2phonelen.append(phone_count)
            valid_duration += duration
            self.index2num_frames.append(duration * 80 + phone_count)
        # 保存过滤后的所有的wavpath
        self.wav_paths = wav_paths_filtered # 这个也可以考虑保存到文件中，下次直接读取
        # write to cache file
        if not os.path.exists("filtered_wav_paths_cache.pkl"):
            with open("filtered_wav_paths_cache.pkl", "wb") as f:
                pickle.dump(self.wav_paths, f)
        # valid duration in hours
        logger.info("Valid duration: %.2f hours" % (all_duration / 3600)) #700
        logger.info("Filtered duration: %.2f hours" % (valid_duration / 3600)) #500
        logger.info("Data filtered successfully")
        logger.info("Number of wavs after filtering: %d" % len(self.wav_paths))
        del wav_paths_filtered, all_duration, valid_duration

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
        phone_id = meta['phone_id'].split()
        phone_id = torch.tensor([int(i) for i in phone_id], dtype=torch.long)
        phone_id = torch.cat([torch.tensor(LANG2CODE[meta['language']], dtype=torch.long).reshape(1), phone_id]) # add language token
        return dict(
            speech=speech_tensor,
            phone_id=phone_id,
        )


if __name__ == '__main__':
    # 创建数据集实例
    dataset = EmiliaDataset(AK, SK, bucket_name)
    print(dataset.__getitem__(0))
    # Test loading a specific item from the dataset
    # for idx in tqdm.tqdm(range(len(dataset))):
    #     batch = dataset[idx]
    #     print(dataset.wav_paths[idx])