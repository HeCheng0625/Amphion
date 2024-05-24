# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from tqdm import tqdm
import pickle
import librosa
import numpy as np
from transformers import SeamlessM4TFeatureExtractor


class SoundStormDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        assert isinstance(dataset, str)

        self.cfg = cfg

        # the path of the processed data
        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        # the name of the meta file, for example: "valid.json" and "train.json"
        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        # the path of the meta file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)

        # the metadata of your data, which is a list of dict
        # for example: [{"Uid": "61-70968-0060", "num_frames": 160000, "text": ..., "path": ...}]
        # uid is the unique identifier of the speech (e.g. the file name of the speech),
        # num_frames is the number of frames of the speech,
        # text is the text of the speech,
        # path is the path of the speech
        # you can change the content of the metadata according to your data
        self.metadata = self.get_metadata()

        # the sorted list of speech index according to the number of frames, which is used for bucketing
        self.all_num_frames = []
        for i in range(len(self.metadata)):
            self.all_num_frames.append(self.metadata[i]["num_frames"])
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )

        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("metadata len: ", len(metadata))

        return metadata

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        single_feature = dict()

        # load speech
        speech, _ = librosa.load(utt_info["path"], sr=self.cfg.preprocess.sample_rate)
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
        inputs = self.processor(speech_16k, sampling_rate=16000)
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]
        # get speech mask
        speech_frames = len(speech) // self.cfg.preprocess.hop_size
        mask = np.ones(speech_frames)

        single_feature.update(
            {
                "input_features": input_features,
                "attention_mask": attention_mask,
                "speech": speech,
                "mask": mask,
            }
        )

        return single_feature

    def get_num_frames(self, index):
        utt_info = self.metadata[index]
        return utt_info["num_frames"]


class SoundStormCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # input_features: [seq_len, feature_dim]
        # attention_mask: [seq_len]

        for key in batch[0].keys():
            if key == "input_features":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).float() for utt in batch], batch_first=True
                )
            if key == "attention_mask":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).float() for utt in batch], batch_first=True
                )
            if key == "speech":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).float() for utt in batch], batch_first=True
                )
            if key == "mask":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).long() for utt in batch], batch_first=True
                )
            if key == "phone_mask":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).long() for utt in batch], batch_first=True
                )
            
            if key == "phone_id":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]) for utt in batch], batch_first=True
                )
                

        return packed_batch_features
