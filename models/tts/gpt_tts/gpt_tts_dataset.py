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
from models.tts.gpt_tts.g2p_old_en import process, PHPONE2ID
from g2p_en import G2p
import librosa


class GPTTTSDataset(torch.utils.data.Dataset):
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

        self.g2p = G2p()

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

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("metadata len: ", len(metadata))

        return metadata

    def get_phone_id(self, text):
        # convert text to phone id, you need to modify this function according to your g2p method
        txt_struct, txt = process(text, self.g2p)
        phone_seq = [p for w in txt_struct for p in w[1]]
        phone_id = [PHPONE2ID[p] for p in phone_seq]
        return phone_id

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        single_feature = dict()

        # load speech
        speech = librosa.load(utt_info["path"], sr=self.cfg.preprocess.sample_rate)[0]
        # get phone id
        text = utt_info["text"]
        phone_id = self.get_phone_id(text)

        single_feature.update(
            {
                "speech": speech,
                "phone_id": phone_id,
            }
        )

        return single_feature

    def get_num_frames(self, index):
        utt_info = self.metadata[index]
        return utt_info["num_frames"]


class GPTTTSCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # speech
        # mask
        # phone_id
        # phone_id_mask

        for key in batch[0].keys():
            try:
                if key == "phone_id":
                    phone_id = [torch.LongTensor(b["phone_id"]) for b in batch]
                    phone_id_mask = [torch.ones(len(b["phone_id"])) for b in batch]
                    packed_batch_features["phone_id"] = pad_sequence(
                        phone_id,
                        batch_first=True,
                        padding_value=0,
                    )
                    packed_batch_features["phone_id_mask"] = pad_sequence(
                        phone_id_mask,
                        batch_first=True,
                        padding_value=0,
                    )
                if key == "speech":
                    speech = [torch.FloatTensor(b["speech"]) for b in batch]
                    speech_mask = [
                        torch.ones(int(len(b["speech"]) // self.cfg.preprocess.hop_size))
                        for b in batch
                    ]
                    packed_batch_features["speech"] = pad_sequence(
                        speech,
                        batch_first=True,
                        padding_value=0,
                    )
                    packed_batch_features["speech_mask"] = pad_sequence(
                        speech_mask,
                        batch_first=True,
                        padding_value=0,
                    )
                else:
                    pass
            except Exception as e:
                print("Get data from oss failed: {}".format(e))

        return packed_batch_features


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
