import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from tqdm import tqdm
import pickle
from transformers import LlamaConfig


class TTMIDIDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):

        self.cfg = cfg
        self.metafile_path = (
            "/home/t-zeqianju/yuancwang/AmphionOpen/data/processed_data/train.json"
        )

        self.metadata = self.get_metadata()
        if is_valid:
            self.metadata = self.metadata[:32]
        # print(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        tokens = self.metadata[idx]["tokens"]
        # 2d -> 1d [[1,2,3], [4,5,6]] -> [1,2,3,4,5,6]
        tokens = [item for sublist in tokens for item in sublist]
        return {"tokens": tokens}

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("metadata len: ", len(metadata))
        return metadata


class TTMIDICollator(object):
    def __init__(self, cfg, llama_config):
        self.cfg = cfg
        self.llama_config = LlamaConfig.from_pretrained(llama_config)

    def __call__(self, batch):
        packed_batch_features = dict()
        tokens = [b["tokens"] for b in batch]
        tokens = self.add_bos(tokens, self.llama_config.bos_token_id)
        tokens = self.add_eos(tokens, self.llama_config.eos_token_id)
        tokens = self.add_padding(tokens, self.llama_config.pad_token_id)
        attention_mask = self.add_mask(tokens, self.llama_config.pad_token_id)
        labels = self.add_labels(tokens, self.llama_config.pad_token_id)

        packed_batch_features["input_token_ids"] = torch.LongTensor(tokens)
        packed_batch_features["attention_mask"] = torch.LongTensor(attention_mask)
        packed_batch_features["labels"] = torch.LongTensor(labels)

        return packed_batch_features

    def add_padding(self, input_ids, pad_token_id, max_length=None):
        # input_ids: List[List[int]]
        # pad_token_id: int
        # max_length: int
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        for ids in input_ids:
            padded_ids = ids + [pad_token_id] * (max_length - len(ids))
            padded_input_ids.append(padded_ids)
        return padded_input_ids

    def add_bos(self, input_ids, bos_token_id):
        # input_ids: List[List[int]]
        # bos_token_id: int
        bos_input_ids = [[bos_token_id] + ids for ids in input_ids]
        return bos_input_ids

    def add_eos(self, input_ids, eos_token_id):
        # input_ids: List[List[int]]
        # eos_token_id: int
        eos_input_ids = [ids + [eos_token_id] for ids in input_ids]
        return eos_input_ids

    def add_mask(self, input_ids, pad_token_id):
        # input_ids: List[List[int]]
        # pad_token_id: int
        attention_mask = [
            [int(token_id != pad_token_id) for token_id in ids] for ids in input_ids
        ]
        return attention_mask

    def add_labels(self, input_ids, pad_token_id):
        # input_ids: List[List[int]]
        # pad_token_id: int
        labels = [
            [token_id if token_id != pad_token_id else -100 for token_id in ids]
            for ids in input_ids
        ]
        return labels
