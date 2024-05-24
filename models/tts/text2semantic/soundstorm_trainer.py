# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json
import time
import torch
import numpy as np
from utils.util import Logger, ValueWindow
from torch.utils.data import ConcatDataset, DataLoader
from models.tts.base.tts_trainer import TTSTrainer
from models.base.base_trainer import BaseTrainer
from models.base.base_sampler import VariableSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.optim import Adam, AdamW
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from transformers import get_inverse_sqrt_schedule

import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from einops import rearrange

from models.tts.text2semantic.soundstorm_dataset import (
    SoundStormCollator,
)
from models.tts.gpt_tts.gpt_tts_dataset import batch_by_size
from models.codec.kmeans.kmeans_model import KMeans, KMeansEMA
from models.tts.text2semantic.soundstorm_model import SoundStorm
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from transformers import Wav2Vec2BertModel

import safetensors


class SoundStormTrainer(TTSTrainer):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                os.makedirs(os.path.join(self.exp_dir, "checkpoint"), exist_ok=True)
                self.log_file = os.path.join(
                    os.path.join(self.exp_dir, "checkpoint"), "train.log"
                )
                self.logger = Logger(self.log_file, level=self.args.log_level).logger

        self.time_window = ValueWindow(100)

        if self.accelerator.is_main_process:
            # Log some info
            self.logger.info("=" * 56)
            self.logger.info("||\t\t" + "New training process started." + "\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")

        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(
                    f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
                )
                self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )

        # setup model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(self.model)
                self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
                self.logger.info(
                    f"Model parameters: {self._count_parameters(self.model)/1e6:.2f}M"
                )

        # setup semantic model
        self._build_semantic_model()

        # setup kmeans model
        self._build_kmeans_model()

        # setup acoustic model
        # self._build_acoustic_model()

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
                )

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            if self.accelerator.is_main_process:
                self.logger.info("Initializing accelerate...")
            start = time.monotonic_ns()
            self.train_dataloader = self.accelerator.prepare(
                self.train_dataloader,
            )

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        end = time.monotonic_ns()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # Resume or Finetune
        with self.accelerator.main_process_first():
            if args.resume:
                ## Automatically resume according to the current exprimental name
                print(
                    "Automatically resuming from latest checkpoint in {}...".format(
                        self.checkpoint_dir
                    )
                )
                start = time.monotonic_ns()
                ckpt_path = self._load_model(
                    checkpoint_dir=self.checkpoint_dir, resume_type=args.resume_type
                )
                end = time.monotonic_ns()
                print(f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        # Only for TTS tasks
        self.task_type = "TTS"
        if self.accelerator.is_main_process:
            self.logger.info("Task type: {}".format(self.task_type))

    def _count_parameters(self, model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            # kwargs_handlers=[ddp_kwargs]
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def _build_model(self):
        model = SoundStorm(**self.cfg.model.soundstorm)
        return model

    def _build_kmeans_model(self):
        if self.cfg.model.kmeans.type == "kmeans":
            kmeans_model = KMeans(cfg=self.cfg.model.kmeans.kmeans)
        elif self.cfg.model.kmeans.type == "kmeans_ema":
            kmeans_model = KMeansEMA(cfg=self.cfg.model.kmeans.kmeans)
        kmeans_model.eval()
        pretrained_path = self.cfg.model.kmeans.pretrained_path
        if ".bin" in pretrained_path:
            kmeans_model.load_state_dict(torch.load(pretrained_path))
        elif ".safetensors" in pretrained_path:
            safetensors.torch.load_model(kmeans_model, pretrained_path)
        kmeans_model.to(self.accelerator.device)
        self.kmeans_model = kmeans_model

    def _build_semantic_model(self):
        self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model.eval()
        self.semantic_model.to(self.accelerator.device)
        self.layer_idx = 15
        if self.layer_idx == 23:
            self.output_idx = 0
        else:
            self.output_idx = self.layer_idx + 2
        stat_mean_var = torch.load(self.cfg.model.kmeans.stat_mean_var_path)
        self.semantic_mean = stat_mean_var["mean"]
        self.semantic_std = torch.sqrt(stat_mean_var["var"])
        self.semantic_mean = self.semantic_mean.to(self.accelerator.device)
        self.semantic_std = self.semantic_std.to(self.accelerator.device)
        print(
            "semantic mean: ", self.semantic_mean, "semantic std: ", self.semantic_std
        )

    def _build_acoustic_model(self):
        # codec encoder and codec decoder
        self.codec_encoder = CodecEncoder(cfg=self.cfg.model.codec.encoder)
        self.codec_decoder = CodecDecoder(cfg=self.cfg.model.codec.decoder)
        self.codec_encoder.load_state_dict(
            torch.load(self.cfg.model.codec.encoder.pretrained_path)
        )
        self.codec_decoder.load_state_dict(
            torch.load(self.cfg.model.codec.decoder.pretrained_path)
        )
        self.codec_decoder = self.codec_decoder.quantizer  # we only need the quantizer
        self.codec_encoder.eval()
        self.codec_decoder.eval()
        self.codec_encoder.to(self.accelerator.device)
        self.codec_decoder.to(self.accelerator.device)

    @torch.no_grad()
    def _extract_acoustic_code(self, speech):
        vq_emb = self.codec_encoder(speech.unsqueeze(1))
        _, vq, _, _, _ = self.codec_decoder(vq_emb)
        acoustic_code = vq.permute(
            1, 2, 0
        )  # (num_quantizer, T, C) -> (T, C, num_quantizer)
        return acoustic_code

    @torch.no_grad()
    def _extract_semantic_code(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.output_idx]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        semantic_code, _ = self.kmeans_model.quantize(feat)  # (B, T)
        return semantic_code

    def _build_dataset(self):
        if (
            hasattr(self.cfg.train, "use_emilia_dataset")
            and self.cfg.train.use_emilia_dataset
        ):
            from models.tts.text2semantic.soundstorm_emilia_dataset import (
                SoundStormDataset,
            )
        else:
            from models.tts.text2semantic.soundstorm_dataset import SoundStormDataset
        return SoundStormDataset, SoundStormCollator

    def _build_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize......")
            Dataset, Collator = self._build_dataset()
            if (
                hasattr(self.cfg.train, "use_emilia_dataset")
                and self.cfg.train.use_emilia_dataset
            ):
                train_dataset = Dataset(cfg=self.cfg)
            else:
                train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)
            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            np.random.seed(980209)
            np.random.shuffle(batch_sampler)
            print(batch_sampler[:1])
            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]

            train_loader = DataLoader(
                train_dataset,
                collate_fn=train_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False, use_random_sampler=True
                ),
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

            valid_loader = None

        else:
            print("Use Normal Batchsize......")
            Dataset, Collator = self._build_dataset()
            if (
                hasattr(self.cfg.train, "use_emilia_dataset")
                and self.cfg.train.use_emilia_dataset
            ):
                train_dataset = Dataset(cfg=self.cfg)
            else:
                train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=train_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            valid_loader = None
            self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

    def _build_optimizer(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.train.adam,
        )
        return optimizer

    def _build_scheduler(self):
        lr_scheduler = get_inverse_sqrt_schedule(
            optimizer=self.optimizer,
            # num_warmup_steps=self.cfg.train.lr_warmup_steps,  # TODO: need to check wheather need to multiply by num_processes
            num_warmup_steps=self.cfg.train.lr_warmup_steps
            * self.accelerator.num_processes,
        )
        return lr_scheduler

    def _build_criterion(self):
        criteria = dict()
        criteria["l1_loss"] = torch.nn.L1Loss(reduction="mean")
        criteria["l2_loss"] = torch.nn.MSELoss(reduction="mean")
        criteria["ce_loss"] = torch.nn.CrossEntropyLoss(reduction="none")
        return criteria

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        input_features = batch["input_features"]
        attention_mask = batch["attention_mask"]
        # speech = batch["speech"]
        x_mask = batch["mask"]
        
        phone_id = batch['phone_id']
        phone_mask = batch['phone_mask']

        semantic_code = self._extract_semantic_code(
            input_features, attention_mask
        )  # (B, T)
        
        seq_len = semantic_code.shape[1]
        x_mask = x_mask[:, :seq_len]
        
        out = self.model(
            phone_ids = phone_id,
            phone_mask=phone_mask,
            target_ids = semantic_code,
            target_mask=x_mask,
        )
        
        # logits = out.logits # logits: (B, T, H)
        total_loss = out.loss
        print(total_loss)
        
        # acoustic_code = self._extract_acoustic_code(speech)  # (B, T, num_quantizer)
        # print(
        #     "semantic code: ",
        #     semantic_code.shape,
        #     "acoustic code: ",
        #     acoustic_code.shape,
        # )

        # seq_len = min(semantic_code.shape[1], acoustic_code.shape[1])
        # semantic_code = semantic_code[:, :seq_len]
        # acoustic_code = acoustic_code[:, :seq_len, :]
        # x_mask = x_mask[:, :seq_len]

        # logits, mask_layer, final_mask, x0, prompt_len, mask_prob = self.model(
        #     x0=acoustic_code, x_mask=x_mask, cond=None, cond_code=semantic_code
        # )
        # print(
        #     "logits: ",
        #     logits.shape,
        #     "mask_layer: ",
        #     mask_layer.shape,
        #     "final_mask: ",
        #     final_mask.shape,
        #     "prompt_len: ",
        #     prompt_len.shape,
        # )
        # logits: (B, T, codebook_size)
        # mask_layer: (1,)
        # final_mask: (B, T, 1)
        # prompt_len: (B,)

        # final_mask = final_mask.squeeze(-1)

        # calculate the loss
        # logits: (B, T, codebook_size), target: (B, T), final_mask: (B, T), only calculate the loss on the masked part (mask is 1)
        # target = acoustic_code[:, :, mask_layer.item()]
        # ce_loss = F.cross_entropy(
        #     logits.permute(0, 2, 1), target, reduction="none"
        # )  # (B, T)
        # ce_loss = ce_loss * final_mask  # (B, T)
        # ce_loss = ce_loss.sum() / final_mask.sum()
        # total_loss += ce_loss
        train_losses["ce_loss"] = total_loss
        # # caluate accuracy top 1
        # acc = (logits.argmax(-1) == target).float()
        # acc = acc * final_mask
        # acc = acc.sum() / final_mask.sum()
        # # caluate accuracy top 5
        # acc5 = torch.topk(logits, 5, dim=-1)[1]
        # acc5 = torch.sum(acc5 == target.unsqueeze(-1), dim=-1).float()
        # acc5 = acc5 * final_mask
        # acc5 = acc5.sum() / final_mask.sum()
        # # caluate accuracy top 10
        # acc10 = torch.topk(logits, 10, dim=-1)[1]
        # acc10 = torch.sum(acc10 == target.unsqueeze(-1), dim=-1).float()
        # acc10 = acc10 * final_mask
        # acc10 = acc10.sum() / final_mask.sum()

        # train_losses["acc"] = acc
        # train_losses["acc5"] = acc5
        # train_losses["acc10"] = acc10

        # train_losses["mask_layer"] = mask_layer
        # train_losses["mask_prob"] = mask_prob

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()

        train_losses["batch_size"] = input_features.shape[0]

        return (total_loss.item(), train_losses, train_stats)

    @torch.inference_mode()
    def _valid_step(self, batch):
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        input_features = batch["input_features"]
        attention_mask = batch["attention_mask"]
        speech = batch["speech"]
        x_mask = batch["mask"]

        semantic_code = self._extract_semantic_code(
            input_features, attention_mask
        )  # (B, T)
        acoustic_code = self._extract_acoustic_code(speech)  # (B, T, num_quantizer)

        seq_len = min(semantic_code.shape[1], acoustic_code.shape[1])
        semantic_code = semantic_code[:, :seq_len]
        acoustic_code = acoustic_code[:, :seq_len, :]
        x_mask = x_mask[:, :seq_len]

        logits, mask_layer, final_mask, x0, prompt_len, mask_prob = self.model(
            x0=acoustic_code, x_mask=x_mask, cond=None, cond_code=semantic_code
        )

        final_mask = final_mask.squeeze(-1)

        # calculate the loss
        # logits: (B, T, codebook_size), target: (B, T), final_mask: (B, T), only calculate the loss on the masked part (mask is 1)
        target = acoustic_code[:, :, mask_layer.item()]
        ce_loss = F.cross_entropy(
            logits.permute(0, 2, 1), target, reduction="none"
        )  # (B, T)
        ce_loss = ce_loss * final_mask  # (B, T)
        ce_loss = ce_loss.sum() / final_mask.sum()
        total_loss += ce_loss
        valid_losses["ce_loss"] = ce_loss
        # caluate accuracy
        acc = (logits.argmax(-1) == target).float()
        acc = acc * final_mask
        acc = acc.sum() / final_mask.sum()
        valid_losses["acc"] = acc

        for item in valid_losses:
            valid_losses[item] = valid_losses[item].item()

        return (total_loss.item(), valid_losses, valid_stats)

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = dict()

        for batch in self.valid_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss = total_loss
            for key, value in valid_losses.items():
                epoch_losses[key] = value

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
        ema_loss = None

        for batch in self.train_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1
            ema_loss = (
                0.98 * ema_loss + 0.02 * self.current_loss
                if ema_loss is not None
                else self.current_loss
            )
            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss = total_loss
                for key, value in train_losses.items():
                    epoch_losses[key] = value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                if (
                    self.accelerator.is_main_process
                    and self.batch_count
                    % (10 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

                if self.step % self.cfg.train.save_checkpoints_steps == 0:
                    self.save_checkpoint()

                if self.accelerator.is_main_process:
                    if self.step % 100 == 0:
                        print(f"EMA Loss: {ema_loss:.6f}")

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            keep_last = self.keep_last[0]
            # 读取self.checkpoint_dir所有的folder
            all_ckpts = os.listdir(self.checkpoint_dir)

            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            all_ckpts = list(all_ckpts)
            if len(all_ckpts) > keep_last:
                # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                all_ckpts = sorted(
                    all_ckpts, key=lambda x: int(x.split("_")[1].split("-")[1])
                )
                for ckpt in all_ckpts[:-keep_last]:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))
            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            self.logger.info("Saving state to {}...".format(path))
            self.accelerator.save_state(path)
            self.logger.info("Finished saving state.")

    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        # if self.accelerator.is_main_process:
        #     self._dump_cfg(self.config_save_path)

        # self.optimizer.zero_grad()

        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            if self.accelerator.is_main_process:
                self.logger.info("\n")
                self.logger.info("-" * 32)
                self.logger.info("Epoch {}: ".format(self.epoch))

            # Do training & validating epoch
            train_total_loss, train_losses = self._train_epoch()
            if isinstance(train_losses, dict):
                for key, loss in train_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            valid_total_loss, valid_losses = 0.0, 0.0
            # if isinstance(valid_losses, dict):
            #     for key, loss in valid_losses.items():
            #         if self.accelerator.is_main_process:
            #             self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
            #         self.accelerator.log(
            #             {"Epoch/Train {} Loss".format(key): loss},
            #             step=self.epoch,
            #         )

            if self.accelerator.is_main_process:
                self.logger.info("  |- Train/Loss: {:.6f}".format(train_total_loss))
                self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_total_loss))
            self.accelerator.log(
                {
                    "Epoch/Train Loss": train_total_loss,
                    "Epoch/Valid Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_total_loss
                    ),
                )
            )
        self.accelerator.end_training()

    def echo_log(self, losses, mode="Training"):
        message = [
            "{} - Epoch {} Step {}: [{:.3f} s/step]".format(
                mode, self.epoch + 1, self.step, self.time_window.average
            )
        ]

        for key in sorted(losses.keys()):
            if isinstance(losses[key], dict):
                for k, v in losses[key].items():
                    message.append(
                        str(k).split("/")[-1] + "=" + str(round(float(v), 5))
                    )
            else:
                message.append(
                    str(key).split("/")[-1] + "=" + str(round(float(losses[key]), 5))
                )
        self.logger.info(", ".join(message))
