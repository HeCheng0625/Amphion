from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)


class GPTTTS(nn.Module):
    def __init__(
        self,
        phone_vocab_size=644,
        target_vocab_size=8192,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        pad_token_id=8838,
        bos_target_id=8839,
        eos_target_id=8840,
        bos_phone_id=8841,
        eos_phone_id=8842,
        max_position_embeddings=2048,
        cfg=None,
    ):
        super(GPTTTS, self).__init__()

        phone_vocab_size = (
            cfg.phone_vocab_size
            if cfg is not None and hasattr(cfg, "phone_vocab_size")
            else phone_vocab_size
        )
        target_vocab_size = (
            cfg.target_vocab_size
            if cfg is not None and hasattr(cfg, "target_vocab_size")
            else target_vocab_size
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        intermediate_size = (
            cfg.intermediate_size
            if cfg is not None and hasattr(cfg, "intermediate_size")
            else intermediate_size
        )
        num_hidden_layers = (
            cfg.num_hidden_layers
            if cfg is not None and hasattr(cfg, "num_hidden_layers")
            else num_hidden_layers
        )
        num_attention_heads = (
            cfg.num_attention_heads
            if cfg is not None and hasattr(cfg, "num_attention_heads")
            else num_attention_heads
        )
        pad_token_id = (
            cfg.pad_token_id
            if cfg is not None and hasattr(cfg, "pad_token_id")
            else pad_token_id
        )
        bos_target_id = (
            cfg.bos_target_id
            if cfg is not None and hasattr(cfg, "bos_target_id")
            else bos_target_id
        )
        eos_target_id = (
            cfg.eos_target_id
            if cfg is not None and hasattr(cfg, "eos_target_id")
            else eos_target_id
        )
        bos_phone_id = (
            cfg.bos_phone_id
            if cfg is not None and hasattr(cfg, "bos_phone_id")
            else bos_phone_id
        )
        eos_phone_id = (
            cfg.eos_phone_id
            if cfg is not None and hasattr(cfg, "eos_phone_id")
            else eos_phone_id
        )
        max_position_embeddings = (
            cfg.max_position_embeddings
            if cfg is not None and hasattr(cfg, "max_position_embeddings")
            else max_position_embeddings
        )

        self.config = LlamaConfig(
            vocab_size=phone_vocab_size + target_vocab_size + 10,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_target_id,
            eos_token_id=eos_target_id,
        )
        self.phone_vocab_size = phone_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pad_token_id = pad_token_id
        self.bos_target_id = bos_target_id
        self.eos_target_id = eos_target_id
        self.bos_phone_id = bos_phone_id
        self.eos_phone_id = eos_phone_id
        self.model = LlamaForCausalLM(self.config)

    def forward(
        self,
        phone_ids,
        phone_mask,
        target_ids,
        target_mask,
        prompt_ids=None,
        prompt_mask=None,
        return_labels=False,
    ):
        if prompt_ids is None:
            phone_ids, phone_mask, phone_label = self.add_phone_eos_bos_label(
                phone_ids,
                phone_mask,
                self.eos_phone_id,
                self.bos_phone_id,
                self.pad_token_id,
            )
            target_ids, target_mask, target_label = self.add_target_eos_bos_label(
                target_ids,
                target_mask,
                self.eos_target_id,
                self.bos_target_id,
                self.pad_token_id,
            )
            input_token_ids = torch.cat([phone_ids, target_ids], dim=-1)
            attention_mask = torch.cat([phone_mask, target_mask], dim=-1)
            labels = torch.cat([phone_label, target_label], dim=-1)

        else:
            phone_ids, phone_mask, phone_label = self.add_phone_eos_bos_label(
                phone_ids,
                phone_mask,
                self.eos_phone_id,
                self.bos_phone_id,
                self.pad_token_id,
            )

            prompt_ids, prompt_mask, prompt_label = self.add_prompt_bos_label(
                prompt_ids,
                None,
                self.bos_target_id,
                self.pad_token_id,
            )
            target_ids, target_mask, target_label = self.add_target_eos_label(
                target_ids,
                target_mask,
                self.eos_target_id,
                self.pad_token_id,
            )

            input_token_ids = torch.cat([phone_ids, prompt_ids, target_ids], dim=-1)
            attention_mask = torch.cat([phone_mask, prompt_mask, target_mask], dim=-1)
            labels = torch.cat([phone_label, prompt_label, target_label], dim=-1)

        if not return_labels:
            out = self.model(
                input_token_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        else:
            out = self.model(
                input_token_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            out["labels"] = labels
        return out

    def add_phone_eos_bos_label(
        self, phone_ids, phone_mask, phone_eos_id, phone_bos_id, pad_token_id
    ):
        # phone_ids: [B, T]
        # phone_mask: [B, T]

        phone_ids = phone_ids + self.target_vocab_size * phone_mask

        phone_ids = phone_ids * phone_mask
        phone_ids = F.pad(phone_ids, (0, 1), value=0) + phone_eos_id * F.pad(
            1 - phone_mask, (0, 1), value=1
        )
        phone_mask = F.pad(phone_mask, (1, 0), value=1)
        phone_ids = phone_ids * phone_mask + pad_token_id * (1 - phone_mask)
        phone_ids = F.pad(phone_ids, (1, 0), value=phone_bos_id)
        phone_mask = F.pad(phone_mask, (1, 0), value=1)
        phone_label = -100 * torch.ones_like(phone_ids)
        return phone_ids, phone_mask, phone_label

    def add_target_eos_bos_label(
        self, target_ids, target_mask, target_eos_id, target_bos_id, pad_token_id
    ):
        # target_ids: [B, T]
        # target_mask: [B, T]
        target_ids = target_ids * target_mask
        target_ids = F.pad(target_ids, (0, 1), value=0) + target_eos_id * F.pad(
            1 - target_mask, (0, 1), value=1
        )
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_ids = target_ids * target_mask + pad_token_id * (1 - target_mask)
        target_ids = F.pad(target_ids, (1, 0), value=target_bos_id)
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_label = target_ids * target_mask + (-100) * (1 - target_mask)
        return target_ids, target_mask, target_label

    def add_prompt_bos_label(
        self, prompt_ids, prompt_mask=None, prompt_bos_id=None, pad_token_id=None
    ):
        # prompt_ids: [B, T]
        # prompt_mask: [B, T]
        if prompt_mask is not None:
            prompt_ids = prompt_ids * prompt_mask
            prompt_ids = F.pad(prompt_ids, (1, 0), value=prompt_bos_id)
            prompt_mask = F.pad(prompt_mask, (1, 0), value=1)
            prompt_ids = prompt_ids * prompt_mask + pad_token_id * (1 - prompt_mask)
            prompt_label = prompt_ids * prompt_mask + (-100) * (1 - prompt_mask)
            return prompt_ids, prompt_mask, prompt_label
        else:
            prompt_ids = F.pad(prompt_ids, (1, 0), value=prompt_bos_id)
            prompt_mask = torch.ones_like(prompt_ids)
            prompt_label = -100 * torch.ones_like(prompt_ids)
            return prompt_ids, prompt_mask, prompt_label

    def add_target_eos_label(
        self, target_ids, target_mask, target_eos_id, pad_token_id
    ):
        # used for instruction fine-tuning
        target_ids = target_ids * target_mask
        target_ids = F.pad(target_ids, (0, 1), value=0) + target_eos_id * F.pad(
            1 - target_mask, (0, 1), value=1
        )
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_ids = target_ids * target_mask + pad_token_id * (1 - target_mask)
        target_label = target_ids * target_mask + (-100) * (1 - target_mask)
        return target_ids, target_mask, target_label

    def sample_hf(
        self,
        phone_ids,
        prompt_ids,
        max_length=2000,
        temperature=1.0,
        top_k=100,
        top_p=0.9,
        do_sample=True,
        repeat_penalty=1.0,
        classifer_free_guidance=0,
    ):
        phone_mask = torch.ones_like(phone_ids)
        if prompt_ids is not None:
            prompt_mask = torch.ones_like(prompt_ids)
        phone_ids, _, _ = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        if prompt_ids is not None:
            prompt_ids, _, _ = self.add_target_eos_bos_label(
                prompt_ids,
                prompt_mask,
                self.eos_target_id,
                self.bos_target_id,
                self.pad_token_id,
            )
            prompt_ids = prompt_ids[:, :-1]

        if prompt_ids is not None:
            input_token_ids = torch.cat([phone_ids, prompt_ids], dim=-1)
        else:
            input_token_ids = phone_ids

        input_length = input_token_ids.shape[1]

        if classifer_free_guidance < 1:
            generated_ids = self.model.generate(
                input_token_ids,
                do_sample=do_sample,
                max_length=max_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_target_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
            )

        else:
            neg_prompt = input_token_ids[:, -1:]
            generated_ids = self.model.generate(
                input_token_ids,
                do_sample=do_sample,
                max_length=max_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_target_id,
                top_k=top_k,
                repetition_penalty=repeat_penalty,
                logits_processor=LogitsProcessorList(
                    [
                        CFGLogits(classifer_free_guidance, neg_prompt, self.model),
                        TemperatureLogitsWarper(temperature),
                        TopPLogitsWarper(top_p),
                    ]
                ),
            )

        gen_tokens = generated_ids[:, input_length:-1]

        return gen_tokens

    def remove_pad_and_eos(self, gen_tokens):
        gen_tokens = gen_tokens.cpu().numpy()
        gen_tokens = np.where(
            (gen_tokens == self.pad_token_id) | (gen_tokens == self.eos_target_id),
            -1,
            gen_tokens,
        )
        gen_tokens = gen_tokens.tolist()
        # remove -1
        gen_tokens = [
            gen_token[: gen_token.index(-1)] if -1 in gen_token else gen_token
            for gen_token in gen_tokens
        ]
        return gen_tokens

    # TODO: add classifer free guidance
    def sample_hf_batch(
        self,
        phone_ids,
        phone_mask,
        prompt_ids=None,
        prompt_mask=None,
        max_length=2048,
        temperature=0.95,
        top_k=1000,
        top_p=0.8,
        do_sample=True,
        repeat_penalty=1.0,
    ):
        phone_ids, phone_mask, _ = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        input_prompt_mask = prompt_mask
        if prompt_ids is not None:
            prompt_ids, prompt_mask, _ = self.add_prompt_bos_label(
                prompt_ids,
                prompt_mask,
                self.bos_target_id,
                self.pad_token_id,
            )
        else:
            # prompt_ids is self.bos_target_id; shape: [B, 1]
            prompt_ids = (
                torch.ones(phone_ids.shape[0], 1).long().to(phone_ids.device)
                * self.bos_target_id
            )
            prompt_mask = torch.ones_like(prompt_ids).to(phone_mask.device)

        attention_mask = torch.cat([phone_mask, prompt_mask], dim=-1)
        input_token_ids = torch.cat([phone_ids, prompt_ids], dim=-1).long()

        generated_ids = self.model.generate(
            input_token_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_length=max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_target_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
        )

        if input_prompt_mask is None:
            input_length = phone_ids.shape[1] + 1  # +1 for bos_target_id
        else:
            input_length = input_token_ids.shape[1]

        gen_tokens = generated_ids[:, input_length:]

        return self.remove_pad_and_eos(gen_tokens)


class CFGLogits(LogitsProcessor):
    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `uncond` branch. Finally, according to CFG Rescale, the reweighted logits are interpolated back with weight
    `rescale_factor` the conditional ones to smooth the effect and increase output quality.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        uncond (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary for the unconditional branch.
        model:
            The LM computing the unconditional scores. Supposedly the same as the one computing the conditional scores.
            Both models must use the same tokenizer.
    """

    def __init__(self, guidance_scale, uncond, model):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.model = model
        self.out = None

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        if self.out is None:
            self.out = self.model(self.uncond, use_cache=True)
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)
        out = (
            self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        )
        return out
