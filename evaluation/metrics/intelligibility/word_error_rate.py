# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import whisper

from torchmetrics import WordErrorRate


def extract_wer(
    content_gt=None,
    audio_ref=None,
    audio_deg=None,
    fs=None,
    language="chinese",
    remove_space=True,
    remove_punctuation=True,
    mode="gt_audio",
    model=None,
):
    """Compute Word Error Rate (WER) between the predicted and the ground truth audio.
    content_gt: the ground truth content.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    mode: "gt_content" computes the WER between the predicted content obtained from the whisper model and the ground truth content.
          both content_gt and audio_deg are needed.
          "gt_audio" computes the WER between the extracted ground truth and predicted contents obtained from the whisper model.
          both audio_ref and audio_deg are needed.
    """
    # Get ground truth content
    if mode == "gt_content":
        assert content_gt != None
        if language == "chinese":
            prompt = "以下是普通话的句子"
            result_deg = model.transcribe(
                audio_deg, language="zh", verbose=True, initial_prompt=prompt
            )
        elif language == "english":
            result_deg = model.transcribe(audio_deg, language="en", verbose=True)
    elif mode == "gt_audio":
        assert audio_ref != None
        if language == "chinese":
            prompt = "以下是普通话的句子"
            result_ref = model.transcribe(
                audio_ref, language="zh", verbose=True, initial_prompt=prompt
            )
            result_deg = model.transcribe(
                audio_deg, language="zh", verbose=True, initial_prompt=prompt
            )
        elif language == "english":
            result_ref = model.transcribe(audio_ref, language="en", verbose=True)
            result_deg = model.transcribe(audio_deg, language="en", verbose=True)
        content_gt = result_ref["text"]
        if remove_space:
            content_gt = content_gt.replace(" ", "")
        if remove_punctuation:
            content_gt = content_gt.replace(".", "")
            content_gt = content_gt.replace("'", "")
            content_gt = content_gt.replace("-", "")
            content_gt = content_gt.replace(",", "")
            content_gt = content_gt.replace("!", "")
            content_gt = content_gt.lower()

    # Get predicted content
    content_pred = result_deg["text"]
    if remove_space:
        content_pred = content_pred.replace(" ", "")
    if remove_punctuation:
        content_pred = content_pred.replace(".", "")
        content_pred = content_pred.replace("'", "")
        content_pred = content_pred.replace("-", "")
        content_pred = content_pred.replace(",", "")
        content_pred = content_pred.replace("!", "")
        content_pred = content_pred.lower()
    wer = WordErrorRate()

    return wer(content_pred, content_gt).numpy().tolist()
