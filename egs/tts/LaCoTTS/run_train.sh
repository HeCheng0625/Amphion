# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config_base.json"
exp_name="latent_codec_gpt_tts"
checkpoint_path="$work_dir/exps/latent_codec_gpt_tts/epoch-0002_step-0009800_loss-7.734620/"
resume_type="resume"
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 
export PHONEMIZER_ESPEAK_PATH=/usr/bin/espeak-ng 

######## Train Model ###########
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    "${work_dir}"/bins/tts/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    --resume \
    --resume_type=$resume_type \
    --checkpoint_path=$checkpoint_path 