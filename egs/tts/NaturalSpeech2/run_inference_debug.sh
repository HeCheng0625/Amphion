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
exp_config="$exp_dir/exp_config.json"
exp_name="ns2_libritts"
ref_audio="/mnt/data2/wangyuancheng/ns2_ckpts/ns2_offical_demo/reference/reference_2.wav"
checkpoint_path="/mnt/data2/wangyuancheng/ns2_ckpts/ns2_mel_debug/550k"
output_dir="$work_dir/output"
vocoder_config_path="/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/config.json"
vocoder_path="/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000"
mode="single"
inference_step=500

export CUDA_VISIBLE_DEVICES="1"

######## Parse Command Line Arguments ###########
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --text)
    text="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

######## Train Model ###########
python "${work_dir}"/bins/tts/inference.py \
    --config=$exp_config \
    --text="$text" \
    --mode=$mode \
    --checkpoint_path=$checkpoint_path \
    --vocoder_config_path=$vocoder_config_path \
    --vocoder_path=$vocoder_path \
    --ref_audio=$ref_audio \
    --inference_step=$inference_step \
    --output_dir=$output_dir \