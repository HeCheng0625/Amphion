# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))
mkdir -p $work_dir/terminal

export TMPDIR=$work_dir/terminal
export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 
export PHONEMIZER_ESPEAK_PATH=/usr/bin/espeak-ng 

######## Train Model ###########
python "${work_dir}"/models/tts/gpt_tts/emilia_dataset.py