#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-11-07)

set -e

trials="exp/test_fusion/sap+rank/trials"
score="exp/test_fusion/sap+rank/sap_lr_train_nosil_task_1_nosil_submean_norm.score"
out="exp/test_fusion/sap+rank/fusion"
. subtools/parse_options.sh
. subtools/score/process.sh
. subtools/score/score.sh
. subtools/path.sh

# add the score of two files.


subtools/computeEER.sh --write-file ${out}.eer $trials $score
subtools/computeCavg.py -pairs $trials $score > ${out}.Cavg