#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-11-07)

set -e
. subtools/parse_options.sh
. subtools/score/process.sh
. subtools/score/score.sh
. subtools/path.sh

if [[ $# != 4 ]];then
echo "[exit] Num of parameters is not equal to 4"
echo "usage:$0 <trials> <score_1> <score_2> <out_dir>"
exit 1
fi

trials=$1
score_1=$2
score_2=$3
out=$4

# trials="exp/test_fusion/transformer+rank_task1/trials"
# score_1="exp/test_fusion/transformer+rank_task1/transformer_lr_train_nosil_task_1_nosil_norm"
# score_2="exp/test_fusion/transformer+rank_task1/rank_lr_train_nosil_task_1_nosil_submean_norm.score"
# out="exp/test_fusion/transformer+rank_task1/transformer+rank"


# add the score of two files.

# echo $score_2 | awk '{print $2}'
cat $score_1 | awk '{print $2}' > label.temp
cat $score_1 | awk '{print $1}' > utt_id.temp
cat $score_1 | awk '{print $3}' > score1.temp
cat $score_2 | awk '{print $3}' > score2.temp
paste score1.temp score2.temp > score.temp
awk '{print $1+$2}' score.temp > score_.temp
paste utt_id.temp label.temp score_.temp -d ' '> ${out}/fusion.score

rm label.temp score1.temp score2.temp score.temp score_.temp utt_id.temp


subtools/computeEER.sh --write-file ${out}/fusion.eer $trials ${out}/fusion.score
subtools/computeCavg.py -pairs $trials ${out}/fusion.score > ${out}/fusion.Cavg