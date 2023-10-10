#!/bin/bash

# Copyright xmuspeech (Author: Snowdar, Zheng Li 2020-05-30)
# Update by Zheng Li 2020-07-23
# Update info: For open-set dialect identification task (task 2), a new script named computeCavg_unknown.py was provided to compute Cavg and it will be used in the final test procedure.
#              As for task 1 and task 3, the computation of Cavg in computeCavg.py is not changed.
#
# Corresponding email: ap_olr@163.com
# Please refer to https://speech.xmu.edu.cn/ or http://olr.cslt.org for more info

### A record of baselines of AP20-OLR. Suggest to execute every script one by one.

#prepare data
# ../lre07/v1/prepare_data.sh

### Start 
# Prepare data sets, all of them contain at least wav.scp, utt2lang, spk2utt and utt2spk;
# spk2utt/utt2spk could be fake, e.g. the utt-id is just the spk-id, in the test set.
#Training set

#train=data/train
train=data/mfcc_20_5.0/train
# train=data/fbank_40_5.0/train
#AP20-OLR-test
#task1_test=data/task1_test
#task2_test=data/task2_test
#task3_test=data/task3_test
# task1_test_3sec=data/mfcc_20_5.0/lre07_test_3sec
# task1_test_10sec=data/mfcc_20_5.0/lre07_test_10sec
# task1_test_30sec=data/mfcc_20_5.0/lre07_test_30sec
task_2_enroll=data/mfcc_20_5.0/task_2_enroll
# task1_test_3sec=data/fbank_40_5.0/lre07_test_3sec
# task1_test_10sec=data/fbank_40_5.0/lre07_test_10sec
# task1_test_30sec=data/fbank_40_5.0/lre07_test_30sec
#AP20-OLR-ref-enroll
#task1_enroll=data/task1_enroll
#task2_enroll=data/task2_enroll
#task3_enroll=data/task3_enroll
task1_enroll=data/mfcc_20_5.0/lre07_enroll_volume_aug
# task1_enroll=data/fbank_40_5.0/lre07_enroll_volume_aug
#AP20-OLR-ref-dev
#task1_dev=data/task1_dev
#task2_dev=data/task2_dev
# task1_dev=data/mfcc_20_5.0/lre07_dev

prefix=mfcc_20_5.0
# prefix=fbank_40_5.0

stage=7


if [ $stage -eq 1 ]; then
  echo "start execute stage 1"
  # Get the copies of dataset which is labeled by a prefix like mfcc_23_pitch or fbank_40_pitch etc.
  #subtools/newCopyData.sh mfcc_20_5.0 "train task1_test task2_test task3_test task1_enroll task2_enroll task3_enroll task1_dev task2_dev"
  # subtools/newCopyData.sh mfcc_20_5.0 "train lre07_test lre07_enroll lre07_dev"
  # subtools/newCopyData.sh mfcc_20_5.0 "lre07_test lre07_enroll"
  # subtools/newCopyData.sh mfcc_20_5.0 "lre07_test_3sec lre07_test_10sec lre07_test_30sec"
  # subtools/newCopyData.sh mfcc_20_5.0 "train lre07_test_3sec lre07_test_10sec lre07_test_30sec lre07_enroll"
  subtools/newCopyData.sh mfcc_20_5.0 "train"
  # subtools/newCopyData.sh fbank_40_5.0 "train lre07_test_3sec lre07_test_10sec lre07_test_30sec lre07_enroll"
fi
# echo "-----------------------------------"
# exit(0)



if [ $stage -eq 2 ]; then
  echo "start execute stage 2"
  # Augment trainset by clean:aug=1:2 with speed (0.9,1.0,1.1) and volume perturbation; Make features for trainset;  Compute VAD for augmented trainset
  # subtools/concatSpFeats.sh --stage 0 --endstage 1 --volume true --datasets "train" --prefix mfcc_20_5.0 --feat_type mfcc \
  #                          --feat_conf subtools/conf/sre-mfcc-20.conf --vad_conf conf/vad-5.0.conf --pitch true --suffix aug
  # subtools/concatSpFeats.sh --stage 0 --endstage 1 --volume true --datasets "train" --prefix mfcc_20_5.0 --feat_type mfcc \
  #                           --feat_conf subtools/conf/sre-mfcc-20.conf --vad_conf subtools/conf/vad-5.0.conf --pitch true --suffix aug
  # subtools/concatSpFeats.sh --stage 1 --endstage 1 --volume false --datasets "train" --prefix mfcc_20_5.0 --feat_type mfcc \
  #                           --feat_conf subtools/conf/sre-mfcc-20.conf --vad_conf subtools/conf/vad-5.0.conf --pitch true --suffix normal
  # subtools/concatSpFeats.sh --stage 0 --endstage 1 --volume true --datasets "train" --prefix fbank_40_5.0 --feat_type fbank \
  #                           --feat_conf subtools/conf/sre-fbank-40.conf --vad_conf subtools/conf/vad-5.0.conf --pitch true --suffix aug
  # subtools/computeVad.sh $train subtools/conf/vad-5.0.conf
  subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf $train mfcc \
                                  subtools/conf/sre-mfcc-20.conf

  subtools/computeVad.sh $train subtools/conf/vad-5.0.conf
fi
# echo "-----------------------------------"
# exit(0)

if [ $stage -eq 3 ]; then
  echo "start execute stage 3"
  # Augment trainset by clean:aug=1:2 with speed (0.9,1.0,1.1) and volume perturbation; Make features for trainset;  Compute VAD for augmented trainset
  # subtools/concatSpFeats.sh --stage 0 --endstage 1 --volume true --datasets "train" --prefix mfcc_20_5.0 --feat_type mfcc \
  #                          --feat_conf subtools/conf/sre-mfcc-20.conf --vad_conf conf/vad-5.0.conf --pitch true --suffix aug
  # subtools/concatSpFeats.sh --stage 0 --endstage 1 --volume true --datasets "lre07_enroll" --prefix mfcc_20_5.0 --feat_type mfcc \
  #                           --feat_conf subtools/conf/sre-mfcc-20.conf --vad_conf subtools/conf/vad-5.0.conf --pitch true --suffix aug
  # subtools/concatSpFeats.sh --stage 0 --endstage 1 --volume true --datasets "lre07_enroll" --prefix fbank_40_5.0 --feat_type fbank \
  #                           --feat_conf subtools/conf/sre-fbank-40.conf --vad_conf subtools/conf/vad-5.0.conf --pitch true --suffix aug
  subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf $task_2_enroll mfcc \
                                  subtools/conf/sre-mfcc-20.conf

  subtools/computeVad.sh $task_2_enroll subtools/conf/vad-5.0.conf
fi

# echo "-----------------------------------"
# exit(0)

if [ $stage -eq 4 ]; then
  echo "start execute stage 4"
  # Make features for testsets, enrollsets and development sets
  #for data in $task1_test $task2_test $task3_test $task1_enroll $task2_enroll $task3_enroll $task1_dev $task2_dev; do
  #  subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf $data mfcc \
  #                                subtools/conf/sre-mfcc-20.conf
  #done
  for data in $task1_test_3sec $task1_test_10sec $task1_test_30sec; do
    subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf $data mfcc \
                                  subtools/conf/sre-mfcc-20.conf
  done
  # for data in $task1_test_3sec $task1_test_10sec $task1_test_30sec; do
  #   subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf $data fbank \
  #                                 subtools/conf/sre-fbank-40.conf
  # done
fi

# echo "-----------------------------"
# exit(0)




# Compute VAD for testsets, enrollsets and development sets
#for data in $task1_test $task2_test $task3_test $task1_enroll $task2_enroll $task3_enroll $task1_dev $task2_dev; do
#  subtools/computeVad.sh $data subtools/conf/vad-5.0.conf
#done
if [ $stage -eq 5 ]; then
  echo "start execute stage 5"
  for data in $task1_test_3sec $task1_test_10sec $task1_test_30sec; do
    subtools/createVisualVad.sh $data subtools/conf/vad-5.0.conf
  done
fi

# echo "-----------------------------"
# exit(0)

if [ $stage -eq 6 ]; then
  echo "start execute stage 6"
  ## Pytorch x-vector model training
  # Training (preprocess -> get_egs -> training -> extract_xvectors)
  # The launcher is a python script which is the main pipeline for it is independent with the data preparing and the scoring.
  # Both this launcher just train a extended xvector baseline system, and other methods like multi-gpu training, 
  # AM-softmax loss etc. could be set by yourself. 

  # First, execute the "python3.8 run_pytorch_xvector.py --stage=0" to get the egs, execute to the train echop 1 and then ctrl+c exit.
  # Secondly, execute muti-gpu-trainer "subtools/runPytorchLauncher.sh run_pytorch_xvector.py --gpu-id=0,1,2 --multi-gpu-solution="ddp"",
  # to get the echop 21 parameter eg. 21.params.
  # Then execute "python3.8 run_pytorch_xvector.py --stage=4" to extract xvector features.

  # Otherwise, execute the "python3.8 run_pytorch_xvector.py --stage=0" which the gpu-num is assigned to 1.
  # It will the run the three steps mentioned above automatically.
  # python3.8 run_pytorch_xvector.py --stage=0
  # python3.8 run_pytorch_xvector.py --stage=4
  # python3.8 run_pytorch_xvector_resnet.py --stage=3 --gpu-id=1
  python3.8 run_pytorch_xvector_resnet.py --stage=4
  # python3.8 run_pytorch_xvector_dnn.py --stage=0
  # python3.8 run_pytorch_xvector_dnn.py --stage=3 --gpu-id=1

  # python3.8 run_pytorch_xvector_resnet_fusion_rank_stat.py --stage=4

  # subtools/runPytorchLauncher.sh run_pytorch_xvector_resnet.py --gpu-id=0,1 --multi-gpu-solution="ddp"
  # subtools/runPytorchLauncher.sh run_pytorch_xvector_resnet.py --gpu-id=2,3 --multi-gpu-solution="ddp"
  # subtools/runPytorchLauncher.sh run_pytorch_xvector.py --gpu-id=0,1,2,3 --multi-gpu-solution="ddp"

  # python3.8 run_pytorch_xvector.py --stage=3
  # python3.8 run_rankpooling_svm.py --stage=3
  # subtools/pytorch/pipeline/extract_xvectors_for_pytorch.sh --model 21.params --cmn true --nj 5 --use-gpu true --gpu-id '' \
  #   --force false --nnet-config config/far.extract.config exp/pytorch_xvector data/mfcc_20_5.0/train_volume_aug_nosil exp/pytorch_xvector/far_epoch_21/train_volume_aug_nosil

  ## Kaldi x-vector model training
  # Training (preprocess -> get_egs -> training -> extract_xvectors)
  
  # bash run_kaldi_xvector.sh

  ## Kaldi i-vector model training
  # Training (preprocess -> get_egs -> training -> extract_ivectors)
  
  # bash run_kaldi_ivector.sh
fi

# echo "-----------------------------"
# exit(0)


if [ $stage -eq 7 ]; then
  echo "start execute stage 7"
  ### Back-end scoring: lda100 -> submean -> norm -> LR

  # For AP20-OLR-ref-dev, the referenced development sets are used to help estimate
  # the system performance when participants repeat the baseline systems or prepare their own systems.
  # Task 1: Cross-channel LID; Task2 : dialect identification; Task3: no ref-development set provided

  # for exp in exp/pytorch_xvector/far_epoch21 exp/pytorch_xvector/far_epoch21 exp/kaldi_xvector/embedding1 exp/kaldi_ivector;do
  #   subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0  --enrollset task1_enroll --testset task1_test \
  #                         --lda true --clda 100 --submean true --score "lr" --metric "Cavg"
  #   sh scoreSets_open_set.sh --eval false --vectordir $exp --prefix mfcc_20_5.0  --enrollset task2_enroll --testset task2_test \
  #                         --lda true --clda 100 --submean true --score "lr" --metric "Cavg"
  # done

  # to 
  # durs="3sec 10sec 30sec"
  # durs="3sec" 
  # for exp in exp/pytorch_xvector_bilstm_mean_std/bilstm_no_mean_std_split_up_down_rank_backup/1e-5_backup/near_epoch_21_up_down-inv_concat;do
  for exp in exp/pytorch_xvector_transformer_mean_std_embed/near_epoch_21;do
    
    #---
    
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset train_nosil --testset task_1_nosil \
                        --lda false --clda 100 --submean true --score "lr" --metric "eer"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset train_nosil --testset task_1_nosil \
                        --lda false --clda 100 --submean true --score "lr" --metric "Cavg"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset train_nosil --testset task_1_nosil \
                        --lda false --clda 100 --submean false --score "lr" --metric "eer"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset train_nosil --testset task_1_nosil \
                        --lda false --clda 100 --submean false --score "lr" --metric "Cavg"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset train_nosil --testset task_1_nosil \
                        --lda true --clda 100 --submean true --score "lr" --metric "eer"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset train_nosil --testset task_1_nosil \
                        --lda true --clda 100 --submean true --score "lr" --metric "Cavg"

    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_2_enroll_nosil --testset task_2_nosil \
                        --lda false --clda 100 --submean true --score "lr" --metric "eer"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_2_enroll_nosil --testset task_2_nosil \
                        --lda false --clda 100 --submean true --score "lr" --metric "Cavg"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_2_enroll_nosil --testset task_2_nosil \
                        --lda false --clda 100 --submean false --score "lr" --metric "eer"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_2_enroll_nosil --testset task_2_nosil \
                        --lda false --clda 100 --submean false --score "lr" --metric "Cavg"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_2_enroll_nosil --testset task_2_nosil \
                        --lda true --clda 100 --submean true --score "lr" --metric "eer"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_2_enroll_nosil --testset task_2_nosil \
                        --lda true --clda 100 --submean true --score "lr" --metric "Cavg"
    
    
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_3_enroll_nosil --testset task_3_test_nosil \
                        --lda false --clda 100 --submean true --score "lr" --metric "eer"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_3_enroll_nosil --testset task_3_test_nosil \
                        --lda false --clda 100 --submean true --score "lr" --metric "Cavg"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_3_enroll_nosil --testset task_3_test_nosil \
                        --lda false --clda 100 --submean false --score "lr" --metric "eer"
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0   --enrollset task_3_enroll_nosil --testset task_3_test_nosil \
                        --lda false --clda 100 --submean false --score "lr" --metric "Cavg"
    
  done
fi
# echo "-----------------------------------"
# exit(0)


if [ $stage -eq 8 ]; then
  echo "start execute stage 8"
  # You can compare your results on AP20-OLR-ref-dev with results.txt to check your systems.

  # For AP20-OLR-test, note that in this stage, only scores will be computed, but no metric will be given, by setting --eval true
  # Task 1: Cross-channel LID; Task2 : dialect identification; Task3: noisy LID

  # for exp in exp/pytorch_xvector/far_epoch21 exp/pytorch_xvector/far_epoch21 exp/kaldi_xvector/embedding1 exp/kaldi_ivector;do
  #   for task in 1 2 3;do
  #     subtools/scoreSets.sh --eval true --vectordir $exp --prefix mfcc_20_5.0  --enrollset task${task}_enroll --testset task${task}_test \
  #                           --lda true --clda 100 --submean true --score "lr" --metric "Cavg"
  #     # Transfer the format of score file to requred format.
  #     subtools/score2table.sh $exp/task${task}_test/lr_task${task}_enroll_task${task}_test_lda100_submean_norm.score $exp/task${task}_test/lr_task${task}_enroll_task${task}_test_lda100_submean_norm.score.requred
  #   done
  # done

  for exp in exp/pytorch_xvector/far_epoch21;do
    for task in 1;do
      subtools/scoreSets.sh --eval true --vectordir $exp --prefix mfcc_20_5.0  --enrollset lre07_enroll_volume_aug --testset lre07_test \
                            --lda true --clda 100 --submean true --score "lr" --metric "eer"
      # Transfer the format of score file to requred format.
      subtools/score2table.sh $exp/lre07_test/lr_lre07_enroll_lre07_test_lda100_submean_norm.score $exp/lre07_test/lr_lre07_enroll_lre07_test_lda100_submean_norm.score.requred
    done
  done

fi

echo "### All done ###"
