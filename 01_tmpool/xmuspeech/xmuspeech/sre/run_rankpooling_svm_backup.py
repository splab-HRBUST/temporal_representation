#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar, Zheng Li 2020-05-30)
# Apache 2.0

import sys, os
import logging
import argparse
import traceback
import time
import math
import numpy as np

import torch

sys.path.insert(0, 'subtools/pytorch')

import libs.egs.egs as egs
import libs.training.optim as optim
import libs.training.lr_scheduler as learn_rate_scheduler
import libs.training.trainer as trainer
import libs.support.kaldi_common as kaldi_common
import libs.support.utils as utils
from  libs.support.logging_stdout import patch_logging_stream

from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
import threading
import time
import copy
import libs.support.kaldi_io as kaldi_io



"""A launcher script with python version (Snowdar's launcher to do experiments w.r.t snowdar-xvector.py).
Python version is gived (rather than Shell) to have more freedom, such as decreasing limitation of parameters that transfering 
them to python from shell.

Note, this launcher does not contain dataset preparation, augmentation, extracting acoustic features and back-end scoring etc.
    1.See subtools/recipe/voxceleb/runVoxceleb.sh to get complete stages.
    2.See subtools/newCopyData.sh, subtools/makeFeatures.sh.sh, subtools/computeVad.sh, subtools/augmentDataByNoise.sh and 
          subtools/scoreSets.sh and run these script separately before or after running this launcher.

How to modify this launcher:
    1.Prepare your kaldi format dataset and model.py (model blueprint);
    2.Give the path of dataset, model blueprint, etc. in main parameters field;
    3.Change the imported name of model in 'model = model_py.model_name(...)' w.r.t model.py by yourself;
    4.Modify any training parameters what you want to change (epochs, optimizer and lr_scheduler etc.);
    5.Modify parameters of extracting in stage 4 w.r.t your own training config;
    6.Run this launcher.

Conclusion: preprare -> config -> run.

How to run this launcher to train a model:
    1.For CPU-based training case. The key option is --use-gpu.
        python3 launcher.py --use-gpu=false
    2.For single-GPU training case (Default).
        python3 launcher.py
    3.For DDP-based multi-GPU training case. Note --nproc_per_node is equal to number of gpu id in --gpu-id.
        python3 -m torch.distributed.launch --nproc_per_node=2 launcher.py --gpu-id=0,1
    4.For Horovod-based multi-GPU training case. Note --np is equal to number of gpu id in --gpu-id.
        horovodrun -np 2 launcher.py --gpu-id=0,1
    5.For all of above, a runLauncher.sh script has been created to launch launcher.py conveniently.
      The key option to use single or multiple GPU is --gpu-id.
      The subtools/runPytorchLauncher.sh is a soft symbolic which is linked to subtools/pytorch/launcher/runLauncher.sh, 
      so just use it.

        [ CPU ]
            subtools/runPytorchLauncher.sh launcher.py --use-gpu=false

        [ Single-GPU ]
        (1) Auto-select GPU device
            subtools/runPytorchLauncher.sh launcher.py
        (2) Specify GPU device
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2

        [ Multi-GPU ]
        (1) Use DDP solution (Default).
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2,3 --multi-gpu-solution="ddp"
        (2) Use Horovod solution.
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2,3 --multi-gpu-solution="horovod"

If you have any other requirements, you could modify the codes in anywhere. 
For more details of multi-GPU devolopment, see subtools/README.md.
"""

# Logger
# Change the logging stream from stderr to stdout to be compatible with horovod.
torch.backends.cudnn.enabled = False
patch_logging_stream(logging.INFO)

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
        description="""Train xvector framework with pytorch.""",
        formatter_class=argparse.RawTextHelpFormatter,
        conflict_handler='resolve')

parser.add_argument("--stage", type=int, default=4,
                    help="The stage to control the start of training epoch (default 3).\n"
                         "    stage 0: vad-cmn (preprocess_to_egs.sh).\n"
                         "    stage 1: remove utts (preprocess_to_egs.sh).\n"
                         "    stage 2: get chunk egs (preprocess_to_egs.sh).\n"
                         "    stage 3: training.\n"
                         "    stage 4: extract xvector.")

parser.add_argument("--endstage", type=int, default=4,
                    help="The endstage to control the endstart of training epoch (default 4).")

parser.add_argument("--train-stage", type=int, default=-1,
                    help="The stage to control the start of training epoch (default -1).\n"
                         "    -1 -> creating model_dir.\n"
                         "     0 -> model initialization (e.g. transfer learning).\n"
                         "    >0 -> recovering training.")

parser.add_argument("--force-clear", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="Clear the dir generated by preprocess.")

parser.add_argument("--use-gpu", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="Use GPU or not.")

parser.add_argument("--gpu-id", type=str, default="2",
                    help="If NULL, then it will be auto-specified.\n"
                         "If giving multi-gpu like --gpu-id=1,2,3, then use multi-gpu training.")

parser.add_argument("--multi-gpu-solution", type=str, default="ddp",
                    choices=["ddp", "horovod"],
                    help="if number of gpu_id > 1, this option will be valid to init a multi-gpu solution.")

parser.add_argument("--benchmark", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="If true, save training time but require a little more gpu-memory.")

parser.add_argument("--run-lr-finder", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="If true, run lr finder rather than training.")

parser.add_argument("--sleep", type=int, default=0,
                    help="The waiting time to launch a launcher.")

parser.add_argument("--local_rank", type=int, default=0,
                    help="Do not delete it when using DDP-based multi-GPU training.\n"
                         "It is important for torch.distributed.launch.")

parser.add_argument("--port", type=int, default=29500+808,
                    help="This port is used for DDP solution in multi-GPU training.")

args = parser.parse_args()
##
######################################################### PARAMS ########################################################
##
##--------------------------------------------------##
## Control options
stage = max(0, args.stage)
endstage = min(4, args.endstage)
train_stage = max(-1, args.train_stage)
##--------------------------------------------------##
## Preprocess options
force_clear=args.force_clear
preprocess_nj = 20
cmn = True # Traditional cmn process.

chunk_size = 100
limit_utts = 1

# sample_type="speaker_balance" # sequential | speaker_balance
sample_type="speaker_balance" # sequential | speaker_balance
# chunk_num=-1 # -1 means using scale, 0 means using max and >0 means itself.
chunk_num = -1


overlap=0.1
scale=1.5 # Get max / num_spks * scale for every speaker.
valid_split_type="--total-spk" # --total-spk or --default
valid_utts = 1024
valid_chunk_num_every_utt = 2
##--------------------------------------------------##
## Training options
use_gpu = args.use_gpu # Default true.
benchmark = args.benchmark # If true, save much training time but require a little more gpu-memory.
gpu_id = args.gpu_id # If NULL, then it will be auto-specified.
run_lr_finder = args.run_lr_finder

egs_params = {
    "aug":None, # None or specaugment. If use aug, you should close the aug_dropout which is in model_params.
    "aug_params":{"frequency":0.2, "frame":0.2, "rows":4, "cols":4, "random_rows":True,"random_cols":True}
}

loader_params = {
    "use_fast_loader":True, # It is a queue loader to prefetch batch and storage.
    "max_prefetch":10,
    "batch_size":512,
    # "batch_size":256, 
    "shuffle":True, 
    "num_workers":2,
    "pin_memory":False, 
    "drop_last":True,
}

# Difine model_params by model_blueprint w.r.t your model's __init__(model_params).
model_params = {
    "extend":True, "SE":False, "se_ratio":4, "training":True, "extracted_embedding":"far",

    "aug_dropout":0., "hidden_dropout":0., 
    "dropout_params":{"type":"default", "start_p":0., "dim":2, "method":"uniform",
                      "continuous":False, "inplace":True},

    "tdnn_layer_params":{"nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
                         "bn-relu":False, 
                         "bn":True, 
                         "bn_params":{"momentum":0.5, "affine":False, "track_running_stats":True}},

    "pooling":"rank", # statistics, lde, attentive, multi-head, multi-resolution
    "pooling_params":{"num_nodes":1500,
                      "num_head":1,
                      "share":True,
                      "affine_layers":1,
                      "hidden_size":64,
                      "context":[0],
                      "temperature":False, 
                      "fixed":True
                      },
    "tdnn6":True, 
    "tdnn7_params":{"nonlinearity":"default", "bn":True},

    "margin_loss":False, 
    "margin_loss_params":{"method":"am", "m":0.2, "feature_normalize":True, 
                          "s":30, "mhe_loss":False, "mhe_w":0.01},
    "use_step":False, 
    "step_params":{"T":None,
                   "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
                   "s":False, "s_tuple":(30, 12), "s_list":None,
                   "t":False, "t_tuple":(0.5, 1.2), 
                   "p":False, "p_tuple":(0.5, 0.1)}
}

optimizer_params = {
    "name":"adamW",
    "learn_rate":0.001,
    "beta1":0.9,
    "beta2":0.999,
    "beta3":0.999,
    "weight_decay":3e-1,  # Should be large for decouped weight decay (adamW) and small for L2 regularization (sgd, adam).
    "lookahead.k":5,
    "lookahead.alpha":0.,  # 0 means not using lookahead and if used, suggest to set it as 0.5.
    "gc":False # If true, use gradient centralization.
}

lr_scheduler_params = {
    "name":"warmR",
    "warmR.lr_decay_step":0, # 0 means decay after every epoch and 1 means every iter. 
    "warmR.T_max":3,
    "warmR.T_mult":2,
    "warmR.factor":1.0,  # The max_lr_decay_factor.
    "warmR.eta_min":4e-8,
    "warmR.log_decay":False
}

epochs = 21 # Total epochs to train. It is important.

report_times_every_epoch = None
report_interval_iters = 100 # About validation computation and loss reporting. If report_times_every_epoch is not None, 
                            # then compute report_interval_iters by report_times_every_epoch.
suffix = "params" # Used in saved model file.
##--------------------------------------------------##
## Other options
exist_model=""  # Use it in transfer learning.
##--------------------------------------------------##
## Main params
#traindata="data/mfcc_20_5.0/train_aug"
traindata="data/mfcc_20_5.0/train_volume_aug"
egs_dir="exp/egs/train" + "_" + sample_type + "_max"

model_blueprint="subtools/pytorch/model/snowdar-xvector.py"
model_dir="exp/pytorch_xvector"
##--------------------------------------------------##
##
######################################################### START #########################################################
##
#### Set seed
utils.set_all_seed(1024)
##
#### Init environment
# It is used for multi-gpu training if used (number of gpu-id > 1).
# And it will do nothing for single-GPU training.
utils.init_multi_gpu_training(args.gpu_id, args.multi_gpu_solution, args.port)
##
#### Set sleep time for a rest
# Use it to run a launcher with a countdown function when there are no extra GPU memory 
# but you really want to go to bed and know when the GPU memory will be free.
if args.sleep > 0 and utils.is_main_training(): 
    logger.info("This launcher will sleep {}s before starting...".format(args.sleep))
    time.sleep(args.sleep)
##
#### Auto-config params
# If multi-GPU used, it will auto-scale learning rate by multiplying number of processes.
optimizer_params["learn_rate"] = utils.auto_scale_lr(optimizer_params["learn_rate"])
# It is used for model.step() defined in model blueprint.
if lr_scheduler_params["name"] == "warmR" and model_params["use_step"]:
    model_params["step_params"]["T"]=(lr_scheduler_params["warmR.T_max"], lr_scheduler_params["warmR.T_mult"])
##
#### Preprocess
if stage <= 2 and endstage >= 0 and utils.is_main_training():
    # Here only give limited options because it is not convenient.
    # Suggest to pre-execute this shell script to make it freedom and then continue to run this launcher.
    kaldi_common.execute_command("sh subtools/pytorch/pipeline/preprocess_to_egs.sh "
                                 "--stage {stage} --endstage {endstage} --valid-split-type {valid_split_type} "
                                 "--nj {nj} --cmn {cmn} --limit-utts {limit_utts} --min-chunk {chunk_size} --overlap {overlap} "
                                 "--sample-type {sample_type} --chunk-num {chunk_num} --scale {scale} --force-clear {force_clear} "
                                 "--valid-num-utts {valid_utts} --valid-chunk-num {valid_chunk_num_every_utt} "
                                 "{traindata} {egs_dir}".format(stage=stage, endstage=endstage, valid_split_type=valid_split_type, 
                                 nj=preprocess_nj, cmn=str(cmn).lower(), limit_utts=limit_utts, chunk_size=chunk_size, overlap=overlap, 
                                 sample_type=sample_type, chunk_num=chunk_num, scale=scale, force_clear=str(force_clear).lower(), 
                                 valid_utts=valid_utts, valid_chunk_num_every_utt=valid_chunk_num_every_utt, traindata=traindata, 
                                 egs_dir=egs_dir))

#### Train model
#===============================================================================================================================================
#global_lock()
global_lock = threading.Lock()

# global temporal_frames, frame_lables
frames_fd = open("data/frames.csv",'a')
labels_fd = open("data/labels.csv",'a')

test_frames_fd = open("data/test_frames.csv",'a')
test_labels_fd = open("data/test_labels.csv",'a')


temporal_frames = np.empty(shape=[0, 46])
frame_labels = np.empty(shape=[0, 1])

test_temporal_frames = np.empty(shape=[0, 46])
test_frame_labels = np.empty(shape=[0, 1])

# threads = []

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter, inputs, frame_labels, start_ind, end_ind):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

        self.inputs = inputs
        self.frame_labels = frame_labels
        self.start_ind = start_ind
        self.end_ind = end_ind
    
    def run(self):
        self.run_thread_func(self.inputs, self.frame_labels, self.start_ind, self.end_ind)

    def run_thread_func(self, inputs, frame_labels, start_ind, end_ind):
        frame_dim = inputs.shape[1]*2
        temporal_frames_temp = np.empty(shape=[0, frame_dim])
        frame_lables_temp = np.empty(shape=[0, 1])

        for frame_ind in range(start_ind, end_ind):
            frame = inputs[frame_ind]
            label = frame_labels[frame_ind].reshape((1,1))
            # print(label.shape)
            W = get_model_params(smooth(frame), frame_ind)
            W = W.reshape((1,frame_dim))
            temporal_frames_temp = np.concatenate((temporal_frames_temp, W), axis=0)
            frame_lables_temp = np.concatenate((frame_lables_temp, label), axis=0)

        #get_lock()
        global temporal_frames, frame_lables, frames_fd, labels_fd
        global_lock.acquire()
        # temporal_frames = np.concatenate((temporal_frames, temporal_frames_temp), axis=0)
        # frame_lables = np.concatenate((frame_lables, frame_lables_temp), axis=0)
        # np.save(frames_fd, temporal_frames_temp)
        # np.save(labels_fd, frame_lables_temp)
        np.savetxt(frames_fd,temporal_frames_temp,fmt='%f',delimiter=',')
        np.savetxt(labels_fd,frame_lables_temp,fmt='%d',delimiter=',')
        #free_lock()
        global_lock.release()


        


def smooth(inputs):
    inputs_ = inputs
    
    frames_ = inputs_
    # ---
    frames_ = frames_.cumsum(1)
    frames_ = frames_ / (torch.Tensor(range(0, frames_.shape[1])) + 1)
    # normalize
    #=======================================================
    norms = torch.norm(frames_, p=2, dim=0, keepdim=True)
    frames_ = frames_ / norms
    #=======================================================

    # nonlinear
    #==============================================
    frames_pos = copy.deepcopy(frames_)
    frames_neg = copy.deepcopy(frames_)

    frames_pos = torch.clamp(frames_pos, min=0)
    frames_neg = torch.clamp(frames_neg, max=0)

    frames_pos = torch.sqrt(frames_pos)
    frames_neg = torch.sqrt(-frames_neg)

    ret_frames = torch.cat((frames_pos, frames_neg), 0)
    #=================================================

    return ret_frames

def get_model_params(inputs, sample_num):
    inputs_ = inputs

    frames_ = inputs_.t()

    # times_ = torch.tensor(range(0, frames_.shape[1]))
    times_pre = torch.ones(1, frames_.shape[0])
    times_ = times_pre.cumsum(1).t()

    X = frames_.numpy()
    Y = times_.numpy()

    # svr = SVR(kernel='linear', C = 1e2, max_iter = 200)
    svr = SVR(kernel='linear', C = 1e2, cache_size = 2048, tol = 1e-3, max_iter = 1e4)
    # svr = SVR(kernel='linear', C = 100)

    svr.fit(X, Y.ravel())
    W = svr.coef_[0]
    W_norm = np.linalg.norm(x=W, ord=2)
    W = W / W_norm
    b = svr.intercept_
    
    print("[R^2 score = {}, frame # = {}]".format(svr.score(X,Y), X.shape))

    return W


def run_threads(inputs, input_labels, threads_num):
    threads = []
    samples_num = inputs.shape[0]

    start_ind = end_ind = 0
    increase_step = int(samples_num / threads_num)
    end_ind += increase_step
    for i in range(0, threads_num):
        threads.append(myThread(i, "Thread-{}".format(i), i, inputs, input_labels, start_ind, end_ind))
        start_ind += increase_step
        end_ind += increase_step

    for t in threads:
        t.start()

    for t in threads:
        t.join()

class myThread_train (threading.Thread):
    def __init__(self, threadID, name, counter, input, label):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

        self.input = input
        self.label = label

    def run(self):
        W = get_model_params(smooth(self.input), 0)
        # print(W.shape)
        # print(W)
        # np.savetxt(frames_fd,W.reshape(1,46),fmt='%f',delimiter=',')
        # np.savetxt(labels_fd,np.array([self.label]),fmt='%d',delimiter=',')
        global temporal_frames, frame_labels
        temporal_frames = np.concatenate((temporal_frames, W.reshape(1,46)), axis=0)
        frame_labels = np.concatenate((frame_labels, np.array([self.label]).reshape(1,1)), axis=0)

class myThread_test (threading.Thread):
    def __init__(self, threadID, name, counter, input, label):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

        self.input = input
        self.label = label

    def run(self):
        W = get_model_params(smooth(self.input), 0)
        # print(W.shape)
        # print(W)
        # np.savetxt(frames_fd,W.reshape(1,46),fmt='%f',delimiter=',')
        # np.savetxt(labels_fd,np.array([self.label]),fmt='%d',delimiter=',')
        global test_temporal_frames, test_frame_labels
        test_temporal_frames = np.concatenate((test_temporal_frames, W.reshape(1,46)), axis=0)
        test_frame_labels = np.concatenate((test_frame_labels, np.array([self.label]).reshape(1,1)), axis=0)



def run_threads_train(frame, label):
    thread = myThread_train(1,"Thread-1",1,frame.T,label)
    thread.start()
    thread.join()

def run_threads_test(frame, label):
    thread = myThread_test(1,"Thread-1",1,frame.T,label)
    thread.start()
    thread.join()


def load_frames(fd, skiprows, max_rows, dtype, delimiter):
    return np.loadtxt(fd,skiprows=skiprows,max_rows=max_rows,dtype = dtype,delimiter=delimiter)


def map_language_to_label(language):
    sets = {
        "arabic": 0,
        "bengali": 1,
        "chinese": 2,
        "english": 3,
        "farsi": 4,
        "german": 5,
        "hindustani": 6,
        "japanese": 7,
        "korean": 8,
        "russian": 9,
        "spanish": 10,
        "tamil": 11,
        "thai": 12,
        "vietnamese": 13
    }

    return sets.get(language)




if stage <= 3 <= endstage:


    #=======================================================================================================================================
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    if utils.is_main_training(): logger.info("Get model_blueprint from model directory.")
    # Save the raw model_blueprint in model_dir/config and get the copy of model_blueprint path.
    model_blueprint = utils.create_model_dir(model_dir, model_blueprint, stage=train_stage)

    if utils.is_main_training(): logger.info("Load egs to bunch.")
    # The dict [info] contains feat_dim and num_targets.
    bunch, info = egs.BaseBunch.get_bunch_from_egsdir(egs_dir, egs_params, loader_params)



    for bunch_ind, data in enumerate(bunch.train_loader):
        inputs, labels = data
        labels = labels.numpy()
        # labels = labels.reshape((512,1))

        

        print("inputs shape = {} | labels shape = {}".format(inputs.shape, labels.shape))
        
        time.sleep(1000)
        
        run_threads(inputs, labels, threads_num = 32)
        print("[iter/total : {}/{}] [inputs shape = {}] [lables shape = {}]".format(bunch_ind, bunch.__len__(), inputs.shape, labels.shape))
    #==========================================================================================================================================


    
    #========================================================================================================================================
    start_ind = end_ind = 0
    increase_step = 10000
    end_ind += increase_step

    bunch_num = 188
    for bunch_ind in range(0, bunch_num-1):
        fd = open("data/labels.csv",'rb')
        fd_ = open("data/frames.csv",'rb')
        # frames = np.loadtxt(fd_,skiprows=start_ind,max_rows=increase_step,dtype = np.float,delimiter=',')
        # labels = np.loadtxt(fd,skiprows=start_ind,max_rows=increase_step,dtype = np.int,delimiter=',')
        frames = load_frames(fd_,skiprows=start_ind,max_rows=increase_step,dtype = np.float,delimiter=',')
        labels = load_frames(fd,skiprows=start_ind,max_rows=increase_step,dtype = np.float,delimiter=',')
        svm.fit(frames,labels)
        start_ind += increase_step
        end_ind += increase_step
        print("[iter/total = {}/{}]".format(bunch_ind, bunch_num-1))
        fd.close()
        fd_.close()


    fd = open("data/labels.csv",'rb')
    fd_ = open("data/frames.csv",'rb')

    test_frames = np.loadtxt(fd_,skiprows=1880000,max_rows=2000,dtype = np.float,delimiter=',')
    test_labels = np.loadtxt(fd,skiprows=1880000,max_rows=2000,dtype = np.int,delimiter=',')

    fd.close()
    fd_.close()

    print(test_frames.shape)
    print("========fit model=============")
    svm.fit(frames,labels)
    print("========predict step===========")
    predict_test_labels = svm.predict(test_frames)
    predict_train_labels = svm.predict(frames)
    print("test accuracy score: {0:.2f}".format(accuracy_score(predict_test_labels,test_labels)))
    print("Congratulate! All done!")
    
    
    #===============================================================================================================================================







# if stage <= 3 <= endstage:

#     #==========================================================================================================================================


#     # svm = SVC(kernel = 'rbf', C=20, verbose=2)
#     svm = SVC(kernel = 'rbf', C=20)
#     # extract features and train svm
#     # feats_rspecifier_file = "raw_mfcc_pitch_train_volume_aug.1.ark"
#     frame_counter = 0
#     train_file_num = 30
#     for file_ind in range(1, train_file_num+1):
#         frame_counter = 0
#         feats_rspecifier = "exp/features/mfcc/data_mfcc_20_5.0_train_volume_aug/raw_mfcc_pitch_train_volume_aug.{}.ark".format(file_ind)
#         with kaldi_io.open_or_fd(feats_rspecifier, "rb") as r:
            
#             while(True):
#                 key = kaldi_io.read_key(r)
#                 if key:
#                     label = map_language_to_label(key.partition('_')[0])
                
#                 if not key:
#                     break


#                 feats = kaldi_io.read_mat(r)


#                 # W = get_model_params(smooth(feats.T), 0)
#                 # np.savetxt(frames_fd,W.reshape(1,46),fmt='%f',delimiter=',')
#                 # np.savetxt(labels_fd,np.array([label]),fmt='%d',delimiter=',')
#                 run_threads_train(feats, label)

#                 # print("feats shape = {}, label = {}".format(feats.shape, label))
#                 print("[file # {} | iter = {}]".format(file_ind, frame_counter))
#                 # print(frame_labels.shape)
#                 # print(temporal_frames.shape)
#                 frame_counter += 1
#                 # time.sleep(1)

#     np.savetxt(frames_fd,temporal_frames,fmt='%f',delimiter=',')
#     np.savetxt(labels_fd,frame_labels,fmt='%f',delimiter=',')


#     frames_fd.close()
#     labels_fd.close()
    
#     print("========fit model=============")
#     svm.fit(temporal_frames,frame_labels)

#     #======================================================================================================================================


#     #======================================================================================================================================
#     # test step
    

#     test_file_num = 20
    
#     for file_ind in range(1, test_file_num+1):
#         frame_counter = 0
#         feats_rspecifier = "exp/features/mfcc/data_mfcc_20_5.0_lre07_test_10sec/raw_mfcc_pitch_lre07_test_10sec.{}.ark".format(file_ind)
#         with kaldi_io.open_or_fd(feats_rspecifier, "rb") as r:
            
#             while(True):
#                 key = kaldi_io.read_key(r)
#                 if key:
#                     label = map_language_to_label(key.partition('_')[0])
                
#                 if not key:
#                     break


#                 feats = kaldi_io.read_mat(r)


#                 run_threads_test(feats, label)

#                 print("[file # {} | iter = {}]".format(file_ind, frame_counter))
#                 # time.sleep(1)
    
#     np.savetxt(test_frames_fd,test_temporal_frames,fmt='%f',delimiter=',')
#     np.savetxt(test_labels_fd,test_frame_labels,fmt='%f',delimiter=',')


#     test_frames_fd.close()
#     test_labels_fd.close()
    

#     print("========predict step===========")
#     predict_test_labels = svm.predict(test_temporal_frames)
#     print("test accuracy score: {0:.2f}".format(accuracy_score(predict_test_labels,test_frame_labels)))
#     print("Congratulate! All done!")


#     time.sleep(1000)





































































































































































































































    

#     if utils.is_main_training(): logger.info("Create model from model blueprint.")
#     # Another way: import the model.py in this python directly, but it is not friendly to the shell script of extracting and
#     # I don't want to change anything about extracting script when the model.py is changed.
#     model_py = utils.create_model_from_py(model_blueprint)
#     model = model_py.Xvector(info["feat_dim"], info["num_targets"], **model_params)

#     # If multi-GPU used, then batchnorm will be converted to synchronized batchnorm, which is important 
#     # to make peformance stable. 
#     # It will change nothing for single-GPU training.
#     model = utils.convert_synchronized_batchnorm(model)

#     if utils.is_main_training(): logger.info("Define optimizer and lr_scheduler.")
#     optimizer = optim.get_optimizer(model, optimizer_params)
#     lr_scheduler = learn_rate_scheduler.LRSchedulerWrapper(optimizer, lr_scheduler_params)

#     # Record params to model_dir
#     utils.write_list_to_file([egs_params, loader_params, model_params, optimizer_params, 
#                               lr_scheduler_params], model_dir+'/config/params.dict')

#     if utils.is_main_training(): logger.info("Init a simple trainer.")
#     # Package(Elements:dict, Params:dict}. It is a key parameter's package to trainer and model_dir/config/.
#     package = ({"data":bunch, "model":model, "optimizer":optimizer, "lr_scheduler":lr_scheduler},
#             {"model_dir":model_dir, "model_blueprint":model_blueprint, "exist_model":exist_model, 
#             "start_epoch":train_stage, "epochs":epochs, "use_gpu":use_gpu, "gpu_id":gpu_id, "max_change":10.,
#             "benchmark":benchmark, "suffix":suffix, "report_times_every_epoch":report_times_every_epoch,
#             "report_interval_iters":report_interval_iters, "record_file":"train.csv"})

#     trainer = trainer.SimpleTrainer(package)

#     if run_lr_finder and utils.is_main_training():
#         trainer.run_lr_finder("lr_finder.csv", init_lr=1e-8, final_lr=10., num_iters=2000, beta=0.98)
#         endstage = 3 # Do not start extractor.
#     else:
#         trainer.run()


# #### Extract xvector
# if stage <= 4 <= endstage and utils.is_main_training():
#     # There are some params for xvector extracting.
#     data_root = "data" # It contains all dataset just like Kaldi recipe.
#     prefix = "mfcc_20_5.0" # For to_extracted_data.

#     to_extracted_positions = ["far"] # Define this w.r.t extracted_embedding param of model_blueprint.
#     # to_extracted_data = ["train","task1_test","task2_test","task3_test","task1_enroll","task2_enroll","task3_enroll","task1_dev"," task2_dev"] # All dataset should be in data_root/prefix.
    
#     #Assumeing the first element of "to_extracted_data" is the train set.
#     to_extracted_data = ["train_volume_aug_nosil","lre07_test_3sec","lre07_test_10sec","lre07_test_30sec","lre07_enroll_volume_aug"] # All dataset should be in data_root/prefix.
#     to_extracted_epochs = ["21"] # It is model's name, such as 10.params or final.params (suffix is w.r.t package).

#     nj = 5
#     force = False
#     use_gpu = False
#     gpu_id = ""
#     sleep_time = 10

#     print("mark1")
#     # Run a batch extracting process.
#     try:
#         for position in to_extracted_positions:
#             # Generate the extracting config from nnet config where 
#             # which position to extract depends on the 'extracted_embedding' parameter of model_creation (by my design).
#             model_blueprint, model_creation = utils.read_nnet_config("{0}/config/nnet.config".format(model_dir))
#             print("----------------{0}/config/nnet.config".format(model_dir))
#             model_creation = model_creation.replace("training=True", "training=False") # To save memory without loading some independent components.
#             model_creation = model_creation.replace(model_params["extracted_embedding"], position)
#             print("----------------{0}.extract.config".format(position))
#             extract_config = "{0}.extract.config".format(position)
#             print("---------------{0}/config/{1}".format(model_dir, extract_config))
#             utils.write_nnet_config(model_blueprint, model_creation, "{0}/config/{1}".format(model_dir, extract_config))
#             for epoch in to_extracted_epochs:
#                 model_file = "{0}.{1}".format(epoch, suffix)
#                 point_name = "{0}_epoch_{1}".format(position, epoch)

#                 # If run a trainer with background thread (do not be supported now) or run this launcher extrally with stage=4 
#                 # (it means another process), then this while-listen is useful to start extracting immediately (but require more gpu-memory).
#                 model_path = "{0}/{1}".format(model_dir, model_file)
#                 while True:
#                     if os.path.exists(model_path):
#                         break
#                     else:
#                         time.sleep(sleep_time)
#                 print("mark2")
#                 for data in to_extracted_data:
#                     datadir = "{0}/{1}/{2}".format(data_root, prefix, data)
#                     outdir = "{0}/{1}/{2}".format(model_dir, point_name, data)
#                     # Use a well-optimized shell script (with multi-processes) to extract xvectors.
#                     # Another way: use subtools/splitDataByLength.sh and subtools/pytorch/pipeline/onestep/extract_embeddings.py 
#                     # with python's threads to extract xvectors directly, but the shell script is more convenient.
#                     kaldi_common.execute_command("sh subtools/pytorch/pipeline/extract_xvectors_for_pytorch.sh "
#                                                 "--model {model_file} --cmn {cmn} --nj {nj} --use-gpu {use_gpu} --gpu-id '{gpu_id}' "
#                                                 " --force {force} --nnet-config config/{extract_config} "
#                                                 "{model_dir} {datadir} {outdir}".format(model_file=model_file, cmn=str(cmn).lower(), nj=nj,
#                                                 use_gpu=str(use_gpu).lower(), gpu_id=gpu_id, force=str(force).lower(), extract_config=extract_config,
#                                                 model_dir=model_dir, datadir=datadir, outdir=outdir))
#     except BaseException as e:
#         if not isinstance(e, KeyboardInterrupt):
#             traceback.print_exc()
#         sys.exit(1)


# #### Congratulate! All done.