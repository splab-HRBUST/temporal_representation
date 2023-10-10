import sys
import os
import argparse
import traceback
import torch
import numpy as np
import kaldi_io

sys.path.insert(0, 'subtools/pytorch')




NUM = 500
DIM = 3072
LANG = 6

#our train
ark_file_path = "/opt/kaldi/egs/xmuspeech/sre/exp/pytorch_xvector_bilstm_mean_std/bilstm_no_mean_std_posneg_split_up_down_rank_backup/1e-4_backup/near_epoch_21_up_down-inv_concat/task_2_enroll_nosil/xvector_submean_norm.ark"
method_name = "our_train"



dic_cnt = {'Tibet': 0,    #0
       'Uyghu': 0,    #1
       'ja-jp': 0,    #2
       'ru-ru': 0,    #3
       'vi-vn': 0,    #4
       'zh-cn': 0}    #5

label = {'Tibet': 0,    #0
         'Uyghu': 1,    #1
         'ja-jp': 2,    #2
         'ru-ru': 3,    #3
         'vi-vn': 4,    #4
         'zh-cn': 5}    #5

r = kaldi_io.open_or_fd(ark_file_path, "rb")

value_pairs = kaldi_io.read_vec_flt_ark(r)
vec_array = np.empty([NUM * LANG, DIM], dtype = float) 
label_array = np.empty([NUM * LANG, 1], dtype = int)

index = 0
for (k, v) in value_pairs:
    if(k[0] == "j" and k[1] == "a"):
        lang_cate = "ja-jp"
    elif(k[0] == "z" and k[1] == "h"):
        lang_cate = "zh-cn"
    elif(k[0] == "r" and k[1] == "u"):
        lang_cate = "ru-ru"
    elif(k[0] == "v" and k[1] == "i"):
        lang_cate = "vi-vn"
    else:
        lang_cate, id = k.split("_", 1)

    if dic_cnt[lang_cate] < NUM:
        label_array[index] = label[lang_cate]
        vec_array[index] = v
        dic_cnt[lang_cate] += 1
        index += 1

#print(label_array.shape)
#print(vec_array.shape)

np.save("npys-500/" + method_name + "_labels", label_array)
np.save("npys-500/" + method_name + "_vecs", vec_array)



