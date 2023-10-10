import sys
import os
import argparse
import traceback
import torch
import numpy as np
import kaldi_io

sys.path.insert(0, 'subtools/pytorch')


NUM = 100
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
#vec_array = np.empty([NUM * LANG, DIM], dtype = float) 
#label_array = np.empty([NUM * LANG, 1], dtype = int)
vec_array = np.empty([0, DIM], dtype = float) 
label_array = np.empty([0, 1], dtype = int)


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
    """
    if dic_cnt[lang_cate] < NUM:
        label_array[index] = label[lang_cate]
        vec_array[index] = v
        dic_cnt[lang_cate] += 1
        index += 1
    """
    label_array = np.append(label_array, label[lang_cate])
    vec_array = np.concatenate((vec_array, v.reshape(1, DIM)),axis=0)
    dic_cnt[lang_cate] += 1
    index += 1
    if index % 100 == 0:
        print(index) 


print(label_array.shape)
print(vec_array.shape)




mean_vectors = []
for cl in range(0,6):
    mean_vectors.append(np.mean(vec_array[np.where(label_array == cl)[0]], axis=0))
    #print(cl, mean_vectors[cl-1], mean_vectors[cl-1].shape)
# print(dic_cnt)


S_W = np.zeros((DIM,DIM))


for cl,mv in zip(range(0,6), mean_vectors):
    cnt = 0
    class_sc_mat = np.zeros((DIM,DIM))                  # scatter matrix for every class
    for row in vec_array[np.where(label_array == cl)[0]]:
        row, mv = row.reshape(DIM,1), mv.reshape(DIM,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
        print('inner product', cnt)
        cnt += 1
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)

overall_mean = np.mean(vec_array, axis=0)
 
S_B = np.zeros((DIM,DIM))
for i,mean_vec in enumerate(mean_vectors):  
    n = vec_array[np.where(label_array == i)[0],:].shape[0]
    mean_vec = mean_vec.reshape(DIM,1) # make column vector
    overall_mean = overall_mean.reshape(DIM,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
 
print('between-class Scatter Matrix:\n', S_B)

print('j-measure:', np.trace(np.linalg.pinv(S_W).dot(S_B)))