# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-06-05)

import sys
import os
import argparse
import traceback
import torch
import numpy as np

sys.path.insert(0, 'subtools/pytorch')

import libs.support.utils as utils
import libs.support.kaldi_io as kaldi_io

# Parse
parser = argparse.ArgumentParser(description="Extract embeddings form a piece of feats.scp or pipeline")


parser.add_argument("--nnet-config", type=str, default="",
                        help="This config contains model_blueprint and model_creation.")

parser.add_argument("--model-blueprint", type=str, default=None,
                        help="A *.py which includes the instance of nnet in this training.")

parser.add_argument("--model-creation", type=str, default=None,
                        help="A command to create the model class according to the class \
                        declaration in --model-path, such as using Xvector(40,2) to create \
                        a Xvector nnet.")

parser.add_argument("--use-gpu", type=str, default='true',
                    choices=["true", "false"],
                    help="If true, use GPU to extract embeddings.")

parser.add_argument("--gpu-id", type=str, default="",
                        help="Specify a fixed gpu, or select gpu automatically.")

parser.add_argument("model_path", metavar="model-path", type=str,
                    help="The model used to extract embeddings.")
                
parser.add_argument("feats_rspecifier", metavar="feats-rspecifier",
                    type=str, help="")
parser.add_argument("vectors_wspecifier", metavar="vectors-wspecifier",
                    type=str, help="")

                


print(' '.join(sys.argv))

args = parser.parse_args()

# Start

try:
    if args.nnet_config != "":
        model_blueprint, model_creation = utils.read_nnet_config(args.nnet_config)
    elif args.model_blueprint is not None and args.model_creation is not None:
        model_blueprint = args.model_blueprint
        model_creation = args.model_creation
    else:
        raise ValueError("Expected nnet_config or (model_blueprint, model_creation) to exist.")

    model = utils.create_model_from_py(model_blueprint, model_creation)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)

    # Select device
    model = utils.select_model_device(model, args.use_gpu, gpu_id=args.gpu_id)

    model.eval()

    with kaldi_io.open_or_fd(args.feats_rspecifier, "rb") as r, \
        kaldi_io.open_or_fd(args.vectors_wspecifier, 'wb') as w:
        
        while(True):
            key = kaldi_io.read_key(r)
            
            if not key:
                break

            print("Process utterance for key {0}".format(key))

            feats = kaldi_io.read_mat(r)
            # type_1 = ["statistic","very_far",-1]
            # type_2 = ["rank","very_far",[1,1e3]]
            # type_3 = ["rank","very_far",1e3]
            # type_1 = "statistic"
            # type_2 = "rank_1"
            type_3 = "up_rank_1"
            type_4 = "up_inv_rank_1"
            type_5 = "down_rank_1"
            type_6 = "down_inv_rank_1"
            # embedding_1 = model.extract_embedding_plus(feats,pooling_type=type_1)
            # embedding_2 = model.extract_embedding_plus(feats,pooling_type=type_2)
            embedding_3 = model.extract_embedding_plus(feats,pooling_type=type_3)
            embedding_4 = model.extract_embedding_plus(feats,pooling_type=type_4)
            embedding_5 = model.extract_embedding_plus(feats,pooling_type=type_5)
            embedding_6 = model.extract_embedding_plus(feats,pooling_type=type_6)
            # print(embedding_2)
            # print(embedding_3)
            # print("embedding shape1 = {}, embedding shape2 = {}, embedding shape3 = {}".format(embedding_1.shape, embedding_2.shape, embedding_3.shape))
            # embedding = torch.cat((embedding_1,embedding_2,embedding_3),dim=0)
            embedding = torch.cat((embedding_3, embedding_4, embedding_5, embedding_6),dim=0)

            # print(embedding.shape)
            kaldi_io.write_vec_flt(w, embedding.numpy(), key=key)

            # mat = np.array([embedding.numpy()]).T
            # kaldi_io.write_mat(w, mat)
            

except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)
        


