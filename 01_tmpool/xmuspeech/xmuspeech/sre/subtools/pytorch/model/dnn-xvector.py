# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-05)

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "subtools/pytorch")

import libs.support.utils as utils
from libs.nnet import *
import math


class SimpleDnn(torch.nn.Module):
    def __init__(self, inputs_dim, simplenet_params = [], fc_params={}):
        super(SimpleDnn, self).__init__()
        self.inputs_dim = inputs_dim
        self.layer_num = len(simplenet_params)
        self.output_dim = simplenet_params[self.layer_num - 1]
        self.aug_dropout = torch.nn.Dropout2d(p=0.5)
        assert self.layer_num >= 1

        if self.layer_num >= 1:
            self.fc1 = ReluBatchNormTdnnLayer(inputs_dim, simplenet_params[0], **fc_params)
        else:
            self.fc1 = None

        if self.layer_num >= 2:
            self.fc2 = ReluBatchNormTdnnLayer(simplenet_params[0], simplenet_params[1], **fc_params)
        else:
            self.fc2 = None
        
        if self.layer_num >= 3:
            self.fc3 = ReluBatchNormTdnnLayer(simplenet_params[1], simplenet_params[2], **fc_params)
        else:
            self.fc3 = None
        
        if self.layer_num >= 4:
            self.fc4 = ReluBatchNormTdnnLayer(simplenet_params[2], simplenet_params[3], **fc_params)
        else:
            self.fc4 = None

    def forward(self, inputs):
        x = inputs

        if self.fc1 is not None:
            x = self.fc1(x)
        if self.fc2 is not None:
            x = self.fc2(x)
        if self.fc3 is not None:
            x = self.fc3(x)
        if self.fc4 is not None:
            x = self.fc4(x)

        return x

    def get_output_dim(self):
        return self.output_dim

    def extra_repr(self):
        pass

    @classmethod
    def thop_count(self, m, x, y):
        pass






class FusionNetXvector(TopVirtualNnet):
    """ A standard x-vector framework """
    
    def init(self, inputs_dim, num_targets, nonlinearity="relu", aug_dropout=0.2, 
                training=True, extracted_embedding="near", simplenet_params = [512, 512, 256, 256],
                fc_params = {}, use_step = False, planes = []):

        fc_params={
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":False,
            "bn":True,  
            "bn_params":{"momentum":0.5, "affine":True, "track_running_stats":True}
            }

        self.aug_dropout = torch.nn.Dropout2d(p=0.5)

        self.dim = math.ceil(inputs_dim / 2)

        self.c1 = torch.nn.Parameter(torch.ones(1) * 0.5)
        # self.c2 = torch.nn.Parameter(torch.ones(1)*0.25)

        self.SimpleDnn1 = SimpleDnn(inputs_dim = self.dim*2, simplenet_params = [self.dim*2, 256], fc_params = fc_params)
        self.SimpleDnn2 = SimpleDnn(inputs_dim = self.dim*2, simplenet_params = [self.dim*2, 256], fc_params = fc_params)


        layer_num = len(planes)

        assert layer_num >= 1

        if layer_num >= 1:
            self.fc1 = ReluBatchNormTdnnLayer(inputs_dim, planes[0], **fc_params)
        else:
            self.fc1 = None

        if layer_num >= 2:
            self.fc2 = ReluBatchNormTdnnLayer(planes[0], planes[1], **fc_params)
        else:
            self.fc2 = None
        
        if layer_num >= 3:
            self.fc3 = ReluBatchNormTdnnLayer(planes[1], planes[2], **fc_params)
        else:
            self.fc3 = None



        # Do not need when extracting embedding.
        if training :
            self.loss = SoftmaxLoss(planes[layer_num - 1], num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            # self.transform_keys = ["tdnn1","tdnn2","tdnn3","tdnn4","tdnn5","stats","tdnn6","tdnn7"]
    
    def normalize(self, tensor):
        batch_num = tensor.shape[0]
        for ind in range(0, batch_num):
            frame = tensor[ind]
            norm = torch.norm(frame, p=2, dim=0)
            frame /= norm
            tensor[ind] = frame
        return tensor

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index,  frames-dim-index, frames-index]
        """
        x = inputs
        # print("x shape = {}".format(x.shape))
        # print(self.dim)
        # mean_std = x[:, 0:self.dim*2, :]
        # nonlinear_normed_rank = x[:, self.dim*2:self.dim*4, :]
        rank1 = x[:, self.dim*0:self.dim*1, :]
        rank2 = x[:, self.dim*1:self.dim*2, :]


        # rank1 = rank1 * (1 - self.c1)
        # rank2 = rank2 * self.c1

        # validate efficiency.
        # print("mean norm = {}".format(x[0][self.dim*0:self.dim*1, :].norm(dim=0, p=2)))
        # print("std norm = {}".format(x[0][self.dim*1:self.dim*2, :].norm(dim=0, p=2)))
        # print("rank1 norm = {}".format(x[0][self.dim*2:self.dim*3, :].norm(dim=0, p=2)))
        # print("rank2 norm = {}".format(x[0][self.dim*3:self.dim*4, :].norm(dim=0, p=2)))

        #previous simple net
        # mean_std_embedding = self.SimpleDnn1(mean_std)
        # mean_std_embedding = mean_std
        # rank_embedding = self.SimpleDnn2(rank2_)
        # rank_embedding = self.SimpleDnn2(nonlinear_normed_rank)
        # rank_embedding = nonlinear_normed_rank


        #cat to fusion embedding
        # fusion_embedding = torch.cat((mean_std_embedding, rank_embedding), dim=2)
        # fusion_embedding = torch.cat((mean_std_embedding, rank_embedding), dim=1)
        # fusion_embedding = torch.cat((rank1, rank2), dim=2)

        # # print(fusion_embedding.shape)
        # counts = fusion_embedding.shape[2]
        # fusion_mean = fusion_embedding.sum(dim=2, keepdim=True) / counts
        # fusion_max = torch.max(fusion_embedding, dim=2)[0].unsqueeze(2)
        # fusion_min = -torch.min(fusion_embedding, dim=2)[0].unsqueeze(2)

        # x = torch.cat((fusion_mean, fusion_max, fusion_min), dim=1)
        # x = torch.cat((rank1, rank2), dim=1)
        
        
        x = rank1 + rank2
        x = x / x[0].norm(dim=0, p=2)
        
        # print("self.c1 = {}".format(self.c1))
        
        # x = nonlinear_normed_rank
        # print(x)
        # x = self.fc1(x)
        # if self.fc1 is not None:
        #     x = self.auto(self.fc1, x)
        #     # x = self.aug_dropout(x)
        # # x = self.fc1(x)
        # if self.fc2 is not None:
        #     x = self.auto(self.fc2, x)
        #     # x = self.aug_dropout(x)
        # # x = self.fc2(x)

        # if self.fc3 is not None:
        #     x = self.fc3(x)
        #     # x = self.aug_dropout(x)

        outputs = x

        return outputs


    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using DnnXvector model function.
        e.g.:
            m=DnnXvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """
        return self.loss(inputs, targets)

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        # mean_std = x[:, 0:self.dim*2, :]
        # nonlinear_normed_rank = x[:, self.dim*2:self.dim*4, :]
        # dim = self.dim 
        dim = 1536
        rank1 = x[:, dim*0:dim*1, :]
        rank2 = x[:, dim*3:dim*4, :]


        # c = 0.0
        # rank1 = (1 - c) * rank1
        # rank2 = c * rank2 
        # rank1 = (1 - self.c1) * rank1
        # rank2 = self.c1 * rank2


        #previous simple net
        # mean_std_embedding = self.SimpleDnn1(mean_std)
        # mean_std_embedding = mean_std
        # rank_embedding = self.SimpleDnn2(rank2_)
        # rank_embedding = self.SimpleDnn2(nonlinear_normed_rank)
        # rank_embedding = nonlinear_normed_rank


        #cat to fusion embedding
        # fusion_embedding = torch.cat((mean_std_embedding, rank_embedding), dim=2)
        # fusion_embedding = torch.cat((mean_std_embedding, rank_embedding), dim=1)
        # fusion_embedding = torch.cat((rank1, rank2), dim=2)

        # counts = fusion_embedding.shape[2]
        # fusion_mean = fusion_embedding.sum(dim=2, keepdim=True) / counts
        # fusion_max = torch.max(fusion_embedding, dim=2)[0].unsqueeze(2)
        # fusion_min = -torch.min(fusion_embedding, dim=2)[0].unsqueeze(2)

        # x = torch.cat((fusion_mean, fusion_max, fusion_min), dim=1)
        # x = mean_std
        # x = torch.cat((mean * nonlinear_normed_rank, std * nonlinear_normed_rank), dim=1)
        x = torch.cat((rank1, rank2), dim=1)
        # x = rank1 + rank2
        x = x / x[0].norm(dim=0, p=2)
        # x = nonlinear_normed_rank
        # x = self.fc1(x)
        # if self.fc1 is not None:
        #     x = self.auto(self.fc1, x)
        #     # x = self.aug_dropout(x)
        # # x = self.fc1(x)
        # if self.fc2 is not None:
        #     x = self.auto(self.fc2, x)
        #     # x = self.aug_dropout(x)
        # # x = self.fc2(x)

        # if self.fc3 is not None:
        #     x = self.fc3(x)
        #   # x = self.aug_dropout(x)

        xvector = x

        return xvector


# Test.
if __name__ == "__main__":
    print(DnnXvector(23, 1211))
