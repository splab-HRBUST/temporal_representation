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
                fc_params = {}, use_step = False, planes = [512, 512, 256]):

        fc_params={
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":False,
            "bn":True,  
            "bn_params":{"momentum":0.5, "affine":True, "track_running_stats":True}
            }


        self.dim = math.ceil(inputs_dim / 4)


        self.SimpleDnn1 = SimpleDnn(inputs_dim = self.dim, simplenet_params = simplenet_params, fc_params = fc_params)
        self.SimpleDnn2 = SimpleDnn(inputs_dim = self.dim, simplenet_params = simplenet_params, fc_params = fc_params)
        self.SimpleDnn3 = SimpleDnn(inputs_dim = self.dim, simplenet_params = simplenet_params, fc_params = fc_params)
        self.SimpleDnn4 = SimpleDnn(inputs_dim = self.dim, simplenet_params = simplenet_params, fc_params = fc_params)


        layer_num = len(planes)

        assert layer_num >= 1

        if layer_num >= 1:
            self.fc1 = ReluBatchNormTdnnLayer(simplenet_params[len(simplenet_params) - 1] * 4, planes[0], **fc_params)
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
        # x = x.unsqueeze(2)
        # print("x shape = {}".format(x.shape))
        # split tensor
        mean_ = self.normalize(x[:, 0:self.dim, :])
        std_= self.normalize(x[:, self.dim:self.dim*2, :])
        rank1_ = x[:, self.dim*2:self.dim*3, :]
        rank2_ = x[:, self.dim*3:self.dim*4, :]

        #previous simple net
        mean_embedding = self.SimpleDnn1(mean_)
        std_embedding = self.SimpleDnn2(std_)
        rank1_embedding = self.SimpleDnn3(rank1_)
        rank2_embedding = self.SimpleDnn4(rank2_)


        #cat to fusion embedding
        # print("mean embedding shape {}, rank1_embedding shape {}, rank2_embedding shape {}".format(mean_embedding.shape, rank1_embedding.shape, rank2_embedding.shape))
        fusion_embedding = torch.cat((mean_embedding, rank1_embedding, rank2_embedding), dim=2)


        # print("fusion embedding shape = {}".format(fusion_embedding.shape))
        counts = fusion_embedding.shape[2]
        fusion_mean = fusion_embedding.sum(dim=2, keepdim=True) / counts
        fusion_max = torch.max(fusion_embedding, dim=2)[0].unsqueeze(2)
        fusion_min = torch.min(fusion_embedding, dim=2)[0].unsqueeze(2)
        # fusion_max = fusion_embedding.max(dim=2, keepdim=True)
        # fusion_min = fusion_embedding.min(dim=2, keepdim=True)

        # print("fusion_mean shape {}, fusion_max shape {}, fusion_min shape {}".format(fusion_mean.shape, fusion_max.shape, fusion_min.shape))
        # print("fusion_mean shape {}, fusion_max shape {}, fusion_min shape {}".format(fusion_mean.shape,len(fusion_max),""))

        # print(fusion_mean[0][0][0], fusion_max[0][0][0],fusion_min[0][0][0], std_embedding[0][0][0])
        x = torch.cat((fusion_mean, fusion_max, fusion_min, std_embedding), dim=1)
        
        # print("forward x shape = {}".format(x.shape))
        # x = self.fc1(x)
        if self.fc1 is not None:
            x = self.auto(self.fc1, x)
        # x = self.fc1(x)
        if self.fc2 is not None:
            x = self.auto(self.fc2, x)
        # x = self.fc2(x)

        if self.fc3 is not None:
            x = self.fc3(x)

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

        # split tensor
        mean_ = self.normalize(x[:, 0:self.dim, :])
        std_= self.normalize(x[:, self.dim:self.dim*2, :])
        rank1_ = x[:, self.dim*2:self.dim*3, :]
        rank2_ = x[:, self.dim*3:self.dim*4, :]

        #pre simple net
        mean_embedding = self.SimpleDnn1(mean_)
        std_embedding = self.SimpleDnn2(std_)
        rank1_embedding = self.SimpleDnn3(rank1_)
        rank2_embedding = self.SimpleDnn4(rank2_)


        #cat to fusion embedding
        # fusion_embedding = torch.cat((mean_embedding, rank1_embedding, rank2_embedding), axis=2, dim=1)
        fusion_embedding = torch.cat((mean_embedding, rank1_embedding, rank2_embedding), dim=2)

        counts = fusion_embedding.shape[2]
        fusion_mean = fusion_embedding.sum(dim=2, keepdim=True) / counts
        fusion_max = torch.max(fusion_embedding, dim=2)[0].unsqueeze(2)
        fusion_min = torch.min(fusion_embedding, dim=2)[0].unsqueeze(2)

        x = torch.cat((fusion_mean, fusion_max, fusion_min, std_embedding), dim=1)

        if self.fc1 is not None:
            x = self.auto(self.fc1, x)
        # x = self.fc1(x)
        if self.fc2 is not None:
            x = self.auto(self.fc2, x)
        # x = self.fc2(x)

        if self.fc3 is not None:
            x = self.fc3(x)

        xvector = x

        return xvector


# Test.
if __name__ == "__main__":
    print(DnnXvector(23, 1211))
