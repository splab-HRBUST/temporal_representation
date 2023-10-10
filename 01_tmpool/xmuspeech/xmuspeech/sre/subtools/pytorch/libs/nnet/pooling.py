# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29 2020-06-10)

from sklearn.svm import SVR as skSVR
from .svr_torch import SVR
import argparse

import gc

import numpy as np
import copy

import torch

import torch.nn.functional as F

from libs.support.utils import to_device

from .components import *

import torch
import torch.nn as nn


from torch.autograd import Function

from .svr_batch import solve_batch_forward, solve_batch_backward
from .temp_botneck import temporal_bottleneck_layer
from .point_wise import point_wise_map
from .norm import l2_norm
from.layer_norm import LayerNorm
from .slide_window import slide_window

# fix random seed
# torch.manual_seed(808)

def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()

def to_torch(x):
    # convert numpy array to torch tensor
    # return torch.from_numpy(x).type(dtype).to(device)
    return torch.from_numpy(x).type(torch.float32).cuda()
    # return torch.from_numpy(x).type(torch.float32)


class FuncSvrBatch(Function):
    @staticmethod
    def forward(ctx, X, y, C, eps):
        u_batch, sv_travse_batch = solve_batch_forward(to_numpy(X), to_numpy(y), C, eps, n_jobs_forward=10)
        
        ctx.save_for_backward(X, y, to_torch(u_batch), to_torch(sv_travse_batch))
        return to_torch(u_batch)

    #[512, 768, 1]
    @staticmethod
    def backward(ctx, grad_output):
        # X, y, C, eps, u_batch, sv_travse_batch, = ctx.saved_tensors
        C = 1e5
        eps = 1
        X, y, u_batch, sv_travse_batch, = ctx.saved_tensors
        grad = solve_batch_backward(to_numpy(X), to_numpy(y),  \
            to_numpy(u_batch), C, eps, to_numpy(sv_travse_batch), to_numpy(grad_output), n_jobs_backward=10)
        return to_torch(grad), None, None, None


# class FuncSvrBatch(Function):
#     @staticmethod
#     def forward(ctx, X, y, C, eps):
#         w, m = solve_func_svr_batch(to_numpy(X),to_numpy(y), C, eps)
#         ctx.save_for_backward(to_torch(m))
#         return to_torch(w)

#     #[512, 768, 1]
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad, = ctx.saved_tensors
#         return grad, None, None, None



class LocalRnn(nn.Module):
    
    def __init__(self, dim=768, num_layers=1, bi_direction=False, window_size=3, stride=1):
        super(LocalRnn, self).__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.bi_direction = bi_direction
        self.window_size = window_size
        self.stride = stride

        # rnn cell
        if torch.cuda.is_available():
            self.rnn = nn.RNN(dim, dim, num_layers, nonlinearity='relu').cuda()
        else:
            self.rnn = nn.RNN(dim, dim, num_layers, nonlinearity='relu')
    
    def pad_zeros(self, x, window_size=3, left_position=True):
        batch = x.shape[0]
        dim = x.shape[1]
        len = x.shape[2]

        if torch.cuda.is_available():
            zeros = torch.zeros(batch, dim, window_size - 1).cuda()
        else:
            zeros = torch.zeros(batch, dim, window_size - 1)

        if left_position:
            x = torch.cat((zeros, x), dim=2)
        else:
            x = torch.cat((x, zeros), dim=2)
        return x





    def forward(self, x):
        batch = x.shape[0]
        dim = x.shape[1]
        length = x.shape[2]

        x_backup = x.clone()

        # init hidden layer and cell layer
        if torch.cuda.is_available():
            local_encode = torch.zeros(batch, dim, length).cuda()
            h0 = torch.zeros(self.num_layers, batch, dim).cuda()
        else:
            local_encode = torch.zeros(batch, dim, length)
            h0 = torch.zeros(self.num_layers, batch, dim)

        # x_padded = self.pad_zeros(x, self.window_size, left_position=True)
        # transpose tensor
        # x_padded = torch.transpose(x_padded, 1, 2)
        # x_padded = torch.transpose(x_padded, 0, 1)
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 0, 1)
        local_encode = torch.transpose(local_encode, 1, 2)
        local_encode = torch.transpose(local_encode, 0, 1)


        for i in range(0, length, self.stride):
            if i - self.window_size + 1 < 0:
                start_ind = 0
                end_ind = i + 1
            else:
                start_ind = i - self.window_size + 1
                end_ind = i + 1
            _, local_encode[i] = self.rnn(x[start_ind: end_ind, :, :], h0)

        

        # transpose tensor
        local_encode = torch.transpose(local_encode, 0, 1)
        local_encode = torch.transpose(local_encode, 1, 2)

        layer_norm = torch.nn.LayerNorm(x_backup.size()[2:], eps=1e-12, elementwise_affine=True).cuda()

        return layer_norm(local_encode + x_backup)
        # return local_encode


class LocalRnnRankPooling(torch.nn.Module):
    """ A rank poolling layer"""
    def __init__(self, input_dim):
        super(LocalRnnRankPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = 768
        self.local_rnn = LocalRnn(dim=input_dim, num_layers=1, window_size=3, stride=1)



    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        
        X = inputs
        X = self.local_rnn(X)
        #X = point_wise_map(temporal_bottleneck_layer(X))
        # X = l2_norm(X)
        y = torch.ones(X.shape[0], X.shape[2])
        y = torch.cumsum(y, 1)

        w = FuncSvrBatch.apply(X, y, 1e3, 1e-1)
        # w = l2_norm(w)
        return w
    


    def get_output_dim(self):
        return self.output_dim
    
    def extra_repr(self):
        pass

    @classmethod
    def thop_count(self, m, x, y):
        pass



class BiLSTM(nn.Module):
    
    def __init__(self, dim=768, num_layers=1, full_state=False):
        super(BiLSTM, self).__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.full_state = full_state

        # lstm
        # self.lstm = nn.LSTM(dim, dim, num_layers, bidirectional=True, bias=False).cuda()
        self.lstm = nn.LSTM(dim, dim, num_layers, bidirectional=True, bias=False)
        self.output_dim = dim * 2




    def forward(self, x):
        batch = x.shape[0]
        dim = x.shape[1]

        # init hidden layer and cell layer
        h0 = torch.zeros(self.num_layers * 2, batch, dim).cuda()
        # h0 = torch.zeros(self.num_layers * 2, batch, dim)
        c0 = torch.zeros(self.num_layers * 2, batch, dim).cuda()
        # c0 = torch.zeros(self.num_layers * 2, batch, dim)

        # transpose tensor
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 0, 1)
        # lstm
        yn, (hn, _) = self.lstm(x, (h0, c0))

        if self.full_state:
            lstm_out = yn
        else:
            lstm_out = hn
        

        # transpose tensor
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        if not self.full_state:
            # first_direction = lstm_out[:, :, 0: 1]
            # second_direction = lstm_out[:, :, 1: 2]
            first_direction = lstm_out[:, :, -2: -1]
            second_direction = lstm_out[:, :, -1: ]
            ret = torch.cat((first_direction, second_direction), dim=1)
        else:
            ret = lstm_out
        
        # return first_direction + second_direction
        return ret


    
    def get_output_dim(self):
        return self.output_dim

class BiLstmRankPooling(torch.nn.Module):
    """ A rank poolling layer"""
    def __init__(self, input_dim):
        super(BiLstmRankPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim * 2
        self.bilstm = BiLSTM(dim=input_dim, num_layers=1, full_state=True)


    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        
        X = inputs
        #X = point_wise_map(temporal_bottleneck_layer(X))
        X = self.bilstm(X)
        X = torch.tanh(X)
        # X = l2_norm(X)
        y = torch.ones(X.shape[0], X.shape[2])
        y = torch.cumsum(y, 1)

        w = FuncSvrBatch.apply(X, y, 1e3, 1e-1)
        w = l2_norm(w)
        return w
    


    def get_output_dim(self):
        return self.output_dim
    
    def extra_repr(self):
        pass

    @classmethod
    def thop_count(self, m, x, y):
        pass

class BilstmMultiHeadAttentionPooling(torch.nn.Module):
    """Implement multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Safari, Pooyan, and Javier Hernando. 2019. “Self Multi-Head Attention for Speaker 
               Recognition.” ArXiv Preprint ArXiv:1906.09890.
    Note, in this paper, affine_layers is default to 1, and final_dim is 1 which means the weights are shared.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=1, **options):
        super(BilstmMultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.stddev = stddev
        self.stddev_attention = stddev_attention
        self.num_head = num_head
        print("head num = {}".format(self.num_head))

        if self.stddev :
            self.output_dim = 4 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if not options["split_input"]:
                raise ValueError("split_input==False is not valid for this MultiHeadAttentionPooling.")
            options.pop("split_input")

        # In this pooling, the special point is that inputs will be splited.
        self.attention = AttentionAlphaComponent(input_dim * 2, num_head=num_head, split_input=True, share=share, 
                                                 affine_layers=affine_layers, bias=False, **options)
        self.bilstm = BiLSTM(dim=input_dim, num_layers=1, full_state=True)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        inputs = self.bilstm(inputs)

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, splited-features, frames]
        # for another case.
        # inputs: [batch, head, splited-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, self.num_head, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, self.num_head, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim


class BiLstmAttentiveStatisticsPooling(torch.nn.Module):
    """ An attentive statistics pooling.
    Reference: Okabe, Koji, Takafumi Koshinaka, and Koichi Shinoda. 2018. "Attentive Statistics Pooling 
               for Deep Speaker Embedding." ArXiv Preprint ArXiv:1803.10963.
    """
    def __init__(self, input_dim, affine_layers=2, hidden_size=64, context=[0], stddev=True, stddev_attention=True, eps=1.0e-10):
        super(BiLstmAttentiveStatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev :
            self.output_dim = 4 * input_dim
        else :
            self.output_dim = input_dim

        self.eps = eps
        self.stddev_attention = stddev_attention

        self.attention = AttentionAlphaComponent(input_dim * 2, num_head=1, share=True, affine_layers=affine_layers, 
                                                 hidden_size=hidden_size, context=context)

        self.bilstm = BiLSTM(dim=input_dim, num_layers=1, full_state=True)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        inputs = self.bilstm(inputs)

        alpha = self.attention(inputs)

        # Weight avarage
        mean = torch.sum(alpha * inputs, dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                var = torch.sum(alpha * inputs**2, dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=self.eps))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim

class BiLstmStatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""
    def __init__(self, input_dim, stddev=True, unbiased=False, eps=1.0e-10):
        super(BiLstmStatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        self.eps = eps
        # Used for unbiased estimate of stddev
        self.unbiased = unbiased

        self.bilstm = BiLSTM(dim=input_dim, num_layers=1, full_state=True)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        inputs = self.bilstm(inputs)
        # Get the num of frames
        counts = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / counts

        if self.stddev :
            if self.unbiased and counts > 1:
                counts = counts - 1

            # The sqrt (as follows) is deprecated because it results in Nan problem.
            # std = torch.unsqueeze(torch.sqrt(torch.sum((inputs - mean)**2, dim=2) / counts), dim=2)
            # There is a eps to solve this problem.
            # Another method: Var is equal to std in "cat" way, actually. So, just use Var directly.

            var = torch.sum((inputs - mean)**2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim * 2
    
    def extra_repr(self):
        return '{input_dim}, {output_dim}, stddev={stddev}, unbiased={unbiased}, eps={eps}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        pass
        # To do
        # x = x[0]

        # kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        # bias_ops = 1 if m.bias is not None else 0

        # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        # total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        # m.total_ops += torch.DoubleTensor([int(total_ops)])

class LSTM(nn.Module):
    
    def __init__(self, dim=768, num_layers=1, full_state=False):
        super(LSTM, self).__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.full_state = full_state

        # lstm
        #self.lstm = nn.LSTM(dim, dim, num_layers).cuda()
        self.lstm = nn.LSTM(dim, dim, num_layers)
        self.output_dim = dim



    def forward(self, x):
        batch = x.shape[0]
        dim = x.shape[1]

        # init hidden layer and cell layer
        #h0 = torch.randn(1, batch, dim).cuda()
        h0 = torch.randn(1, batch, dim)
        # h0 = torch.zeros(1, batch, dim).cuda()
        # c0 = torch.randn(1, batch, dim).cuda()
        c0 = torch.randn(1, batch, dim)
        # c0 = torch.zeros(1, batch, dim).cuda()

        # transpose tensor
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 0, 1)
        # lstm
        yn, (hn, _) = self.lstm(x, (h0, c0))

        if self.full_state:
            lstm_out = yn
        else:
            lstm_out = hn

        # transpose tensor
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = torch.tanh(lstm_out)

        return lstm_out

    def get_output_dim(self):
        return self.output_dim

        
class LstmRankPooling(torch.nn.Module):
    """ A lstm-rank poolling layer"""
    def __init__(self, input_dim):
        super(LstmRankPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = 768
        self.lstm = LSTM(dim = input_dim, num_layers = 1, full_state = True)



    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        
        X = inputs
        X = self.lstm(X)
        #X = point_wise_map(temporal_bottleneck_layer(X))
        X = l2_norm(X)
        y = torch.ones(X.shape[0], X.shape[2])
        y = torch.cumsum(y, 1)

        w = FuncSvrBatch.apply(X, y, 1e3, 1e-1)
        w = l2_norm(w)
        return w
    


    def get_output_dim(self):
        return self.output_dim
    
    def extra_repr(self):
        pass

    @classmethod
    def thop_count(self, m, x, y):
        pass
        # To do
        # x = x[0]

        # kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        # bias_ops = 1 if m.bias is not None else 0

        # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        # total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        # m.total_ops += torch.DoubleTensor([int(total_ops)])




## Pooling ✿

class RankPooling(torch.nn.Module):
    """ A rank poolling layer"""
    def __init__(self, input_dim):
        super(RankPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = 768



    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        
        X = inputs
        # X = point_wise_map(X)
        # X = slide_window(X, type='TVM')
        # X = l2_norm(X)
        y = torch.ones(X.shape[0], X.shape[2])
        y = torch.cumsum(y, 1)

        w = FuncSvrBatch.apply(X, y, 1e-5, 0.0)
        # w = l2_norm(w)
        return w
    


    def get_output_dim(self):
        return self.output_dim
    
    def extra_repr(self):
        pass

    @classmethod
    def thop_count(self, m, x, y):
        pass
        # To do
        # x = x[0]

        # kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        # bias_ops = 1 if m.bias is not None else 0

        # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        # total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        # m.total_ops += torch.DoubleTensor([int(total_ops)])

class RankPoolingPlus(torch.nn.Module):
    """ A rank poolling layer"""
    def __init__(self, input_dim, regu_c):
        super(RankPoolingPlus, self).__init__()

        self.input_dim = input_dim * 2
        self.output_dim = self.input_dim
        self.regu_c = regu_c
    


    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        
        # X has negative values!
        X = inputs
        # X = slide_window(X, type='TVM')
        # X = l2_norm(X)
        X = point_wise_map(X)

        y = torch.ones(X.shape[0], X.shape[2])
        y = torch.cumsum(y, 1)

        w = FuncSvrBatch.apply(X, y, self.regu_c, 0.0)
        # print(w.shape)
        # w = l2_norm(w)


        return w
        


    def get_output_dim(self):
        return self.output_dim * 2
    
    def extra_repr(self):
        pass

    @classmethod
    def thop_count(self, m, x, y):
        pass
        # To do
        # x = x[0]

#==========================================================================
# torch svr
#==========================================================================
# class RankPooling(torch.nn.Module):
#     """ A rank poolling layer"""
#     def __init__(self, input_dim):
#         super(RankPooling, self).__init__()

#         self.input_dim = input_dim
#         self.output_dim = self.input_dim*2
    
#     def smooth(self, inputs):
#         inputs_ = inputs
#         for samples_index in range(0, inputs_.shape[0]):
#             frames_ = inputs_[samples_index]
#             # ---
#             frames_ = frames_.cumsum(1)
#             frames_ = frames_ / (torch.Tensor(range(0, frames_.shape[1])) + 1)
#             # united
#             norms = torch.norm(frames_, p=2, dim=0, keepdim=True)
#             frames_ = frames_ / norms

#             inputs_[samples_index] = frames_
        
#         return inputs_

#     def get_model_params(self, inputs):
#         inputs_ = inputs
#         temp_tensor_ = torch.Tensor(inputs_.shape[0],inputs_.shape[1]*2, 1)

#         for samples_index in range(0, inputs_.shape[0]):
#             frames_ = inputs_[samples_index]

#             # times_ = torch.tensor(range(0, frames_.shape[1]))
#             times_pre = torch.ones(1, frames_.shape[1])
#             times_ = times_pre.cumsum(1).t()
#             svr = SVR(input_dim=frames_.shape[0], regu_coef=100, margin=1, lr=0.0001, batchsize=10, epoch=200)
#             svr.train(frames_, times_)
#             w = svr.get_model_params()

#             w = w.reshape(frames_.shape[0],1)

#             if samples_index % 100 == 0:
#                 print(svr.get_loss_list())

#             temp_tensor_[samples_index] = torch.cat((w, w), dim=0)
            
#         return temp_tensor_


#     def forward(self, inputs):
#         """
#         @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
#         """
#         assert len(inputs.shape) == 3
#         assert inputs.shape[1] == self.input_dim
#         print(inputs.shape[1], inputs.shape[2])
#         w = self.get_model_params(self.smooth(inputs))
#         # w = self.get_model_params(inputs)
#         w = w.type(torch.FloatTensor)
#         # w = torch.randn(inputs.shape[0],self.input_dim*2,1)
#         # print("executed rank pooling forward once.")
#         return w


#     def get_output_dim(self):
#         return self.output_dim
    
#     def extra_repr(self):
#         pass

#     @classmethod
#     def thop_count(self, m, x, y):
#         pass
#         # To do
#         # x = x[0]

#         # kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
#         # bias_ops = 1 if m.bias is not None else 0

#         # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
#         # total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

#         # m.total_ops += torch.DoubleTensor([int(total_ops)])
class StatisticsAugumentPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""
    def __init__(self, input_dim, stddev=True, unbiased=False, eps=1.0e-10):
        super(StatisticsAugumentPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim
        
        # self.output_dim *= 2

        self.eps = eps
        # Used for unbiased estimate of stddev
        self.unbiased = unbiased

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        # Get the num of frames
        counts = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / counts
        zeros = torch.zeros([1, inputs.shape[1] * 2])
        if self.stddev :
            if self.unbiased and counts > 1:
                counts = counts - 1

            # The sqrt (as follows) is deprecated because it results in Nan problem.
            # std = torch.unsqueeze(torch.sqrt(torch.sum((inputs - mean)**2, dim=2) / counts), dim=2)
            # There is a eps to solve this problem.
            # Another method: Var is equal to std in "cat" way, actually. So, just use Var directly.

            var = torch.sum((inputs - mean)**2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            mean_std_aug = torch.cat((mean, std, zeros), dim=1)
            # mean_std = torch.cat((mean, std), dim=1)

            return mean_std_aug
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim
    
    def extra_repr(self):
        return '{input_dim}, {output_dim}, stddev={stddev}, unbiased={unbiased}, eps={eps}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        pass
        # To do
        # x = x[0]

        # kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        # bias_ops = 1 if m.bias is not None else 0

        # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        # total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        # m.total_ops += torch.DoubleTensor([int(total_ops)])

class StatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""
    def __init__(self, input_dim, stddev=True, unbiased=False, eps=1.0e-10):
        super(StatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        self.eps = eps
        # Used for unbiased estimate of stddev
        self.unbiased = unbiased

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        # Get the num of frames
        counts = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / counts

        if self.stddev :
            if self.unbiased and counts > 1:
                counts = counts - 1

            # The sqrt (as follows) is deprecated because it results in Nan problem.
            # std = torch.unsqueeze(torch.sqrt(torch.sum((inputs - mean)**2, dim=2) / counts), dim=2)
            # There is a eps to solve this problem.
            # Another method: Var is equal to std in "cat" way, actually. So, just use Var directly.

            var = torch.sum((inputs - mean)**2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim
    
    def extra_repr(self):
        return '{input_dim}, {output_dim}, stddev={stddev}, unbiased={unbiased}, eps={eps}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        pass
        # To do
        # x = x[0]

        # kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        # bias_ops = 1 if m.bias is not None else 0

        # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        # total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        # m.total_ops += torch.DoubleTensor([int(total_ops)])

class FreeStatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""
    def __init__(self, stddev=True, unbiased=False, eps=1.0e-10):
        super(FreeStatisticsPooling, self).__init__()

        self.stddev = stddev

        self.eps = eps
        # Used for unbiased estimate of stddev
        self.unbiased = unbiased

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """

        inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[len(inputs.shape)-1])

        # Get the num of frames
        counts = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / counts

        if self.stddev :
            if self.unbiased and counts > 1:
                counts = counts - 1

            # The sqrt (as follows) is deprecated because it results in Nan problem.
            # std = torch.unsqueeze(torch.sqrt(torch.sum((inputs - mean)**2, dim=2) / counts), dim=2)
            # There is a eps to solve this problem.
            # Another method: Var is equal to std in "cat" way, actually. So, just use Var directly.

            var = torch.sum((inputs - mean)**2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

class LDEPooling(torch.nn.Module):
    """A novel learnable dictionary encoding layer.
    Reference: Weicheng Cai, etc., "A NOVEL LEARNABLE DICTIONARY ENCODING LAYER FOR END-TO-END 
               LANGUAGE IDENTIFICATION", icassp, 2018
    """
    def __init__(self, input_dim, c_num=64, eps=1.0e-10):
        super(LDEPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim * c_num
        self.eps = eps

        self.mu = torch.nn.Parameter(torch.randn(input_dim, c_num))
        self.s = torch.nn.Parameter(torch.ones(c_num))

        self.softmax_for_w = torch.nn.Softmax(dim=3)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        r = inputs.transpose(1,2).unsqueeze(3) - self.mu
        # Make sure beta=self.s**2+self.eps > 0
        w = self.softmax_for_w(- (self.s**2 + self.eps) * torch.sum(r**2, dim=2, keepdim=True))
        e = torch.mean(w * r, dim=1)

        return e.reshape(-1, self.output_dim, 1)

    def get_output_dim(self):
        return self.output_dim


# Attention-based
class AttentionAlphaComponent(torch.nn.Module):
    """Compute the alpha with attention module.
            alpha = softmax(v'·f(w·x + b) + k) or softmax(v'·x + k)
    where f is relu here and bias could be lost.
    Support: 
            1. Single or Multi-head attention
            2. One affine or two affine
            3. Share weight (last affine = vector) or un-shared weight (last affine = matrix)
            4. Self-attention or time context attention (supported by context parameter of TdnnAffine)
            5. Different temperatures for different heads.
    """
    def __init__(self, input_dim, num_head=1, split_input=True, share=True, affine_layers=2, 
                 hidden_size=64, context=[0], bias=True, temperature=False, fixed=True):
        super(AttentionAlphaComponent, self).__init__()
        assert num_head >= 1
        # Multi-head case.
        if num_head > 1:
            if split_input:
                # Make sure fatures/planes with input_dim dims could be splited to num_head parts.
                assert input_dim % num_head == 0
            if temperature:
                if fixed:
                    t_list = []
                    for i in range(num_head):
                        t_list.append([[max(1, (i // 2) * 5)]])
                    # shape [1, num_head, 1, 1]
                    self.register_buffer('t', torch.tensor([t_list]))
                else:
                    # Different heads have different temperature.
                    # Use 1 + self.t**2 in forward to make sure temperature >= 1.
                    self.t = torch.nn.Parameter(torch.zeros(1, num_head, 1, 1))

        self.input_dim = input_dim
        self.num_head = num_head
        self.split_input = split_input
        self.share = share
        self.temperature = temperature
        self.fixed = fixed

        if share:
            # weight: [input_dim, 1] or [input_dim, hidden_size] -> [hidden_size, 1]
            final_dim = 1
        elif split_input:
            # weight: [input_dim, input_dim // num_head] or [input_dim, hidden_size] -> [hidden_size, input_dim // num_head]
            final_dim = input_dim // num_head
        else:
            # weight: [input_dim, input_dim] or [input_dim, hidden_size] -> [hidden_size, input_dim]
            final_dim = input_dim

        first_groups = 1
        last_groups = 1

        if affine_layers == 1:
            last_affine_input_dim = input_dim
            # (x, 1) for global case and (x, h) for split case.
            if num_head > 1 and split_input:
               last_groups = num_head
            self.relu_affine = False
        elif affine_layers == 2:
            last_affine_input_dim = hidden_size * num_head
            if num_head > 1:
                # (1, h) for global case and (h, h) for split case.
                last_groups = num_head
                if split_input:
                    first_groups = num_head
            # Add a relu-affine with affine_layers=2.
            self.relu_affine = True
            self.first_affine = TdnnAffine(input_dim, last_affine_input_dim, context=context, bias=bias, groups=first_groups)
            self.relu = torch.nn.ReLU(inplace=True)
        else:
            raise ValueError("Expected 1 or 2 affine layers, but got {}.",format(affine_layers))

        self.last_affine = TdnnAffine(last_affine_input_dim, final_dim * num_head, context=context, bias=bias, groups=last_groups)
        # Dim=2 means to apply softmax in different frames-index (batch is a 3-dim tensor in this case).
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        if self.temperature:
            batch_size = inputs.shape[0]
            chunk_size = inputs.shape[2]

        x = inputs
        if self.relu_affine:
            x = self.relu(self.first_affine(x))
        if self.num_head > 1 and self.temperature:
            if self.fixed:
                t = self.t
            else:
                t = 1 + self.t**2
            x = self.last_affine(x).reshape(batch_size, self.num_head, -1, chunk_size) / t
            return self.softmax(x.reshape(batch_size, -1, chunk_size))
        else:
            return self.softmax(self.last_affine(x))


class AttentiveStatisticsPooling(torch.nn.Module):
    """ An attentive statistics pooling.
    Reference: Okabe, Koji, Takafumi Koshinaka, and Koichi Shinoda. 2018. "Attentive Statistics Pooling 
               for Deep Speaker Embedding." ArXiv Preprint ArXiv:1803.10963.
    """
    def __init__(self, input_dim, affine_layers=2, hidden_size=64, context=[0], stddev=True, stddev_attention=True, eps=1.0e-10):
        super(AttentiveStatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        self.eps = eps
        self.stddev_attention = stddev_attention

        self.attention = AttentionAlphaComponent(input_dim, num_head=1, share=True, affine_layers=affine_layers, 
                                                 hidden_size=hidden_size, context=context)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        alpha = self.attention(inputs)

        # Weight avarage
        mean = torch.sum(alpha * inputs, dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                var = torch.sum(alpha * inputs**2, dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=self.eps))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim


class MultiHeadAttentionPooling(torch.nn.Module):
    """Implement multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Safari, Pooyan, and Javier Hernando. 2019. “Self Multi-Head Attention for Speaker 
               Recognition.” ArXiv Preprint ArXiv:1906.09890.
    Note, in this paper, affine_layers is default to 1, and final_dim is 1 which means the weights are shared.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=1, **options):
        super(MultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.stddev = stddev
        self.stddev_attention = stddev_attention
        self.num_head = num_head

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if not options["split_input"]:
                raise ValueError("split_input==False is not valid for this MultiHeadAttentionPooling.")
            options.pop("split_input")

        # In this pooling, the special point is that inputs will be splited.
        self.attention = AttentionAlphaComponent(input_dim, num_head=num_head, split_input=True, share=share, 
                                                 affine_layers=affine_layers, bias=False, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, splited-features, frames]
        # for another case.
        # inputs: [batch, head, splited-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, self.num_head, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, self.num_head, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim


class GlobalMultiHeadAttentionPooling(torch.nn.Module):
    """Implement global multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Zhiming Wang, Kaisheng Yao, Xiaolong Li, Shuo Fang. "MULTI-RESOLUTION MULTI-HEAD 
               ATTENTION IN DEEP SPEAKER EMBEDDING." ICASSP, 2020.
    It is not equivalent to multi-head attention pooling even when
               input_dim of global multi-head = 1/num_head * input_dim of multi-head.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=2, **options):
        super(GlobalMultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.num_head = num_head
        self.stddev = stddev
        self.stddev_attention = stddev_attention

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if options["split_input"]:
                raise ValueError("split_input==True is not valid for GlobalMultiHeadAttentionPooling.")
            options.pop("split_input")
        if "temperature" in options.keys():
            if options["temperature"]:
                raise ValueError("temperature==True is not valid for GlobalMultiHeadAttentionPooling.")
            options.pop("temperature")

        # In this pooling, the special point is that all (global) features of inputs will be used.
        self.attention = AttentionAlphaComponent(input_dim, num_head=num_head, split_input=False, share=share, 
                                                 temperature=False, affine_layers=affine_layers, bias=True, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, all-features, frames]
        # for another case.
        # inputs: [batch, 1, all-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, 1, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, 1, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim * self.num_head


class MultiResolutionMultiHeadAttentionPooling(torch.nn.Module):
    """Implement multi-resolution global multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Zhiming Wang, Kaisheng Yao, Xiaolong Li, Shuo Fang. "MULTI-RESOLUTION MULTI-HEAD 
               ATTENTION IN DEEP SPEAKER EMBEDDING." ICASSP, 2020.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=2, **options):
        super(MultiResolutionMultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.num_head = num_head
        self.stddev = stddev
        self.stddev_attention = stddev_attention

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if options["split_input"]:
                raise ValueError("split_input==True is not valid for MultiResolutionMultiHeadAttentionPooling.")
            options.pop("split_input")
        if "temperature" in options.keys():
            if not options["temperature"]:
                raise ValueError("temperature==False is not valid for MultiResolutionMultiHeadAttentionPooling.")
            options.pop("temperature")

        # In this pooling, the special point is that all (global) features of inputs will be used and
        # the temperature will be added.
        self.attention = AttentionAlphaComponent(input_dim, num_head=num_head, split_input=False, temperature=True, 
                                                 share=share, affine_layers=affine_layers, bias=True, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, all-features, frames]
        # for another case.
        # inputs: [batch, 1, all-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, 1, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, 1, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim * self.num_head
