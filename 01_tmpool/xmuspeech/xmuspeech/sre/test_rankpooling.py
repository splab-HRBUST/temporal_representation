# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV

import numpy as np

import torch
import torch.nn.functional as F
import copy
import time

# from libs.support.utils import to_device

# from .components import *

## Pooling ✿

## Pooling ✿
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
        # print("counts = {}".format(counts))

        mean = inputs.sum(dim=2, keepdim=True) / counts
        # print("mean shape = {}".format(mean.shape))
        # print(inputs.numpy().shape)

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



class RankPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""
    def __init__(self, input_dim):
        super(RankPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = self.input_dim
    
    def smooth(self, inputs):
        inputs_ = inputs
        for samples_index in range(0, inputs_.shape[0]):
            frames_ = inputs_[samples_index]
            frames_ = frames_.cumsum(1)
            frames_ = frames_ / (torch.Tensor(range(0, frames_.shape[1])) + 1)
            #united
            lengths = torch.norm(frames_, p=2, dim=0, keepdim=True)
            frames_ = frames_ / lengths

            inputs_[samples_index] = frames_

        return inputs_

    def get_model_params(self, inputs):
        inputs_ = inputs
        temp_tensor_ = torch.Tensor(inputs_.shape[0],inputs_.shape[1]*2, 1)

        for samples_index in range(0, inputs_.shape[0]):
            frames_ = inputs_[samples_index]

            # times_ = torch.tensor(range(0, frames_.shape[1]))
            times_pre = torch.ones(1, frames_.shape[1])
            times_ = times_pre.cumsum(1).t()
            
            m4 = torch.matmul(frames_, times_)
            m1 = torch.matmul(frames_, frames_.t())
            m2 = torch.matmul(m1, torch.eye(frames_.shape[0]))
            m3 = m2.inverse()
            #m4 = torch.matmul(frames_.t(), times_)
            w = torch.matmul(m3, m4)
            temp_tensor_[samples_index] = torch.cat((w, w), dim=0)
            
        return temp_tensor_
    


    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        # Get the num of frames
        counts = inputs.shape[2]

        # mean = inputs.sum(dim=2, keepdim=True) / counts

        w = self.get_model_params(self.smooth(inputs))
        # print(w, w.shape)

        return w

        # if self.stddev :
        #     if self.unbiased and counts > 1:
        #         counts = counts - 1

        #     # The sqrt (as follows) is deprecated because it results in Nan problem.
        #     # std = torch.unsqueeze(torch.sqrt(torch.sum((inputs - mean)**2, dim=2) / counts), dim=2)
        #     # There is a eps to solve this problem.
        #     # Another method: Var is equal to std in "cat" way, actually. So, just use Var directly.

        #     var = torch.sum((inputs - mean)**2, dim=2, keepdim=True) / counts
        #     std = torch.sqrt(var.clamp(min=self.eps))
        #     return torch.cat((mean, std), dim=1)
        # else:
        #     return mean

    def get_output_dim(self):
        return self.output_dim
    
    def extra_repr(self):
        pass
        # return '{input_dim}, {output_dim}, stddev={stddev}, unbiased={unbiased}, eps={eps}'.format(**self.__dict__)

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

def smooth(inputs):
    inputs_ = inputs.numpy()
    for samples_index in range(0, inputs_.shape[0]):
        frames_ = np.cumsum(inputs_[samples_index], axis=1)
        # print(frames_[0][9],frames_[0][10])
        # print(inputs_[samples_index][0][9], inputs_[samples_index][0][10])
        # print("")
        for frames_index in range(0, frames_.shape[1]):
            frames_[:, frames_index] = frames_[:, frames_index] / (frames_index + 1)
        inputs_[samples_index] = frames_

    return torch.from_numpy(inputs_)

def smooth_simpler(inputs):
        inputs_ = inputs
        for samples_index in range(0, inputs_.shape[0]):
            frames_ = inputs_[samples_index]
            frames_ = frames_.cumsum(1)
            frames_ = frames_ / (torch.Tensor(range(0, frames_.shape[1])) + 1)
            #united
            lengths = torch.norm(frames_, p=2, dim=0, keepdim=True)
            frames_ = frames_ / lengths

            inputs_[samples_index] = frames_

        return inputs_

def get_model_params(inputs):
        inputs_ = inputs

        for samples_index in range(0, inputs_.shape[0]):
            frames_ = inputs_[samples_index]

            # times_ = torch.tensor(range(0, frames_.shape[1]))
            times_pre = torch.ones(1, frames_.shape[1])
            times_ = times_pre.cumsum(1).t()

            m1 = torch.matmul(frames_.t(), frames_)
            m2 = torch.matmul(m1, torch.eye(frames_.shape[1]))
            m3 = m2.inverse()
            m4 = torch.matmul(frames_.t(), times_)
            w = torch.matmul(m3, m4)

            
        
        return torch.cat((w, w), dim=1)

def test(frames):
    frames_ = frames
    frames_ = np.cumsum(frames_, axis=1)
    frames_ = frames_ / (np.array(range(0, frames_.shape[1])) + 1)
    #united
    lengths = np.linalg.norm(frames_, axis=0, keepdims=True)
    frames_ = frames_ / lengths
    print(frames_)
    

# Test.
if __name__ == "__main__":
    # Let bach-size:128, fbank:40, frames:200.
    tensor = torch.randn(128, 100, 200)

    
    
    print("Test RankPooling ...")
    start = time.perf_counter()
    rank_pooling = RankPooling(100)
    print(rank_pooling(tensor).shape)
    end = time.perf_counter()
    t1=end-start
    print("Runtime is ：",t1)

    print("\n")
    print("Test StatisticsPooling")
    start = time.perf_counter()
    stat_pooling = StatisticsPooling(100,stddev=True)
    print(stat_pooling(tensor).shape)
    end = time.perf_counter()
    t2=end-start
    print("Runtime is ：",t2)

    print("Time durition rate: {}".format(t1/t2))
    
    #----------------------------------------------------------------------------------
    # stat_pooling = StatisticsPooling(40,stddev=False)

    # tensor_copy = copy.deepcopy(tensor)

    # tensor_copy_2 = copy.deepcopy(tensor)

    # smoothed_tensor_simpler = smooth_simpler(tensor_copy)

    # smoothed_tensor = smooth(tensor_copy_2)

    # print(smoothed_tensor.shape)
    # print(smoothed_tensor_simpler[0][0][0],smoothed_tensor_simpler[0][0][1], smoothed_tensor_simpler[0][0][2], smoothed_tensor_simpler[0][0][3])
    # print(smoothed_tensor[0][0][0],smoothed_tensor[0][0][1], smoothed_tensor[0][0][2], smoothed_tensor[0][0][3])
    # print(tensor[0][0][0], tensor[0][0][1], tensor[0][0][2], tensor[0][0][3])


    # print(smoothed_tensor)
    # model_params = get_model_params(smoothed_tensor_simpler)
    # print(model_params.shape)
    
    
    # test smooth matrix operation
    # arr = np.array([[1, 0, 0],
    #                 [0, 2, 0],
    #                 [0, 0, 3]])
    # test(arr)

    # print(stat_pooling)
    # print(stat_pooling(tensor).shape)
    print("\n")

    print("Test done.")