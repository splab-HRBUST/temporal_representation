from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV

import numpy as np

import torch
import torch.nn.functional as F
import copy
import time

# from libs.support.utils import to_device

# from .components import *

## Pooling ✿

class RankPooling(torch.nn.Module):
    """ A rank poolling layer"""
    def __init__(self, input_dim):
        super(RankPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = self.input_dim
    
    def smooth(self, inputs):
        inputs_ = inputs.numpy()
        for samples_index in range(0, inputs_.shape[0]):
            frames_ = np.cumsum(inputs_[samples_index], axis=1)
            frames_ = frames_ / (np.array(range(0, frames_.shape[1])) + 1)
            #united
            lengths = np.linalg.norm(frames_, axis=0, keepdims=True)
            frames_ = frames_ / lengths

            inputs_[samples_index] = frames_

        return torch.from_numpy(inputs_)




    def get_model_params(self, inputs):
        inputs_ = inputs.numpy()

        inputs_temp_ = np.empty(shape = (inputs_.shape[0],inputs_.shape[1],1))
        for samples_index in range(0, inputs_.shape[0]):
            frames_ = inputs_[samples_index]

            times_ = np.array(range(0, frames_.shape[1]))

            svr = SVR(kernel ='linear')

            model = svr.fit(frames_.T, times_.ravel())
            w = model.coef_[0]
            # print(model.dual_coef_.shape)
            w = w.reshape(frames_.shape[0],1)
            inputs_temp_[samples_index] = w
        
        return torch.from_numpy(inputs_temp_)


    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        w = self.get_model_params(self.smooth(inputs))

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

    

# Test.
if __name__ == "__main__":
    # Let bach-size:128, fbank:40, frames:200.
    tensor = torch.randn(128, 100, 200)
    # tensor_ = torch.randn(128, 100, 200)
    print("Test RankPooling ...")
    start = time.perf_counter()
    rank_pooling = RankPooling(100)
    print(rank_pooling(tensor).shape)
    end = time.perf_counter()
    t1=end-start
    print("Runtime is ：",t1)

    # print("\n")
    # print("Test StatisticsPooling")
    # start = time.perf_counter()
    # stat_pooling = StatisticsPooling(100,stddev=False)
    # print(stat_pooling(tensor).shape)
    # end = time.perf_counter()
    # t2=end-start
    # print("Runtime is ：",t2)

    # print("Time durition rate: {}".format(t1/t2))
