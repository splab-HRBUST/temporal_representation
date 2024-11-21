import torch as t
import torch.nn as nn

from src.layers.pooling import (
    MeanStdStatPool1D,
    MeanStatPool1D,
    AttentiveStatPool1D,
    QuantilePool1D,
    IndexPool1D,
    NoPooling,
    MaxPool1D,
)

class cnn_mapping(nn.Module):
    def __init__(self,feature_num,dim_num):
        super().__init__()
        self.in_channels=feature_num
        self.dim_num =dim_num
        self.cnn_list =nn.ModuleList([
            nn.Conv1d(
                        in_channels=self.in_channels,
                        out_channels=self.in_channels,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    ) for i in range(self.dim_num)
        ])
        # self.pooling = MeanStatPool1D(dim_to_reduce=1)
    def forward(self, input: t.Tensor):
        #print(self.device)
        assert len(input.shape)==3
        out =t.zeros((input.shape[0],input.shape[1],input.shape[2]),dtype=t.float)
        out=out.to(input.device)
        for index, cnn in enumerate(self.cnn_list):
            out+=cnn(input)
        out = out / self.dim_num

        out=t.transpose(out,2,1)

        return out


class fc_mapping_relu(nn.Module):
    def __init__(self,feature_num,dim_num):
        super().__init__()
        self.in_feature=feature_num
        self.dim_num =dim_num
        self.cnn_list =nn.ModuleList([
            nn.Sequential(
                    nn.Linear(
                        in_features=self.in_feature,
                        out_features=self.in_feature,
                        bias=True
                    ),
                    nn.ReLU(),
                )for i in range(self.dim_num-1)
        ])
        self.cnn_list.append(
            nn.Sequential(
                nn.Linear(
                        in_features=self.in_feature,
                        out_features=self.in_feature,
                        bias=True
                )
            )
        )
        #self.pooling = MeanStatPool1D(dim_to_reduce=1)
    def forward(self, input: t.Tensor):
        # print(self.device)
        assert len(input.shape)==3
        input=t.transpose(input, 2, 1)
        for index, cnn in enumerate(self.cnn_list):
             out=cnn(input)

        return out




class fc_mapping(nn.Module):
    def __init__(self,feature_num,dim_num):
        super().__init__()
        self.in_feature=feature_num
        self.dim_num =dim_num
        self.cnn_list =nn.ModuleList([
            nn.Linear(
                        in_features=self.in_feature,
                        out_features=self.in_feature,
                        bias=True
                    )for i in range(self.dim_num)
        ])
        #self.pooling = MeanStatPool1D(dim_to_reduce=1)
    def forward(self, input: t.Tensor):
        # print(self.device)
        assert len(input.shape)==3
        input=t.transpose(input, 2, 1)
        out =t.zeros((input.shape[0],input.shape[1],input.shape[2]),dtype=t.float)
        out=out.to(input.device)
        for index, cnn in enumerate(self.cnn_list):
             out+=cnn(input)
        out = out / self.dim_num

        return out

class fc_relu_fc_mapping(nn.Module):
    def __init__(self,feature_num,bn_num):
        super().__init__()
        self.linear =nn.Linear(
                        in_features=feature_num,
                        out_features=bn_num,
                        bias=False
                    )
        self.relu=nn.ReLU()
        self.linear1=nn.Linear(
                        in_features=bn_num,
                        out_features=feature_num,
                        bias=False
                    )
        #self.pooling = MeanStatPool1D(dim_to_reduce=1)
    def forward(self, input: t.Tensor):
        # print(self.device)
        return self.linear1(self.relu(self.linear(input)))

class fc_tanh_fc_mapping(nn.Module):
    def __init__(self,feature_num,bn_num):
        super().__init__()
        self.linear =nn.Linear(
                        in_features=feature_num,
                        out_features=bn_num,
                        bias=False
                    )
        self.relu=nn.Tanh()
        self.linear1=nn.Linear(
                        in_features=bn_num,
                        out_features=feature_num,
                        bias=False
                    )
        #self.pooling = MeanStatPool1D(dim_to_reduce=1)
    def forward(self, input: t.Tensor):
        # print(self.device)
        return self.linear1(self.relu(self.linear(input)))