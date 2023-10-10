# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-28)

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "subtools/pytorch")

import libs.support.utils as utils
from libs.nnet import *
from libs.nnet.transformer import TransformerEncoder


class ResNetXvector(TopVirtualNnet):
    """ A resnet x-vector framework """
    
    def init(self, inputs_dim, num_targets, aug_dropout=0., tail_dropout=0., training=True, extracted_embedding="near", 
             resnet_params={}, pooling="lstm", pooling_params={}, fc1=True, fc1_params={}, fc2_params={}, margin_loss=False, margin_loss_params={},
             use_step=False, step_params={}, transfer_from="softmax_loss"):

        ## Params.
        default_resnet_params = {
            "head_conv":True, "head_conv_params":{"kernel_size":3, "stride":1, "padding":1},
            "head_maxpool":False, "head_maxpool_params":{"kernel_size":3, "stride":1, "padding":1},
            "block":"BasicBlock",
            "layers":[3, 4, 6, 3],
            "planes":[32, 64, 128, 256], # a.k.a channels.
            "convXd":2,
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "full_pre_activation":True,
            "zero_init_residual":False
            }

        default_pooling_params = {
            "num_head":1,
            "hidden_size":64,
            "share":True,
            "affine_layers":1,
            "context":[0],
            "stddev":False,
            "temperature":False, 
            "fixed":True
        }
        
        default_fc_params = {
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":False, 
            "bn":True, 
            "bn_params":{"momentum":0.5, "affine":True, "track_running_stats":True}
            }

        default_margin_loss_params = {
            "method":"am", "m":0.2, "feature_normalize":True, 
            "s":30, "mhe_loss":False, "mhe_w":0.01
            }
        
        default_step_params = {
            "T":None,
            "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
            "s":False, "s_tuple":(30, 12), "s_list":None,
            "t":False, "t_tuple":(0.5, 1.2), 
            "p":False, "p_tuple":(0.5, 0.1)
            }

        resnet_params = utils.assign_params_dict(default_resnet_params, resnet_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        ## Var.
        self.extracted_embedding = extracted_embedding # only near here.
        self.use_step = use_step
        self.step_params = step_params
        self.convXd = resnet_params["convXd"]
        
        ## Nnet.
        self.aug_dropout = torch.nn.Dropout2d(p=aug_dropout) if aug_dropout > 0 else None

        # [batch, 1, feats-dim, frames] for 2d and  [batch, feats-dim, frames] for 1d.
        # Should keep the channel/plane is always in 1-dim of tensor (index-0 based).
        inplanes = 1 if self.convXd == 2 else inputs_dim
        self.resnet = ResNet(inplanes, **resnet_params)

        # It is just equal to Ceil function.
        resnet_output_dim = (inputs_dim + self.resnet.get_downsample_multiple() - 1) // self.resnet.get_downsample_multiple() \
                            * self.resnet.get_output_planes() if self.convXd == 2 else self.resnet.get_output_planes()
        self.resnet_output_dim = resnet_output_dim
        # Pooling
        stddev = pooling_params.pop("stddev")
        print("Pooling: {}".format(pooling))
        print("stddev: {}".format(stddev))
        if pooling == "lde":
            self.stats = LDEPooling(resnet_output_dim, c_num=pooling_params["num_head"])
        elif pooling == "attentive":
            self.stats = AttentiveStatisticsPooling(resnet_output_dim, hidden_size=pooling_params["hidden_size"], 
                                                    context=pooling_params["context"], stddev=stddev)
        elif pooling == "multi-head":
            self.stats = MultiHeadAttentionPooling(resnet_output_dim, stddev=stddev, **pooling_params)
        elif pooling == "multi-resolution":
            self.stats = MultiResolutionMultiHeadAttentionPooling(resnet_output_dim, **pooling_params)
        elif pooling == "rank":
            print("----------------rank pooling-------------------")
            self.stats = RankPooling(resnet_output_dim)
        elif pooling == "stats-aug":
            print("---------------stats aug pooling---------------")
            self.stats = StatisticsAugumentPooling(resnet_output_dim, stddev=stddev)
        elif pooling == "lstm":
            self.stats = LSTM(dim = resnet_output_dim, num_layers = 1, full_state = False)
        elif pooling == "bilstm":
            # self.stats = BiLSTM(dim = resnet_output_dim, num_layers = 1, full_state = False)
            self.stats = BiLSTM(dim = resnet_output_dim, num_layers = 2, full_state = False)
        elif pooling == "lstm-rank":
            self.stats = LstmRankPooling(resnet_output_dim)
        elif pooling == "biLstm-attentive-statistics-pooling":
            self.stats = BiLstmAttentiveStatisticsPooling(resnet_output_dim, hidden_size=pooling_params["hidden_size"], 
                                                    context=pooling_params["context"], stddev=stddev)
        elif pooling == "bilstm-multi-head-attentive-statistics-pooling":
            self.stats = BilstmMultiHeadAttentionPooling(resnet_output_dim, stddev=stddev, **pooling_params)
        elif pooling == "bilstm-rank-pooling":
            self.stats = BiLstmRankPooling(resnet_output_dim)
        elif pooling == "local-rnn-rank-pooling":
            self.stats = LocalRnnRankPooling(resnet_output_dim)
        elif pooling == "bilstm-statistics-pooling":
            self.stats = BiLstmStatisticsPooling(resnet_output_dim, stddev=stddev)
        elif pooling == "transformer-mean-std":
            self.transformer = TransformerEncoder(resnet_output_dim,  embed=False, attention_dim=768)
            self.stats = StatisticsPooling(resnet_output_dim, stddev=stddev)
        else:
            self.stats = StatisticsPooling(resnet_output_dim, stddev=stddev)
        

        self.fc1 = ReluBatchNormTdnnLayer(self.stats.get_output_dim(), resnet_params["planes"][3], **fc1_params) if fc1 else None

        if fc1:
            fc2_in_dim = resnet_params["planes"][3]
        else:
            fc2_in_dim = self.stats.get_output_dim()

        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, resnet_params["planes"][3], **fc2_params)

        self.tail_dropout = torch.nn.Dropout2d(p=tail_dropout) if tail_dropout > 0 else None

        ## Do not need when extracting embedding.
        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(resnet_params["planes"][3], num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(resnet_params["planes"][3], num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["resnet", "stats", "fc1", "fc2"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight":"loss.weight"} 

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = inputs
        # print("raw input x shape = {}".format(x.shape))
        x = self.auto(self.aug_dropout, x) # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, 1, frames-dim-index, frames-index]
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        # [samples-index, channel, frames-dim-index, frames-index] -> [samples-index, channel*frames-dim-index, frames-index]
        # print("pre reshape resnet output = {}".format(x.shape))
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) if self.convXd == 2 else x

        # print("resnet output = {}".format(x.shape))
        x = torch.transpose(x,1,2)
        # print("resnet output = {}".format(x.shape))
        x,_ = self.transformer(x)
        x = torch.transpose(x,1,2)
        x = self.stats(x)
        # print(x.shape)
        # print("stats output = {}".format(x.shape))
        x = self.auto(self.fc1, x)
        #print("fc1 output = {}".format(x.shape))
        x = self.fc2(x)
        #print("fc2 output = {}".format(x.shape))
        x = self.auto(self.tail_dropout, x)
        # print("tail_dropout = {}".format(x.shape))

        return x


    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
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
        # Tensor shape is not modified in libs.nnet.resnet.py for calling free, such as using this framework in cv.
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) if self.convXd == 2 else x

        # stats = self.stats(x)
        x = torch.transpose(x,1,2)
        x,_ = self.transformer(x)
        x = torch.transpose(x,1,2)
        x = self.stats(x)

        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
        elif self.extracted_embedding == "very_far":
            xvector = x
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.fc1, x)
            xvector = self.fc2.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(self.extracted_embedding))

        return xvector

    @for_extract_embedding_plus(maxChunk=10000, isMatrix=True)
    def extract_embedding_plus(self, inputs, pooling_type):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        # Tensor shape is not modified in libs.nnet.resnet.py for calling free, such as using this framework in cv.
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) if self.convXd == 2 else x



        if pooling_type == "rank_1":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1.0)
        elif pooling_type == "rank_1e-1":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-1)
        elif pooling_type == "rank_1e-2":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-2)
        elif pooling_type == "rank_1e-3":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-3)
        elif pooling_type == "rank_1e-4":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-4)
        elif pooling_type == "rank_1e-5":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-5)
        elif pooling_type == "rank_1e1":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e1)
        elif pooling_type == "rank_1e2":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e2)
        elif pooling_type == "rank_1e3":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e3)
        elif pooling_type == "rank_1e4":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e4)
        elif pooling_type == "rank_1e5":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e5)
        elif pooling_type == "inv_rank_1":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1.0)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e-1":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-1)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e-2":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-2)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e-3":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-3)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e-4":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-4)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e-5":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e-5)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e1":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e1)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e2":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e2)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e3":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e3)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e4":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e4)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "inv_rank_1e5":
            stats = RankPoolingPlus(self.resnet_output_dim, regu_c = 1e5)
            x = torch.flip(x, dims=[2])
        elif pooling_type == "statistic":
            stats = StatisticsPooling(self.resnet_output_dim, stddev=True)

        x = stats(x)

        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
        elif self.extracted_embedding == "very_far":
            xvector = x
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.fc1, x)
            xvector = self.fc2.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(self.extracted_embedding))

        return xvector


    def get_warmR_T(T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i


    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end)/(T_i-1) * (T_cur%T_i)


    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch*epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"], 
                                 self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur*epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(*self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]


# Test.
if __name__ == "__main__":
    # Let bach-size:128, fbank:40, frames:200.
    tensor = torch.randn(128, 40, 200)
    print("Test resnet2d ...")
    resnet2d = ResNetXvector(40, 1211, resnet_params={"convXd":2})
    print(resnet2d)
    print(resnet2d(tensor).shape)
    print("\n")
    print("Test resnet1d ...")
    resnet1d = ResNetXvector(40, 1211, resnet_params={"convXd":1})
    print(resnet1d)
    print(resnet1d(tensor).shape)

    print("Test done.")
