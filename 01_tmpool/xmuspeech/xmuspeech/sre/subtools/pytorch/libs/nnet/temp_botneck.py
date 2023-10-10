import torch
import torch.nn as nn



def conv1x3(in_planes, out_planes, Conv=nn.Conv1d, stride=1, groups=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes, out_planes, Conv=nn.Conv2d, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)



# def temporal_bottleneck_layer(inputs):
#     x = inputs

#     batch = x.shape[0]
#     dim = x.shape[1]
#     num = x.shape[2]

#     # local temporal layer
#     Conv1x3 = conv1x3(dim, dim).cuda()
#     bn1 = nn.BatchNorm1d(dim).cuda()
#     relu = nn.ReLU(inplace=True).cuda()
#     x = Conv1x3(x)
#     x = bn1(x)
#     x = relu(x)


#     # bottleneck layer
#     Conv3x1 = conv1x3(num, num).cuda()
#     bn2 = nn.BatchNorm1d(dim).cuda()
#     x = x.permute(0, 2, 1)
#     x = Conv3x1(x)
#     x = x.permute(0, 2, 1)
#     x = bn2(x)
#     x = x.permute(0, 2, 1)
#     x = relu(x)


#     # max pooling layer
#     pool3x1 = nn.MaxPool1d(kernel_size=3).cuda()
#     bn3 = nn.BatchNorm1d(int(dim/3)).cuda()
#     x = pool3x1(x)
#     x = x.permute(0, 2, 1)
#     x = bn3(x)
#     x = relu(x)

#     return x

def temporal_bottleneck_layer(inputs):
    x = inputs

    batch = x.shape[0]
    dim = x.shape[1]
    num = x.shape[2]

    conv = conv3x3(dim, dim)

    x = conv(x)
    return x


# test
# x = torch.ones(512, 768, 13)
# print(x)

# x = temporal_bottleneck_layer(x)
# print(x)