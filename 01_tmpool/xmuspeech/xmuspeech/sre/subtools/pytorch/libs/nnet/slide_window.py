import torch

def slide_window(inputs, type='MA', window_size=3, stride=1):
    x = inputs
    batch = x.shape[0]
    dim = x.shape[1]
    length = x.shape[2]
    
    # assert length >= window_size, 'frames length is less than window size.'

    if type == 'MA':
        length_smoothed = length - (window_size - 1)
        x_smoothed = torch.zeros(batch, dim, length_smoothed).cuda()
    elif type == 'TVM':
        length_smoothed = length
        x_smoothed = torch.zeros(batch, dim, length_smoothed).cuda()

    

    # window smooth with stride
    if type == 'MA':
        for i in range(0, length_smoothed, stride):
            start_ind = i
            end_ind = i + window_size
            x_smoothed[:, :, i: i+1] = torch.sum(x[:, :, start_ind: end_ind], dim=2, keepdim=True) / (end_ind - start_ind)
    # time varying mean
    elif type == 'TVM':
        x_smoothed = x.cumsum(dim=2) / torch.ones(batch, 1, length).cumsum(dim=2).cuda()

    return x_smoothed


