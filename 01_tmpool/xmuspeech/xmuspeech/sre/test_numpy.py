import numpy as np

fd = open("data/labels.csv",'rb')
labels = np.loadtxt(fd,dtype = np.int,delimiter=',')
print(labels.shape)

fd_ = open("data/frames.csv",'rb')
frames = np.loadtxt(fd_,dtype = np.float,delimiter=',')
print(frames.shape)