import numpy as np
from numpy import random
from scipy.sparse import dia
from threadpoolctl import threadpool_limits
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from sklearn.svm import SVR as skSVR
import time

from torch.autograd import grad



def forward_and_backward_proc(X, y, C, eps):
    u, sv_indexs = forward_proc(X, y, C, eps)
    grad_mat = backward_proc(X, y, u, C, eps, sv_indexs)
    return u , grad_mat



def forward_proc(X, y, C, eps):
    dim = X.shape[1]
    svr = skSVR(kernel='linear', C = C, epsilon = eps, cache_size = 256, tol = 1e-3, max_iter = -1, verbose = False)
    svr.fit(X,y)

    #normed_coef = svr.coef_[0].reshape(dim, 1)
    #normed_coef = normed_coef / np.linalg.norm(normed_coef)
    return svr.coef_[0].reshape(dim, 1), svr.support_

def backward_proc(X, y, u, C, eps, sv_indexs):
        # calculate the hinge loss of svr
        def et(u, v, t, eps):
            delta = t - u.T @ v
            ret = 0.0
            if delta >= eps:
                ret = u.T @ v - t + eps
            elif delta <= -eps:
                ret = u.T @ v - t - eps
            return ret

        # calculate Hessine diagnal inverse 
        def HessineInv(X, C, sv_indexs):
            dim = X.shape[1]
            diag_num = np.zeros((1, dim))
            for i in range(len(sv_indexs)):
                diag_num += X[sv_indexs[i]] * X[sv_indexs[i]]
            diag_num = diag_num * C + np.zeros((1, dim))
            diag_num = 1 / diag_num
            
            return diag_num
            


        def F_xy(X, y, u, eps, C, sv_index, I):
            dim = X.shape[1]
            v = X[sv_index].reshape(dim, 1)
            t = y[sv_index]
            M1 = et(u, v, t ,eps) * I
            M2 = v @ u.T
            grad_tensor_t = C * (M1 - M2)

            return grad_tensor_t

            
        n = X.shape[0]
        dim = X.shape[1]
        HessineInvVec = HessineInv(X, C, sv_indexs)

        travse = np.zeros(n)
        for i in range(len(sv_indexs)):
            travse[sv_indexs[i]] = 1.0

        grad_tensor = np.zeros((n, dim))
        I = np.eye(dim)
        for i in range(n):
            if travse[i] == 1.0:
                grad_tensor[i] = HessineInvVec @ F_xy(X, y, u, eps, C, i, I)


        return grad_tensor.T




def solve_batch_forward_and_backward(X, y, C, eps, proc, n_jobs_forward = -1):
    # solve proc 
    batch_size = X.shape[0]
    frame_dim = X.shape[1]
    frame_num = X.shape[2]
    ret_batch_u = np.zeros((batch_size, frame_dim, 1))
    ret_batch_grad = np.zeros((batch_size, frame_dim, frame_num))

    if n_jobs_forward == -1:
        n_jobs_forward = mp.cpu_count()
    n_jobs_forward = min(batch_size, n_jobs_forward)


    if n_jobs_forward == 1:
        # serial
        for i in range(batch_size):
            ret_batch_u[i], ret_batch_grad[i] = proc(X[i].T, y[i], C, eps)


    else:
        # thread pool
        pool = ThreadPool(processes=n_jobs_forward)
        args = [(Xi.T, yi, C, eps) for Xi, yi in zip(X, y)]
        with threadpool_limits(limits=1):
            results = pool.starmap(proc, args)
        pool.close()
        ret_batch_u =np.array([r[0] for r in results])o
        ret_batch_grad = np.array([r[1] for r in results])
    
    return ret_batch_u, ret_batch_grad
    


# return normed w and grad_mat m
def solve_func_svr_batch(X, y, C, eps, jobs=10):
    return solve_batch_forward_and_backward(X, y, C, eps, forward_and_backward_proc, jobs)



# X = np.random.randn(512, 512, 13)
# y = np.random.randn(512, 13)

# start_time = time.time()
# w, m = solve_func_svr_batch(X, y, 1e3, 1e-3)
# print(m)
# print(w.shape, m.shape)
# print("elaspe time = {}".format(time.time() - start_time))
