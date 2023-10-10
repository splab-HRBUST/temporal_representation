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
    return u, grad_mat



def forward_proc(X, y, C, eps):
    dim = X.shape[1]
    svr = skSVR(kernel='linear', C = C, epsilon = eps, cache_size = 256, tol = 1e-3, max_iter = -1, verbose = False)
    svr.fit(X,y)
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
            # return 1


        # calculate Sherman-Morrison inverse for fyy
        # def ShMoInv(X, C, sv_indexs):
        #     dim = X.shape[1]
        #     n = len(sv_indexs)
        #     H = np.eye(dim)
        #     for i in range(n):
        #         num = (H @ X[sv_indexs[i]].reshape(dim,1) @ (C * X[sv_indexs[i]].reshape(1,dim)) @ H)
        #         den = (1 + (C * X[sv_indexs[i]].reshape(1,dim)) @ (H @ X[sv_indexs[i]].reshape(dim,1)))
        #         if den == 0.0:
        #             den += 1e-9
        #         H = H - num / den
        #     return H

        def ShMoInv(X, C, sv_indexs):
            dim = X.shape[1]
            diag_num = np.ones((1, dim))
            for i in range(len(sv_indexs)):
                diag_num += X[sv_indexs[i]] * X[sv_indexs[i]]
            diag_num = np.ones((1, dim)) / diag_num
            
            # return np.diag(np.squeeze(diag_num, axis=0))
            return diag_num
            


        def F_xy(X, y, u, eps, C, sv_index, I):
            dim = X.shape[1]
            v = X[sv_index].reshape(dim, 1)
            t = y[sv_index]
            
            D = I
            
            M1 = et(u, v, t ,eps) * D 
            # M2 = u.T @ D @ v 
            M2 = v @ u.T
            grad_tensor_t = C * (M1 - M2)
            # grad_tensor_t = C * M1
            
            return grad_tensor_t

        # def F_xy(X, y, u, eps, C, sv_index):
        #     dim = X.shape[1]
        #     return np.eye(dim)

            
        n = X.shape[0]
        dim = X.shape[1]
        ShMoInVec = ShMoInv(X, C, sv_indexs)
        # print(ShMoI.shape)
        # ShMoI = np.eye(dim)
        # DiagShMoI = np.diag(ShMoI)
        travse = np.zeros(n)
        for i in range(len(sv_indexs)):
            travse[sv_indexs[i]] = 1.0

        # grad_tensor = np.zeros((n, dim, dim))
        grad_tensor = np.zeros((n, 1, dim))
        I = np.eye(dim)
        for i in range(n):
            if travse[i] == 1.0:
                # grad_tensor[i] = ShMoI @ F_xy(X, y, u, eps, C, i, I)
                grad_tensor[i] = ShMoInVec @ F_xy(X, y, u, eps, C, i, I)
                # grad_tensor[i] = ShMoInVec @ ShMoInVec

        return np.mean(grad_tensor, axis=1).T




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
        # print("serial")
        # serial
        for i in range(batch_size):
            ret_batch_u[i], ret_batch_grad[i] = proc(X[i].T, y[i], C, eps)


    else:
        # print("thread pool")
        # thread pool
        pool = ThreadPool(processes=n_jobs_forward)
        args = [(Xi.T, yi, C, eps) for Xi, yi in zip(X, y)]
        with threadpool_limits(limits=1):
            results = pool.starmap(proc, args)
        pool.close()
        # ret_batch_u = np.array(results[0])
        ret_batch_u =np.array([r[0] for r in results])
        # ret_batch_grad = np.array(results[1])
        ret_batch_grad = np.array([r[1] for r in results])
    
    # print("elaspe time = {}".format(time.time() - start_time))
    return ret_batch_u, ret_batch_grad
    


X = np.random.randn(512, 768, 13)
y = np.random.randn(512, 13)

start_time = time.time()
w, m = solve_batch_forward_and_backward(X, y, 1e3, 1e-3, forward_and_backward_proc, 10)
print("elaspe time = {}".format(time.time() - start_time))
print(m)
print(w.shape, m.shape)
