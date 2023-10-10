import numpy as np
from threadpoolctl import threadpool_limits
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from sklearn.svm import SVR, LinearSVR
import time




def forward_and_backward_proc(X, y, C, eps):
    # X (n, dim)
    u, sv_indexs = forward_proc(X, y, C, eps)
    grad_mat = backward_proc(X, y, u, C, eps, sv_indexs)
    return u, grad_mat



def forward_proc(X, y, C, eps):
    def et(u, v, t, eps):
        delta = t - u.T @ v
        ret = 0.0
        if delta >= eps:
            ret = u.T @ v - t + eps
        elif delta <= -eps:
            ret = u.T @ v - t - eps
        return ret

    n, dim = X.shape
    # print("C = {}".format(C))
    svr = LinearSVR(epsilon=eps, tol=1e-2, C=C, loss='squared_epsilon_insensitive', dual=True, fit_intercept=False, random_state=0, max_iter=1000000)
    svr.fit(X,y)
    u = svr.coef_.reshape(dim, 1)
    sv_travse = np.zeros(n)

    if X.shape[1] % 41 == 0:
        print("[R^2 score = {}]".format(svr.score(X,y)))

    for i in range(n):
        if abs(et(u, X[i].T, y[i], eps)) != 0.0:
            sv_travse[i] = 1.0

    return u, sv_travse.reshape(n, 1)

def backward_proc(X, y, u, C, eps, sv_travse, grad_output):
        # calculate the hinge loss of svr
        def et(u, v, t, eps):
            delta = t - u.T @ v
            ret = 0.0
            if delta >= eps:
                ret = u.T @ v - t + eps
            elif delta <= -eps:
                ret = u.T @ v - t - eps
            print(ret)
            return ret

        # calculate Hessine diagnal inverse 
        def HessineInv(X, C, sv_travse):
            dim = X.shape[1]
            diag_num = np.zeros((1, dim))
            for i in range(len(sv_travse)):
                if sv_travse[i] == 1.0:
                    diag_num += X[i] * X[i]

            diag_num = diag_num * C + np.ones((1, dim))
            diag_num = 1 / diag_num
            
            return diag_num
            

        def F_xy(X, y, u, eps, C, sv_index, I):
            dim = X.shape[1]
            v = X[sv_index].reshape(dim, 1)
            t = y[sv_index]
            M1 = et(u, v, t ,eps) * I
            M2 = v @ u.T
            grad_tensor = C * (M1 - M2)

            return grad_tensor

    
        n, dim = X.shape
        HessineInvVec = HessineInv(X, C, sv_travse)

        grad_tensor = np.zeros((n, dim))
        I = np.eye(dim)
        for i in range(n):
            if sv_travse[i] == 1.0:
                grad_tensor[i] = (grad_output * HessineInvVec) @ F_xy(X, y, u, eps, C, i, I)


        # return grad_tensor.T / dim
        return grad_tensor.T

def solve_batch_forward(X, y, C, eps, proc=forward_proc, n_jobs_forward = -1):
    # solve proc 
    batch_size, frame_dim, frame_num = X.shape

    ret_batch_u = np.zeros((batch_size, frame_dim, 1))
    ret_batch_travse = np.zeros((batch_size, frame_num, 1))

    if n_jobs_forward == -1:
        n_jobs_forward = mp.cpu_count()
    n_jobs_forward = min(batch_size, n_jobs_forward)

    # print("n jobs {}".format(n_jobs_forward))
    if n_jobs_forward == 1:
        # serial
        for i in range(batch_size):
            ret_batch_u[i], ret_batch_travse[i] = proc(X[i].T, y[i], C, eps)
        # print("ret_batch_u shape {}".format(ret_batch_u.shape))
        # print("solve_batch_forward u[0][1] {} u[0][2] {}".format(ret_batch_u[0][1], ret_batch_u[0][2]))


    else:
        # thread pool
        pool = ThreadPool(processes=n_jobs_forward)
        args = [(Xi.T, yi, C, eps) for Xi, yi in zip(X, y)]
        with threadpool_limits(limits=1):
            results = pool.starmap(proc, args)
        pool.close()
        ret_batch_u =np.array([r[0] for r in results])
        ret_batch_travse = np.array([r[1] for r in results])
    
    return ret_batch_u, ret_batch_travse




def solve_batch_backward(X, y, u, C, eps, sv_travses, grad_outputs, proc=backward_proc, n_jobs_backward = -1):
    # solve proc 
    batch_size, frame_dim, frame_num = X.shape

    ret_batch_grad = np.zeros((batch_size, frame_dim, frame_num))

    if n_jobs_backward == -1:
        n_jobs_backward = mp.cpu_count()
    n_jobs_backward = min(batch_size, n_jobs_backward)


    if n_jobs_backward == 1:
        # serial
        for i in range(batch_size):
            ret_batch_grad[i] = proc(X[i].T, y[i], u[i], C, eps, sv_travses[i], grad_outputs[i].T)


    else:
        # thread pool
        pool = ThreadPool(processes=n_jobs_backward)
        args = [(Xi.T, yi, ui, C, eps, sv_travse, grad_output.T) for Xi, yi, ui, sv_travse, grad_output in zip(X, y, u, sv_travses, grad_outputs)]
        with threadpool_limits(limits=1):
            results = pool.starmap(proc, args)
        pool.close()
        ret_batch_grad = np.array(results)
    
    return ret_batch_grad

    

# BATCH=3
# return normed w and grad_mat m
def solve_func_svr_batch(X, y, C, eps, jobs=10):
    u_batch, sv_travse_batch = solve_batch_forward(X, y, C, eps, n_jobs_forward=jobs)

    print(sv_travse_batch)
    grad_outputs = np.random.randn(BATCH, 5, 1)
    grad = solve_batch_backward(X, y, u_batch, C, eps, sv_travse_batch, grad_outputs, n_jobs_backward=jobs)
    return grad

