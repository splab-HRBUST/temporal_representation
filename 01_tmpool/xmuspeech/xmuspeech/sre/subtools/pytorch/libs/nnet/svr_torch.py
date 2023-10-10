import matplotlib as plt
import torch
import copy

from libs.support.utils import to_device

# SVR
def kernel(x1, x2):
    return x1.t() @ x2

class SupportVector:
    def __init__(self, i, x, y, eps):
        self.i = i
        self.x = x
        # self.g = torch.zeros(2)
        self.g = torch.cuda.FloatTensor(2).fill_(0)
        # self.alpha = torch.zeros(2)
        self.alpha = torch.cuda.FloatTensor(2).fill_(0)

        self.g[0] = eps + y
        self.g[1] = eps - y
        self.k = kernel(x, x)

class SVR:
    def __init__(self, kernel = "linear", eps = 0.1, C=1.0, tol = 1e-3, max_iter = -1, cache_size = 256, verbose = False):
        self.eps = eps
        self.C = C
        self.tol = tol 
        self.type = kernel
        self.max_iter = max_iter
        self.verbose = verbose
        self.cache_size = cache_size

        self.TAU = 1e-12
        self.vectors = []
        self.K = None
        self.K_rows_memo = None

        self.svmin = None
        self.svmax = None
        self.gmin = float('inf')
        self.gmax = -float('inf')
        self.gmin_index = -1
        self.gmax_index = -1
        self.b = 0.0
        self.X = None
        # self.W = None


    def fit(self, X, y):
        n = X.shape[0]
        dim = X.shape[1]
        self.X = X
        # self.K = torch.zeros(n, n)
        self.K = torch.cuda.FloatTensor(n, n).fill_(0)
        # self.K_rows_memo = torch.zeros(n)
        self.K_rows_memo = torch.cuda.FloatTensor(n).fill_(0)
        # self.W = torch.zeros(dim)
        self.W = torch.cuda.FloatTensor(dim).fill_(0)

        for i in range(n):
            self.vectors.append(SupportVector(i, X[i], y[i], self.eps))
        
        self.minmax()

        phase = min(n, 1000)
        count = 1
        while self.smo(self.tol):
            if count % phase == 0 and self.verbose:
                print("{} SMO iteration".format(count))
            count += 1
            if self.max_iter > 0 and count > self.max_iter:
                break
        
        nsv = 0
        bsv = 0

        for i in range(n):
            v = self.vectors[i]
            if v.alpha[0] == v.alpha[1]:
                self.vectors[i] = None
            else:
                nsv += 1
                if v.alpha[0] == self.C or v.alpha[1] == self.C:
                    bsv += 1
        
        alphas = torch.zeros(nsv)
        sv_indexs = []
        Xs = []

        i = 0
        # for v in self.vectors:
        for j in range(len(self.vectors)):
            v = self.vectors[j]
            if v != None:
                sv_indexs.append(v.i)
                alphas[i] = v.alpha[1] - v.alpha[0]
                self.W = self.W + self.X[j].t() * alphas[i]
                i += 1
        
        self.sv_indexs = sv_indexs
        self.alphas = alphas
        self.coef_ = torch.unsqueeze(self.W, 0)

    def predict(self, X):
        n = X.shape[0]
        pred_y = torch.zeros(n)
        for i in range(n):
            pred_y[i] = self.W @ X[i].t() + self.b
        return pred_y


    def score(self, X, y):
        return 1 - ((y - self.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    
    def minmax(self):
        self.gmin = float('inf')
        self.gmax = -float('inf')

        for i in range(len(self.vectors)):
            v = self.vectors[i]
            g = -v.g[0]
            a = v.alpha[0]

            if (g < self.gmin and a > 0.0):
                self.svmin = v
                self.gmin = g
                self.gmin_index = 0
            
            if (g > self.gmax and a < self.C):
                self.svmax = v
                self.gmax = g
                self.gmax_index = 0
            
            g = v.g[1]
            a = v.alpha[1]
            if (g < self.gmin and a < self.C):
                self.svmin = v
                self.gmin = g
                self.gmin_index = 1
            
            if (g > self.gmax and a > 0.0):
                self.svmax = v
                self.gmax = g
                self.gmax_index = 1

    def gram(self, v):
        # print(self.K_rows_memo[v.i])
        if self.K_rows_memo[v.i] == 0:
            n = self.K.shape[1]
            for i in range(n):
                self.K[v.i][i] = kernel(v.x, self.X[i].t())
            self.K_rows_memo[v.i] = 1
        return self.K[v.i]
    
    def smo(self, tol):


        v1 = self.svmax
        i = self.gmax_index
        old_alpha_i = v1.alpha[i].clone()
        
        k1 = self.gram(v1)

        v2 = self.svmin
        j = self.gmin_index
        old_alpha_j = v2.alpha[j].clone()

        #second order working set selection
        best = 0.0
        gi = 0.0
        if i == 0:
            gi = -v1.g[0]
        else:
            gi = v1.g[1]

        # for v in self.vectors:
        for k in range(len(self.vectors)):
            v = self.vectors[k]

            curv = v1.k + v.k - 2 * k1[v.i]
            if curv <= 0.0:
                curv = self.TAU
            
            gj = -v.g[0]
            if v.alpha[0] > 0.0 and gj < gi:
                gain = - (gi - gj) ** 2 / curv
                if gain < best:
                    best = gain
                    v2 = v
                    j = 0
                    old_alpha_j = v2.alpha[0].clone()
                    # old_alpha_j = v2.alpha[0]

            gj = v.g[1]
            if v.alpha[1] < self.C and gj < gi:
                gain = - (gi - gj) ** 2 / curv
                if gain < best:
                    best = gain
                    v2 = v
                    j = 1
                    old_alpha_j = v2.alpha[1].clone()
                    # old_alpha_j = v2.alpha[1]
        
        
        
        k2 = self.gram(v2)
        # determine curvature
        curv = v1.k + v2.k - 2 * k1[v2.i]
        if curv <= 0.0:
            curv = self.TAU
        
        if i != j:
            delta = (-v1.g[i] - v2.g[j]) / curv
            diff = v1.alpha[i] - v2.alpha[j]
            v1.alpha[i] += delta
            v2.alpha[j] += delta

            if (diff > 0.0):
                # region III
                if (v2.alpha[j] < 0.0):
                    v2.alpha[j] = 0.0
                    v1.alpha[i] = diff
            else:
                # region IV
                if (v1.alpha[i] < 0.0):
                    v1.alpha[i] = 0.0
                    v2.alpha[j] = -diff
            
            if diff > 0:
                #region I
                if (v1.alpha[i] > self.C):
                    v1.alpha[i] = self.C
                    v2.alpha[j] = self.C - diff
            else:
                #region II
                if (v2.alpha[j] > self.C):
                    v2.alpha[j] = self.C
                    v1.alpha[i] = self.C + diff
        else:
            delta = (v1.g[i] - v2.g[j]) / curv
            sum = v1.alpha[i] + v2.alpha[j]
            v1.alpha[i] -= delta
            v2.alpha[j] += delta

            if (sum > self.C):
                if (v1.alpha[i] > self.C):
                    v1.alpha[i] = self.C
                    v2.alpha[j] = sum - self.C
            else:
                if (v2.alpha[j] < 0):
                    v2.alpha[j] = 0
                    v1.alpha[i] = sum
            
            if (sum > self.C):
                if (v2.alpha[j] > self.C):
                    v2.alpha[j] = self.C
                    v1.alpha[i] = sum - self.C
            else:
                if (v1.alpha[i] < 0):
                    v1.alpha[i] = 0.0
                    v2.alpha[j] = sum
        
        
        delta_alpha_i = v1.alpha[i] - old_alpha_i
        delta_alpha_j = v2.alpha[j] - old_alpha_j
        
        si = 2 * i - 1
        sj = 2 * j - 1
        # for v in self.vectors:
        for i in range(len(self.vectors)):
            v = self.vectors[i]
            v.g[0] -= si * k1[v.i] * delta_alpha_i + sj * k2[v.i] * delta_alpha_j
            v.g[1] += si * k1[v.i] * delta_alpha_i + sj * k2[v.i] * delta_alpha_j
        
        self.minmax()

        self.b = -(self.gmax + self.gmin) / 2
        return abs(self.gmax - self.gmin) > tol