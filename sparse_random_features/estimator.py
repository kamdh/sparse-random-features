from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, LogisticRegression
import numpy as np
import scipy.sparse

class SparseRFRegressor(BaseEstimator, RegressorMixin):
    """
    Random features with sparse weight regression
    """
    def __init__(self, deg=None, width=1000, weights="gaussian",
                     nonlinearity=np.cos,
                     nonneg=False, alpha=1., W_scale=1., b_scale=np.pi,
                     clf=None,
                     weight_fun=None, multiscale=None,
                     biregular=False, clf_args={}):
        self.deg = deg
        self.width = width
        self.weights = weights
        self.nonlinearity = nonlinearity
        self.nonneg = nonneg
        self.alpha = alpha
        self.W_scale = W_scale
        self.b_scale = b_scale
        self.clf = clf
        self.weight_fun = weight_fun
        self.multiscale = multiscale
        self.biregular = biregular
        self.clf_args = clf_args
        #self._fitted = False

    def fit(self, X, y):
        # check parameters
        n = X.shape[1]
        if self.clf is None:
            self.clf = Ridge(alpha=self.alpha)
        else:
            self.clf = self.clf(**self.clf_args)
        if self.deg is None:
            self.deg = n
        if self.weight_fun is not None:
            self.W_ = self.weight_fun(self.width, n, self.deg)
        else:
            self.W_ = \
              sparse_regular_expansive_matrix(self.width, n, self.deg,
                            self.nonneg, self.weights, self.biregular).T \
                            * self.W_scale
        if self.multiscale is not None:
            scale_mat = np.diag(np.repeat(self.multiscale,
                                int(self.width / len(self.multiscale))))
            self.W_ = self.W_ @ scale_mat
        self.b_ = 2 * (np.random.rand(1, self.width) - 0.5) * self.b_scale
        H = self.nonlinearity(X @ self.W_ - self.b_) / np.sqrt(self.width)
        # fit classifier
        self.clf.fit(H, y)
        self._fitted = True
        return self
        
    def predict(self, X):
        H = self.nonlinearity(X @ self.W_ - self.b_) / np.sqrt(self.width)
        return self.clf.predict(H)

    def transform(self, X):
        H = self.nonlinearity(X @ self.W_ - self.b_) / np.sqrt(self.width)
        return H

class SparseRFClassifier(BaseEstimator, ClassifierMixin):
    """
    Random features with sparse weight classification
    """
    def __init__(self, deg=None, width=1000, weights="gaussian", nonlinearity=np.cos,
                     nonneg=False, C=1., W_scale=1., b_scale=np.pi, clf=None,
                     weight_fun=None, multiscale=None,
                     biregular=False, clf_args={}):
        self.deg = deg
        self.width = width
        self.weights = weights
        self.nonlinearity = nonlinearity
        self.nonneg = nonneg
        self.C = C
        self.W_scale = W_scale
        self.b_scale = b_scale
        self.clf = clf
        self.weight_fun = weight_fun
        self.multiscale = multiscale
        self.biregular = biregular
        self.clf_args = clf_args
        # self._fitted = False

    def fit(self, X, y):
        n = X.shape[1]
        if clf is None:
            self.clf = LogisticRegression(C=self.C)
        else:
            self.clf = self.clf(**self.clf_args)
        if self.deg is None:
            self.deg = n
        if self.weight_fun is not None:
            self.W_ = self.weight_fun(self.width, n, self.deg)
        else:
            self.W_ = \
              sparse_regular_expansive_matrix(self.width, n, self.deg,
                            self.nonneg, self.weights, self.biregular).T \
                                        * self.W_scale
        if self.multiscale is not None:
            scale_mat = np.diag(np.repeat(self.multiscale,
                                int(self.width / len(self.multiscale))))
            self.W_ = self.W_ @ scale_mat
        self.b_ = 2 * (np.random.rand(1, self.width) - 0.5) * self.b_scale
        H = self.nonlinearity(X @ self.W_ - self.b_) / np.sqrt(self.width)
        self.clf.fit(H, y)
        self._fitted = True
        return self
        
    def predict(self, X):
        H = self.nonlinearity(X @ self.W_ - self.b_) / np.sqrt(self.width)
        return self.clf.predict(H)

    def transform(self, X):
        H = self.nonlinearity(X @ self.W_ - self.b_) / np.sqrt(self.width)
        return H    

def sparse_regular_expansive_matrix(M, N, deg, nonneg=False, \
                        weights='gaussian', biregular=False):
    '''
    Generates a size (M, N) random matrix with degree deg entries per row

    Weights are generated from the specified distribution and scaled 
    by 1 / sqrt(deg).

    Parameters
    ----------
    M : number of rows

    N : number of columns

    deg : in-degree, number of nonzero entries per row

    nonneg : boolean, default False
        If True, makes the weights nonnegative
    
    weights : string or function, default 'gaussian'
        If 'gaussian', entries are drawn ~ N(0, 1).
        Can be 'uniform', corresponding to ~ U(-sqrt(3), sqrt(3)).
        or can be function handle taking arguments M, N.

    Returns
    -------
    weight_matrix : scipy.sparse.coo_matrix of shape (M, N)
    
    '''
    from numpy.random import randn, choice, rand
    M = int(M)
    N = int(N)
    deg = int(deg)
    if deg == N:
        # complete graph (not sparse)
        if weights == 'gaussian':
            J = randn(M, N) / np.sqrt(deg)
        elif weights == 'uniform':
            J = np.sqrt(3) * 2 * (rand(M, N) - 0.5) / np.sqrt(deg)
        else:
            J = weights(M, N) / np.sqrt(deg)
        if nonneg:
            J = np.abs(J) 
        return J
    elif biregular:
        Jsparsity = _bipartite_biregular_matrix(M, N, deg)
        if weights == 'gaussian':
            J = randn(M, N) / np.sqrt(deg)
        elif weights == 'uniform':
            J = np.sqrt(3) * 2 * (rand(M, N) - 0.5) / np.sqrt(deg)
        else:
            J = weights(M, N) / np.sqrt(deg)
        if nonneg:
            J = np.abs(J)
        J *= Jsparsity
        return J
    else:
        rows = np.zeros(M * deg)
        cols = np.zeros(M * deg)
        # sample sparsity pattern
        for i in np.arange(M):
            rows[i * deg:(i+1) * deg] = i
            # cols[i * deg:(i+1) * deg] = sample(np.arange(N), int(deg))
            cols[i * deg:(i+1) * deg] = choice(np.arange(N), deg, replace=False)
        if weights == 'gaussian':
            data = randn(M * deg)
        elif weights == 'uniform':
            data = np.sqrt(3) * 2 * (rand(M * deg) - 0.5)
        else:
            data = weights(M * deg)
        data /= np.sqrt(deg)
        if nonneg:
            data = np.abs(data)
        J = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(M, N))
        return J.toarray()

def _bipartite_biregular_matrix(M, N, deg):
    import networkx as nx
    from networkx.algorithms.bipartite import configuration_model \
     as bipartite_configuration_model

    def _extract_blocks(A, n1, n2):
            X = A[0:n1, n1:n1+n2]
            Xt = A[n1:n1+n2, 0:n1]
            return X, Xt

    deg_out = int(M * deg / N)
    assert M * deg == N * deg_out, \
      "Unbalanced model: N, M, d1, d2 = %d, %d, %d, %d" % (N, M, deg, deg_out)

    g = bipartite_configuration_model([deg] * M, [deg_out] * N,
           create_using=nx.Graph())
    # # we may have to configure multiple graphs to ensure no parallel edges
    # while True:
    #     g = bipartite_configuration_model([deg] * M, [deg_out] * N,
    #                                         create_using=nx.Graph())
    #     deg_list = set(g.degree)
    #     deg_desire = [(i, deg) for i in range(M)]
    #     deg_desire.extend([(i + M, deg_out) for i in range(N)])
    #     deg_desire = set(deg_desire)
    #     if deg_list == deg_desire:
    #         break
    X, Xt = _extract_blocks(nx.to_numpy_matrix(g), M, N)
    assert X.shape == (M, N), "Wrong size X returned"
    return X
