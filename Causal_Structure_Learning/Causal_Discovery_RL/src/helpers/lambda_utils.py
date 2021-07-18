import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import pdist, squareform


def BIC_input_graph(X, g, reg_type='LR', score_type='BIC'):

    """

    Parameters
    ----------
    X Input data
    g Input graph
    reg_type Regressor
    score_type score function type

    Returns The score of graph g given input data, reg_type and score_type
    -------

    """

    RSS_ls = []

    m, n = X.shape

    if reg_type in ('LR', 'QR'):
        reg = LinearRegression()
    else:
        reg =GaussianProcessRegressor()

    poly = PolynomialFeatures()

    for i in range(n):
        y_ = X[:, [i]]
        inds_x = list(np.abs(g[i])>0.1)

        if np.sum(inds_x) < 0.1: 
            y_pred = np.mean(y_)
        else:
            X_ = X[:, inds_x]
            if reg_type == 'QR':              
                X_ = poly.fit_transform(X_)[:, 1:] 
            elif reg_type == 'GPR':                
                med_w = np.median(pdist(X_, 'euclidean'))
                X_ = X_ / med_w
            reg.fit(X_, y_)
            y_pred = reg.predict(X_)
        RSSi = np.sum(np.square(y_ - y_pred))

        if reg_type == 'GPR':
            RSS_ls.append(RSSi+1.0)
        else:
            RSS_ls.append(RSSi)

    if score_type == 'BIC':
        return np.log(np.sum(RSS_ls)/m+1e-8)
    elif score_type == 'BIC_different_var':
        return np.sum(np.log(np.array(RSS_ls)/m)+1e-8)
    
    
def BIC_lambdas(X, g_l=None, g_u=None, g_true=None, reg_type='LR', score_type='BIC'):

    """
    :param X: dataset
    :param g_l: input graph to get score lower bound
    :param g_u: input graph to get score upper bound
    :param g_true: input true graph
    :param reg_type:
    :param score_type:
    :return: score lower bound, score upper bound, true score (only for monitoring)
    """
        
    n, d = X.shape

    if score_type == 'BIC':
        bic_penalty = np.log(n) / (n*d)
    elif score_type == 'BIC_different_var':
        bic_penalty = np.log(n) / n

    g_l = np.ones(d)-np.eye(d) if g_l is None else g_l # default g_l for BIC score: complete graph except digonals
    g_u = np.zeros((d, d)) if g_u is None else g_u # default g_u for BIC score: empty graph

    s_l = BIC_input_graph(X, g_l, reg_type, score_type)
    s_u = BIC_input_graph(X, g_u, reg_type, score_type) 

    if g_true is None:
        s_true = s_l - 10
    else:
        print(BIC_input_graph(X, g_true, reg_type, score_type))
        print(g_true)
        print(bic_penalty)
        s_true = BIC_input_graph(X, g_true, reg_type, score_type) + np.sum(g_true) * bic_penalty
    
    return s_l, s_u, s_true

