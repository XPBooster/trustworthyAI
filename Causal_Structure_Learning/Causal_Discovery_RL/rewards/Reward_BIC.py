import numpy as np
from scipy.linalg import expm as matrix_exponential
from scipy.spatial.distance import pdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import PolynomialFeatures
import logging

class Reward(object):

    def __init__(self, config, inputdata, sl, su, lambda1_upper, verbose_flag=True):
        """

        Parameters
        ----------
        config: config file of the project
        inputdata: the entire input data
        sl: the lower bound of score
        su: the upper bound of score
        lambda1_upper: the upper bound of lambda1
        verbose_flag
        """
        self.config = config
        self.baseint = 2**self.config.num_nodes
        self.graph2score = {}   # key: graph_id, values: score, cycness
        self.node2rss = {}      # store RSS for reuse
        self.inputdata = inputdata
        self.n_samples = inputdata.shape[0]
        self.verbose = verbose_flag
        self.sl = sl
        self.su = su
        self.lambda1_upper = lambda1_upper
        self.bic_penalty = np.log(inputdata.shape[0])/inputdata.shape[0]


    def cal_rewards(self, graphs, lambda1, lambda2):

        return np.array([self.calculate_reward_single_graph(graph, lambda1, lambda2) for graph in graphs])


    ####### regression 

    def calculate_yerr(self, X_train, y_train):
        """

        Parameters
        ----------
        X_train: the masked training set as source nodes
        y_train: the selected target node

        Returns: the error vector with shape (batch,)
        -------

        """
        def calculate_LR(X_train, y_train):
            # faster than LinearRegression() from sklearn
            ones = np.ones((X_train.shape[0], 1), dtype=np.float32)
            X = np.hstack((X_train, ones))
            XtX = X.T.dot(X)
            Xty = X.T.dot(y_train)
            theta = np.linalg.solve(XtX, Xty)
            y_err = X.dot(theta) - y_train
            return y_err

        def calculate_QR(X_train, y_train):

            poly = PolynomialFeatures()
            X_train = poly.fit_transform(X_train)[:,1:]
            return calculate_LR(X_train, y_train)

        def calculate_GPR(X_train, y_train):

            med_w = np.median(pdist(X_train, 'euclidean'))
            gpr = GPR().fit(X_train/med_w, y_train)
            return (y_train.reshape(-1,1) - gpr.predict(X_train/med_w).reshape(-1,1)).reshape(-1)

        if self.config.reg_type == 'LR':
            return calculate_LR(X_train, y_train)
        elif self.config.reg_type == 'QR':
            return calculate_QR(X_train, y_train)
        elif self.config.reg_type == 'GPR':
            return calculate_GPR(X_train, y_train)
        elif self.config.reg_type not in ('LR', 'QR', 'GPR'):
            raise ValueError('Reg type not supported')

    ####### score calculations

    def calculate_reward_single_graph(self, graph, lambda1, lambda2):

        graph_id, node_id = [], []
        graph = graph.numpy()
        for i in range(self.config.num_nodes):
            graph[i][i] = 0
            tt = np.int32(graph[i])
            node_id.append(self.baseint * i + np.int(''.join([str(ad) for ad in tt]), 2))
            graph_id.append(np.int(''.join([str(ad) for ad in tt]), 2))

        graph_id = tuple(graph_id)

        if graph_id in self.graph2score:
            score_cyc = self.graph2score[graph_id]
            reward = self.penalized_score(score_cyc, lambda1, lambda2)
            return reward, score_cyc[0], score_cyc[1]

        RSS_ls = []

        for i in range(self.config.num_nodes):
            col = graph[i]
            if node_id[i] in self.node2rss:
                RSS_ls.append(self.node2rss[node_id[i]])
                continue

            # no parents, then simply use mean
            if col.sum() < 0.1:
                y_err = self.inputdata[:, i]
                y_err = y_err - np.mean(y_err)

            else:
                cols_binary = col > 0.5
                X_train = self.inputdata[:, cols_binary]
                y_train = self.inputdata[:, i]
                y_err = self.calculate_yerr(X_train, y_train)

            RSSi = np.sum(np.square(y_err))

            # if the regresors include the true parents, GPR would result in very samll values, e.g., 10^-13
            # so we add 1.0, which does not affect the monotoniticy of the score
            if self.config.reg_type == 'GPR':
                RSSi += 1.0

            RSS_ls.append(RSSi)
            self.node2rss[node_id[i]] = RSSi
        if self.config.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls)/self.n_samples+1e-8) + graph.sum()*self.bic_penalty/self.config.num_nodes
        elif self.config.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(RSS_ls/self.n_samples+1e-8)) + np.sum(graph)*self.bic_penalty
        elif self.config.score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')

        score = self.score_normalize(BIC)
        cycness = self.score_acyclic(graph)
        reward = self.penalized_score((score, cycness), lambda1, lambda2)
        sparse_reg = self.config.l1_graph_reg * np.sum(graph)
        reward, score = reward + sparse_reg, score + sparse_reg
        self.graph2score[graph_id] = (score, cycness)

        return reward, score, cycness

    #### helper
    def score_acyclic(self, graph):
        return np.trace(matrix_exponential(np.array(graph))) - self.config.num_nodes

    def score_normalize(self, s):
        return (s-self.sl)/(self.su-self.sl)*self.lambda1_upper

    def penalized_score(self, score_cyc, lambda1, lambda2):
        score, cyc = score_cyc
        return score + lambda1*np.float(cyc>1e-5) + lambda2*cyc
    
    def update_scores(self, score_cyc_list, lambda1, lambda2):
        reward = []
        for score_cyc in score_cyc_list:
            reward.append(self.penalized_score(score_cyc, lambda1, lambda2))
        return reward
    
    def update_all_scores(self, lambda1, lambda2):
        score_cycs = list(self.graph2score.items())
        ls = []
        for graph_int, score_cyc in score_cycs:
            ls.append((graph_int, (self.penalized_score(score_cyc, lambda1, lambda2), score_cyc[0], score_cyc[1])))
        return sorted(ls, key=lambda x: x[1][0])
