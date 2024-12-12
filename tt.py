import numpy as np
import math
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import gammaln
from scipy import stats
from scipy.stats import geninvgauss
from functools import reduce
from sklearn.linear_model import LassoCV
import itertools
import time

class BayesTensorReg:
    def __init__(self):
        # Initialization method (empty, no parameters needed for now)
        self.alpha_store = None
        self.beta_store = None
        self.gam_store = None
        self.tau2_store = None
        self.phi_store = None
        self.varphi_store = None
        self.lambda_store = None
        self.omega_store = None

    def getouter_list(self, bet):
        d = len(bet)
        if d == 1:
            return bet[0]
        elif d == 2:
            return np.outer(bet[0], bet[1]).reshape(bet[0].shape + bet[1].shape)
        else:
            return np.outer(self.getouter_list(bet[:-1]), bet[-1]).reshape(self.getouter_list(bet[:-1]).shape + bet[-1].shape)

    def getmean(self, X, beta, rank, rank_exclude=None):
        if rank_exclude is None:
            rank_exclude = []

        idx = [i for i in range(rank) if i not in rank_exclude]
        B_list = [self.getouter_list([beta_elem[r, :] for beta_elem in beta]) for r in idx]
        B = reduce(lambda x, y: x + y, B_list)

        def compute_sum(xx, bb):
            return np.sum(xx * bb)

        mu_B = np.array([compute_sum(xx, B) for xx in X])
        return mu_B

    def logsum(self, lx):
        max_lx = np.max(lx)
        return max_lx + np.log(np.sum(np.exp(lx - max_lx)))

    def TP_rankR(self, X_allr):
        R = X_allr[0].shape[1] if len(X_allr[0].shape) > 1 else None
        if R is None:
            return self.getouter_list(X_allr)
        else:
            dims = [x.shape[0] for x in X_allr]
            Y = np.zeros(dims)
            for r in range(R):
                outer_results = [x[:, r] for x in X_allr]
                Y += self.getouter_list(outer_results)
            return Y

    def getBeta_mcmc(self, beta_store):
        nsweep = len(beta_store)
        d = len(beta_store[0])
        rank = beta_store[0][0].shape[0]
        p = [beta_store[0][x].shape[1] for x in range(d)]
        Beta_mcmc = np.zeros((nsweep, np.prod(p)))

        for i in range(nsweep):
            coef = np.zeros(np.prod(p))
            for r in range(rank):
                outer_list = [beta_store[i][x][r, :] for x in range(d)]
                coef += self.getouter_list(outer_list).flatten()
            Beta_mcmc[i, :] = coef
        return Beta_mcmc

    #### Main function ####
    def tensor_reg(self, z_train, x_train, y_train, a_lam=None, b_lam=None, phi_alpha=None, nsweep=1e3, rank=5, burn=0.30,
                   nskip=3, scale=True, plot=False):

        n = len(y_train)
        p = x_train.shape[1:]
        d = len(x_train.shape) - 1
        pgamma = z_train.shape[1]

        #### Standardize ####
        my = np.mean(y_train)
        sy = np.std(y_train, ddof=1) if scale else 1
        if scale:
            obs = (y_train - my) / sy
        else:
            obs = y_train

        if scale:
            mz = np.mean(z_train, axis=0)
            sz = np.array([np.max(z_train[:, i]) - np.min(z_train[:, i]) for i in range(pgamma)])
            sz[sz == 0] = 1
            Zt = (z_train - mz) / sz

            mx = np.mean(x_train, axis=0)
            sx = np.apply_along_axis(lambda z: np.nanmax(z) - np.nanmin(z), axis=0, arr=x_train)
            sx[sx == 0] = 1
            Xt = (x_train - mx) / sx
        else:
            Zt = z_train
            Xt = x_train

        #### MCMC setup ####
        ZZ = np.dot(Zt.T, Zt)
        vecXt = Xt.reshape(n, np.prod(p))
        vecXt = np.hstack((z_train, vecXt))

        las = LassoCV(cv=5).fit(vecXt, y_train)
        beta_init = las.coef_
        gam = beta_init[:pgamma]

        ## Hyperparameter initialization
        if a_lam is None:
            a_lam = np.array([3.0] * rank)
        if b_lam is None:
            b_lam = a_lam**(1 / (2 * d))
        if phi_alpha is None:
            phi_alpha = np.array([1 / rank] * rank)

        ## Initialize tensors
        tau2 = 1 / np.random.gamma(size=1, shape=2.5, scale=1 / 2.5)
        phi = np.random.dirichlet(phi_alpha, size=1)
        varphi = np.random.gamma(size=1, shape=rank, scale=1 / (phi_alpha[1] * rank))

        tau_r = (phi * varphi).flatten()

        ## Initialize MCMC storage
        alpha_store = [None] * nsweep
        beta_store = [[np.random.normal(size=(rank, p[j])) for j in range(d)] for _ in range(int(nsweep))]
        gam_store = np.zeros((int(nsweep), pgamma))
        tau2_store = np.zeros(int(nsweep))
        c0_store = np.zeros(int(nsweep))
        phi_store = np.zeros((int(nsweep), rank))
        varphi_store = np.zeros(int(nsweep))
        lambda_store = np.zeros((int(nsweep), rank, d))
        omega_store = [[np.random.exponential(scale=.5 * (a_lam[j] / b_lam[j]), size=(rank, p[j])) for j in range(d)] for _ in range(int(nsweep))]

        ## Run MCMC
        start_time = time.time()

        for sweep in range(int(nsweep)):
            ## Compute tensor mean
            tens_mean = self.getmean(Xt, beta_store[sweep], rank)

            ## Update gamma
            Sig_g = np.linalg.inv(np.diag(np.ones(pgamma)) + ZZ / tau2)
            mu_g = Sig_g @ np.dot(Zt.T, (obs - tens_mean)) / tau2
            gam = mu_g + np.linalg.cholesky(Sig_g) @ np.random.normal(size=pgamma)

            ## Update tau2
            a_tau = 2.5 + n / 2
            b_tau = 2.5 + 0.5 * np.sum((obs - tens_mean) ** 2)
            tau2 = 1 / np.random.gamma(shape=a_tau, scale=1 / b_tau)

            ## Update beta, lambda, and omega
            for rr in range(rank):
                for j in range(d):
                    tens_mu_r = self.getmean(Xt, beta_store[sweep], rank, rank_exclude=[rr])

                    betj = self.getouter_list([beta_elem[rr, :] for k, beta_elem in enumerate(beta_store[sweep]) if k != j])

                    H = np.zeros((n, p[j]))
                    for i in range(n):
                        if d == 2:
                            if j == 0:
                                H[i, :] = [np.sum(Xt[i, k, :] * betj) for k in range(p[j])]
                            elif j == 1:
                                H[i, :] = [np.sum(Xt[i, :, k] * betj) for k in range(p[j])]
                        elif d == 3:
                            if j == 0:
                                H[i, :] = [np.sum(Xt[i, k, :, :] * betj) for k in range(p[j])]
                            elif j == 1:
                                H[i, :] = [np.sum(Xt[i, :, k, :] * betj) for k in range(p[j])]
                            elif j == 2:
                                H[i, :] = [np.sum(Xt[i, :, :, k] * betj) for k in range(p[j])]

                    HH = np.dot(H.T, H)
                    diag_elements = 1 / omega_store[sweep][j][rr, :] / tau_r[rr]
                    diag_matrix = np.diag(diag_elements)
                    chol_matrix = cholesky(HH / tau2 + diag_matrix)

                    K = np.linalg.inv(chol_matrix.T @ chol_matrix)
                    mm = obs - tens_mu_r
                    bet_mu_jr = K @ (H.T @ mm / tau2)
                    beta_store[sweep][j][rr, :] = bet_mu_jr + cholesky(K) @ np.random.randn(p[j])

                    ## Update lambda
                    shape = a_lam[rr] + p[j]
                    rate = b_lam[rr] + np.sum(np.abs(beta_store[sweep][j][rr, :])) / np.sqrt(tau_r[rr])
                    lambda_store[sweep][rr, j] = np.random.gamma(shape, 1.0 / rate)

                    ## Update omega
                    omega_store[sweep][j][rr, :] = np.array([geninvgauss.rvs(0.5, beta_store[sweep][j][rr, kk] ** 2 / tau_r[rr], scale=lambda_store[sweep][rr, j] ** 2) for kk in range(p[j])])

            ## Store results for each sweep
            tau2_store[sweep] = tau2
            gam_store[sweep, :] = gam
            c0_store[sweep] = 0
            phi_store[sweep, :] = phi.flatten()
            varphi_store[sweep] = varphi

        elapsed_time = time.time() - start_time

        ## Output dictionary containing results
        out = {
            "nsweep": nsweep,
            "rank": rank,
            "p": p,
            "d": d,
            "Zt": Zt,
            "Xt": Xt,
            "obs": obs,
            "tau2_store": tau2_store,
            "gam_store": gam_store,
            "beta_store": beta_store,
            "phi_store": phi_store,
            "varphi_store": varphi_store,
            "omega_store": omega_store,
            "lambda_store": lambda_store,
            "time": elapsed_time
        }

        return out


    def predict(X, Beta, y = None, Train = False, Test = False):
        err = 0
        for i in range(X.shape[1]):
            err += (np.tensordot(X[i], Beta, axes=((0, 1, 2), (0, 1, 2)))-y[i])**2
            mse = err/X.shape[1]
        rmse = mse/np.var(y)
        return rmse