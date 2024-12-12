import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.linalg import cholesky, inv
import math
import time
from functools import reduce
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.special import gammaln, logsumexp
from sklearn.linear_model import LassoCV
import itertools
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import geninvgauss


import numpy as np
import math
from scipy.linalg import cho_factor, cho_solve, cholesky
from sklearn.linear_model import LassoCV
from functools import reduce
from scipy.special import gammaln
from scipy import stats
import itertools
import pandas as pd
import time

class BayesTensorReg:
    
    def __init__(self, rank=5, nsweep=1000, burn=0.30, nskip=3):
        self.rank = rank
        self.nsweep = nsweep
        self.burn = burn
        self.nskip = nskip

    def getouter_list(self, bet):
        d = len(bet)
        if d == 1:
            return bet[0]
        elif d == 2:
            return np.outer(bet[0], bet[1]).reshape(bet[0].shape + bet[1].shape)
        else:
            return np.outer(self.getouter_list(bet[:-1]), bet[-1]).reshape(
                self.getouter_list(bet[:-1]).shape + bet[-1].shape
            )

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

    ####main function####
    def tensor_reg(self, z_train, x_train, y_train, a_lam, b_lam, phi_alpha, nsweep=1e3, rank=5, burn=0.30,
                nskip=3, scale=True, plot=False):
        
        
        n = len(y_train)
        p = x_train.shape[1:]
        d = len(x_train.shape)-1
        pgamma = z_train.shape[1]
        

        #### standarize ####
        my = np.mean(y_train)
        sy = np.std(y_train, ddof=1) if scale else 1
        if scale:
            obs = (y_train - my) / sy
        else:
            obs = y_train


        if scale:
            mz = np.mean(z_train, axis=0)
            sz = np.array([np.max(z_train[:, i]) - np.min(z_train[:, i]) for i in range(pgamma)])
            sz[sz==0] = 1
            Zt = np.zeros_like(z_train, dtype=float)
            for jj in range(pgamma):
                Zt[:,jj] = (z_train[:,jj] - mz[jj]) / sz[jj] 
        
            Xt = np.zeros_like(x_train, dtype=float)
            mx = np.mean(x_train, axis=0)
            def range_diff(z):
                return np.nanmax(z) - np.nanmin(z)
            sx = np.apply_along_axis(range_diff, axis=0, arr=x_train)
            sx[sx == 0] <- 1
    
            if d == 2:
                for jj in range(n):
                    Xt[jj,:,:] = (x_train[jj,:,:] - mx) / sx
            elif d == 3:
                for jj in range(n):
                    Xt[jj,:,:,:] = (x_train[jj,:,:,:] - mx) / sx
    
        else:
            Zt = z_train
            Xt = x_train
        

        x_train_nona = Xt
        #### MCMC setup ####
        ZZ = np.dot(Zt.T, Zt)
        vecXt = Xt.reshape(n, np.prod(p))
        vecXt = np.hstack((z_train, vecXt))
        las = LassoCV(cv=5).fit(vecXt, y_train)
        beta_init = las.coef_
        gam = beta_init[0:pgamma]

        ##hyper-par initialize
        a_lam = None; b_lam = None; phi_alpha = None
        if a_lam is None:
            a_lam = [3.0] * rank
            a_lam = np.array(a_lam)
        if b_lam is None:
            b_lam = a_lam**(1/(2*d))
        if phi_alpha is None:
            phi_alpha = np.array([1/rank]*rank)
        phi_a0 = np.sum(phi_alpha)
        a_vphi = phi_a0
        b_vphi = phi_alpha[1] * rank**(1/d)
        c0 = 0
        s0 = 1; a_t = 2_5/2; b_t = 2.5/2 * s0**2

        ##fix randomness for now
        tau2  = 1 / np.random.gamma(size = 1, shape = a_t, scale=1/b_t)
        #tau2 = 2
        phi = np.random.dirichlet(phi_alpha, size = 1)
        varphi = np.random.gamma(size = 1, shape = a_vphi, scale=1/b_vphi)
        #varphi = 0.5
        tau_r = phi * varphi
        # Define the flatten function
        def flatten(nested_list):
            return [item for sublist in nested_list for item in sublist]
        tau_r = flatten(tau_r)
        
        lambda_ = np.array([1.5]*rank*d).reshape((rank, d))
        omega = [None]*d
        for x in range(d):
            omega[x] = np.random.exponential(scale=.5*(a_lam[1]/b_lam[1]), size=(rank,p[x]))

        beta = [None]*d
        for x in range(d):
            beta[x] = np.random.normal(size = (rank,p[x]))

        ##initialize tensor margins
        alpha_store = [None]*nsweep
        c0_store = [None]*nsweep
        gam_store = np.array([None]*nsweep*pgamma).reshape(nsweep,pgamma)
        tau2_store = [None]*nsweep
        phi_store = np.array([None]*nsweep*rank).reshape(nsweep,rank)
        phi_store
        varphi_store = np.array([None]*nsweep*rank).reshape(-1,1)
        varphi_store
        beta_store = [[None]*d]*nsweep
        for x in range(nsweep):
            for y in range(d):
                beta_store[x][y] = np.array([None]*rank*p[y]).reshape(rank,p[y])
        beta_store
        omega_store = [[None]*d]*nsweep
        for x in range(nsweep):
            for y in range(d):
                omega_store[x][y] = np.array([None]*rank*p[y]).reshape(rank,p[y])
        lambda_store = np.array([None]*nsweep*rank*d).reshape(nsweep,rank, d)
        #hyppar_store = np.array([None]*nsweep*rank*d).reshape(nsweep,rank, 2)
        hyppar_store = np.zeros((nsweep, rank, 2))

        alam_seq = np.linspace(2.1, d + 1, num=5)
        zeta_max = np.ceil(10 * rank**(1 / (2 * d)) / 2) / 10
        zeta_seq = np.linspace(0.5, zeta_max, num=5)
        param_grid = list(itertools.product(alam_seq, zeta_seq))
        par_grid = pd.DataFrame(param_grid, columns=['alam', 'zeta'])
        alam_seq = np.linspace(2.1, d + 1, num=5)
        zeta_max = np.ceil(10 * rank**(1 / (2 * d)) / 2) / 10
        zeta_seq = np.linspace(0.5, zeta_max, num=5)
        alam_grid, zeta_grid = np.meshgrid(alam_seq, zeta_seq)
        par_grid = pd.DataFrame({
            'alam': alam_grid.flatten(),
            'zeta': zeta_grid.flatten()
        })
        par_grid = par_grid.values
        alpha_grid = np.linspace(rank**(-d), rank**(-0.1), num=10)
        M=20
        score_store = np.array([None]*nsweep*len(alpha_grid)).reshape(nsweep,len(alpha_grid))

        #### MCMC run ####
        start_time = time.time()
        for sweep in range(nsweep):
            tens_mean = self.getmean(x_train_nona, beta, rank)
            Cjr = np.zeros((d, rank))
            for rr in range(rank):
                for jj in range(d):
                    bb = np.sum(np.abs(beta[jj][rr, :]))
                    Cjr[jj, rr] = bb / np.sqrt(tau_r[rr])
                    #print("Cjr[jj, rr]: ", Cjr[jj, rr])
            def mfun(z, rank, p, Cjr):
                o = [gammaln(z[0] + p[x]) - gammaln(z[0]) + z[0] * math.log(z[1] * z[0]) - (z[0] + p[x]) * math.log(z[1] * z[0] + Cjr[x][rank]) for x in range(d)]
                return sum(o)
            
            ll = np.zeros((par_grid.shape[0], rank))
            for rr in range(rank):
                for z in range(par_grid.shape[0]):
                    result = mfun(par_grid[z], rr, p, Cjr)
                    ll[z, rr] = result
            




            par_wt = np.apply_along_axis(lambda z: np.exp(z - self.logsum(z)), axis=0, arr=ll)
            #par_wt = np.apply_along_axis(lambda z: np.exp(z - np.log(np.sum(z))), axis=0, arr=ll)
            par_wt = np.nan_to_num(par_wt, nan=0.0, posinf=0.0, neginf=0.0)
            for i in range(par_wt.shape[1]):
                par_wt[:,i] = par_wt[:,i]/np.sum(par_wt[:,i])
            par_wt = np.nan_to_num(par_wt, nan=0.0, posinf=0.0, neginf=0.0)
            # Convert the cleaned NumPy array back to a list
            #cleaned_list = cleaned_array.tolist()
            #print("par_wt:", sum(par_wt)) 





            indices = np.arange(par_grid.shape[0])
            ixx = np.zeros(par_wt.shape[1], dtype=int)
            for i in range(par_wt.shape[1]):
                ixx[i] = np.random.choice(indices, size=1, p=par_wt[:, i])[0]
            
            for rr in range(rank):
                a_lam[rr] = par_grid[ixx[rr], 0]
                b_lam[rr] = par_grid[ixx[rr], 1] * a_lam[rr]
            np.set_printoptions(precision=10, suppress=False)

            ##update gamma
            diag_pgamma = np.diag(np.ones(pgamma))
            cho_factor_matrix = cho_factor(diag_pgamma + ZZ / tau2)
            Sig_g = cho_solve(cho_factor_matrix, np.eye(pgamma))
            mu_g = np.dot(Sig_g, np.dot(Zt.T, (obs - c0 - tens_mean)) / tau2)
            rnorm_pgamma = np.random.normal(size=pgamma)
            gam = mu_g + np.dot(np.linalg.cholesky(Sig_g), rnorm_pgamma)

            ## update alpha (intercept)
            pred_mean = np.dot(Zt, gam)
            mu_c0 = np.mean(obs - pred_mean - tens_mean)
            c0 = np.random.normal(loc=mu_c0, scale=np.sqrt(tau2 / n))
            
            ## update tau2
            a_tau = a_t + n / 2
            b_tau = b_t + 0.5 * np.sum((obs - c0 - pred_mean - tens_mean)**2)
            tau2 = 1 / stats.gamma.rvs(a=a_tau, scale=1/b_tau)

            ## update (alpha, phi, varphi)
            def draw_phi_tau(alpha_grid):
                length = len(alpha_grid)

                # Precompute Cr matrix
                Cr = np.array([
                    [
                        np.dot(beta[jj][rr, :], np.dot(np.diag(1 / omega[jj][rr, :]), beta[jj][rr, :]))
                        for rr in range(rank)
                    ]
                    for jj in range(d)
                ])

                def score_fn(phi_alpha, phi_s, varphi_s, Cstat):
                    def ldirdens(v, a):
                        c1 = gammaln(np.sum(a))
                        c2 = np.sum(gammaln(a))
                        return (c1 - c2) + np.sum((a - 1) * np.log(np.maximum(v, 1e-10)))  

                    ldir = np.apply_along_axis(ldirdens, 1, phi_s, a=phi_alpha)

                    lvarphi = stats.gamma.logpdf(varphi_s, a=np.sum(phi_alpha), scale=1/b_vphi)
                    
                    dnorm_log = -np.sum(Cstat, axis=1) / (2 * np.maximum(varphi_s, 1e-10)) 
                    dnorm_log -= (np.sum(p) / 2) * np.array([np.sum(np.log(np.maximum(varphi_s[ii] * phi_s[ii, :], 1e-10))) for ii in range(len(varphi_s))])  # Avoid log(0)
                    
                    return dnorm_log + ldir + lvarphi

                if length > 1:
                    phi = np.zeros((M * length, rank))
                    varphi = np.zeros((M * length, 1))
                    Cstat = np.zeros((M * length, rank))
                    
                    for jj in range(length):
                        m_phialpha = np.full(rank, alpha_grid[jj])
                        m_phia0 = np.sum(m_phialpha)
                        m_avphi = m_phia0

                        # Draw phi
                        Cr1 = np.sum(Cr, axis=0)
                        phi_a = np.array([geninvgauss.rvs(m_phialpha[rr] - np.sum(p)/2, Cr1[rr], scale=2 * b_vphi, size=M) for rr in range(rank)]).T
                        phi_a = np.apply_along_axis(lambda z: z / np.sum(z), 1, phi_a)

                        # Draw varphi
                        Cr2 = np.apply_along_axis(lambda z: Cr1 / np.maximum(z, 1e-10), 1, phi_a)  # Avoid division by zero
                        varphi_a = np.array([geninvgauss.rvs(m_avphi - rank * np.sum(p)/2, 2 * b_vphi, scale=np.sum(z)) for z in Cr2]).flatten()

                        phi[jj * M:(jj + 1) * M, :] = phi_a
                        varphi[jj * M:(jj + 1) * M, 0] = varphi_a
                        Cstat[jj * M:(jj + 1) * M, :] = Cr2

                    scores = [score_fn(np.full(rank, z), phi, varphi, Cstat) for z in alpha_grid]
                    scores = np.array(scores)
                    lmax = np.max(scores)
                    normalized_scores = np.array([np.mean(np.exp(score - lmax)) for score in scores])
                    normalized_scores /= np.sum(normalized_scores)  # Ensure scores sum to 1
                else:
                    m_phialpha = np.full(rank, alpha_grid[0])
                    m_phia0 = np.sum(m_phialpha)
                    m_avphi = m_phia0

                    Cr1 = np.sum(Cr, axis=0)

                    # Draw phi
                    phi = np.array([geninvgauss.rvs(m_phialpha[rr] - np.sum(p) / 2, 2 * b_vphi, scale=Cr1[rr], size=1) for rr in range(rank)]).flatten()
                    phi = phi / np.sum(phi)

                    # Draw varphi
                    Cr2 = Cr1 / np.maximum(phi, 1e-10)  # Avoid division by zero
                    varphi = geninvgauss.rvs(m_avphi - rank * np.sum(p) / 2, 2 * b_vphi, scale=np.sum(Cr2), size=1)

                    scores = score_fn(m_phialpha, np.array([phi]), np.array([varphi]), np.array([Cr2]))
                    scores = np.array([scores])
                    lmax = np.max(scores)
                    normalized_scores = np.array([np.mean(np.exp(scores - lmax))])
                    normalized_scores /= np.sum(normalized_scores)  # Ensure scores sum to 1

                return {'phi': phi, 'varphi': varphi, 'scores': normalized_scores}
            ## sample astar
            o = draw_phi_tau(alpha_grid)
            scores = o['scores']
            normalized_scores = scores / np.sum(scores)
            astar = np.random.choice(alpha_grid, size=1, p=normalized_scores)
            score_store[sweep, :] = normalized_scores

            # Sample (phi, varphi) based on astar
            o = draw_phi_tau(astar)
            phi = o['phi']
            varphi = o['varphi']

            # Calculate tau.r
            tau_r = varphi * phi
            #print("tau_r: ", tau_r)
            # Define phi.alpha, phi.a0, and a.vphi
            phi_alpha = np.full(rank, astar)
            phi_a0 = np.sum(phi_alpha)
            a_vphi = phi_a0

            ## update rank specific params
            lambda_ = np.zeros((rank, len(beta)))
            for r in range(rank):
                for j in range(d):
                    tens_mu_r = self.getmean(x_train_nona, beta, rank, [r])
                        
                    betj = self.getouter_list([beta_elem[r, :] for k, beta_elem in enumerate(beta) if k != j])
                        
                    H = np.full((n, p[j]), np.nan)
                    for i in range(n):
                        if d == 2:
                            if j == 0:
                                H[i, :] = [np.sum(x_train_nona[i, k, :] * betj) for k in range(p[j])]
                            elif j == 1:
                                    H[i, :] = [np.sum(x_train_nona[i, :, k] * betj) for k in range(p[j])]
                        elif d == 3:
                            if j == 0:
                                H[i, :] = [np.sum(x_train_nona[i, k, :, :] * betj) for k in range(p[j])]
                            elif j == 1:
                                H[i, :] = [np.sum(x_train_nona[i, :, k, :] * betj) for k in range(p[j])]
                            elif j == 2:
                                H[i, :] = [np.sum(x_train_nona[i, :, :, k] * betj) for k in range(p[j])]
                    #print("H: ", H)
                    HH = np.dot(H.T, H)
                    #print("HH: ", HH)
                    diag_elements = 1 / omega[j][r, :] / tau_r[r]
                    diag_matrix = np.diag(diag_elements)
                    #print("HH / tau2 + diag_matrix: ", HH / tau2 + diag_matrix)
                    chol_matrix = cholesky(HH / tau2 + diag_matrix)
                    K = inv(chol_matrix.T @ chol_matrix)
                    #K = inv(chol_matrix)
                    #print("K:", K)
                    
                    ##update betas
                    mm = obs - c0 - pred_mean - tens_mu_r
                    bet_mu_jr = K @ ((H.T/tau2)@ mm)
                    chol_K = cholesky(K, lower=True)
                    beta[j][r, :] = bet_mu_jr + chol_K @ np.random.randn(p[j])

                    ## update lambda.jr
                    shape = a_lam[r] + p[j]
                    rate = b_lam[r] + np.sum(np.abs(beta[j][r, :])) / np.sqrt(tau_r[r]) 
                    lambda_[r, j] = np.random.gamma(shape, 1.0 / rate)
                    ## update omega.jr
                    omega[j][r, :] = [geninvgauss.rvs(0.5, beta[j][r, kk]**2 / tau_r[r], scale=lambda_[r, j]**2) for kk in range(p[j])]
            
            ## store params
            tau2_store[sweep] = tau2
            c0_store[sweep] = c0
            if z_train is not None:
                gam_store[sweep, :] = gam
            else:
                gam_store[sweep] = gam
            alpha_store[sweep] = astar
            phi_store[sweep, :] = phi
            varphi_store[sweep, :] = varphi
            beta_store[sweep] = beta
            omega_store[sweep] = omega
            lambda_store[sweep, :, :] = lambda_
            for rr in range(rank):
                hyppar_store[sweep, rr, :] = [a_lam[rr], b_lam[rr]]


            
            
            #if sweep % 5 == 0:
                #print(f"{sweep}, tau2: {tau2 * sy**2:.3f}, (alpha, a.lam, b.lam): {astar:.3f}, {a_lam[r]:.3f}, {b_lam[r]:.3f}")
        
        # Example time-consuming operation
        time.sleep(0.01)
        end_time = time.time()
        elapsed_time = abs(end_time - start_time)
        print('Time out:', elapsed_time)

        out = {
        "nsweep": nsweep,
        "rank": rank,
        "p": p,
        "d": d,
        "par_grid": par_grid,
        "alpha_grid": alpha_grid,
        "my": my,
        "sy": sy,
        "mz": mz,
        "sz": sz,
        "mx": mx,
        "sx": sx,
        "Zt": Zt,
        "Xt": Xt,
        "obs": obs,
        "a_t": a_t,
        "b_t": b_t,
        "tau2_store": tau2_store,
        "c0_store": c0_store,
        "gam_store": gam_store,
        "alpha_store": alpha_store,
        "beta_store": beta_store,
        "phi_store": phi_store,
        "varphi_store": varphi_store,
        "omega_store": omega_store,
        "lambda_store": lambda_store,
        "hyppar_store": hyppar_store,
        "score_store": score_store,
        "time": elapsed_time
        }
        return out

    def predict(self, X, Beta, y = None, Train = False, Test = False):
        err = 0
        for i in range(X.shape[1]):
            err += (np.tensordot(X[i], Beta, axes=((0, 1, 2), (0, 1, 2)))-y[i])**2
            mse = err/X.shape[1]
        rmse = mse/np.var(y)
        return rmse