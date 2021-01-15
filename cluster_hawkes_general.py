import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.sparse import identity
import math

'''
lost function:

L(alpha, beta, mu) = \sum_{i}\sum_{j}\sum_{tau} - log(mu_{ij} + alpha_{ij} beta_{ij} \M_{ij}(tau))
                           - \sum_{i}\sum_{j} (A element-wise product T)
                           + \sum+{i}\sum{j}(\mu_{ij}*X^i_{jk})
                           + k\sum_{i}\sum_{j} alpha_{ij}

stopping rule:
bFlag = 1: |f(t) - f(t-1)| <= tol
bFlag = 2: |f(t) - f(t-1)| <= tol * f(t-1)
bFlag = 3: run iterMax iterations
bFlag = 4: gradient does not change much
'''


def clusterhawkes(X, rho1, rho2, k, beta, maxIter, bFlag, prior=None, gamma_theta=None, gamma_k=None):
    num_student = len(X)
    num_hw = len(X[0][0, :])
    K = len(X[0][:, 0])
    U0 = np.ones((num_hw, num_student)) * 0.1
    # W0 = np.random.rand(num_hw, num_student)
    W0 = np.ones((num_hw, num_student))
    # U0 = np.random.rand(num_hw, num_student)
    eta = rho2 / rho1
    c = rho1 * eta * (1 + eta)
    M0 = identity(num_student) * k / num_student
    # bFlag = 1
    tol = 10 ** -16
    # maxIter = 10
    Wz = W0
    Wz_old = W0
    Mz = M0
    Mz_old = M0
    Uz = U0
    Uz_old = U0
    t = 1
    t_old = 0
    iter = 0
    gamma = 1
    gamma_inc = 1.5
    # beta = 0.7
    funcVal = []

    def compute_T(X):
        T_func = []
        for i in range(len(X)):
            Xi = X[i]
            Xik = np.array([X[i][:, -1]] * K)
            Ti = np.exp(-beta * np.subtract(Xik.T, Xi)) - 1
            # print(np.subtract(Xik.T, Xi))
            T_func.append(np.sum(Ti, axis=1))
        return np.array(T_func).T

    def compute_M(X):
        M = []
        for i in range(num_student):
            Mi = []
            for j in range(num_hw):
                Mi_hwj = [0]
                for tau in range(1, K):
                    Mi_hwj.append((1 + Mi_hwj[tau - 1]) * np.exp(-beta * (X[i][j, tau] - X[i][j, tau - 1])))
                Mi.append(Mi_hwj)
            M.append(np.array(Mi))
        return M

    def multi_transpose(X):
        XT = np.array([np.zeros(X[0].T.shape)] * num_student)
        for i in range(len(X)):
            XT[i] = X[i].T
        return XT

    def bsa_ihb(a, b, r, u, l):
        break_flag = 0
        t_l = np.divide(a, b)
        t_u = np.divide(np.subtract(a, u), b)
        T = np.concatenate((t_l, t_u), axis=0)
        t_L = float('inf')
        t_U = float('-inf')
        g_tL = 0
        g_tU = 0
        iteration = 0
        while len(T) > 0:
            iteration = iteration + 1
            g_t = 0
            t_hat = np.median(T)
            U = np.less(t_hat, t_u)
            M = (np.less_equal(t_u, t_hat)) & (np.less_equal(t_hat, t_l))
            if sum(U):
                g_t = g_t + np.sum(np.matmul(np.transpose(b[U]), u[U]))
            if sum(M):
                g_t = g_t + np.sum(np.multiply(b[M], (a[M] - t_hat * b[M])))

            if g_t > r:
                t_L = t_hat
                T = T[T > t_hat]
                g_tL = g_t
            elif g_t < r:
                t_U = t_hat
                T = T[T < t_hat]
                g_tU = g_t
            else:
                t_star = t_hat
                break_flag = 1
                break
        if break_flag == 0:
            t_star = t_L - (g_tL - r) * (t_U - t_L) / (g_tU - g_tL)
        x_star = np.minimum(np.maximum(l, a - t_star * b), u)
        return x_star, t_star, iteration

    def singular_projection(Msp, k):
        try:
            EValue, EVector = LA.eig(Msp)
        except TypeError:
            EValue, EVector = LA.eig(np.array(Msp, dtype=float))
        Pz = np.real(EVector)
        diag_EValue = np.real(np.diag(EValue)).diagonal()
        DiagSigz, _, _ = bsa_ihb(diag_EValue, np.ones(diag_EValue.shape), k, np.ones(diag_EValue.shape), 0)
        Mzp = np.matmul(np.matmul(Pz, np.diag(DiagSigz)), np.transpose(Pz))
        Mzp_Pz = Pz
        Mzp_DiagSigz = DiagSigz
        return Mzp, Mzp_Pz, Mzp_DiagSigz

    def gamma_pdf(k, theta, X):
        return 1 / math.factorial(k) / np.power(theta, k) * X ** (k - 1) * np.exp(-X / theta)

    def gamma_grad(gamma_k, gamma_theta, X):
        dom = np.sum(np.sum([gamma_pdf(k, theta, X) for k, theta in zip(gamma_k, gamma_theta)]))
        num = 1 / X * np.sum(np.array(gamma_k) - 1) - np.sum([1 / np.array(gamma_theta)])
        return num / dom

    def funVal_eval(W, U, M_Pz, M_DiagSigz):
        invIM = M_Pz * (np.diag(1. / (eta + M_DiagSigz))) * M_Pz
        invEtaMWt = np.matmul(invIM, W.T)
        f = 0
        for j in range(num_hw):
            for i in range(num_student):
                mu = U[j, i]
                alpha = W[j, i]
                f = f - np.sum(np.log(M_tau[i][j] * alpha * beta + mu)) + mu * X[i][j, -1]
        f = f - np.sum(np.multiply(W, T))
        f = f + c * np.trace(np.matmul(W, invEtaMWt))
        if prior == 'gamma':
            gamm_f = 1 / k * np.log(np.sum([gamma_pdf(k, theta, W) for k, theta in zip(gamma_k, gamma_theta)]))
            # for i in range(k):
            #     gamm_f = np.sum(-np.log(gamma_theta[k]**gamma_k[i])*math.factorial(gamma_k[i]) + (gamma_k[i] - 1 )*\
            #                     np.log(W) - W/gamma_theta[i])
            f = f + gamm_f
        return f

    def gradVal(W, U, M):
        IM = eta * identity(num_student).toarray() + M
        invEtaMWt = np.linalg.lstsq(IM, W.T, rcond=None)[0]
        WTW = np.matmul(W.T, W)
        WTWIM = np.linalg.lstsq(WTW, IM, rcond=None)[0]
        grad_M = - c * np.linalg.lstsq(WTWIM, IM, rcond=None)[0]
        grad_W = []
        grad_U = []
        for j in range(num_hw):
            g_wj = []
            g_uj = []
            for i in range(num_student):
                mu = U[j, i]
                alpha = W[j, i]

                g_wji = - np.sum(np.divide(beta * M_tau[i][j], mu + alpha * beta * M_tau[i][j])) - T[j, i]
                g_wj.append(g_wji)
                g_uji = - np.sum(1 / (mu + alpha * beta * M_tau[i][j])) + X[i][j, K - 1]
                g_uj.append(g_uji)
            grad_U.append(np.array(g_uj))
            grad_W.append(np.array(g_wj))
        grad_W = np.array(grad_W) + 2 * c * invEtaMWt.T
        grad_U = np.array(grad_U)

        f = 0
        for j in range(num_hw):
            for i in range(num_student):
                mu = U[j, i]
                alpha = W[j, i]

                f = f - np.sum(np.log(M_tau[i][j] * alpha * beta + mu)) + mu * X[i][j, -1]
        f = f - np.sum(np.multiply(W, T))
        f = f + c * np.trace(np.matmul(W, invEtaMWt))

        if prior == 'gamma':
            grad_W = grad_W + gamma_grad(gamma_k, gamma_theta, W)
            f = f + 1 / k * np.log(np.sum([gamma_pdf(k, theta, W) for k, theta in zip(gamma_k, gamma_theta)]))
        return grad_W, grad_U, grad_M, f

    X = multi_transpose(X)
    T = compute_T(X)
    # print(T[0])
    M_tau = compute_M(X)
    # print(T.shape)
    # print(T[0])
    while iter < maxIter:
        # print(iter)
        alpha = (t_old - 1) / t
        Ws = (1 + alpha) * Wz - alpha * Wz_old

        Ws[Ws <= 0] = 10 ** -4
        Ws[Ws >= 1] = 1

        Ws = np.array(Ws, dtype=float)
        Ms = (1 + alpha) * Mz - alpha * Mz_old
        Us = (1 + alpha) * Uz - alpha * Uz_old
        Us[Us <= 0] = 10 ** -4
        Us = np.array(Us, dtype=float)
        gWs, gUs, gMs, Fs = gradVal(Ws, Us, Ms)

        inneriter = 0

        while True:
            inneritermax = 300
            Fzp_list = []
            Wzp = Ws - gWs / gamma
            Wzp[Wzp <= 0] = 10 ** -4
            Wzp[Wzp >= 1] = 1
            Wzp = np.array(Wzp, dtype=float)
            Uzp = Us - gUs / gamma
            # print(Uzp, gUs / gamma)
            Uzp[Uzp <= 0] = 10 ** -4
            Uzp = np.array(Uzp, dtype=float)
            # print(Uzp)
            Mzp, Mzp_Pz, Mzp_DiagSigz = singular_projection(Ms - gMs / gamma, k)

            Fzp = funVal_eval(Wzp, Uzp, Mzp_Pz, Mzp_DiagSigz)
            delta_Wzs = Wzp - Ws
            delta_Mzs = Mzp - Ms
            delta_Uzs = Uzp - Us
            norm_Wzs = np.linalg.norm(delta_Wzs, 'fro') ** 2
            norm_Uzs = np.linalg.norm(delta_Uzs, 'fro') ** 2
            norm_Mzs = np.linalg.norm(delta_Mzs, 'fro') ** 2
            r_sum = (norm_Wzs + norm_Mzs + norm_Uzs) / 3
            Fzp_gamma = Fs + np.sum(np.multiply(delta_Wzs, gWs)) \
                        + np.sum(np.multiply(delta_Mzs, gMs)) \
                        + np.sum(np.multiply(delta_Uzs, gUs)) \
                        + norm_Wzs + norm_Mzs + norm_Uzs
            # print('r_sum: ')
            # print(r_sum)
            # print(Fzp - Fzp_gamma)
            Fzp_list.append(Fzp - Fzp_gamma)

            if r_sum <= np.finfo(float).eps:
                bFlag = 4
                # print('line 232, bflag==4')
                break
            if Fzp <= Fzp_gamma:
                # print('line 235, Fzp<Fzp_gamma')
                break
            if (len(Fzp_list) > 1):
                if abs(Fzp_list[-1] - Fzp_list[-2]) == 0:
                    break
            # if (gUs/gamma)[0,0]<=10**-52:
            #     break
            else:
                gamma = gamma * gamma_inc
                inneriter = inneriter + 1
                if inneriter == inneritermax:
                    break

                # print(gamma, inneriter)

        Wz_old = Wz
        Mz_old = Mz
        Uz_old = Uz

        Wz = Wzp
        Mz = Mzp
        Uz = Uzp
        funcVal.append(Fzp)
        # print('bflag =')
        # print(bFlag)
        # try:
        #     print(abs(funcVal[-2]-funcVal[-1]),funcVal[-1])
        # except IndexError:
        #     pass

        # print(funcVal)
        if bFlag == 4:
            break

        if bFlag == 1:
            # print('line'+str(202))
            if iter >= 2:
                # print('line'+str(204))
                # print(funcVal[-1], funcVal[-2])
                if abs(funcVal[-1] - funcVal[-2]) <= tol:
                    # print('line' + str(211))
                    break
        elif bFlag == 2:
            if iter >= 2:
                if abs(funcVal[-1] - funcVal[-2]) <= tol * funcVal[-1]:
                    # print('line' + str(211))
                    break
        # elif bFlag == 2:
        #     if funcVal[-1] <= tol:
        #         break
        elif bFlag == 3:
            if iter >= maxIter:
                break

        iter = iter + 1
        # print(iter)
        t_old = t
        t = 0.5 * (1 + (1 + 4 * t ** 2) ** 0.5)

    W = Wzp
    M = Mzp
    U = Uzp
    return W, funcVal, M, U
