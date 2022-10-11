import ctf
from ctf import random
import time
import sys
glob_comm = ctf.comm()
import numpy as np
status_prints = False

def get_objective(T,U,V,W,omega,regParam):
    L = ctf.tensor(T.shape, sp=T.sp)
    L.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
    normL = ctf.vecnorm(L)
    if T.sp == True:
        RMSE = normL/(T.nnz_tot**.5)
    else:
        nnz_tot = ctf.sum(omega)
        RMSE = normL/(nnz_tot**.5)
    objective = normL + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * regParam
    return [objective, RMSE]

def run_CCD(T,U,V,W,omega,regParam,num_iter,time_limit,objective_frequency,use_MTTKRP=True):
    U_vec_list = []
    V_vec_list = []
    W_vec_list = []
    r = U.shape[1]
    for f in range(r):
        U_vec_list.append(U[:,f])
        V_vec_list.append(V[:,f])
        W_vec_list.append(W[:,f])

    ite = 0
    objectives = []
    t_obj_calc = 0.

    while True:
        R = ctf.copy(T)
        # R -= ctf.einsum('ijk, ir, jr, kr -> ijk', omega, U, V, W)
        R -= ctf.TTTP(omega, [U,V,W])
        # R += ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,0], V[:,0], W[:,0])
        R += ctf.TTTP(omega, [U[:,0], V[:,0], W[:,0]])
        if ite % objective_frequency == 0:
            [objective, RMSE] = get_objective(T,U,V,W,omega,regParam)
            objectives.append(objective)

        for f in range(r):
            # update U[:,f]
            if use_MTTKRP:
                alphas = ctf.tensor(R.shape[0])
                #ctf.einsum('ijk -> i', ctf.TTTP(R, [None, V_vec_list[f], W_vec_list[f]]),out=alphas)
                ctf.MTTKRP(R, [alphas, V_vec_list[f], W_vec_list[f]], 0)
            else:
                alphas = ctf.einsum('ijk, j, k -> i', R, V_vec_list[f], W_vec_list[f])

            if use_MTTKRP:
                betas = ctf.tensor(R.shape[0])
                #ctf.einsum('ijk -> i', ctf.TTTP(omega, [None, V_vec_list[f]*V_vec_list[f], W_vec_list[f]*W_vec_list[f]]),out=betas)
                ctf.MTTKRP(omega, [betas, V_vec_list[f]*V_vec_list[f], W_vec_list[f]*W_vec_list[f]], 0)
            else:
                betas = ctf.einsum('ijk, j, j, k, k -> i', omega, V_vec_list[f], V_vec_list[f], W_vec_list[f], W_vec_list[f])

            U_vec_list[f] = alphas / (regParam + betas)
            U[:,f] = U_vec_list[f]

            if glob_comm.rank() == 0 and status_prints == True:
                print('ctf.einsum() takes {}'.format(t1-t0))
                print('ctf.einsum() takes {}'.format(t2-t1))

            # update V[:,f]
            if glob_comm.rank() == 0 and status_prints == True:
                print('updating V[:,{}]'.format(f))
            if use_MTTKRP:
                alphas = ctf.tensor(R.shape[1])
                #ctf.einsum('ijk -> j', ctf.TTTP(R, [U_vec_list[f], None, W_vec_list[f]]),out=alphas)
                ctf.MTTKRP(R, [U_vec_list[f], alphas, W_vec_list[f]], 1)
            else:
                alphas = ctf.einsum('ijk, i, k -> j', R, U_vec_list[f], W_vec_list[f])

            if use_MTTKRP:
                betas = ctf.tensor(R.shape[1])
                #ctf.einsum('ijk -> j', ctf.TTTP(omega, [U_vec_list[f]*U_vec_list[f], None, W_vec_list[f]*W_vec_list[f]]),out=betas)
                ctf.MTTKRP(omega, [U_vec_list[f]*U_vec_list[f], betas, W_vec_list[f]*W_vec_list[f]], 1)
            else:
                betas = ctf.einsum('ijk, i, i, k, k -> j', omega, U_vec_list[f], U_vec_list[f], W_vec_list[f], W_vec_list[f])

            V_vec_list[f] = alphas / (regParam + betas)
            V[:,f] = V_vec_list[f]

            if glob_comm.rank() == 0 and status_prints == True:
                print('updating W[:,{}]'.format(f))
            if use_MTTKRP:
                alphas = ctf.tensor(R.shape[2])
                #ctf.einsum('ijk -> k', ctf.TTTP(R, [U_vec_list[f], V_vec_list[f], None]),out=alphas)
                ctf.MTTKRP(R, [U_vec_list[f], V_vec_list[f], alphas], 2)
            else:
                alphas = ctf.einsum('ijk, i, j -> k', R, U_vec_list[f], V_vec_list[f])

            if use_MTTKRP:
                betas = ctf.tensor(R.shape[2])
                #ctf.einsum('ijk -> k', ctf.TTTP(omega, [U_vec_list[f]*U_vec_list[f], V_vec_list[f]*V_vec_list[f], None]),out=betas)
                ctf.MTTKRP(omega, [U_vec_list[f]*U_vec_list[f], V_vec_list[f]*V_vec_list[f], betas], 2)
            else:
                betas = ctf.einsum('ijk, i, i, j, j -> k', omega, U_vec_list[f], U_vec_list[f], V_vec_list[f], V_vec_list[f])

            W_vec_list[f] = alphas / (regParam + betas)
            W[:,f] = W_vec_list[f]

            R -= ctf.TTTP(omega, [U_vec_list[f], V_vec_list[f], W_vec_list[f]])

            if f+1 < r:
                R += ctf.TTTP(omega, [U_vec_list[f+1], V_vec_list[f+1], W_vec_list[f+1]])
        ite += 1

    [objective, RMSE] = get_objective(T,U,V,W,omega,regParam)

    if glob_comm.rank() == 0:
        print('CCD amortized seconds per sweep: {}'.format(duration/ite))
        print('Time/CCD Iteration: {}'.format(duration/ite))
        print('Objective after',duration,'seconds (',ite,'iterations) is: {}'.format(objective))
        print('RMSE after',duration,'seconds (',ite,'iterations) is: {}'.format(RMSE))
