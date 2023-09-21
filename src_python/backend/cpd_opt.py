 #!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import time
import csv
import ctf
import random

# Normalize each column within a factor matrix
def normalize(X):
    # Re-normalize by iterating over each column of each factor matrix
    order = len(X)
    temp = X[0].to_nparray()    # choice of index 0 is arbitrary
    rank = len(temp[0,:])
    for j in range(rank):
        weight = 1
        # Iterate over the j'th column of all d factor matrices
        for k in range(order):
            temp = X[k].to_nparray()	# choice of index 0 is arbitrary
            nrm = la.norm(temp[:,j])
            weight *= nrm
            temp[:,j] /= nrm
            X[k] = ctf.from_nparray(temp)
        weight = weight**(1./order)
        for k in range(order):
            temp = X[k].to_nparray()        # choice of index 0 is arbitrary
            temp[:,j] *= weight
            X[k] = ctf.from_nparray(temp)
    return X


def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        print(xn,fxn,Dfxn,fxn/Dfxn)
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

def subtract_sparse(T,M):
    [inds,data] = T.read_local_nnz()
    [inds,data2] = M.read_local_nnz()

    new_data = data-data2
    new_tensor = ctf.tensor(T.shape, sp=T.sp)
    new_tensor.write(inds,new_data)
    return new_tensor

def elementwise_prod(T,M):
    [inds,data] = T.read_local_nnz()
    [inds,data2] = M.read_local_nnz()

    new_data= data2*data
    new_tensor = ctf.tensor(T.shape, sp=T.sp)
    new_tensor.write(inds,new_data)
    return new_tensor

def elementwise_exp(T):
    [inds,data] = T.read_local_nnz()
    new_data = np.exp(data)

    new_tensor = ctf.tensor(T.shape, sp=T.sp)
    new_tensor.write(inds,new_data)
    return new_tensor

def elementwise_log(T):
    [inds,data] = T.read_local_nnz()
    new_data = np.log(data)

    new_tensor = ctf.tensor(T.shape, sp=T.sp)
    new_tensor.write(inds,new_data)
    return new_tensor

class MLogQ2():
    #Current implementation is using \lambda  = e^m and replacing it in the function to get: e^m - xm
    def __init__(self,tenpy, T, Omega, A, tol, num_newton_iterations):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A
        self.tol = tol
        self.max_newton_iterations = num_newton_iterations

    def Get_RHS(self,num,regu,mu):
        M = self.tenpy.TTTP(self.Omega,self.A)
        Constant = M.copy()
        M_reciprocal1 = M.copy()
        M_reciprocal2 = M.copy()

        [inds,data] = M.read_local_nnz()
        new_data = -1./data
        M_reciprocal1.write(inds,new_data)
        new_data2 = 1./(data**2)			# Confirmed sign is correct
        M_reciprocal2.write(inds,new_data2)
        # Confirmed that swapping the negatives between new_data and new_data2 fails.

        [inds,t_data] = self.T.read_local_nnz()
        t_data = np.log(t_data / data)
        M.write(inds,t_data)
        ctf.Sparse_mul(M,M_reciprocal1)

        lst_mat = []
        for j in range(len(self.A)):
            if j != num :
                lst_mat.append(self.A[j])
            else:
                lst_mat.append(self.tenpy.zeros(self.A[num].shape))

        self.tenpy.MTTKRP(M,lst_mat,num)
        grad = lst_mat[num] + regu*self.A[num] - mu/2./self.A[num]

        [inds,data] = Constant.read_local_nnz()
        Constant.write(inds,np.ones(len(inds)) + t_data)
        ctf.Sparse_mul(Constant,M_reciprocal2)
        return [grad,Constant]

    def step(self,regu,barrier_start,barrier_stop,barrier_reduction_factor):
        newton_count = 0
        # Sweep over each factor matrix.
        for i in range(len(self.A)):
            lst_mat = []
            for j in range(len(self.A)):
                lst_mat.append(self.A[j].copy())
            # Minimize convex objective -> Newton's method
            # Reset barrier coefficient to starting value
            mu = barrier_start
            converge_list = np.ones(self.A[i].shape[0])
            converge_count = 0
            # Optimize factor matrix i by solving each row's nonlinear loss via multiple steps of Newtons method.
            while (mu >= barrier_stop):
                t=0
                prev_step_nrm = np.inf
                while (t<self.max_newton_iterations):
                    t += 1
                    [g,m] = self.Get_RHS(i,regu,mu)
                    grad_nrm = self.tenpy.vecnorm(g)

                    if self.tenpy.name() == "numpy": 
                        delta = self.tenpy.Solve_Factor(m,lst_mat,g,i,0,regu,mu)
                    else:
                        self.tenpy.Solve_Factor(m,lst_mat,g,i,0,regu,mu)
                        delta = lst_mat[i]
                    step_nrm = self.tenpy.vecnorm(delta)/self.tenpy.vecnorm(self.A[i])
                    """
                    if (step_nrm > 10*prev_step_nrm):
                        print("Break early due to large step: %f,%f"%(prev_step_nrm,step_nrm))
                        break
                    """
                    prev_step_nrm = step_nrm
                    # Verify that following update of factor matrix, every element is positive.

                    temp_update = self.A[i] - delta

                    [inds,data] = temp_update.read_local()
                    data[data<=0]=1e-6        # hacky reset
                    self.A[i].write(inds,data)
                    lst_mat[i] = self.A[i].copy()
                    """
                    while (np.any(data<=0)):
                        print("barrier val - ", mu)
                        print("newton iter - ", t)
                        print("updated factor matrix data - ", data)
                        assert(0)
                        [delta_inds,delta_data] = delta.read_local()
                        delta_data /= 2
                        delta.write(delta_inds,delta_data)
                        temp_update = self.A[i] - delta
                        [inds,data] = temp_update.read_local()
                        #data[data<=0]=1e-6        # hacky reset
                        #print("Neg values!")
                        #self.A[i].write(inds,data)
                        #break
                    #else:
                    """
                    #print(i,ii,mu,step_nrm)
                    if (step_nrm <= self.tol or converge_count == self.A[i].shape[0]):
                        break
                mu /= barrier_reduction_factor
                #print("Newton iteration %d: step_nrm - "%(t), step_nrm)
                #self.A[i] -= delta 
                newton_count += t
        return self.A,newton_count


class MLogQAbs():
    def __init__(self,tenpy, T, Omega, A, tol, num_newton_iterations):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A
        self.tol = tol
        self.max_newton_iterations = num_newton_iterations

    def Get_RHS(self,num,regu,mu):
        M = self.tenpy.TTTP(self.Omega,self.A)
        Constant = M.copy()
        M_reciprocal1 = M.copy()
        M_reciprocal2 = M.copy()

        [inds,data] = M.read_local_nnz()
        new_data = -1./data
        M_reciprocal1.write(inds,new_data)
        new_data2 = 1./(data**2)                        # Confirmed sign is correct
        M_reciprocal2.write(inds,new_data2)
        # Confirmed that swapping the negatives between new_data and new_data2 fails.

        [inds,t_data] = self.T.read_local_nnz()
        t_data = np.log(t_data / data)
        cc_data = np.abs(t_data)
        M.write(inds,t_data / cc_data)
        ctf.Sparse_mul(M,M_reciprocal1)

        lst_mat = []
        for j in range(len(self.A)):
            if j != num :
                lst_mat.append(self.A[j])
            else:
                lst_mat.append(self.tenpy.zeros(self.A[num].shape))

        self.tenpy.MTTKRP(M,lst_mat,num)
        grad = lst_mat[num] + 2*regu*self.A[num] - mu/self.A[num]

        [inds,data] = Constant.read_local_nnz()
        Constant.write(inds,(cc_data + t_data*(cc_data - t_data/cc_data))/cc_data**2)
        ctf.Sparse_mul(Constant,M_reciprocal2)
        return [grad,Constant]

    def step(self,regu,barrier_coeff,barrier_reduction_factor):
        newton_count = 0
        # Sweep over each factor matrix.
        for i in range(len(self.A)):
            lst_mat = []
            for j in range(len(self.A)):
                lst_mat.append(self.A[j].copy())
            # Minimize convex objective -> Newton's method
            # Reset barrier coefficient to starting value
            mu = barrier_coeff
            converge_list = np.ones(self.A[i].shape[0])
            converge_count = 0
            # Optimize factor matrix i by solving each row's nonlinear loss via multiple steps of Newtons method.
            prev_step_nrm = np.inf
            t=0
            while (t<self.max_newton_iterations):
                t += 1
                [g,m] = self.Get_RHS(i,regu,mu)
                grad_nrm = self.tenpy.vecnorm(g)

                if self.tenpy.name() == "numpy": 
                    delta = self.tenpy.Solve_Factor(m,lst_mat,g,i,regu)
                else:
                    self.tenpy.Solve_Factor(m,lst_mat,g,i,regu,mu)
                    delta = lst_mat[i]
                step_nrm = self.tenpy.vecnorm(delta)/self.tenpy.vecnorm(self.A[i])
                """
                if (step_nrm > 10*prev_step_nrm):
                    print("Break early due to large step: %f,%f"%(prev_step_nrm,step_nrm))
                    break
                """
                prev_step_nrm = step_nrm
                # Verify that following update of factor matrix, every element is positive.

                temp_update = self.A[i] - delta

                [inds,data] = temp_update.read_local()
                while (np.any(data<=0)):
                    print("barrier val - ", mu)
                    print("newton iter - ", t)
                    print("updated factor matrix data - ", data)
                    assert(0)
                    [delta_inds,delta_data] = delta.read_local()
                    delta_data /= 2
                    delta.write(delta_inds,delta_data)
                    temp_update = self.A[i] - delta
                    [inds,data] = temp_update.read_local()
                    #data[data<=0]=1e-6        # hacky reset
                    #print("Neg values!")
                    #self.A[i].write(inds,data)
                    #break
                #else:
                mu /= barrier_reduction_factor
                print("Newton iteration %d: step_nrm - "%(t), step_nrm)
                self.A[i] -= delta 
                lst_mat[i] = self.A[i].copy()
                if (step_nrm <= self.tol or converge_count == self.A[i].shape[0]):
                    break
            newton_count += t
        return self.A,newton_count

class MSE():
    def __init__(self,tenpy, T, Omega, A ):
        self.tenpy = tenpy
        self.T = T
        self.Omega = Omega
        self.A = A

    def Get_RHS(self,num,reg):
        # Note: reg has been adjusted via multiplication by nnz
        lst_mat = []
        for j in range(len(self.A)):
            if j != num :
                lst_mat.append(self.A[j])
            else:
                lst_mat.append(self.tenpy.zeros(self.A[num].shape))

        self.tenpy.MTTKRP(self.T,lst_mat,num)
        # Notice that grad should be negative, but it is not!
        #         This is taken into account when we subtract the step from the FM in 'step(..)'
        # Notice: no factors of 2. These are divided away automatically, as both the loss, regularization terms, etc. have them.
        grad = lst_mat[num] - reg*self.A[num]
        return grad

    def step(self,reg,nnz):
        # Sweep over each factor matrix, stored as an entry in list 'A'
        # This outer loop determines which factor matrix we are optimizing over.
        for i in range(len(self.A)):
            lst_mat = []
            # Extract all factor matrices for this optimization (i)
            for j in range(len(self.A)):
                if i != j :
                    lst_mat.append(self.A[j])
                else:
                    lst_mat.append(self.tenpy.zeros(self.A[i].shape))
            # Extract the rhs from a MTTKRP of the the sparse tensor T and all but the i'th factor matrix
            # MTTKRP - Matricized Tensor Times Khatri-Rao Product
            # The Tensor is T, and the Khatri-Rao Product is among all but the i'th factor matrix.
            g = self.Get_RHS(i,reg*nnz)
            if self.tenpy.name() == "numpy": 
                self.A[i] = self.tenpy.Solve_Factor(self.Omega,lst_mat,g,i,reg*nnz,0,0)
            else:
                self.tenpy.Solve_Factor(self.Omega,lst_mat,g,i,reg*nnz,0,0)
                self.A[i] = lst_mat[i]
        # Return the updated factor matrices
        return self.A


###############################################################################3
"""
2 Methods:
        1. Alternating Least-Squares (ALS)
        2. Alternating Minimization via Newtons Method (AMN)
"""


def cpd_als(error_metric, tenpy, T_in, O, X, reg,tol,max_nsweeps):
    assert(error_metric == "MSE")
    # X - model parameters, framed as a guess
    # O - sparsity pattern encoded as a sparse matrix
    # T_in - sparse tensor of data
    # Assumption is that the error is MSE with optional regularization
    if tenpy.name() == 'ctf':
        nnz = len(O.read_local_nnz()[0])
    else:
        nnz = np.sum(O)
    opt = MSE(tenpy, T_in, O, X)
    err=np.inf
    n_newton_iterations=0
    for i in range(max_nsweeps):
        # Tensor-times-tensor product with sparsity pattern and the factor matrices
        # The point of this product is to extract the relevant approximate entries
        #    from the contracted factor matrices. If not for the TTTP with the sparsity
        #    matrix, this operation would explode memory footprint.
        M = tenpy.TTTP(O,X)
        # M has same sparsity pattern as X, which has same sparsity pattern as T_in
        # Now, add M with T_in
        if tenpy.name() =='ctf':
            ctf.Sparse_add(M,T_in,beta=-1)
        else:
            M = T_in - M
        err = (tenpy.vecnorm(M)/np.sqrt(nnz))**2
        reg_loss = 0
        for j in range(len(X)):
            [inds,data] = X[j].read_local_nnz()
            #reg_loss += la.norm(np.log(data),2)**2
            reg_loss += la.norm(data,2)**2
        reg_loss *= reg
        #print("(Loss,Regularization component of objective,Objective) at AMN sweep %d: %f,%f,%f)"%(nsweeps,err,reg_loss,err+reg_loss))
        print("%d,%f,%f,%f)"%(i,err,reg_loss,err+reg_loss))
 
        if err < tol:
            break

        # Update model parameters X, one step at a time
        X = opt.step(reg,nnz)
        X = normalize(X)
    return (X,err,i)


def cpd_amn(error_metric,tenpy, T_in, O, X, reg, tol,\
            max_nsweeps, tol_newton, max_newton_iter, barrier_start, barrier_stop, barrier_reduction_factor):
    # Establish solver
    if (error_metric == "MLogQ2"):
        opt = MLogQ2(tenpy, T_in, O, X, tol_newton, max_newton_iter)
    elif (error_metric == "MLogQAbs"):
        opt = MLogQAbs(tenpy, T_in, O, X, tol_newton, max_newton_iter)
    else:
        assert(0)

    if tenpy.name() == 'ctf':
        nnz = len(O.read_local_nnz()[0])
    else:
        nnz = np.sum(O)

    reg *= nnz
    barrier_start *= nnz
    barrier_stop *= nnz

    err_prev = np.inf
    X_prev = []
    for i in range(len(X)):
        X_prev.append(X[i].copy())
    err=np.inf
    n_newton_iterations=0
    TT = T_in.copy()
    ctf.Sparse_log(TT)
    for i in range(max_nsweeps):
        M = tenpy.TTTP(O,X)
        P = M.copy()
        if error_metric == "MLogQ2" or error_metric == "MLogQAbs":
            ctf.Sparse_log(P)
        else:
            assert(0)
        if tenpy.name() =='ctf':
            ctf.Sparse_add(P,TT,alpha=-1)
        else:
            P = TT - P
        if error_metric == "MLogQ2":
            err = tenpy.vecnorm(P)**2/nnz
        elif error_metric == "MLogQAbs":
            err = tenpy.abs_sum(P)/nnz
        else:
            assert(0)
        reg_loss = 0
        for j in range(len(X)):
            [inds,data] = X[j].read_local_nnz()
            reg_loss += la.norm(data,2)**2
        reg_loss *= (reg/nnz)
        #print("(Loss,Regularization component of objective,Objective) at AMN sweep %d: %f,%f,%f)"%(i,err,reg_loss,err+reg_loss))
        print("%d,%f,%f,%f)"%(i,err,reg_loss,err+reg_loss))
        if (abs(err) > 10*abs(err_prev)):
            err = err_prev
            for j in range(len(X)):
                X[j] = X_prev[j].copy()
            break
        if (i>0 and abs(err)<tol):
            break
        P.set_zero()
        M.set_zero()
        err_prev = err
        for j in range(len(X)):
            X_prev[j] = X[j].copy()
        X,_n_newton_iterations = opt.step(reg,barrier_start,barrier_stop,barrier_reduction_factor)
        X = normalize(X)
        n_newton_iterations += _n_newton_iterations
    print("Break with %d AMN sweeps, %d total Newton iterations"%(i,n_newton_iterations))
    return (X,err,i,n_newton_iterations)
