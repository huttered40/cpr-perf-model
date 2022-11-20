 #!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import time
import csv
import ctf
import random

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

    def special_routine(self,niter,regu,barrier_coeff,barrier_reduction_factor):
        # First just get T
        Tdense = self.T.to_nparray()
        Odense = self.Omega.to_nparray()
	lst_mat = []
	for j in range(len(self.A)):
	    lst_mat.append(self.A[j].to_nparray())

        n_newton_iterations = 0
        n_newton_restarts = 0
        # Iterate over each factor matrix
        for i in range(len(lst_mat)):
            if (i==0): 
                factors = [1,2]
            elif (i==1): 
                factors = [0,2]
            elif (i==2): 
                factors = [0,1]
            # Iterate over each row of factor matrix i
            for j in range(lst_mat[i].shape[0]):
                # Newton method to update row j of factor matrix i
                #print("Initial Factor %d, Row %d: FM - "%(i,j), lst_mat[i][j,:])
                # Specify the derivative
                mu = barrier_coeff
                """
                if (i==0):
                    f = lambda x: (-2/x)*((Tdense[j,0,0]-np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][0,0]))+\
                                      (Tdense[j,0,1]-np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][1,0]))+\
                                      (Tdense[j,1,0]-np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][0,0]))+\
                                      (Tdense[j,1,1]-np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][1,0])))+\
                                      2*regu*x - mu/x
                    # Specify the Hessian
                    Df = lambda x: (-2/(x**2))*((-1-Tdense[j,0,0]+np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][0,0]))+\
                                      (-1-Tdense[j,0,1]+np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][1,0]))+\
                                      (-1-Tdense[j,1,0]+np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][0,0]))+\
                                      (-1-Tdense[j,1,1]+np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][1,0])))+\
                                      2*regu + mu/x**2
                elif (i==1):
                    f = lambda x: (-2/x)*((Tdense[0,j,0]-np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][0,0]))+\
                                      (Tdense[0,j,1]-np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][1,0]))+\
                                      (Tdense[1,j,0]-np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][0,0]))+\
                                      (Tdense[1,j,1]-np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][1,0])))+\
                                      2*regu*x - mu/x
                    # Specify the Hessian
                    Df = lambda x: (-2/(x**2))*((-1-Tdense[0,j,0]+np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][0,0]))+\
                                      (-1-Tdense[0,0,1]+np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][1,0]))+\
                                      (-1-Tdense[1,j,0]+np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][0,0]))+\
                                      (-1-Tdense[1,j,1]+np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][1,0])))+\
                                      2*regu + mu/x**2
                elif (i==2):
                    f = lambda x: (-2/x)*((Tdense[0,0,j]-np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][0,0]))+\
                                      (Tdense[0,1,j]-np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][1,0]))+\
                                      (Tdense[1,0,j]-np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][0,0]))+\
                                      (Tdense[1,1,j]-np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][1,0])))+\
                                      2*regu*x - mu/x
                    # Specify the Hessian
                    Df = lambda x: (-2/(x**2))*((-1-Tdense[0,0,j]+np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][0,0]))+\
                                      (-1-Tdense[0,1,j]+np.log(x*lst_mat[factors[0]][0,0] * lst_mat[factors[1]][1,0]))+\
                                      (-1-Tdense[1,0,j]+np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][0,0]))+\
                                      (-1-Tdense[1,1,j]+np.log(x*lst_mat[factors[0]][1,0] * lst_mat[factors[1]][1,0])))+\
                                      2*regu + mu/x**2
                lst_mat[i][j,:] = newton(f,Df,lst_mat[i][j,:],1e-6,20)
                #"""
                bad_count=0
                k=0
                while (k<50):
                    n_newton_iterations += 1
		    # Form Hessian H
		    H = np.zeros(shape=(lst_mat[i].shape[1],lst_mat[i].shape[1]))
                    H_ref = 0
		    # Form gradient f
		    f = 2*regu*lst_mat[i][j,:]
		    f_ref = 0
		    for k1 in range(lst_mat[factors[0]].shape[0]):
			for k2 in range(lst_mat[factors[1]].shape[0]):
			    temp = lst_mat[factors[0]][k1,:] * lst_mat[factors[1]][k2,:]
			    m = 0
			    for r in range(lst_mat[i].shape[1]):
				m += lst_mat[factors[0]][k1,r] * lst_mat[factors[1]][k2,r] * lst_mat[i][j,r]
                            t_val = 0
                            if (i==0): 
                                t_val = Tdense[j,k1,k2]
                                if (Odense[j,k1,k2]==0):
                                    continue
                            elif (i==1): 
                                t_val = Tdense[k1,j,k2]
                                if (Odense[k1,j,k2]==0):
                                    continue
                            elif (i==2): 
                                t_val = Tdense[k1,k2,j]
                                if (Odense[k1,k2,j]==0):
                                    continue
                            #print("t=%g\t\tm=%g\t\t(t_val-np.log(m))/m=%g\t\ttemp*(-2)*(t_val-np.log(m))/m=%g"%(t_val,m,(t_val-np.log(m))/m,temp*(-2)*(t_val-np.log(m))/m))
			    f += temp*(-2)*(t_val-np.log(m))/m
			    f_ref += (t_val-np.log(lst_mat[factors[0]][k1,0] * lst_mat[factors[1]][k2,0] * lst_mat[i][j,0]))
			    for r in range(lst_mat[i].shape[1]):
                                temp1 = lst_mat[factors[0]][k1,:] * lst_mat[factors[1]][k2,:]
                                temp1 *= (1+t_val-np.log(m))*lst_mat[factors[0]][k1,r] * lst_mat[factors[1]][k2,r]
                                temp1 *= (2./m**2)
                                H[:,r] += temp1
                            H_ref += (-1-t_val+np.log(lst_mat[factors[0]][k1,0] * lst_mat[factors[1]][k2,0] * lst_mat[i][j,0]))
		    for r in range(lst_mat[i].shape[1]):
			H[r,r] += 2*regu
                    f_ref /= lst_mat[i][j,0]
                    f_ref -= regu*lst_mat[i][j,0]
                    f_ref *= (-2)
                    # Below: log-barrier term
                    f_ref += ((-1.*mu)/lst_mat[i][j,0])
                    H_ref *= (-2./lst_mat[i][j,0]**2)
                    H_ref += 2*regu
                    # Below: log-barrier term
                    H_ref += (mu / (lst_mat[i][j,0]**2))

                    # Below: log-barrier term
                    f += ((-1.*mu)/lst_mat[i][j,:])
                    for r in range(lst_mat[i].shape[1]):
                        #barrier_update = mu*np.ones(lst_mat[i].shape[1]) / (lst_mat[i][j,:]**2)
                        barrier_update = np.zeros(lst_mat[i].shape[1])
                        barrier_update[r] = mu/(lst_mat[i][j,r]**2)
                        H[:,r] += barrier_update
		    # Solve system of linear equations
                    print("H - ", H)
                    #print("H_ref - ", H_ref)
                    print("f - ", f)
                    #print("f_ref - ", f_ref)
                    if (lst_mat[i].shape[1] == 1):
                        delta = 1./H[0,0] * f[0]
		        lst_mat[i][j,:] = lst_mat[i][j,:] - delta
                    else:
                        try:
		            delta = la.solve(H,f.reshape(f.shape[0],1))
			    # your code that will (maybe) throw
			except np.linalg.LinAlgError as e:
			    if 'Singular matrix' in str(e):
                                print("Singular matrix")
                                n_newton_restarts += 1
				# your error handling block
				k=0
				lst_mat[i][j,:] = 2**((-1)*bad_count)*np.ones(lst_mat[i][j,:].shape)
				for r in range(lst_mat[i].shape[1]):
				    lst_mat[i][j,r] /= 2**r
				bad_count += 1
				if (bad_count>20):
				    print("Out of retrys")
                                    return (n_newton_iterations,n_newton_restarts)
				mu = barrier_coeff
                                continue
			    else:
                                return (n_newton_iterations,n_newton_restarts)
				raise
		        lst_mat[i][j,:] = lst_mat[i][j,:] - delta[:,0]
		    # Update
                    #print("Factor %d, Row %d, Newton iteration %d: delta - "%(i,j,k), delta)
                    #print("Factor %d, Row %d, Newton iteration %d: FM - "%(i,j,k), lst_mat[i][j,:])
                    if (la.norm(delta)<1e-5):
                        #print("Converged!")
                        break
                    if (np.any(lst_mat[i][j,:]<0) or la.norm(delta)>10.):
                        n_newton_restarts += 1
                        #print("BAD INITIAL GUESS")
                        k=0
                        lst_mat[i][j,:] = 2**((-1)*bad_count)*np.ones(lst_mat[i][j,:].shape)
                        for r in range(lst_mat[i].shape[1]):
                            lst_mat[i][j,r] /= 2**r
                        bad_count += 1
                        if (bad_count>20):
                            print("Out of retrys")
                            return (n_newton_iterations,n_newton_restarts)
                        mu = barrier_coeff
                    else:
                        k += 1
                        mu /= barrier_reduction_factor
                #"""
	for j in range(len(self.A)):
	    self.A[j].from_nparray(lst_mat[j])
    
        return (n_newton_iterations,n_newton_restarts)

    def special_routine_opt(self,niter,regu,barrier_coeff,barrier_reduction_factor):
        # First just get T
        Tdense_inds,Tdense_nnz = self.T.read_local_nnz()
        Odense_inds,Odense_nnz = self.Omega.read_local_nnz()
	lst_mat = []
	for j in range(len(self.A)):
	    lst_mat.append(self.A[j].to_nparray())

        print(self.A)

        n_newton_iterations = 0
        n_newton_restarts = 0
	mu = barrier_coeff
        # Iterate over each factor matrix
        for i in range(len(lst_mat)):
            if (i==0): 
                factors = [1,2]
            elif (i==1): 
                factors = [0,2]
            elif (i==2): 
                factors = [0,1]

            # Outer loop over Newton's method
            bad_count_list=[0]*lst_mat[i].shape[0]
            barrier_list=[mu]*lst_mat[i].shape[0]
	    k=0
            converge_info=[False]*lst_mat[i].shape[0]
            converge_count=0
	    while (k<100):
                # Note: below is a different notion of #Newton iterations than method above
		n_newton_iterations += 1
                # Reset the Hessians and gradients
		Hessian_list = []
		gradient_list = []
		# Iterate over each row of factor matrix i
		for j in range(lst_mat[i].shape[0]):
		    Hessian_list.append(np.zeros(shape=(lst_mat[i].shape[1],lst_mat[i].shape[1])))
		    gradient_list.append(2*regu*lst_mat[i][j,:])
		# Iterate over all nnz to set up the Hessian,gradient lists
		for j in range(len(Odense_inds)):
		    # TODO: Note: this indexing might be wrong.
                    """
		    k0 = Tdense_inds[j]%lst_mat[0].shape[0]
		    k1 = (Tdense_inds[j]/lst_mat[0].shape[0])%lst_mat[1].shape[0]
		    k2 = Tdense_inds[j]/(lst_mat[0].shape[0]*lst_mat[1].shape[0])
                    """
		    k2 = Tdense_inds[j]%lst_mat[2].shape[0]
		    k1 = (Tdense_inds[j]/lst_mat[2].shape[0])%lst_mat[1].shape[0]
		    k0 = Tdense_inds[j]/(lst_mat[2].shape[0]*lst_mat[1].shape[0])
		    #print(k0,k1,k2)
		    list_indices = [k0,k1,k2]
                    if (i==0): 
		        temp = lst_mat[factors[0]][k1,:] * lst_mat[factors[1]][k2,:]
                    elif (i==1): 
		        temp = lst_mat[factors[0]][k0,:] * lst_mat[factors[1]][k2,:]
                    elif (i==2): 
		        temp = lst_mat[factors[0]][k0,:] * lst_mat[factors[1]][k1,:]
		    m = 0
		    for r in range(lst_mat[i].shape[1]):
			if (i==0): 
			    m += lst_mat[factors[0]][k1,r] * lst_mat[factors[1]][k2,r] * lst_mat[i][k0,r]
			elif (i==1): 
			    m += lst_mat[factors[0]][k0,r] * lst_mat[factors[1]][k2,r] * lst_mat[i][k1,r]
			elif (i==2): 
			    m += lst_mat[factors[0]][k0,r] * lst_mat[factors[1]][k1,r] * lst_mat[i][k2,r]
		    t_val = Tdense_nnz[j]
                    #print("check  it - ", m,np.log(m),t_val,t_val-np.log(m))
		    gradient_list[list_indices[i]] += temp*(-2)*(t_val-np.log(m))/m
		    for r in range(lst_mat[i].shape[1]):
			if (i==0):
			    temp1 = lst_mat[factors[0]][k1,:] * lst_mat[factors[1]][k2,:]
			    temp1 *= (1+t_val-np.log(m))*lst_mat[factors[0]][k1,r] * lst_mat[factors[1]][k2,r]
			    temp1 *= (2./m**2)
                            #print((2./m**2)*(1+t_val-np.log(m)))
                            #print((1+t_val-np.log(m))*(2./m**2))
			elif (i==1):
			    temp1 = lst_mat[factors[0]][k0,:] * lst_mat[factors[1]][k2,:]
			    temp1 *= (1+t_val-np.log(m))*lst_mat[factors[0]][k0,r] * lst_mat[factors[1]][k2,r]
			    temp1 *= (2./m**2)
			elif (i==2):
			    temp1 = lst_mat[factors[0]][k0,:] * lst_mat[factors[1]][k1,:]
			    temp1 *= (1+t_val-np.log(m))*lst_mat[factors[0]][k0,r] * lst_mat[factors[1]][k1,r]
			    temp1 *= (2./m**2)
			Hessian_list[list_indices[i]][:,r] += temp1
		    for r in range(lst_mat[i].shape[1]):
			Hessian_list[list_indices[i]][r,r] += 2*regu
		    # Below: log-barrier term
		    gradient_list[list_indices[i]] += ((-1.*barrier_list[list_indices[i]])/lst_mat[i][list_indices[i],:])
		    for r in range(lst_mat[i].shape[1]):
			#barrier_update = mu*np.ones(lst_mat[i].shape[1]) / (lst_mat[i][list_indices[i],:]**2)
			barrier_update = np.zeros(lst_mat[i].shape[1])
			barrier_update[r] = barrier_list[list_indices[i]]/(lst_mat[i][list_indices[i],r]**2)
			Hessian_list[list_indices[i]][:,r] += barrier_update
                print("gradient list - ", gradient_list)
                print("Hessian_list - ", Hessian_list)

                # Iterate over each row of factor matrix i
                for j in range(lst_mat[i].shape[0]):
                    if (converge_info[j]==True):
                        continue
		    # Solve system of linear equations
		    if (lst_mat[i].shape[1] == 1):
                        #print(i,j,k,gradient_list[j][0],Hessian_list[j][0,0])
			delta = 1./Hessian_list[j][0,0] * gradient_list[j][0]
			lst_mat[i][j,:] = lst_mat[i][j,:] - delta
		    else:
			try:
			    delta = la.solve(Hessian_list[j],gradient_list[j].reshape(gradient_list[j].shape[0],1))
			    # your code that will (maybe) throw
			except np.linalg.LinAlgError as e:
			    if 'Singular matrix' in str(e):
                                #all_valid = False
		                lst_mat[i][j,:] = 2**((-1)*bad_count_list[j])*np.ones(lst_mat[i][j,:].shape)
				for r in range(lst_mat[i].shape[1]):
				    lst_mat[i][j,r] /= 2**r
                                bad_count_list[j] += 1
                                barrier_list[j] = mu
                                continue
			    else:
				raise
			lst_mat[i][j,:] = lst_mat[i][j,:] - delta[:,0]
		    # Update
		    #print("Factor %d, Row %d, Newton iteration %d: delta - "%(i,j,k), delta)
		    #print("Factor %d, Row %d, Newton iteration %d: FM - "%(i,j,k), lst_mat[i][j,:])
                    print("delta - ", delta)
		    if (la.norm(delta)<1e-5):
			converge_info[j]=True
                        converge_count += 1
		    if (np.any(lst_mat[i][j,:]<0) or la.norm(delta)>10.):
                        #all_valid = False
		        lst_mat[i][j,:] = 2**((-1)*bad_count_list[j])*np.ones(lst_mat[i][j,:].shape)
			for r in range(lst_mat[i].shape[1]):
			    lst_mat[i][j,r] /= 2**r
                        bad_count_list[j] += 1
                        barrier_list[j] = mu
                        continue
                    else:
                        barrier_list[j] /= barrier_reduction_factor

                if (converge_count == lst_mat[i].shape[0]):
                    break
                k += 1
            #print("How many converged for factor matrix %d - %d"%(i,converge_count))

	for j in range(len(self.A)):
	    self.A[j].from_nparray(lst_mat[j])
    
        return (n_newton_iterations,n_newton_restarts)

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
                    #data[data<=0]=1e-6	# hacky reset
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
        new_data2 = 1./(data**2)			# Confirmed sign is correct
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
                    #data[data<=0]=1e-6	# hacky reset
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
	# 	This is taken into account when we subtract the step from the FM in 'step(..)'
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
                self.A[i] = self.tenpy.Solve_Factor(self.Omega,lst_mat,g,i,reg*nnz)
            else:
                self.tenpy.Solve_Factor(self.Omega,lst_mat,g,i,reg*nnz,0)
                self.A[i] = lst_mat[i]
        # Return the updated factor matrices
        return self.A


###############################################################################3
"""
2 Methods:
	1. Alternating Least-Squares (ALS)
	2. Alternating Minimization via Newtons Methods (AMN)
"""


def cpd_als(error_metric, tenpy, T_in, O, X, reg,tol,max_nsweeps):
    assert(error_metric == "MSE")
    # X - model parameters, framed as a guess
    # O - sparsity pattern encoded as a sparse matrix
    # T_in - sparse tensor of data
    # Assumption is that the error is MSE with optional regularization
    if tenpy.name() == 'ctf':
        nnz = np.sum(O.read_all())
    else:
        nnz = np.sum(O)
    opt = MSE(tenpy, T_in, O, X)
    err=np.inf
    n_newton_iterations=0
    for i in range(max_nsweeps+1):
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
        print("Loss %f at ALS sweep %d"%(err,i))
        
        if err < tol or i== max_nsweeps:
            break

        # Update model parameters X, one step at a time
        X = opt.step(reg,nnz)
    return (X,err,i,0)


def cpd_amn(error_metric,tenpy, T_in, O, X, reg, tol, max_iter_amn, tol_newton, max_newton_iter, barrier_start, barrier_reduction_factor):
    # Establish solver
    print(T_in)
    if (error_metric == "MLogQ2"):
        opt = MLogQ2(tenpy, T_in, O, X, tol_newton, max_newton_iter)
    elif (error_metric == "MLogQAbs"):
        opt = MLogQAbs(tenpy, T_in, O, X, tol_newton, max_newton_iter)
    else:
        assert(0)

    if tenpy.name() == 'ctf':
        nnz = np.sum(O.read_all())
    else:
        nnz = np.sum(O)

    reg *= nnz

    err_prev = np.inf
    X_prev = []
    for i in range(len(X)):
        X_prev.append(X[i].copy())
    err=np.inf
    barrier_coeff = barrier_start * nnz
    n_newton_iterations=0
    n_newton_restarts=0
    nsweeps = 0
    TT = T_in.copy()
    ctf.Sparse_log(TT)
    while(nsweeps<max_iter_amn):
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
        print("Loss %f at AMN sweep %d"%(err,nsweeps))
        if (abs(err) > 10*abs(err_prev)):
            err = err_prev
            for i in range(len(X)):
                X[i] = X_prev[i].copy()
            break
        if (nsweeps>0 and abs(err)<tol or i==max_iter_amn):
            break
        P.set_zero()
        M.set_zero()
        err_prev = err
        for i in range(len(X)):
            X_prev[i] = X[i].copy()
        #barrier_coeff /= barrier_reduction		# BAD IDEA
        X,_n_newton_iterations = opt.step(reg,barrier_coeff,barrier_reduction_factor)
        n_newton_iterations += _n_newton_iterations
        nsweeps += 1
    print("Break with %d AMN sweeps, %d total Newton iterations"%(nsweeps,n_newton_iterations))
    return (X,err,n_newton_iterations,n_newton_restarts)
