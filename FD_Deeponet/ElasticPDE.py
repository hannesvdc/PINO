import numpy as np
import scipy.linalg as lg

def counter_to_index(counter):
    N=100
    i = counter // (N+1)
    j = counter % (N+1)
    return (i,j)

def computeSystemMatrix(mu, N=100):
    A = np.zeros((2*N*(N+1), 2*N*(N+1)))
    for counter in range(N*(N+1)): # counter is the index of current (x_i, y_j) point
        (i,j) = counter_to_index(counter)

        # All center-cross variables
        counter_u = counter
        counter_u_left  = counter_u - (N+1)   if i != 0   else []
        counter_u_right = counter_u + (N+1)   if i != N-1 else counter_u
        counter_u_down  = counter_u - 1       if j != 0   else counter_u
        counter_u_up    = counter_u + 1       if j != N   else counter_u
        counter_v       = counter_u + N*(N+1)
        counter_v_left  = counter_v - (N+1) if i != 0   else []
        counter_v_right = counter_v + (N+1) if i != N-1 else counter_u
        counter_v_down  = counter_v - 1     if j != 0   else counter_u
        counter_v_up    = counter_v + 1     if j != N   else counter_u

        # All diagonal variables including all boundary conditions
        if i == 0:
            counter_v_up_left   = []
            counter_v_down_left = []
            counter_u_down_left = []
            counter_u_up_left   = []
        else:
            counter_v_up_left    = counter_v - N         if j != N else counter_v
            counter_u_up_left    = counter_u - N         if j != N else counter_u
            counter_v_down_left  = counter_v - (N + 2)   if j != 0 else counter_v
            counter_u_down_left  = counter_u - (N + 2)   if j != 0 else counter_u
        
        if i == N-1:
            counter_v_up_right  = counter_v
            counter_v_down_right = counter_v
            counter_u_up_right = counter_u
            counter_u_down_right = counter_u
        else:
            counter_v_down_right = counter_v + N   if j != 0 else counter_v
            counter_v_up_right   = counter_v + N + 2 if j != N else counter_v
            counter_u_down_right = counter_u + N   if j != 0 else counter_u
            counter_u_up_right   = counter_u + N + 2 if j != N else counter_u

        # Implement first equation
        row = counter
        A[row, counter_u_left] += 1.0
        A[row, counter] -= 2.0
        A[row, counter_u_right] += 1.0
        A[row, counter_u_up] += (1.0-mu)/2 * 1.0
        A[row, counter] -= (1.0-mu)/2 * 2.0
        A[row, counter_u_down] += (1.0-mu)/2 * 1.0
        A[row, counter_v_up_right] += (1.0+mu)/8
        A[row, counter_v_down_right] -= (1.0+mu)/8
        A[row, counter_v_up_left] -= (1.0+mu)/8
        A[row, counter_v_down_left] += (1.0+mu)/8

        # Implement second equation
        row = row + N*(N+1)
        A[row, counter_v_down] += 1.0
        A[row, counter_v] += -2.0
        A[row, counter_v_up] += 1.0
        A[row, counter_v_left] += (1.0-mu)/2 * 1.0
        A[row, counter_v] += -(1.0-mu)/2 * 2.0
        A[row, counter_v_right] += (1.0-mu)/2 * 1.0
        A[row, counter_u_up_right] += (1.0+mu)/8
        A[row, counter_u_down_right] -= (1.0+mu)/8
        A[row, counter_u_up_left] -= (1.0+mu)/8
        A[row, counter_u_down_left] += (1.0+mu)/8

    return A

def solveElasticPDE(lu, pivot, E, mu, f, N): # f has shape (N+1, 2)
    
    # Put the source term f into the right structure
    dx = 1.0 / N
    b = np.zeros(2*N*(N+1))
    b[N*(N+1)-(N+1):N*(N+1)] = -dx**2 * (1.0 - mu**2) / E * f[:,0]
    b[2*N*(N+1)-(N+1):]      = -dx**2 * (1.0 - mu**2) / E * f[:,1]
    
    # Solve the linear system with A
    print('\nSolving Linear System...')
    q = lg.lu_solve((lu, pivot), b)
    print('Done!')

    # Put q into two (N+1) x (N+1) matrices representing the grid of u and v
    u = np.zeros((N+1, N+1))
    v = np.zeros((N+1, N+1))
    u[:,1:] = np.reshape(q[0:N*(N+1)], (N+1,N), order='F')
    v[:,1:] = np.reshape(q[N*(N+1):], (N+1,N), order='F')

    # return the tuple (u, v)
    return (u, v)