from __future__ import division
import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
import math 
from math import sqrt
from tqdm import tqdm
from numba import cuda, jit, float32



#Ploting
def Simulation_plot(k, X, U, dit, txt,L0):
    
    
    fig, ax = plt.subplots(figsize=(12,12))

    ax.set_xlabel('X - axis', fontsize=14)
    ax.set_ylabel('Y - axis', fontsize=14)


    ax.scatter(X[k,:,0],X[k,:,1], linewidths=0.1, color='cyan')
    ax.quiver(X[k,:,0],X[k,:,1], U[k,:,0],U[k,:,1], color='blue')

    
#     for j in range(10):
#         ax.scatter(X[k-j,:,0],X[k-j,:,1], linewidths=1, color='blue')
        
#     for i in range(N):
#         ax.arrow(X[k,i,0],X[k,i,1], 2*U[k,i,0], 2*U[k,i,1],zorder=0, head_length=0.05, head_width=0.05, color='blue')
        
        
         
    fig.text(.5, .05, txt, ha='center',fontsize=14, linespacing=0.2)
   
    plt.xlim(0,L0)
    plt.ylim(0,L0)

    k= (k//dit)+1
    plt.savefig('Flock2D_PBC_Delay\August18\Frames/frame_VEM'+str(k)+'.png')
        
    plt.close()   
    return

#Calculation of the noise

def Add_noise(U, N, eta):
    
    noise = np.random.uniform(low=-eta, high=eta, size=(N,1)).astype('float32')
  
    for i in range(N):
        
        magnitude= np.sqrt(U[i,0]**2 + U[i,1]**2)
        theta = np.arctan2(U[i,1] , U[i,0]) + noise[i]
        U[i,1] = magnitude * np.sin(theta)
        U[i,0] = magnitude * np.cos(theta)
       
            
    return U

#Normalizing velocity
@cuda.jit
def normalized_U(a,out, v0): 
  
    i, j = cuda.grid(2)
    d1, d2 = cuda.gridsize(2)
  
    for i1 in range(i, a.shape[0], d1): 
        for i2 in range(j, a.shape[1], d2):
            out[i1,i2] = v0*a[i1,i2]/(sqrt(a[i1,0]**2 + a[i1,1]**2)+0.00001)

            
#Normalizing velocity
@cuda.jit
def limited_U(a, out, v_max): 

    i , j= cuda.grid(2)
    d1 , d2= cuda.gridsize(2)

    for i1 in range(i, a.shape[0], d1): 
        for i2 in range(j, a.shape[1], d2):
            if sqrt(a[i1,0]**2 + a[i1,1]**2) > v_max:
                out[i1,i2] = v_max*a[i1,i2]/sqrt(a[i1,0]**2 + a[i1,1]**2)
                           
                                
#Calculation of the Order Parameter        
@cuda.jit
def order_parameter(a,c,out,N): 
  
    i, j = cuda.grid(2)
    d1, d2= cuda.gridsize(2)

    for i1 in range(i, a.shape[0], d1):
        for i2 in range(j, a.shape[1], d2):
            c[i1,i2] = a[i1,i2]/(sqrt(a[i1,0]**2 + a[i1,1]**2)+0.00001)
            
    k = cuda.grid(1)
    d3 = cuda.gridsize(1)
    for j1 in range(k, a.shape[1], d3):
        sum=0
        for s in range(N):
            sum += c[s,j1]
        out[j1] = abs(sum/N)
        
#Calculation of the new position of the particles
@cuda.jit
def Update_X(a, U_in, out, L0,dt):

    i,j = cuda.grid(2)
    d1,d2 = cuda.gridsize(2)

    for i1 in range(i, a.shape[0], d1): 
        for i2 in range(j, a.shape[1], d2): 
            out[i1,i2] = (a[i1,i2] + U_in[i1,i2]*dt)%L0
            

#Calculation of alignment velocity       
@cuda.jit
def U_average(U, A, m, k, n, out):

    i, j = cuda.grid(2)
    d1, d2 = cuda.gridsize(2)

    for i1 in range(i, A.shape[0], d1):
        dm=m[i1]
        for i2 in range(j, U.shape[2], d2):
            
            sum = 0
            for s in range(A.shape[1]): 
                sum += A[i1, s]*U[dm,s ,i2]
            out[k,i1,i2]=(sum + U[k-1,i1,i2])/(n[i1,i2]+1)
            
            
#Calculation of the Adjacency Matrix
@cuda.jit
def Update_A(X, m, A_out, r, L0):

    i1, j1 = cuda.grid(2)
    d1,d2 = cuda.gridsize(2)
  
    for i in range(i1, X.shape[1], d1):
        dm=m[i]
        for j in range(j1, X.shape[1], d2):

            dis1 = sqrt((X[dm,i,0]-X[dm,j,0])**2 + (X[dm,i,1]-X[dm,j,1])**2)
            dis2 = sqrt((X[dm,i,0]-(X[dm,j,0]-L0))**2 + (X[dm,i,1]-X[dm,j,1])**2) 
            dis3 = sqrt((X[dm,i,0]-(X[dm,j,0]+L0))**2 + (X[dm,i,1]-X[dm,j,1])**2)
            dis4 = sqrt((X[dm,i,0]-(X[dm,j,0]-L0))**2 + (X[dm,i,1]-(X[dm,j,1]-L0))**2)
            dis5 = sqrt((X[dm,i,0]-(X[dm,j,0]+L0))**2 + (X[dm,i,1]-(X[dm,j,1]-L0))**2)
            dis6 = sqrt((X[dm,i,0]-X[dm,j,0])**2 + (X[dm,i,1]-(X[dm,j,1]-L0))**2) 
            dis7 = sqrt((X[dm,i,0]-(X[dm,j,0]+L0))**2 + (X[dm,i,1]-(X[dm,j,1]+L0))**2)
            dis8 = sqrt((X[dm,i,0]-X[dm,j,0])**2 + (X[dm,i,1]-(X[dm,j,1]+L0))**2)
            dis9 = sqrt((X[dm,i,0]-(X[dm,j,0]-L0))**2 + (X[dm,i,1]-(X[dm,j,1]+L0))**2)

            dismin=min(dis1, dis2, dis3, dis4, dis5, dis6, dis7, dis8, dis9)
            
            if i!=j and dismin <= r:
                A_out[i,j] = 1 
                
                
def count_n(A):

    n1=(A == 1).sum(axis=1)
    return n1


#Calculation of the Repulsion term
@cuda.jit
def U_repulsion(X, U, m, U_out, r_rep, c_rep, L0):
  
    i1, j1 = cuda.grid(2)

    d1,d2 = cuda.gridsize(2)
  
    for i in range(i1, X.shape[1], d1):
        dm=m[i]
        for j in range(j1, X.shape[1], d2):
            if i!=j:
                a1 = X[dm,i,0]-X[dm,j,0]
                b1 = X[dm,i,1]-X[dm,j,1]
                dis1 = sqrt(a1**2 + b1**2) + 0.00001

                a2 = X[dm,i,0]-(X[dm,j,0]-L0)
                b2 = X[dm,i,1]-X[dm,j,1]
                dis2 = sqrt(a2**2 + b2**2) 

                a3 = X[dm,i,0]-(X[dm,j,0]-L0)
                b3 = X[dm,i,1]-(X[dm,j,1]-L0)
                dis3 = sqrt(a3**2 + b3**2) 

                a4 = X[dm,i,0]-X[dm,j,0]
                b4 = X[dm,i,1]-(X[dm,j,1]-L0)
                dis4 = sqrt(a4**2 + b4**2)

                a5 = X[dm,i,0]-(X[dm,j,0]+L0)
                b5 = X[dm,i,1]-(X[dm,j,1]-L0)
                dis5 = sqrt(a5**2 + b5**2)  

                a6 = X[dm,i,0]-(X[dm,j,0]+L0)
                b6 = X[dm,i,1]-X[dm,j,1]
                dis6 = sqrt(a6**2 + b6**2) 

                a7=X[dm,i,0]-(X[dm,j,0]+L0)
                b7=X[dm,i,1]-(X[dm,j,1]+L0)
                dis7 = sqrt(a7**2 + b7**2) 

                a8=X[dm,i,0]-X[dm,j,0]
                b8=X[dm,i,1]-(X[dm,j,1]+L0)
                dis8 = sqrt(a8**2 + b8**2) 

                a9=X[dm,i,0]-(X[dm,j,0]-L0)
                b9=X[dm,i,1]-(X[dm,j,1]+L0)
                dis9 = sqrt(a9**2 + b9**2) 

                dismin=min(dis1, dis2, dis3, dis4, dis5, dis6, dis7, dis8, dis9)

      
                if dismin < 2*r_rep:

                    if dismin == dis1:
                        U[i,0] += c_rep*(2*r_rep - dis1) * a1/dis1
                        U[i,1] += c_rep*(2*r_rep - dis1) * b1/dis1

                    elif dismin ==dis2:
                        U[i,0] += c_rep*(2*r_rep - dis2) * a2/dis2
                        U[i,1] += c_rep*(2*r_rep - dis2) * b2/dis2

                    elif dismin == dis3:
                        U[i,0] += c_rep*(2*r_rep - dis3) * a3/dis3
                        U[i,1] += c_rep*(2*r_rep - dis3) * b3/dis3

                    elif dismin == dis4:
                        U[i,0] += c_rep*(2*r_rep - dis4) * a4/dis4
                        U[i,1] += c_rep*(2*r_rep - dis4) * b4/dis4

                    elif dismin == dis5:
                        U[i,0] += c_rep*(2*r_rep - dis5) * a5/dis5
                        U[i,1] += c_rep*(2*r_rep - dis5) * b5/dis5

                    elif dismin ==dis6:
                        U[i,0] += c_rep*(2*r_rep - dis6) * a6/dis6
                        U[i,1] += c_rep*(2*r_rep - dis6) * b6/dis6

                    elif dismin == dis7:
                        U[i,0] += c_rep*(2*r_rep - dis7) * a7/dis7
                        U[i,1] += c_rep*(2*r_rep - dis7) * b7/dis7

                    elif dismin == dis8:
                        U[i,0] += c_rep*(2*r_rep - dis8) * a8/dis8
                        U[i,1] += c_rep*(2*r_rep - dis8) * b8/dis8

                    elif dismin == dis9:
                        U[i,0] += c_rep*(2*r_rep - dis9) * a9/dis9
                        U[i,1] += c_rep*(2*r_rep - dis9) * b9/dis9
                        

    U_out=U
     
        
        
##########################       
###### MAIN FUNCTION #####
##########################
        
        
def calc_x_v(eta, rho, T, N, L0, mu, s, T0, r, r_rep, c_rep, v0, v_max,blocks_per_grid, threads_per_block,dt, dit):
    
    from numba import cuda, jit, float32
    txt='N='+str(N)+', L='+str(L0)+', eta='+str(np.round(eta,2))+', rho='+str(rho)+', r='+str(r)+', v0='+str(v0)+', c_rep='+str(c_rep)+ ', r_rep='+str(r_rep)+' ,mu='+str(mu)+', s='+str(s)+', T='+ str(T)
    
    
##########################       
###### INITIALIZING ######
##########################
    
    U=np.zeros((T,N,2), dtype=np.float32)
    X=np.zeros((T,N,2), dtype=np.float32)
    op=np.zeros((T,2), dtype=np.float32)


    #Defining particles initial positions
    X[0,:,:]=np.random.uniform(low=0, high=L0, size=(N,2)).astype('float32')
    
    theta0=np.random.uniform(low=0, high=2*np.pi, size=(N)).astype('float32')
    
    U[0,:,0]=v0*np.cos(theta0[:])
    U[0,:,1]=v0*np.sin(theta0[:])



    #Defining particles initial velocities
    #U[0,:,:]=np.random.normal(0, 1.0, size=(N,2)).astype('float32')


    #Defining particles delay time
    Dt= np.random.normal(mu,s, size=(N)).astype(int)


    #Normalization of the velocity vectors to the magnitude of the velocity(v0)
    U_temp=U[0,:,:]
    U_in = cuda.to_device(U_temp)
    U_out = cuda.to_device(U_in)

    normalized_U[blocks_per_grid, threads_per_block](U_in,U_out,v0)
    cuda.synchronize()
    U[0,:,:] = U_out.copy_to_host()

    #Calculation of initial order parameter
    
    #print("U0=", U)
    U_temp = U[0,:,:]
    op_temp = op[0,:]
    U_in = cuda.to_device(U_temp)
    c = cuda.to_device(U_in)
    op_out = cuda.to_device(op_temp)

    order_parameter[blocks_per_grid, threads_per_block](U_in,c,op_out, N)
    cuda.synchronize()
    op[0,:] = op_out.copy_to_host().astype('float32')
    
    #print("op0=", op)
    

##########################       
###### FIRST ROUND #######
##########################    
    
    for k in tqdm(range(1,T0)):

    #Updating particles velocities
        U[k,:,:] = U[k-1,:,:]

    #Updating particles positions 
        X_temp=X[k-1,:,:]
        U_temp=U[k,:,:]
        X_in = cuda.to_device(X_temp)
        X_out = cuda.to_device(X_in)
        U_in = cuda.to_device(U_temp)

        Update_X[blocks_per_grid, threads_per_block](X_in,U_in,X_out,L0,dt)
        X[k,:,:] = X_out.copy_to_host()

    #Ploting
        if dit==1:
            Simulation_plot(k,X,U,dit,txt,L0)
        elif k%dit==1:
            Simulation_plot(k,X,U,dit,txt,L0)
            
            
    #Calculation of order parameter
    
        #print("U"+str(k)+"=", U)
        U_temp = U[k,:,:]
        op_temp = op[k,:]
        U_in = cuda.to_device(U_temp)
        c = cuda.to_device(U_in)
        op_out = cuda.to_device(op_temp)

        order_parameter[blocks_per_grid, threads_per_block](U_in,c,op_out,N)
        cuda.synchronize()
        op[k,:] = op_out.copy_to_host().astype('float32')
        
        #print("op"+str(k)+"=" , op)

        
###########################       
###### SECOND ROUND #######
###########################

    A = np.zeros((N,N))
    m = np.zeros(N)

    for k in tqdm(range(T0, T)):

        m = abs((k-abs(Dt)).astype(int)-1)   #delay time for each particle.
        U_rep=np.zeros((N,2), dtype=np.float32)

    #Calculation of adjacency matrix
        #print("U"+str(k)+"=", U)
        if r > 0:

            X_in = cuda.to_device(X)
            A_out = cuda.to_device(A)

            Update_A[blocks_per_grid, threads_per_block](X_in, m, A_out,r, L0)
            cuda.synchronize()
            A = A_out.copy_to_host()
          
            n=count_n(A)

     #Calculation of align velocities

            ntiled = np.tile(n.reshape(N,1),(1,2))

            U_in = cuda.to_device(U)
            out = cuda.to_device(U_in)
            A_in = cuda.to_device(A)
            ntiled_in = cuda.to_device(ntiled)


            U_average[blocks_per_grid, threads_per_block](U_in, A_in, m,k, ntiled_in, out)
            cuda.synchronize()
            U = out.copy_to_host().astype('float32')
            
            #print("U-av"+str(k)+"=", U)

            U_temp=U[k,:,:]
            U_in = cuda.to_device(U_temp)
            U_out = cuda.to_device(U_in)

            normalized_U[blocks_per_grid, threads_per_block](U_in,U_out,v0)
            cuda.synchronize()
            U[k,:,:] = U_out.copy_to_host()

        else:
            U[k,:,:]=U[k-1,:,:]
            
        #print("A"+str(k)+"=", A)
        
    #Adding repulsion term to the velocities
        if r_rep > 0:

            X_in = cuda.to_device(X)
            U_in = cuda.to_device(U_rep)
            Urep_out = cuda.to_device(U_in)

            U_repulsion[blocks_per_grid, threads_per_block](X_in, U_in, m, Urep_out,r_rep, c_rep, L0)
            cuda.synchronize()
            U_rep = Urep_out.copy_to_host().astype('float32')
            
            #print("U_rep=" , U_rep)
            U[k,:,:] += U_rep[:,:]

            U_temp=U[k,:,:]
            U_in = cuda.to_device(U_temp)
            U_out = cuda.to_device(U_in)

            limited_U[blocks_per_grid, threads_per_block](U_in,U_out,v_max)
            cuda.synchronize()
            U[k,:,:] = U_out.copy_to_host().astype('float32')
        

    #Adding noise and normalization to the allowed maximum velocity
    
        if eta != 0:
            U_temp = U[k,:,:]
            U[k,:,:] = Add_noise(U_temp, N, eta)
      
        
    #Updating particles positions     

        X_temp=X[k-1,:,:]
        U_temp=U[k,:,:]
        X_in = cuda.to_device(X_temp)
        X_out = cuda.to_device(X_in)
        U_in = cuda.to_device(U_temp)

        Update_X[blocks_per_grid, threads_per_block](X_in,U_in,X_out,L0,dt)
        X[k,:,:] = X_out.copy_to_host() 
        


    #Ploting
    
        if dit==1:
            Simulation_plot(k,X,U,dit,txt,L0)
        elif k%dit==1:
            Simulation_plot(k,X,U,dit,txt,L0)


    #Calculation of order parameter
        #print("U"+str(k)+"=", U)
        U_temp = U[k,:,:]
        op_temp = op[k,:]
        U_in = cuda.to_device(U_temp)
        c = cuda.to_device(U_in)
        op_out = cuda.to_device(op_temp)

        order_parameter[blocks_per_grid, threads_per_block](U_in,c,op_out,N)
        cuda.synchronize()
        op[k,:] = op_out.copy_to_host().astype('float32')
        
        #print("op"+str(k)+"=" , op)
        
        
    #Calculating stationary order parameter
    op_norm=np.sqrt(np.sum(op**2, axis=1))
    sum=0
    for i in range(T-int(T/5), T):
        sum+=op_norm[i]
    op_stationary=np.round(sum/(int(T/5)), 3)
    print(op_stationary)
    
    return op , op_stationary

        
        
        
        