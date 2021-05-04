# -*- coding: utf-8 -*-
"""
Channel Flow with Inlet Velocity

@author: asudha1
"""
#%%
""" RK Liu code (Numba) for contact angle hysteresis """
#%% Import Packages
import numpy as np
import time
import scipy.io as sio
import numba as nb
import pickle as pkl
import math 
#%% parameters
start=time.time() #Determine code execution time
lx=100 #grid dimensions
ly=100
ex= np.array([0.0, 1, 0,-1,  0, 1, -1, -1, 1]) #D2Q9
ey= np.array([0.0, 0, 1, 0, -1, 1,  1, -1, -1])
cc=1 #lattice speed
csqu= pow(cc,2)/3
#Weights
w1=4/9
w2=1/9
w3=1/36
wk=np.zeros(9)
wk[0]=w1
wk[1:5]=w2
wk[5:]=w3
#rsq is the length of the velocities,ie the direction,diagonal is sqrt(2)
rsq= np.zeros(9)
rsq[0]=0;rsq[1:5]=1;rsq[5:]=np.sqrt(2)
nw=1000 #Dump data every nw time steps
beta=0.7 #Parameter adjusting interface thickness
sigma= 0.01; #interface tension
a=25 #interface location
df= 0.000000015; #body force
alphar= 4/9; #parameters that determine initial density
alphab= 4/9;
rhori= 1; #initial density
rhobi= 1;
mp=10
tm= 100000; #max time steps
uib=0.04 #inlet velocity
uir=0.01
Uib=np.zeros((1,ly))
Uir=np.zeros((1,ly))
Uir[:,:]=10**-8
Uir[0,a:ly-a]=uir
# Uir[0,1:mp]=uir
Uib[0,0:a]=uib
Uib[0,ly-a:]=uib
Uib[0,a:ly-a]=10**-8
# Uib[:,:]=10**-8
# Uib[0,mp:ly-1]=uib
#Parameters for calculating pressure
csqr=3*(1-alphar)/5; csqb= 3*(1-alphab)/5;
theta= 0 #tan of contact angle
ta=180
tr=0
sta=50000 #stabilization iteration
#Wall BC weights
t0=0;t1=1./3;t2=1./12;
tp=np.array([t0,t1,t1,t1,t1,t2,t2,t2,t2])
rhow=0.5 #pressure
#%% Choose fluid/solid nodes
# @nb.njit
# def obsta():
#   obst= np.zeros((lx,ly))
#   obst[:,0]=1
#   obst[:,ly-1]=1
#   return obst
# obst=obsta()
#%% Choose fluid/solid nodes
@nb.njit
def obsta1():
    obst= np.zeros((lx,ly))
    obst[:,0]=1
    obst[:,ly-1]=1
    obst[0,1:ly-1]=0.5
    # obst[lx-1,1:a]=0.4
    # obst[lx-1,ly-a:ly-1]=0.4
    obst[lx-1,1:ly-1]=0.6
    # obst[0:2*a,a]=1.5
    # obst[0:2*a,ly-a]=1.5
    return obst
obst=obsta1()
#%% Initialize density and velocity
@nb.njit#(parallel=True)
def initialize():
    ux= np.zeros((lx,ly)) #x velocity
    uy= np.zeros((lx,ly)) #y velocity
    rhor= np.zeros((lx,ly)) #Density of red fluid
    rhob= np.zeros((lx,ly)) #Density of blue fluid
    rhor[0:,a:ly-a]=rhori # Red fluid in the middle
    # rhor[:,0:mp]=rhori
    # rhob[:,mp:ly]=rhobi
    rhob[0:,0:a]=rhobi
    rhob[0:,ly-a:]=rhobi
    rho=rhor+rhob
    # rho[lx-1,:]=rhow
    G= np.zeros((lx,ly)) # Gravity term
    G[:,:]=df
    return ux,uy,rhor,rhob,rho,G
[ux,uy,rhor,rhob,rho,G] = initialize()
#%% Equil Dist. Function
from numba.extending import overload

@overload(np.array)
def np_array_ol(x):
    if isinstance(x, nb.types.Array):
        def impl(x):
            return np.copy(x)
    return impl

#@nb.njit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8,f8,f8[:].f8,f8[:],f8[:],i8,i8)')
@nb.njit#(parallel=True)
def feq(rhor,rhob,ux,uy,alphar,alphab,wk,csqu,ex,ey,lx,ly):
    usqu= ux**2+uy**2
    un= np.zeros((9,lx,ly))
    fer=np.zeros((9,lx,ly))
    feb=np.zeros((9,lx,ly))
    for i in np.arange(0,9,1):
        un[i,:,:]=ex[i]*ux+ey[i]*uy
        if (i==0):
            fer[i,:,:]=wk[i]*rhor*(un[i,:,:]/csqu+ un[i,:,:]**2*4.5 -
                   usqu*1.5) +rhor*alphar
            feb[i,:,:]=wk[i]*rhob*(un[i,:,:]/csqu+ un[i,:,:]**2*4.5 -
                    usqu*1.5) +rhob*alphab

        elif (i>0 and i<5):
           fer[i,:,:]=wk[i]*rhor*(un[i,:,:]*3+ un[i,:,:]**2*4.5 -
                                usqu*1.5) +rhor*(1-alphar)/5
           feb[i,:,:]=wk[i]*rhob*(un[i,:,:]*3+ un[i,:,:]**2*4.5 -
                                 usqu*1.5) +rhob*(1-alphab)/5
        else:
             fer[i,:,:]=wk[i]*rhor*(un[i,:,:]*3+ un[i,:,:]**2*4.5 -
                                usqu*1.5) +rhor*(1-alphar)/20
             feb[i,:,:]=wk[i]*rhob*(un[i,:,:]*3+ un[i,:,:]**2*4.5 -
                                usqu*1.5) +rhob*(1-alphab)/20
    return fer,feb

# Equil. Dist. Fns
[fer,feb]=feq(rhor,rhob,ux,uy,alphar,alphab,wk,csqu,ex,ey,lx,ly)
#feq.parallel_diagnostics(level=4)
#Initial probability distribution of particles
fr= fer; fb=feb;
ff= fr+fb; #Overall prob dist.
#o = np.where(obst==1)
#rho[o]=1
#rhob[o]=rhobi
#rhor[o]=0
#%% Streaming Function
# @nb.njit#(parallel=True)
# def stream(f,lx,ly):
#     fs= np.zeros((9,lx,ly))
#     for x in np.arange(0,lx):
#         for y in np.arange(0,ly):
#             yn= y%(ly-1)+1
#             if (y==ly-1):
#                   yn=0
#             xe= x%(lx-1)+1
#             if (x==lx-1):
#                   xe=0
    
#             ys= ly-1-(ly-y)%(ly)
#             xw= lx-1-(lx-x)%(lx)
#     #Streaming Step
#             fs[1,xe,y]= f[1,x,y];
#             fs[2,x,yn]= f[2,x,y];
#             fs[3,xw,y]= f[3,x,y];
#             fs[4,x,ys]= f[4,x,y];
#             fs[5,xe,yn]= f[5,x,y];
#             fs[6,xw,yn]= f[6,x,y];
#             fs[7,xw,ys]= f[7,x,y];
#             fs[8,xe,ys]= f[8,x,y];
#     for x in np.arange(0,lx):
#         for y in np.arange(0,ly):
#     #Propagation
#             f[1:,x,y]=fs[1:,x,y]
#     return f
#%% Streaming Function
@nb.njit#(parallel=True)#(fastmath=True)
def stream(f,lx,ly):
    fs= np.copy(f)
    # fs=f1
    for x in np.arange(1,lx-1):
        for y in np.arange(1,ly-1):
            yn= y%(ly-1)+1
            if (y==ly-1):
                  yn=0
            xe= x%(lx-1)+1
            if (x==lx-1):
                xe=0
            ys= ly-1-(ly-y)%(ly)
            xw= lx-1-(lx-x)%(lx)
    #Streaming Step
            fs[1,xe,y]= f[1,x,y];
            fs[2,x,yn]= f[2,x,y];
            fs[3,xw,y]= f[3,x,y];
            fs[4,x,ys]= f[4,x,y];
            fs[5,xe,yn]=f[5,x,y];
            fs[6,xw,yn]= f[6,x,y];
            fs[7,xw,ys]= f[7,x,y];
            fs[8,xe,ys]= f[8,x,y];
            
    fs[1,1:lx,0]= f[1,0:lx-1,0];
    fs[2,0:lx,1]= f[2,0:lx,0];
    fs[3,0:lx-1,0]= f[3,1:lx,0];
    fs[5,1:lx,1]=f[5,0:lx-1,0];
    fs[6,0:lx-1,1]= f[6,1:lx,0];
    
    fs[1,1,:]= f[1,0,0:ly];
    fs[2,0,1:ly]= f[2,0,0:ly-1];
    fs[4,0,0:ly-1]= f[4,0,1:ly];
    fs[5,1,1:ly]=f[5,0,0:ly-1];
    fs[8,1,0:ly-1]= f[8,0,1:ly];
    
    fs[1,1:lx,ly-1]= f[1,0:lx-1,ly-1];
    fs[4,0:lx,ly-2]= f[4,0:lx,ly-1];
    fs[3,0:lx-1,ly-1]= f[3,1:lx,ly-1];
    fs[8,1:lx,ly-2]=f[8,0:lx-1,ly-1];
    fs[7,0:lx-1,ly-2]= f[7,1:lx,ly-1];
   
    fs[3,lx-2,0:ly]= f[3,lx-1,0:ly];
    fs[2,lx-1,1:ly]= f[2,lx-1,0:ly-1];
    fs[4,lx-1,0:ly-1]= f[4,lx-1,1:ly];
    fs[6,lx-2,1:ly]=f[6,lx-1,0:ly-1];
    fs[7,lx-2,0:ly-1]= f[7,lx-1,1:ly];
    
    for x in np.arange(0,lx):
        for y in np.arange(0,ly):
    #Propagation
            f[1:,x,y]=fs[1:,x,y]
    return f
#%% Periodic function
@nb.njit#(parallel=True)
def period(f):
    f[:, -1, :] = f[:, 1, :]
    f[:, 0, :] = f[:, -2, :]

    return f
#%% Bounceback function
@nb.njit#(parallel=True)
def bounceback(f,obst,o):
#    o1=np.asarray((o[0]))
    for o1 in np.arange(np.min(o[0]),np.max(o[0])+1):
        oe=o1%(lx-1)+1
        if (o1==lx-1):
            oe=0
        ow= lx-1-(lx-o1)%(lx)
        f[2,o1,0]=f[4,o1,1]
        f[5,ow,0]=f[7,o1,1]
        f[6,oe,0]=f[8,o1,1]
        f[4,o1,-1]=f[2,o1,-2]
        f[7,oe,-1]=f[5,o1,-2]
        f[8,ow,-1]=f[6,o1,-2]
    return f
#%% Bounceback function
@nb.njit#(parallel=True)#(fastmath=True)
def bounceback1(f,obst,o,ox,oo,ov,ui,ex,ey,ux,uy,wk,rho,rhow,csqu): 
    for o2 in np.arange(np.min(ox[1]),np.max(ox[1])+1): 
        #ox corresponds to inlet boundary points (y coordinates)
        on=o2+1
        # if (o2==ly-1): 
        #     on=0
        os= o2-1
        # if (o2==0):
        #     os=ly-1
      
        #Inlet, ui is the velocity at the inlet, 
        f[1,0,o2]=f[3,1,o2]-2*wk[3]*ex[3]*rho[1,o2]*ui[0,o2]/csqu
        f[5,0,o2]=f[7,1,on]-2*wk[7]*ex[7]*rho[1,o2]*ui[0,o2]/csqu
        f[8,0,o2]=f[6,1,os]-2*wk[6]*ex[6]*rho[1,o2]*ui[0,o2]/csqu
        # f[4,0,on]=f[2,1,o2]-2*wk[2]*ex[2]*rho[1,o2]*ui[0,o2]/csqu
        # f[2,0,os]=f[4,1,o2]-2*wk[4]*ex[4]*rho[1,o2]*ui[0,o2]/csqu

    for o2 in np.arange(np.min(oo[1]),np.max(oo[1])+1):   
        #oo corresponds to the points at the outlet 
        ux1= ux[lx-2,o2]+ 0.5*(ux[lx-2,o2]-ux[lx-3,o2]) # determination of velocities at the outlet
        uy1= uy[lx-2,o2]+ 0.5*(uy[lx-2,o2]-uy[lx-3,o2])
        usqu= np.sqrt(ux1**2+uy1**2)
        on=o2+1
        # if (o2==ly-1): 
        #     on=0
        os= o2-1
        # if (rho[lx-2,o2]>=0.5):
        #     rhow=0.5
        # else:
        #     rhow=10**-8
            
        #Outlet, rhow is the density (pressure) at the outlet
        f[3,lx-1,o2]=-f[1,lx-2,o2]+2*wk[1]*rhow*(1+(ex[1]*ux1+ey[1]*uy1)**2/(2*csqu**2)-usqu**2/(2*csqu))
        f[6,lx-1,o2]=-f[8,lx-2,on]+2*wk[8]*rhow*(1+(ex[8]*ux1+ey[8]*uy1)**2/(2*csqu**2)-usqu**2/(2*csqu))
        # f[4,lx-1,on]=-f[2,lx-2,o2]+2*wk[2]*rhow*(1+(ex[2]*ux1+ey[2]*uy1)/(2*csqu**2)-usqu**2/(2*csqu))
        # f[2,lx-1,os]=-f[4,lx-2,o2]+2*wk[4]*rhow*(1+(ex[4]*ux1+ey[4]*uy1)/(2*csqu**2)-usqu**2/(2*csqu))
        f[7,lx-1,o2]=-f[5,lx-2,os]+2*wk[5]*rhow*(1+(ex[5]*ux1+ey[5]*uy1)**2/(2*csqu**2)-usqu**2/(2*csqu))
    for o1 in np.arange(np.min(o[0])+1,np.max(o[0])):
        oe=o1%(lx-1)+1
        if (o1==lx-1):
            oe=0
        ow= lx-1-(lx-o1)%(lx)
        f[2,o1,0]=f[4,o1,1]
        f[5,ow,0]=f[7,o1,1]
        f[6,oe,0]=f[8,o1,1]
        # f[1,ow,0]=f[3,o1,1]
        # f[3,oe,0]=f[1,o1,1]
        f[4,o1,-1]=f[2,o1,-2]
        f[7,oe,-1]=f[5,o1,-2]
        f[8,ow,-1]=f[6,o1,-2]
        # f[1,ow,-1]=f[3,o1,-2]
        # f[3,oe,-1]=f[1,o1,-2]
    #bb for the vertical plates
    for o1 in np.arange(np.min(ov[0]),np.max(ov[0])+1):
        oe=o1%(lx-1)+1
        ow= lx-1-(lx-o1)%(lx)
        f[2,o1,a]=f[4,o1,a+1]
        f[5,o1,a]=f[7,oe,a+1]
        f[6,o1,a]=f[8,ow,a+1]
        f[2,o1,ly-a]=f[4,o1,ly-a+1]
        f[5,o1,ly-a]=f[7,oe,ly-a+1]
        f[6,o1,ly-a]=f[8,ow,ly-a+1]
        f[4,o1,a]=f[2,o1,a-1]
        f[7,o1,a]=f[5,ow,a-1]
        f[8,o1,a]=f[6,oe,a-1]
        f[4,o1,ly-a]=f[2,o1,ly-a-1]
        f[7,o1,ly-a]=f[5,ow,ly-a-1]
        f[8,o1,ly-a]=f[6,oe,ly-a-1]
    # #Edge of vertical plate
    # f[1,0,a]=f[3,1,a]
    # f[5,0,a]=f[7,1,a-1]
    # f[8,0,a]=f[6,1,a+1]
    # f[1,0,ly-a]=f[3,1,ly-a]
    # f[5,0,ly-a]=f[7,1,ly-a-1]
    # f[8,0,ly-a]=f[6,1,ly-a+1]
    #Corners    
    f[2,0,0]=f[4,0,1]
    f[1,0,0]=f[3,1,0]
    f[5,lx-2,0]=f[7,lx-1,1]
    f[6,1,0]=f[8,0,1]
    f[7,1,-1]=f[5,0,-2]
    f[8,-2,-1]=f[6,-1,-2]
    f[4,0,ly-1]=f[2,0,ly-2]
    f[1,0,ly-1]=f[3,1,ly-1]
    f[4,lx-1,ly-1]=f[2,lx-1,ly-2]
    f[3,lx-1,ly-1]=f[1,lx-2,ly-1]
    f[2,lx-1,0]=f[4,lx-1,1]
    f[3,lx-1,0]=f[1,lx-2,0]
    return f
#%% Zou's Bounceback function
@nb.njit#(parallel=True)#(fastmath=True)
def bouncezou(f,obst,o,ox,oo,ui,ex,ey,ux,uy,wk,rho,rhow,cc,csqu):
    for o1 in np.arange(np.min(o[0]),np.max(o[0]+1)):
        oe=o1%(lx-1)+1
        if (o1==lx-1):
            oe=0
        ow= lx-1-(lx-o1)%(lx)
        f[2,o1,0]=f[4,o1,1]
        f[5,ow,0]=f[7,o1,1]
        f[6,oe,0]=f[8,o1,1]
        # f[1,ow,0]=f[3,o1,1]
        # f[3,oe,0]=f[1,o1,1]
        f[4,o1,-1]=f[2,o1,-2]
        f[7,oe,-1]=f[5,o1,-2]
        f[8,ow,-1]=f[6,o1,-2]
        # f[1,ow,-1]=f[3,o1,-2]
        # f[3,oe,-1]=f[1,o1,-2]
    #Inlet zou
    rw=np.zeros((ly))
    for o2 in np.arange(np.min(ox[1]),np.max(ox[1])+1):
        rw[o2]= cc/(cc-ui[0,o2])*(f[0,0,o2]+f[2,0,o2]+f[4,0,o2]+2*(f[3,0,o2] \
                                                    +f[6,0,o2]+f[7,0,o2]))
        f[1,0,o2]=f[3,0,o2]+2/(3*cc)*rw[o2]*ui[0,o2]
        f[5,0,o2]=f[7,0,o2]+1/(6*cc)*rw[o2]*ui[0,o2]-0.5*(f[2,0,o2]-f[4,0,o2])
        f[8,0,o2]=f[6,0,o2]+1/(6*cc)*rw[o2]*ui[0,o2]+0.5*(f[2,0,o2]-f[4,0,o2])
    #Outlet zou
    for o2 in np.arange(np.min(oo[1]),np.max(oo[1])+1):
        # if (rho[lx-2,o2]>0.5):
        #     rhow=0.5
        # else:
        #     rhow=10**-8
        uw=(f[0,lx-1,o2]+f[2,lx-1,o2]+f[4,lx-1,o2]+2*(f[1,lx-1,o2]+f[5,lx-1,o2]+f[8,lx-1,o2]))/rhow -1
        f[3,lx-1,o2]=f[1,lx-1,o2]-2/(3*cc)*rhow*uw
        f[7,lx-1,o2]=f[5,lx-1,o2]-1/(6*cc)*rhow*uw+0.5*(f[2,lx-1,o2]-f[4,lx-1,o2])
        f[6,lx-1,o2]=f[8,lx-1,o2]-1/(6*cc)*rhow*uw-0.5*(f[2,lx-1,o2]-f[4,lx-1,o2])
    # rhow=0.5
    # Bottom inlet Corner
    f[1,0,0]=f[3,0,0]
    f[2,0,0]=f[4,0,0]
    f[5,0,0]=f[7,0,0]
    f[6,0,0]=0.5*(rho[1,1]-(f[0,0,0]+f[1,0,0]+f[2,0,0]+f[3,0,0]+f[4,0,0]+f[5,0,0]+f[7,0,0]))
    f[8,0,0]=f[6,0,0]
    # Top inlet corner
    f[1,0,ly-1]=f[3,0,ly-1]
    f[4,0,ly-1]=f[2,0,ly-1]
    f[8,0,ly-1]=f[6,0,ly-1]
    f[5,0,ly-1]=0.5*(rho[1,ly-2]-(f[0,0,ly-1]+f[1,0,ly-1]+f[2,0,ly-1]+f[3,0,ly-1]+f[4,0,ly-1]+f[6,0,ly-1]+f[8,0,ly-1]))
    f[7,0,ly-1]=f[5,0,ly-1]
    # Top outlet corner
    f[3,lx-1,ly-1]=f[1,lx-1,ly-1]
    f[4,lx-1,ly-1]=f[2,lx-1,ly-1]
    f[7,lx-1,ly-1]=f[5,lx-1,ly-1]
    f[6,lx-1,ly-1]=0.5*(rhow-(f[0,lx-1,ly-1]+f[1,lx-1,ly-1]+f[2,lx-1,ly-1]+f[3,lx-1,ly-1]+f[4,lx-1,ly-1]+f[5,lx-1,ly-1]+f[7,lx-1,ly-1]))
    f[8,lx-1,ly-1]=f[6,lx-1,ly-1]
    # Bottom outlet corner
    f[3,lx-1,0]=f[1,lx-1,0]
    f[2,lx-1,0]=f[4,lx-1,0]
    f[6,lx-1,0]=f[8,lx-1,0]
    f[5,lx-1,0]=0.5*(rhow-(f[0,lx-1,0]+f[1,lx-1,0]+f[2,lx-1,0]+f[3,lx-1,0]+f[4,lx-1,0]+f[6,lx-1,0]+f[8,lx-1,0]))
    f[7,lx-1,0]=f[5,lx-1,0]
    return f
#%% Bounceback for moving boundaries
@nb.njit#(fastmath=True)
def bouncewall(f,obst,o,ex,wk,rho,csqu):
        uw=0.0199
    #    o1=np.asarray((o[0]))
    #    o1=np.arange(np.min(o[0]),np.max(o[0])+1)
        o2=np.array([np.min(o[1]),np.max(o[1])])
#        o2=ly-1
#        for o1 in np.arange(1,lx-1):
        for o1  in np.arange(np.min(o[0]),np.max(o[0])+1):
                oe=o1%(lx-1)+1
                if (o1==lx-1):
                    oe=0
                ow= lx-1-(lx-o1)%(lx)
                f[2,o1,o2[0]]=f[4,o1,o2[0]+1]
                f[5,ow,o2[0]]=f[7,o1,o2[0]+1]
                f[6,oe,o2[0]]=f[8,o1,o2[0]+1]
                f[1,ow,o2[0]]=f[3,o1,o2[0]+1]
                f[3,oe,o2[0]]=f[1,o1,o2[0]+1]
                f[4,o1,o2[1]]=f[2,o1,o2[1]-1]-2*wk[2]*ex[2]*rho[o1,o2[1]-1]*uw/csqu
                f[7,oe,o2[1]]=f[5,o1,o2[1]-1]-2*wk[5]*ex[5]*rho[o1,o2[1]-1]*uw/csqu
                f[8,ow,o2[1]]=f[6,o1,o2[1]-1]-2*wk[6]*ex[6]*rho[o1,o2[1]-1]*uw/csqu
                f[1,ow,o2[1]]=f[3,o1,o2[1]-1]-2*wk[3]*ex[3]*rho[o1,o2[1]-1]*uw/csqu
                f[3,oe,o2[1]]=f[1,o1,o2[1]-1]-2*wk[1]*ex[1]*rho[o1,o2[1]-1]*uw/csqu
        return f
#%% Function for determining u,v,rho
@nb.njit#(parallel=True)
def getuv(obst,ux,uy,rhor,rhob,rho,fr,fb,theta):
            [x,y]= np.where(obst<1)
                 #Density derivatives
            rhon=np.zeros((lx,ly))
            dw1=np.zeros((lx,1))
            dw2=np.zeros((lx,1))
#    dwl1=np.zeros((lx,1))
#    dwl2=np.zeros((lx,1))
#            x= range(min(x),max(x+1))
#            y=range(min(y),max(y+1))
            for i in np.arange(np.min(x),np.max(x)+1):
                for j in np.arange(np.min(y),np.max(y)+1):
                  if(obst[i,j]==0):
                    rhor[i,j]= fr[0,i,j]+fr[1,i,j]+fr[2,i,j]+ fr[3,i,j]+ \
                      fr[4,i,j]+fr[5,i,j]+fr[6,i,j]+fr[7,i,j]+fr[8,i,j];
                    rhob[i,j]= fb[0,i,j]+fb[1,i,j]+fb[2,i,j]+fb[3,i,j]+ \
                    fb[4,i,j]+fb[5,i,j]+fb[6,i,j] +fb[7,i,j]+fb[8,i,j];
                    rho[i,j]=rhor[i,j]+rhob[i,j];
                    ux[i,j]= (fb[1,i,j]-fb[3,i,j]+fb[5,i,j]-fb[6,i,j]-fb[7,i,j] 
                    +fb[8,i,j]+fr[1,i,j]-fr[3,i,j]+fr[5,i,j]-fr[6,i,j]-fr[7,i,j] +fr[8,i,j])/rho[i,j]; 
                    uy[i,j]= (fb[2,i,j]-fb[4,i,j]+fb[5,i,j]+fb[6,i,j]-fb[7,i,j] 
                   -fb[8,i,j]+fr[2,i,j]-fr[4,i,j]+fr[5,i,j]+fr[6,i,j]-fr[7,i,j]-fr[8,i,j])/rho[i,j]
                    rhon[i,j]= (rhor[i,j]-rhob[i,j])/(rhor[i,j]+rhob[i,j])#phase field function
                  elif(obst[i,j]==1.5):
                    rhor[i,j]=rhori
                    rhob[i,j]=0
#            rhor[:,o[1]]=0
#            rhob[:,o[1]]=rhobi
            # rhon[0:2*a,a]= (rhon[0:2*a,a-1]+rhon[0:2*a,a+1])/2
            # rhon[0:2*a,ly-a]= (rhon[0:2*a,ly-a-1]+rhon[0:2*a,ly-a+1])/2
            rhon[0,1:ly-1]=rhon[1,1:ly-1]
            rhon[lx-1,1:ly-1]=rhon[lx-2,1:ly-1]
            p=rho/3 #pressure
 
            for i in np.arange(np.min(x),np.max(x)+1):
                ie= i%(lx-1)+1
                if (i==lx-1):
                    ie=0
                iw=lx-1-(lx-i)%(lx)
                #Central difference near boundary
                dw1[i,0]= (rhon[ie,1]-rhon[iw,1])/2
                dw2[i,0]=(rhon[ie,2]-rhon[iw,2])/2
        #        dwl1[i,0]= (rhon[ie,ly-2]-rhon[iw,ly-2])/2
        #        dwl2[i,0]= (rhon[ie,ly-3]-rhon[iw,ly-3])/2
                rhon[i,0]= rhon[i,1]+theta*np.abs(1.5*dw1[i,0]-0.5*dw2[i,0])
                rhon[i,ly-1]= rhon[i,ly-2]+theta*np.abs(1.5*dw1[i,0]-0.5*dw2[i,0])
            # Corners
            rhon[0,0]= (rhon[1,0]+rhon[0,1])/2
            rhon[0,ly-1]= (rhon[1,ly-1]+rhon[0,ly-2])/2
            rhon[lx-1,0]= (rhon[lx-2,0]+rhon[lx-1,1])/2
            rhon[lx-1,ly-1]= (rhon[lx-2,ly-1]+rhon[lx-1,ly-2])/2
            return rhor,rhob,rho,ux,uy,p,rhon

#%% Collision Function
@nb.njit#(parallel=True)
def collision(obst,ux,uy,rhor,rhob,rho,fr,fb,fer,feb,wk,ex,G,csqu):
    taur=1; #relaxation time
    taub=1
    #Parameters for relaxation parameter
    delt=0.98;
    alp= 2*taur*taub/(taur+taub)
    bet= 2*(taur-alp)/delt;
    kap= -bet/(2*delt);
    eta= 2*(alp-taub)/delt;
    kxi= eta/(2*delt);
    #Parameters for calculating pressure
    # csqr=3*(1-alphar)/5; csqb= 3*(1-alphab)/5;
    [x,y]= np.where(obst<1)
     #Density derivatives
    # dr=np.zeros((lx,ly))
    # db=np.zeros((lx,ly))
    # ddr=np.zeros((lx,ly))
    # ddb=np.zeros((lx,ly))
    # cr=np.zeros((lx,ly))
    # cb=np.zeros((lx,ly))
    ff=np.zeros((9,lx,ly))
    # for i in np.arange(np.min(x),np.max(x)+1):
    #             for j in np.arange(np.min(y),np.max(y)+1):
    #                 #Density Derivatives for eliminating the unwanted terms
    #                 dr[i,j]= 0.5*(rhor[i,j+1]-rhor[i,j-1])
    #                 db[i,j]= 0.5*(rhob[i,j+1]-rhob[i,j-1])
    # for i in np.arange(np.min(x ),np.max(x)+1):
    #             for j in np.arange(np.min(y),np.max(y)+1):
    #                 #Second Density Derivative
    #                 ddr[i,j]= 0.5*(dr[i,j+1]*ux[i,j+1]-dr[i,j-1]*ux[i,j-1]);
    #                 ddb[i,j]= 0.5*(db[i,j+1]*ux[i,j+1]-db[i,j-1]*ux[i,j-1]);
    for i in np.arange(np.min(x),np.max(x)+1):
                for j in np.arange(np.min(y),np.max(y)+1):
                   if(obst[i,j]==0): 
                    phi= (rhor[i,j]-rhob[i,j])/(rhor[i,j]+rhob[i,j])#Relaxation parameter as a function of position
                    #Relaxation time
                    if (phi>delt):
                       tau=taur
                    elif ((phi>0) & (phi<=delt)):
                        tau= alp+bet*phi+kap*phi**2
                    elif ((phi<=0) & (phi>=-delt)):
                        tau= alp+eta*phi+kxi*phi**2
                    elif (phi<-delt):
                        tau=taub
                    
                    # cr[i,j]=(taur-0.5)*(1/3-csqr)*ddr[i,j];
                    # cb[i,j]= (taub-0.5)*(1/3-csqb)*ddb[i,j];
                    for n in np.arange(0,9):
                        fr[n,i,j]= fer[n,i,j]+(1-1/tau)*(fr[n,i,j]-fer[n,i,j])#-cr[i,j]*wk[n]*ex[n]/csqu
                        fb[n,i,j]= feb[n,i,j]+(1-1/tau)*(fb[n,i,j]-feb[n,i,j])#-cb[i,j]*wk[n]*ex[n]/csqu
                        ff[n,i,j]= fr[n,i,j]+fb[n,i,j]#+wk[n]*ex[n]*G[i,j]/csqu
    return fr,fb,ff
#%% Perturbation Redistribution Function
@nb.njit#(parallel=True)
def redistribute(obst,csqu,ux,uy,rhor,rhob,rho,rhon,sigma,fr,fb,ff,ex,ey,lx,ly,wk,rsq,beta):
            [x,y]= np.where((obst!=1))
            fc= np.zeros((lx,ly))
            #Unit normal and derivative 
            dx= np.zeros((lx,ly))
            dy= np.zeros((lx,ly))
            nx= np.zeros((lx,ly))
            ny= np.zeros((lx,ly))
            dxnx=np.zeros((lx,ly))
            dynx=np.zeros((lx,ly))
            dxny=np.zeros((lx,ly))
            dyny=np.zeros((lx,ly))
            coslam= np.zeros((9,lx,ly)) #angle between color gradient and direction
#            eqf= np.zeros((9,lx,ly))
            un=np.zeros((9,lx,ly))
            Fn=np.zeros((9,lx,ly))
            Fu=np.zeros((lx,ly)) 
            upx=np.zeros((lx,ly))
            upy=np.zeros((lx,ly))
            Fix=np.zeros((lx,ly))
            Fiy=np.zeros((lx,ly))
            for i in np.arange(np.min(x),np.max(x)+1):
                for j in np.arange(np.min(y),np.max(y)+1):
                 if(obst[i,j]==0): 
                    jn= j%(ly-1)+1
                    ie= i%(lx-1)+1
                    if (i==lx-1):
                        ie=0
                    js= ly-1-(ly-j)%(ly)
                    iw= lx-1-(lx-i)%(lx)
                    # Gradient of phase field function
                    dx[i,j] = 1/csqu*(wk[1]*ex[1]*rhon[ie,j]+wk[3]*ex[3]*rhon[iw,j]+ wk[5]*ex[5]*rhon[ie,jn]+ \
              wk[6]*ex[6]*rhon[iw,jn]+wk[7]*ex[7]*rhon[iw,js]+wk[8]*ex[8]*rhon[ie,js])
                    dy[i,j] = 1/csqu*(wk[2]*ey[2]*rhon[i,jn]+wk[4]*ey[4]*rhon[i,js]+ \
             wk[5]*ey[5]*rhon[ie,jn]+wk[6]*ey[6]*rhon[iw,jn]+wk[7]*ey[7]*rhon[iw,js]+wk[8]*ey[8]*rhon[ie,js])
                    fc[i,j]= np.sqrt(dx[i,j]**2+dy[i,j]**2)
                    if (fc[i,j]>10**-8):
                            nx[i,j]= -dx[i,j]/fc[i,j]
                            ny[i,j]= -dy[i,j]/fc[i,j]
                            
            for i in np.arange(np.min(x),np.max(x)+1):
                for j in np.arange(np.min(y),np.max(y)+1):
                 if(obst[i,j]==0): 
                    jn= j%(ly-1)+1
                    ie= i%(lx-1)+1
                    if (i==lx-1):
                        ie=0
                    js= ly-1-(ly-j)%(ly)
                    iw= lx-1-(lx-i)%(lx)
                            #Second derivative for curvature
                    dxnx[i,j] = 1/csqu*(wk[1]*ex[1]*nx[ie,j]+wk[3]*ex[3]*nx[iw,j]+ wk[5]*ex[5]*nx[ie,jn]+ \
                    wk[6]*ex[6]*nx[iw,jn]+wk[7]*ex[7]*nx[iw,js]+wk[8]*ex[8]*nx[ie,js])
                    dyny[i,j] = 1/csqu*(wk[2]*ey[2]*ny[i,jn]+wk[4]*ey[4]*ny[i,js]+ \
                    wk[5]*ey[5]*ny[ie,jn]+wk[6]*ey[6]*ny[iw,jn]+wk[7]*ey[7]*ny[iw,js]+wk[8]*ey[8]*ny[ie,js])
                    dynx[i,j] = 1/csqu*(wk[2]*ey[2]*nx[i,jn]+wk[4]*ey[4]*nx[i,js]+ \
                    wk[5]*ey[5]*nx[ie,jn]+wk[6]*ey[6]*nx[iw,jn]+wk[7]*ey[7]*nx[iw,js]+wk[8]*ey[8]*nx[ie,js])
                    dxny[i,j] = 1/csqu*(wk[1]*ex[1]*ny[ie,j]+wk[3]*ex[3]*ny[iw,j]+ wk[5]*ex[5]*ny[ie,jn]+ \
                    wk[6]*ex[6]*ny[iw,jn]+wk[7]*ex[7]*ny[iw,js]+wk[8]*ex[8]*ny[ie,js])
                             #Curvature
                    K= nx[i,j]*ny[i,j]*(dynx[i,j]+dxny[i,j])-nx[i,j]**2*dyny[i,j]-ny[i,j]**2*dxnx[i,j]
                    Fix[i,j]= -0.5*sigma*K*dx[i,j]
                    Fiy[i,j]= -0.5*sigma*K*dy[i,j]
                    upx[i,j]= ux[i,j]+Fix[i,j]/2/rho[i,j]
                    upy[i,j]= uy[i,j]+Fiy[i,j]/2/rho[i,j]
                    Fu[i,j]=Fix[i,j]*upx[i,j]+Fiy[i,j]*upy[i,j]
                    
                    for n in np.arange(0,9):
                        un[n,i,j]=ex[n]*upx[i,j]+ey[n]*upy[i,j]
                        Fn[n,i,j]=ex[n]*Fix[i,j]+ey[n]*Fiy[i,j]
                        #Cases for denominator=0
                        if (fc[i,j]<10**-8 and fc[i,j]>=0):
                          fr[n,i,j]= ff[n,i,j]*rhor[i,j]/rho[i,j]
                          fb[n,i,j]= ff[n,i,j]*rhob[i,j]/rho[i,j]
                        else:
                            if (n==0):
                              ff[n,i,j]= ff[n,i,j]+wk[n]*(( Fn[n,i,j]-Fu[i,j])/csqu+ \
                                (Fix[i,j]*ex[n]*un[n,i,j]+Fiy[i,j]*ey[n]*un[n,i,j])/csqu**2)
                              fr[n,i,j]= rhor[i,j]*ff[n,i,j]/rho[i,j];
                              fb[n,i,j]= rhob[i,j]*ff[n,i,j]/rho[i,j];
                            else:
                              coslam[n,i,j]= (ex[n]*dx[i,j]+ ey[n]*dy[i,j])/(rsq[n]*fc[i,j]); 
                              ff[n,i,j]= ff[n,i,j]+wk[n]*(( Fn[n,i,j]-Fu[i,j])/csqu+ \
                                (Fix[i,j]*ex[n]*un[n,i,j]+Fiy[i,j]*ey[n]*un[n,i,j])/csqu**2)
                              tem= rhor[i,j]*rhob[i,j]/(rho[i,j]);
                          #Redistributed f's
#                              eqf[n,i,j]= wk[n]*rho[i,j];
                              fr[n,i,j]= rhor[i,j]*ff[n,i,j]/rho[i,j]+ beta*tem*wk[n]*coslam[n,i,j];
                              fb[n,i,j]= rhob[i,j]*ff[n,i,j]/rho[i,j]-beta*tem*wk[n]*coslam[n,i,j];
                          
            return fr,fb
#%% Output function
@nb.jit#(parallel=True)
def results(lx,ly,obst,rhor,rhob,rho,ux,uy,p):
     x=np.array([range(0,lx)])
     y=np.array([range(0,ly)])
     print('Iter',t)
     dict={'x':x,'y':y,'rhor':rhor,'rhob':rhob,'rho':rho,'ux':ux,'uy':uy,'p':p,'obst':obst}
     fname= "Itera%s.mat" % t
     sa=sio.savemat(fname,dict)
     return sa
#%% Iteration
o=np.where(obst==1)
ox1=np.where((obst==0.5))
oo2=np.where((obst==0.4))
oo=np.where(obst==0.6)
ov=np.where(obst==1.5)
rhow1=10**-8
for t in range(0,sta+1):
            #Streaming
            fr= stream(fr,lx,ly)
            fb= stream(fb,lx,ly)
              #Bounceback
            # fr=bouncezou(fr,obst,o,ox1,oo,Uir,ex,ey,ux,uy,wk,rhor,rhow1,cc,csqu)
            # fb= bouncezou(fb,obst,o,ox1,oo,Uib,ex,ey,ux,uy,wk,rhob,rhow,cc,csqu)
#Determine u,v,rho
            [rhor,rhob,rho,ux,uy,p,rhon]= getuv(obst,ux,uy,rhor,rhob,rho,fr,fb,theta)
            [fer,feb]=feq(rhor,rhob,ux,uy,alphar,alphab,wk,csqu,ex,ey,lx,ly);
            #collision
            [fr,fb,ff]=collision(obst,ux,uy,rhor,rhob,rho,fr,fb,fer,feb,wk,ex,G,csqu)
            #Redistribution
            [fr,fb]=redistribute(obst,csqu,ux,uy,rhor,rhob,rho,rhon,sigma,fr,fb,ff,ex,ey,lx,ly,wk,rsq,beta)
            #   #Bounceback
            fr=bounceback1(fr,obst,o,ox1,oo,ov,Uir,ex,ey,ux,uy,wk,rho,rhow,csqu)
            fb= bounceback1(fb,obst,o,ox1,oo,ov,Uib,ex,ey,ux,uy,wk,rho,rhow,csqu)
            # fr=bounceback(fr,obst,o)
            # fb=bounceback(fb,obst,o)
            #Periodic BC
            # fr=period(fr)
            # fb=period(fb)
            #Store data after nw iterations
            if (t%(50*nw)==0):
              with open('temp1.pkl','wb') as s:
                    pkl.dump([rhor,rhob,rhon],s)
              results(lx,ly,obst,rhor,rhob,rho,ux,uy,p) 
#%% Hysteresis
# with open('temp1.pkl','rb') as s:
#     rhor,rhob,rhon=pkl.load(s)
# #[i,j]= np.where(obst==0)
# rhor= (1+rhon)/2
# rhob= (1-rhon)/2
# ux=np.zeros((lx,ly))
# uy=np.zeros((lx,ly))
# obst=obsta1()
# #ux[:,ly-1]= 0.00512
# # Equil. Dist. Fns
# [fer,feb]=feq(rhor,rhob,ux,uy,alphar,alphab,wk,csqu,ex,ey,lx,ly)
# #Initial probability distribution of particles
# fr= fer; fb=feb;
# ff= fr+fb; #Overall prob dist.
# ox=np.where(obst==0.5)
# # ox2=np.where(obst==0.4)
# oo=np.where(obst==0.6)
# for t in range(0,tm+1):
#             #Streaming
#             fr= stream(fr,lx,ly)
#             fb= stream(fb,lx,ly)
# #Determine u,v,rho
#             [rhor,rhob,rho,ux,uy,p,rhon]= getuv(obst,ux,uy,rhor,rhob,rho,fr,fb,theta)
#             [fer,feb]=feq(rhor,rhob,ux,uy,alphar,alphab,wk,csqu,ex,ey,lx,ly);
#             [fr,fb,ff]=collision(obst,ux,uy,rhor,rhob,rho,fr,fb,fer,feb,wk,ex,G,csqu)
#             #Redistribution
#             [fr,fb]=redistribute(obst,csqu,ux,uy,rhor,rhob,rho,rhon,sigma,fr,fb,ff,ex,ey,lx,ly,wk,rsq,beta)
#             # o=np.where(obst!=0)
#             #Bounceback
#             fr=bounceback1(fr,obst,o,ox,oo,Uir,ex,ey,ux,uy,wk,rhor,csqu)
#             fb= bounceback1(fb,obst,o,ox,oo,Uib,ex,ey,ux,uy,wk,rhob,csqu)
#             # #Periodic BC
#             # fr=period(fr)
#             # fb=period(fb)
#             #Store data after nw iterations
#             if (t%(50*nw)==0):
#               results(lx,ly,obst,rhor,rhob,rho,ux,uy,p)              
end= time.time()
print('Running Time:',end-start)
#End




