import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as od
import cv2
from matplotlib import rc
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def comat1(m1):
    for i in range(0, n):
       
        #General relationship
        m1[i,i] = P
       
        #Boundary conditions
        if i in range(0, n_x):
            m1[i,i] = m1[i,i] + B
        if i % n_x == 0:
            m1[i,i] = m1[i,i] + A
        if i % n_x == (n_x-1):
            m1[i,i] = m1[i,i] + A
        if i in range(n-n_x, n):
            m1[i,i] = m1[i,i] + B
               
        #Normal conditions for coefficient matrix
        if i >= n_y:
            m1[i, i-n_y] = B
        if i > n_x-1:
            m1[i-n_x, i] = B
        if i % n_y:
            m1[i, i-1] = A
        if i % n_x:
            m1[i-1, i] = A
           
    m1 = m1.tocsr()        
   
    return

def comat0(m0):
    for i in range(0, n):
        m0[i, i] = O
       
        if i in range(0, n_x):
            m0[i,i] = m0[i,i] - B
        if i % n_x == 0:
            m0[i,i] = m0[i,i] - A
        if i % n_x == (n_x-1):
            m0[i,i] = m0[i,i] - A
        if i in range(n-n_x, n):
            m0[i,i] = m0[i,i] - B
           
        if i >= n_y:
            m0[i, i-n_y] = -B
        if i > n_x-1:
            m0[i-n_x, i] = -B
        if i % n_y:
            m0[i, i-1] = -A
        if i % n_x:
            m0[i-1, i] = -A
           
    m0 = m0.tocsr()        
   
    return


def vel_x(i):
   
    x = ((i % n_x) - ((n_x - 1)/2))*dx0
    y = (((n_y - 1)/2) - (i // n_x))*dy0

    vel = -0.05

   
    return vel


def vel_y(i):
   
    x = ((i % n_x) - ((n_x - 1)/2))*dx0
    y = (((n_y - 1)/2) - (i // n_x))*dy0
   
   
    vel = -0.05
   
    return vel


def velocity1(v1):
    for i in range(0, n):
        v1[i, i] = 0
       
        if i % n_x == 0:
            v1[i,i] = v1[i,i] + vel_x(i)/(4*dx0)                        #boundary condition -> Resets diagonal element accounting for no points behind (i=0)
        if i % n_x == (n_x-1):
            v1[i,i] = v1[i,i] - vel_x(i)/(4*dx0)                         #boundary condition -> Resets diagonal element accounting for no points in front (i=n_x)
        if i in range(0, n_y):
            v1[i,i] = v1[i,i] - vel_y(i)/(4*dy0)                 #boundary condition -> Resets diagonal element accounting for no points to the left (k=0)
        if i in range(n-(n_y), n):
            v1[i,i] = v1[i,i] + vel_y(i)/(4*dy0)                  #boundary condition -> Resets diagonal element accounting for no points to the right (k=n_z)

        if i % n_y:
            v1[i, i-1] = vel_x(i)/(4*dx0)                       #general condition -> Adds element describing contribution from point behind considered point (ith-1)
        if i % n_x:                                         #general condition -> Adds element describing contribution from point in front of considered point (ith+1)
            v1[i-1, i] = -vel_x(i)/(4*dx0)
        if i >= (n_x):
            v1[i-(n_x), i] = vel_y(i)/(4*dy0)                #general condition -> Adds element describing contribution from point to the right of considered point (kth+1)
        if i > (n_x-1):
            v1[i, i-(n_x)] = -vel_y(i)/(4*dy0)                #general condition -> Adds element describing contribution from point to the left of considered point (kth-1)
           
   # v1 = v1.tocsr()

    return

def velocity0(v0):
    for i in range(0, n):
        v0[i, i] = 0

        if i % n_x == 0:
            v0[i,i] = v0[i,i] - vel_x(i)/(4*dx0)                        #boundary condition -> Resets diagonal element accounting for no points behind (i=0)
        if i % n_x == (n_x-1):
            v0[i,i] = v0[i,i] + vel_x(i)/(4*dx0)                         #boundary condition -> Resets diagonal element accounting for no points in front (i=n_x)
        if i in range(0, n_x):
            v0[i,i] = v0[i,i] + vel_y(i)/(4*dy0)                 #boundary condition -> Resets diagonal element accounting for no points to the left (k=0)
        if i in range(n-n_x, n):
            v0[i,i] = v0[i,i] - vel_y(i)/(4*dy0)                  #boundary condition -> Resets diagonal element accounting for no points to the right (k=n_z)

        if i % n_y:
            v0[i, i-1] = -vel_x(i)/(4*dx0)                       #general condition -> Adds element describing contribution from point behind considered point (ith-1)
        if i % n_x:
            v0[i-1, i] = vel_x(i)/(4*dx0)                        #general condition -> Adds element describing contribution from point in front of considered point (ith+1)              
        if i >= (n_x):
            v0[i-(n_x), i] = -vel_y(i)/(4*dy0)                #general condition -> Adds element describing contribution from point to the right of considered point (kth+1)
        if i > (n_x-1):
            v0[i, i-(n_x)] = vel_y(i)/(4*dy0)                #general condition -> Adds element describing contribution from point to the left of considered point (kth-1)

  #  v0 = v0.tocsr()
   
    return



def solve_cf(m0,m1,ci):
   
    Z = m0.dot(ci)
    cf = spsolve(m1, Z)
   
    return(cf)
   
def conc_init(n_x, n_y):
    x = np.linspace(-W/2, W/2, n_x)
    y = np.linspace(-H/2, H/2, n_y)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X)**2 + (Y)**2)/0.1)
   
    #path = r'C:\Users\Alex\Desktop\READ.png' #Edit path to match your directory
    #Z = cv2.imread(path, 0)
   
    Z = Z.reshape(n)
    return(Z)
   
def dU_dt(U,t):
    x,vx,y,vy=U
    r = np.sqrt(x**2.0+y**2.0 )
    while(x<a):
        return [vx,0.5*bvspline(x,y,dx=1,dy=0),vy,0.5*bvspline(x,y,dx=0,dy=1)]
    else:
        return [vx, 0.0, vy, 0.0]
#ODE solver
def Up(U0,N,s):
    t0=np.arange(N)*s
    res=od.odeint(dU_dt,U0,t0)
    return [res[:,0],res[:,2]]
   
H = 1                                                #arbitrary length of domain space (x)
W = 1                                                #arbitrary width of domain space (y)
n_x = 70                                            #number of points along length to be calculated
n_y = 70                                             #number of points along width to be calculated

dx0 = H/(n_x-1)                                           #space step (x)
dy0 = W/(n_y-1)                                           #space step (y)
D = 0.0005                                            #diffusion coefficient
t_final = 50                                         #last time point
sig = 0.05                                   # CFL limit (needs to be less than certain value ~ 0.2)
dt = sig*((dx0**2/D)+(dy0**2/D))                       #time step


#Begin by defining the coefficients
A = (D/(2*(dx0**2)))                                  #(Contribution from point behind)
B = (D/(2*(dy0**2)))                                  #(Contribution from point above)
P = (-(2*A) - (2*B) - (1/dt))                      #(Point considered in comp mol)

#concentration conversion matrix - takes real space concentrations and turns into values
O = (D/(dx0**2)+D/(dy0**2)-1/dt)                       #(Point considered by comp mol)

#Selecting the size of the matrix
n = n_x*n_y
t = np.arange(0, t_final, dt)
m1 = lil_matrix((n, n))
m0 = lil_matrix((n, n))
v1 = lil_matrix((n, n))
v0 = lil_matrix((n, n))

ci = conc_init(n_x, n_y)

#Creating our coefficient & velocity matrices
comat1(m1)
comat0(m0)
velocity1(v1)
velocity0(v0)

#Adding the coefficient & velocity matrices together
m1 = (m1 + v1)
m0 = (m0 + v0)
   
a = H/2

x0 = np.linspace(-W/2, W/2, n_x)
y0 = np.linspace(-H/2, H/2, n_y)
X, Y = np.meshgrid(x0, y0)

X0v=[-a for j in np.linspace(-a,a,10)]
Y0v=[j  for j in np.linspace(-a,a,10)]
p0x=[1 for j in X0v]
p0y=[0 for j in Y0v]
U0v=[[X0v[j],p0x[j],Y0v[j],p0y[j]] for j in np.arange(len(X0v))]

FS = 20

for i in range(0, len(t)):
    output = solve_cf(m0,m1,ci).reshape(n_x, n_y)
    ci = solve_cf(m0,m1,ci)
   
    bvspline=RBS(x0,y0,output)

    plt.clf()

    for k in U0v:
        xoutput,youtput=Up(k,500,0.005)
        plt.plot(youtput,-xoutput,'r-',lw=2)
   
    
    plt.rcParams.update({'font.size': FS})
    plt.xlabel("x", fontsize=FS)
    plt.ylabel("y", fontsize=FS)
    
    plt.imshow(output, cmap='Blues', vmin=0, vmax=1, extent=[-H/2,H/2,-W/2,W/2])
    plt.colorbar()
    #plt.xlim(0, W)
    #plt.ylim(0, H)
    plt.show()
    plt.pause(0.01)
   
#bvspline=RBS(x0,y0,output)

#for k in U0v:
#    xoutput,youtput=Up(k,500,0.005)
#    plt.plot(xoutput,youtput,'r-',lw=2)
#plt.clf()
#plt.imshow(bvspline(x0,y0), cmap='Reds', vmin=0, vmax=1, extent=[0,H,0,W])
#plt.show()