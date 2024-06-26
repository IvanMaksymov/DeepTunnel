import numpy as np
import matplotlib.pyplot as plt

m = 9.1093837e-31
hbar = 1.054571817e-34 
qe = 1.60217663e-19 

def diffT(L, E, U0):

    ind1 = np.where((E < U0) & (E >= 0.0))
    ind2 = np.where(E > U0)
    ind3 = np.where(E == U0)
    
    diffT = np.zeros([E.size])    

    alpha = E[ind1] - U0
    kappa1 = np.sqrt(-2.0*m*alpha*qe)/hbar
    beta = U0**2/(4.0*E[ind1]*alpha)
    delta1 = kappa1*L
    T1 = 1.0/(1.0-beta*np.power(np.sinh(kappa1*L),2.0))
    diffT[ind1] = -beta*(np.power(np.sinh(delta1),2.0)/E[ind1] + (np.power(np.sinh(delta1),2.0)-delta1*np.sinh(delta1)*np.cosh(delta1))/alpha)*np.power(T1,2.0)
    
    alpha = E[ind2] - U0
    kappa = np.sqrt(2.0*m*alpha*qe)/hbar
    beta = U0**2/(4.0*E[ind2]*alpha)        
    T2 = 1.0/(1.0+beta*np.power(np.sin(kappa*L),2.0)) 
    delta = kappa*L
    diffT[ind2] = beta*(np.power(np.sin(delta),2.0)/E[ind2] + (np.power(np.sin(delta),2.0)-delta*np.sin(delta)*np.cos(delta))/alpha)*np.power(T2,2.0)

    diffT[ind3] = (4*L**4*U0*m**2*qe**2+6*L**2*hbar**2*m*qe)/(3*L**4*U0**2*m**2*qe**2+12*L**2*U0*hbar**2*m*qe+12*hbar**4)

    return diffT
   
def T(L, E, U0):

    ind1 = np.where((E < U0) & (E >= 0.0))
    ind2 = np.where(E > U0)
    ind3 = np.where(E == U0)
    
    T = np.zeros([E.size])       
    
    alpha = E[ind1] - U0
    kappa1 = np.sqrt(-2.0*m*alpha*qe)/hbar
    beta = U0**2/(4.0*E[ind1]*alpha)
    T[ind1] = 1.0/(1.0-beta*np.power(np.sinh(kappa1*L),2.0))

    alpha = E[ind2] - U0
    kappa = np.sqrt(2.0*m*alpha*qe)/hbar
    beta = U0**2/(4.0*E[ind2]*alpha)        
    T[ind2] = 1.0/(1.0+beta*np.power(np.sin(kappa*L),2.0))

    T[ind3] = 1.0/(1.0+m*L**2*U0*qe/2.0/np.power(hbar,2.0))

    return T

Vpot = 30.0

Nb=31

ProbTrans1Paper = np.zeros((Nb))
ProbTrans2Paper = np.zeros((Nb))
ProbTrans3Paper = np.zeros((Nb))

E = np.zeros((4*Nb))

m1 = 0.5
L = m1*hbar/np.sqrt(2*m*qe*Vpot)
cnt = 0
for i in range(-Nb,3*Nb):
    E[cnt] = i*1.0
    if i == 0:
        E[cnt] = 0.001    
        
    cnt = cnt + 1
    
ProbTrans1Paper = T(L,E,Vpot)
diffProbTransPaper = diffT(L,E,Vpot)

m1 = 1
L = m1*hbar/np.sqrt(2*m*qe*Vpot)
cnt = 0
for i in range(-Nb,3*Nb):
    E[cnt] = i*1.0
    cnt = cnt + 1
    
ProbTrans2Paper = T(L,E,Vpot)

m1 = 1.5
L = m1*hbar/np.sqrt(2*m*qe*Vpot)
cnt = 0
for i in range(-Nb,3*Nb):
    E[cnt] = i*1.0
    cnt = cnt + 1
    
ProbTrans3Paper = T(L,E,Vpot)

E = E/Vpot
    
fig = plt.figure()
p=plt.plot(E, ProbTrans1Paper, linewidth=2, linestyle="-", color='r', label="$\sqrt{2mV_0}a/\hbar=0.5$")
p=plt.plot(E, ProbTrans2Paper, linewidth=2, linestyle="--", color='g', label="$\sqrt{2mV_0}a/\hbar=1.0$")
p=plt.plot(E, ProbTrans3Paper, linewidth=2, linestyle=":", color='b', label="$\sqrt{2mV_0}a/\hbar=1.5$")

x1, y1 = [0, 1], [0, 0]
x2, y2 = [1, 1], [0, 1]
x3, y3 = [1, 3], [1, 1]
plt.plot(x1, y1, linestyle = '-.', color='k', linewidth=1)
plt.plot(x2, y2, linestyle = '-.', color='k', linewidth=1)
plt.plot(x3, y3, linestyle = '-.', color='k', linewidth=1, label="Classical theory")
 
plt.ylabel('$T$',fontsize=14)
plt.xlabel('$E/V_0$',fontsize=14)
plt.xlim([-1, 3])
plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
plt.yticks(fontsize=14)  # Set font size for y-axis tick labels

plt.legend(loc="lower right", frameon=False)

# save the plot
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
 
