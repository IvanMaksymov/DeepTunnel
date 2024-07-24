import numpy as np
import matplotlib.pyplot as plt
import math
import string
from numpy import loadtxt

m = 9.1093837e-31
hbar = 1.054571817e-34 
qe = 1.60217663e-19

mm1 = [0.5, 1.0, 1.5]
ITER = 1000 #NUMBER OF EPOCHS

def diffT(L, E, U0):

    diffT = np.zeros([E.size])       

    for ind in range(E.size):
        if E[ind] < 0.0:
            diffT[ind] = 0.0
            
        if E[ind] < U0 and E[ind] >= 0.0:
            alpha = E[ind] - U0
            kappa1 = np.sqrt(-2.0*m*alpha*qe)/hbar
            beta = U0**2/(4.0*E[ind]*alpha)
            delta1 = kappa1*L
            T1 = 1.0/(1.0-beta*np.power(np.sinh(kappa1*L),2.0))
            diffT[ind] = -beta*(np.power(np.sinh(delta1),2.0)/E[ind] + (np.power(np.sinh(delta1),2.0)-delta1*np.sinh(delta1)*np.cosh(delta1))/alpha)*np.power(T1,2.0)

        if E[ind] > U0:
            alpha = E[ind] - U0
            kappa = np.sqrt(2.0*m*alpha*qe)/hbar
            beta = U0**2/(4.0*E[ind]*alpha)        
            T2 = 1.0/(1.0+beta*np.power(np.sin(kappa*L),2.0)) 
            delta = kappa*L
            diffT[ind] = beta*(np.power(np.sin(delta),2.0)/E[ind] + (np.power(np.sin(delta),2.0)-delta*np.sin(delta)*np.cos(delta))/alpha)*np.power(T2,2.0)

        if E[ind] == U0:
            diffT[ind] = (4*L**4*U0*m**2*qe**2+6*L**2*hbar**2*m*qe)/(3*L**4*U0**2*m**2*qe**2+12*L**2*U0*hbar**2*m*qe+12*hbar**4)

    return diffT.reshape((E.size,1))

def T(L, E, U0):

    T = np.zeros([E.size])
    
    for ind in range(E.size):       
        if E[ind] < 0.0:
            T[ind] = 0.0
            
        if E[ind] < U0 and E[ind] >= 0.0:
            alpha = E[ind] - U0
            kappa1 = np.sqrt(-2.0*m*alpha*qe)/hbar
            beta = U0**2/(4.0*E[ind]*alpha)
            T[ind] = 1.0/(1.0-beta*np.power(np.sinh(kappa1*L),2.0))
    
        if E[ind] > U0:        
            alpha = E[ind] - U0
            kappa = np.sqrt(2.0*m*alpha*qe)/hbar
            beta = U0**2/(4.0*E[ind]*alpha)        
            T[ind] = 1.0/(1.0+beta*np.power(np.sin(kappa*L),2.0))

        if E[ind] == U0:    
            T[ind] = 1.0/(1.0+m*L**2*U0*qe/2.0/np.power(hbar,2.0))

    return T.reshape((E.size,1))

def Softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 
    
# MAIN PROGRAM

X = np.zeros((3,10,10), dtype=int)

# Necker cube
X[0, :, :] = [ [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
               [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
               [1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
               [1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ];
               
X[1, :, :] = [ [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
               [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ];

X[2, :, :] = [ [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
               [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
               [1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
               [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
               [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ];      

alpha = 0.01
TRAINING = 2
NumberInput = 10*10
NumberHidden1 = 20
NumberHidden2 = 20
NumberHidden3 = 20
NumberOutput = TRAINING

W1 = np.zeros((NumberHidden1, NumberInput))
W2 = np.zeros((NumberHidden2, NumberHidden1))
W3 = np.zeros((NumberHidden3, NumberHidden2))
W4= np.zeros((NumberOutput, NumberHidden3))

dW1 = np.zeros((NumberHidden1, NumberInput))
dW2 = np.zeros((NumberHidden2, NumberHidden1))
dW3 = np.zeros((NumberHidden3, NumberHidden2))
dW4= np.zeros((NumberOutput, NumberHidden3))
	
W2transp = np.zeros((NumberHidden1, NumberHidden2))
W3transp = np.zeros((NumberHidden2, NumberHidden3))
W4transp = np.zeros((NumberHidden3, NumberOutput))

y = np.zeros((NumberOutput))
v = np.zeros((NumberOutput))
e = np.zeros((NumberOutput))
delta = np.zeros((NumberOutput))

y1 = np.zeros((NumberHidden1))
v1 = np.zeros((NumberHidden1))
e1 = np.zeros((NumberHidden1))
delta1 = np.zeros((NumberHidden1))

y2 = np.zeros((NumberHidden2))
v2 = np.zeros((NumberHidden2))
e2 = np.zeros((NumberHidden2))
delta2 = np.zeros((NumberHidden2))

y3 = np.zeros((NumberHidden3))
v3 = np.zeros((NumberHidden3))
e3 = np.zeros((NumberHidden3))
delta3 = np.zeros((NumberHidden3))

d = np.zeros((NumberOutput, NumberOutput)) 

illusion = np.zeros((3, 2, 40))

QuantumRandom = loadtxt('quantum_random.dat')  

for barrier in range(3):

    Vpot = 30.0 # eV
    m1 = mm1[barrier]
    L = m1*hbar/np.sqrt(2*m*qe*Vpot)
    ampl = 10.0

    CntRnd = 0    
    for realisation in range(40):
    
        # CORRECT ANSWERS
        for trainingSet in range(TRAINING):
            for j in range(TRAINING):
                if j==trainingSet:
                    d[trainingSet,j] = 1.0	
                else: 
                    d[trainingSet,j] = 0.0
        
        # HIDDEN NEURONS   
        #W1 = 2.0*np.random.rand(NumberHidden1, NumberInput) - 1.0
        for iq in range(NumberHidden1):
            for jq in range(NumberInput):
                W1[iq,jq] = 2.0*QuantumRandom[CntRnd] - 1.0
                CntRnd = CntRnd + 1        
        
        #W2 = 2.0*np.random.rand(NumberHidden2, NumberHidden1) - 1.0
        for iq in range(NumberHidden2):
            for jq in range(NumberHidden1):
                W2[iq,jq] = 2.0*QuantumRandom[CntRnd] - 1.0
                CntRnd = CntRnd + 1
        
        #W3 = 2.0*np.random.rand(NumberHidden3, NumberHidden3) - 1.0
        for iq in range(NumberHidden3):
            for jq in range(NumberHidden3):
                W3[iq,jq] = 2.0*QuantumRandom[CntRnd] - 1.0
                CntRnd = CntRnd + 1
        
        # OUTPUT NEURONS
        #W4 = 2.0*np.random.rand(NumberOutput, NumberHidden3) - 1.0         
        for iq in range(NumberOutput):
            for jq in range(NumberHidden3):
                W4[iq,jq] = 2.0*QuantumRandom[CntRnd] - 1.0
                CntRnd = CntRnd + 1         
        
        # MAIN LOOP
        for iteration in range(ITER):
            for trainingSet in range(TRAINING):
                
                x  = np.reshape(X[trainingSet, :, :], (NumberInput, 1))	
                v1 = np.matmul(W1,x)
                y1 = T(L, v1*ampl, Vpot)
                v2 = np.matmul(W2,y1)
                y2 = T(L, v2*ampl, Vpot)
                v3 = np.matmul(W3, y2)
                y3 = T(L, v3*ampl, Vpot)
                v = np.matmul(W4, y3)
                y  = Softmax(v) 
                	  			        
                e = d[trainingSet,:].reshape((TRAINING, 1)) - y    
                delta = e
        			
                W4transp = np.transpose(W4)
                e3 = np.matmul(W4transp, delta)
            		
                # delta3 = (v3 > 0).*e3
                delta3 = np.multiply(diffT(L, v3*ampl, Vpot),e3)	
        	
                W3transp = np.transpose(W3)
                e2 = np.matmul(W3transp, delta3)
            		
                # delta2 = (v2 > 0).*e2
                delta2 = np.multiply(diffT(L, v2*ampl, Vpot), e2)    		
        
                W2transp = np.transpose(W2)
                e1 = np.matmul(W2transp, delta2)
        
                # delta1 = (v1 > 0).*e1
                delta1 = np.multiply(diffT(L, v1*ampl, Vpot),e1)
        			
                # dW4 = alpha*delta*y3'	
                dW4 = alpha*np.matmul(delta,np.transpose(y3))	
                W4 = W4 + dW4
        			
        		# dW3 = alpha*delta3*y2'
                dW3 = alpha*np.matmul(delta3,np.transpose(y2))	
                W3  = W3 + dW3
            		
                # dW2 = alpha*delta2*y1'
                dW2 = alpha*np.matmul(delta2,np.transpose(y1))     		
                W2 = W2 + dW2
        
                # dW1 = alpha*delta1*x'			
                dW1 = alpha*np.matmul(delta1,np.transpose(x))			
                W1 = W1 + dW1
        	               
        # EXPLOITATION
        trainingSet = 2
        x  = np.reshape(X[trainingSet, :, :], (NumberInput, 1))
        v1 = np.matmul(W1,x)
        y1 = T(L, v1*ampl, Vpot)
        v2 = np.matmul(W2,y1)
        y2 = T(L, v2*ampl, Vpot)
        v3 = np.matmul(W3,y2)
        y3 = T(L, v3*ampl, Vpot)
        v = np.matmul(W4,y3)
        y  = Softmax(v)	
        illusion[barrier, 0, realisation] = y[0]
        illusion[barrier, 1, realisation] = y[1]
        print('Illusion = ', illusion[barrier,: , realisation])       

panel_lbl = ['(a)', '(b)', '(c)']

fig1, axs = plt.subplots(3, 1, layout='constrained')

for barrier in range(3):

    axs[barrier].plot(illusion[barrier, 0, :], linewidth=2, linestyle="-", color='r', label=r"$|0\rangle$")
    axs[barrier].plot(illusion[barrier, 1, :], linewidth=2, linestyle=":", color='b', label=r"$|1\rangle$")
    axs[barrier].set_ylabel('Probab.',fontsize=14)
    if barrier == 2:
        axs[barrier].set_xlabel('Time (arb. units)',fontsize=14)
    axs[barrier].grid(False)
    axs[barrier].set_xlim([-1, 41])
    axs[barrier].set_ylim([-0.05, 1.05])
    axs[barrier].set_yticks([0, 0.5, 1])
    axs[barrier].xaxis.set_tick_params(labelsize=14)
    axs[barrier].yaxis.set_tick_params(labelsize=14)
    axs[barrier].text(-8, 1.05, panel_lbl[barrier], size=14)
    axs[barrier].legend(loc="upper right")

# save the plot
plt.savefig('DeepTunnel_QuantumNecker.pdf', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()  