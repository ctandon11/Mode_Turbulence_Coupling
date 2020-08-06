#!/usr/bin/env python
# coding: utf-8

# # Mode-Turbulence Coupling: Numerical Analysis
# # Explore the Effects of Positive Damping: D = 10^-4, 10^-3, 10^-2, 10^-1, 1
# ## M = 10^3, < Delta_tau > = 2pi, K = 1, eta M^2 = 0.0125

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.interpolate as interpolate


# ## Physical Parameters

# eta * Ma ^2 = 0.0125
Ma = 0.0125 ** (1/2)
K = 1
eta = 1


# ## Functions of Ma

meandeltaT = 2 * np.pi

# variables for generating F2 and G1
stdF2 = eta * Ma**2
limitY2 = eta 


# ## Calculation parameters

L = 1
M = 10**4
M_short = 200
N = 1
m = 10**2

J = 300
dTau = meandeltaT / J

Tr = K * meandeltaT


# ## Initial Conditions

initial_AT = 0
initial_dAdT = 0

# deltaT is the length of each eddy turnover
def fundeltaT(n_bins, M, dTau):
    Tturb = [0]
    deltaT = []
    
    # generate a rayleigh distribution from which to pick values for deltaT from
    rayleigh = np.random.rayleigh(meandeltaT, M)                           
    hist, bin_edges = np.histogram(rayleigh, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    
    for i in range(M):
        # picks a random value from the rayleigh distribution
        r = np.random.rand(1)
        deltaTtemp = inv_cdf(r)
        
        # rounds deltaTtemp up or down to be an interger multiple of dTau
        if (deltaTtemp % dTau < 0.005):
            deltaTtemp2 = deltaTtemp - (deltaTtemp % dTau)
        else:
            deltaTtemp2 = deltaTtemp + dTau - (deltaTtemp % dTau)
        deltaT.append(deltaTtemp2)
        Tturb.append(deltaTtemp2 + Tturb[i])
    
    Tturb.pop(-1)
    Tturb.pop(-1)
    return deltaT, Tturb

# generating F1, F2, G1
def funF_k(std, limit, deltaT, Tturb):
    F_k =[]
    y1 = 0
    for i in range(len(deltaT) - 1):
        
        # Tn is halfway through the eddy
        Tn = ((deltaT[i] + deltaT[i+1]) / 2)
        
        stdevt = ((1 - math.exp(-2 * Tn / Tr)) * std ** 2) ** (1/2)
        meant = math.exp(-Tn / Tr) * y1
        
        # y1 is chosen from markov conditional probability function
        # put limit on y1
            # set y1 > limitY1 so that while loop only ends when y1 < limitY1
        y1 = limit + 1
        while abs(y1) > limit:
            y1 = np.random.normal(meant, stdevt, 2)[0]
        
        # loops through each eddy
        for j in np.linspace(0, deltaT[i], int(deltaT[i] / dTau)):
            x = j / deltaT[i]
            if j == deltaT[i]:
                pass
            else:
                F_k.append(16 * y1 * x**2 * (1 - x)**2)
        
    return F_k

# generating F1
def fundHdt(std, limit, deltaT, Tturb):
    dHdt =[]
    y1 = 0
    for i in range(len(deltaT) - 1):
        
        # Tn is halfway through the eddy
        Tn = ((deltaT[i] + deltaT[i+1]) / 2)
        
        stdevt = ((1 - math.exp(-2 * Tn / Tr)) * std ** 2) ** (1/2)
        meant = math.exp(-Tn / Tr) * y1
        
        # y1 is chosen from markov conditional probability function
        # put limit on y1
            # set y1 > limitY1 so that while loop only ends when y1 < limitY1
        y1 = limit + 1
        while abs(y1) > limit:
            y1 = np.random.normal(meant, stdevt, 2)[0]
        
        # loops through each eddy
        for j in np.linspace(0, deltaT[i], int(deltaT[i] / dTau)):
            x = j / deltaT[i]
            if j == deltaT[i]:
                pass
            else:
                # calculated the derivative of H = 16 * y1 * x**2 * (1 - x)**2
                dHdt.append(1 / deltaT[i] * 32 * y1 * x * (2 * x**2 - 3 * x + 1))
        
    return dHdt

def funAmp(F1, F2, G1, dTau, A, dAdt, D):
    # huen's method
    # u = dA/dt
    
    Amp = [A]
    t = 0.0
    amptime = [t]
    u = dAdt
    
    for i in range(0, int(len(F1)) -2):
        m1 = u
        k1 = -((D + F1[i]) * u ) - ( 1 + G1[i] ) * A + F2[i]
        m2 = u + dTau * k1
        A_2 = A + dTau * m1
        u_2 = m2
        k2 = -((D + F1[i + 1]) * u_2 ) - ( 1 + G1[i + 1] ) * A_2 + F2[i + 1]
        m2 = u + dTau * k2
        t = t + dTau
        A = A + (dTau / 2) * (m1 + m2)
        u = u + (dTau / 2) * (k1 + k2)
        Amp.append(A)
        amptime.append(t)

    Amp = Amp[:-1]
    amptime = amptime[:-1]
    nextAmp = Amp[-1]
    
    return Amp, amptime, nextAmp, u

def FunplotAmp(amptime, Amp, D):
    plt.figure(figsize=(40,10))
    plt.plot(amptime, Amp, 'k')
    plt.title('D = 10** %d' % (m), **title_font)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel(r'$\tau$', **axis_font)
    plt.ylabel('Amplitude', **axis_font)
    plt.tight_layout()
    plt.savefig('m=%d' % (m))
    plt.show()

# flattens nested arrays into one dimensional arrays
def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list

n_bins = 200

axis_font = {'size':'30'}
title_font = {'size':'40'}



# ## Evolution 1


# deltaT is the length of each eddy turnover
# Tturb is the consecutive sum of deltaT's so each value marks a new eddy
deltaT, Tturb = fundeltaT(n_bins, M, dTau)
deltaT = flatten(deltaT)


# generate F1, F2, G1
F1 = fundHdt(stdF2, limitY2, deltaT, Tturb)
F2 = funF_k(stdF2, limitY2, deltaT, Tturb)
G1 = funF_k(stdF2, limitY2, deltaT, Tturb)


# ## D = 10^-3

m = -3
D = 10**m

Amp, amptime, nextAmp, nextdAdt = funAmp(F1, F2, G1, dTau, initial_AT, initial_dAdT, D)

FunplotAmp(amptime, Amp, m)


# ## D = 10^-2

m = -2
D = 10**m

Amp, amptime, nextAmp, nextdAdt = funAmp(F1, F2, G1, dTau, initial_AT, initial_dAdT, D)

FunplotAmp(amptime, Amp, m)


# ## D = 10**-1

m = -1
D = 10**m

Amp, amptime, nextAmp, nextdAdt = funAmp(F1, F2, G1, dTau, initial_AT, initial_dAdT, D)

FunplotAmp(amptime, Amp, m)


# ## D = 1

m=0
D =10**m 

Amp, amptime, nextAmp, nextdAdt = funAmp(F1, F2, G1, dTau, initial_AT, initial_dAdT, D)

FunplotAmp(amptime, Amp, m)




