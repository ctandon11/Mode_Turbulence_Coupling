#!/usr/bin/env python
# coding: utf-8

# # Mode-Turbulence Coupling: Numerical Analysis
# # M = 10^3, < Delta_tau > = 2pi, K = 1, eta M^2 = 0.0125
# # Fourier Transform of F2

import random
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.interpolate as interpolate

n_bins = 200

axis_font = {'size':'30'}
title_font = {'size':'40'}


# ## Physical Parameters

# eta * Ma ^2 = 0.0125
Ma = 0.0125 ** (1/2)
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
dTau = 2 * np.pi / J


# q = w / w0
# -1 <= log q <= 1

logq = np.arange(-1,1,0.01)
q = [10**x for x in logq]


# ## Functions


# deltaT is the length of each eddy turnover
def fundeltaT(n_bins, M, dTau):
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
    
    return deltaT

# Tturb adds up the delta T's
def funTturb(deltaT):
    Tturb=[0]
    for i in range(len(deltaT)-1):
        Tturb.append(deltaT[i]+Tturb[i])
    Tturb.pop(-1)
    return Tturb

# generating F2, G1
def funF_k(std, limit, deltaT, Tturb):
    F_k =[]
    y1 = 0
    time = []
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
                time.append(j + Tturb[i])
                F_k.append(16 * y1 * x**2 * (1 - x)**2)
        
    return F_k, time

def Fourier(F2):
    ReF2 = []
    ImF2 = []

    # w_0 ReF2 = int F2(T)cos(qT) dT
    # w_0 ImF2 = int F2(T)sin(qT) dT

    timestep = 0.0210

    for j in q:
        totalRe = 0
        totalIm = 0
        for i in range(len(time)):
            totalRe += F2[i] * math.cos(j*time[i]) * timestep
            totalIm += F2[i] * math.sin(j*time[i]) * timestep
        ReF2.append(totalRe)
        ImF2.append(totalIm) 
            
    return ReF2, ImF2
    
def Plot(ReF2, ImF2, K):
    f = plt.figure(figsize=(20,10))
    ax = f.add_subplot(211)
    ax2 = f.add_subplot(212)
    ax.set_ylabel('\u03C9\u2080 Re F2(q)', fontsize = 20)
    ax.set_xlabel('log(q)', fontsize = 20)
    ax.plot(logq, ReF2, 'k')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.title.set_text('K = %f' % (K))


    ax2.set_ylabel('\u03C9\u2080 Im F2(q)', fontsize = 20)
    ax2.set_xlabel('log(q)', fontsize = 20)
    ax2.plot(logq, ImF2, 'k')
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax.title.set_text('K = %f' % (K))

    f.savefig('K=%d' % (K))

# flattens nested arrays into one dimensional arrays
def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list

# deltaT is the length of each eddy turnover
deltaT = flatten(fundeltaT(n_bins, M, dTau))

# Tturb is the consecutive sum of deltaT's so each value marks a new eddy
Tturb = funTturb(deltaT)


# ## K = 1

K = 1
Tr = K * meandeltaT

F2, time = funF_k(stdF2, limitY2, deltaT, Tturb)

ReF2, ImF2 = Fourier(F2)
Plot(ReF2, ImF2, K)


# ## K = 0.2

K = 0.2
Tr = K * meandeltaT

F2, time = funF_k(stdF2, limitY2, deltaT, Tturb)

ReF2, ImF2 = Fourier(F2)
Plot(ReF2, ImF2, K)






