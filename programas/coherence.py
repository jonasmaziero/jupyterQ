import pTrace
import math
import numpy as np
import entropy as ent

def coh(rho, name):
    if name == 'l1':
        return coh_l1(rho)
    elif name == 're':
        return coh_re(rho)
    elif name == 'hs':
        return coh_hs(rho)

### l1-norm complementarity
#l1-norm coherence
def coh_l1(rho):  # normalized to [0,1]
    d = rho.shape[0]
    coh = 0.0
    for j in range(0, d-1):
        for k in range(j+1, d):
            coh += math.sqrt((rho[j][k].real)**2.0 + (rho[j][k].imag)**2.0)
    return 2.0*coh#/(d*(d-1))
    #isso eu comentei pra tirar a normalização por enquanto.

#l1-norm predictability
#peguei do coherence no github  e adaptei aqui.
# função para o cálculo da previsibilidade, eq 87 do artigo do Basso_Maziero,
# Quantitative wave-particle duality relations from the density matrix properties

def predict_l1(rho):
    d = rho.shape[0]
    P = 0.0
    for j in range(0,d-1):
        for k in range(j+1,d):
            P += math.sqrt(rho[j,j]*rho[k,k])
    return d-1-2*P

#l1-norm quantum correlation
def qcorr_l1(da,db,rhoAB):
    rhoA = pTrace.ptraceB(da, db, rhoAB)
    qc = 0.0
    for j in range(0,da-1):
        for k in range(j+1,da):
            qc += math.sqrt(rhoA[j,j]*rhoA[k,k]) - abs(rhoA[j,k])
    return 2*qc

###Hilbert-Schmidt complementarity
#hilbert-schmidt coherence
def coh_hs(rho):
    d = rho.shape[0];  C = 0.0
    for j in range(0,d-1):
        for k in range(j+1,d):
            C += (rho[j][k].real)**2.0 + (rho[j][k].imag)**2.0
    return 2*C

#predictability liner de Hilbert-Schmidt
#peguei do coherence no github e adaptei aqui.
def predict_hs_l(rho):
    d = rho.shape[0];  P = 0.0
    for j in range(0,d):
        P += abs(rho[j,j])**2
    return P-(1/d)

#Hilbert-Shmidt-liner quantum correlation
###adaptado do github e do artigo 
def qcorr_hs(rho):
    d = rho.shape[0] #;print(d)
    qc = 0.0
    for j in range(0,d):
        for k in range(0,d):
            qc += abs(rho[j,k])**2
    return 1-qc


#Hilbert-Schmidt-von Neumann quantum correlation FALTA

###Wigner-Yanase complementarity
#coerencia de wigner e yanase
import mat_func as mf
#peguei do coherence no github  e adaptei aqui.
def coh_wy(rho):
    d = rho.shape[0]
    rho_sqrt = mf.mat_sqrt(d,rho)
    C = 0.0
    for j in range(0,d-1):
        for k in range(j+1,d):
            C += abs(rho_sqrt[j,k])**2
    return 2*C

#Wigner-Yanase quantum correlation
#adaptado do artigo e do github jupyterQ/coherence.
# ESTA DIFERENTE DO GITHUB 

def qcorr_wy(rho):
    d = rho.shape[0]
    rho_sqrt = mf.mat_sqrt(d,rho)
    qc = 0.0    
    for j in range(0,d):
        qc += abs(rho_sqrt[j,j])**2 - abs(rho[j,j])**2
    return qc

###Relative entropy complementarity
#coherencia relativa da entropia libPy/coherence
def coh_re(rho):
    d = rho.shape[0]
    pv = np.zeros(d)
    for j in range(0,d):
        pv[j] = rho[j][j].real
    from entropy import shannon, von_neumann
    coh = shannon(pv) - von_neumann(rho)
    return coh/math.log(d,2)

#Relative entropy predictability
#adaptado do github jupyterQ/coherence
def predict_vn(rho):
    d = rho.shape[0]
    P = 0.0
    for j in range(0,d):
        P += abs(rho[j][j]) * math.log(abs(rho[j][j]))
    return math.log(d)+P

### Relative entropy quantum correlation
### ainda temos que verificar se adaptação esta certo
### adaptado do github  
def qcorr_re(rho):
    d = rho.shape[0]
    rhoA = pTrace.pTraceB(rho)
    from entropy import von_neumann
    return von_neumann(rhoA)


def coh_nl(da, db, rho):
    rhoa = pTrace.pTraceL(da, db, rho)
    rhob = pTrace.pTraceR(da, db, rho)
    return coh_l1(da*db, rho)-coh_l1(da, rhoa)-coh_l1(db, rhob)