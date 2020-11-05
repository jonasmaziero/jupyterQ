import numpy as np
import matplotlib.pyplot as plt
import coherence as coh
import entropy as ent
import cmath
import math
import mat_func as mf
import pTrace as pT

# função que chama o arquivo onde está os dados extraídos do ibmqe
# chama as funções para o calcular a coerência, previssibilidade e correlação quântica
# de cada uma das diferentes funções  do artigo citado acima.  
def coh_predic_random():    
    x4 = np.zeros(100); y4 = np.zeros(100)
    x5 = np.zeros(100); y5 = np.zeros(100)
    coh_2q = np.zeros(100);   coh_2q_teo= np.zeros(100)
    previ_2q = np.zeros(100); previ_2q_teo = np.zeros(100)
    qcorr_2q = np.zeros(100); qcorr_2q_teo = np.zeros(100)
    soma_2q = np.zeros(100);  soma_2q_teo = np.zeros(100)
    C_hs_2q = np.zeros(100);  C_hs_2q_teo = np.zeros(100)
    P_hs_2q = np.zeros(100);  P_hs_2q_teo = np.zeros(100)
    C_wy_2q = np.zeros(100);  C_wy_2q_teo = np.zeros(100)
    QC_wy_2q = np.zeros(100); QC_wy_2q_teo = np.zeros(100)
    C_re_2q = np.zeros(100);  C_re_2q_teo = np.zeros(100)
    P_vn_2q = np.zeros(100);  P_vn_2q_teo = np.zeros(100)
    QC_hs_2q = np.zeros(100); QC_hs_2q_teo = np.zeros(100)
    # S entropias
    S_l_2q = np.zeros(100);   S_l_2q_teo = np.zeros(100)
    S_vn_2q = np.zeros(100);  S_vn_2q_teo = np.zeros(100)
    sj_2q = np.zeros(100)
    for j in range(0, 100):
        sj_2q[j] = str(j)  #;print(sj_2q)
        x4[j] = 1.9; x5[j] = 2.1        
        # circuito com 2qubits london
        path4 = '/home/mauro/Documentos/documents/random_circ/2qubits_london/rho_qc/%i.npy' %j
        rho4 = np.load(path4, 'r')          #;print(rho4)
        #rhoc é o rho reduzido que retorna da função traço parcial.
        rhoc = pT.pTraceR(2, 2, rho4)       #;print(rhoc)
        #relaçoes da norma-l1 (l1)
        cl1_c = coh.coh_l1(rhoc)            #;print(cl1_c)
        pl1_c = coh.predict_l1(rhoc)        #;print(pl1_c)
        c_p_c = cl1_c + pl1_c               #;print(c_p_c)
        qc_l1_2q = coh.qcorr_l1(2, 2, rho4) #;print(qc_l1_2q)
        #soma de coh + predic + qcorr
        soma_C_P_QC_2q = c_p_c + qc_l1_2q   #;print(soma_C_P_QC_2q)

        #relações Hilbet-Schmidt
        C_hs_2q[j] = coh.coh_hs(rhoc)
        P_hs_2q[j] = coh.predict_hs_l(rhoc)
        QC_hs_2q[j] = coh.qcorr_hs(rhoc)
        
        #relações Wigner-Yanase (wy)       
        C_wy_2q[j] = coh.coh_wy(rhoc)          
        QC_wy_2q[j] = coh.qcorr_wy(rhoc)

        #relações de entropia realtiva (re)
        C_re_2q[j] = coh.coh_re(rhoc)         
        P_vn_2q[j] = coh.predict_vn(rhoc)     

        #relaçoes entropias
        S_l_2q[j] = 1 - ent.purity(rhoc)       
        S_vn_2q[j] = ent.von_neumann(rhoc)

        #para calcular o rho_tc (teorico de d) 2 qubits london
        path5 = '/home/mauro/Documentos/documents/random_circ/2qubits_london/psi_qc/%i.npy' %j
        psi_5 = np.load(path5, 'r')         #;print(psi_5)      
        proj_psi_5 = mf.proj(4, psi_5)      #;print(proj_psi_5)
        rho_tc = pT.pTraceR(2, 2, proj_psi_5) #;print(rho_tc)
        #relaçoes da norma-l1 (l1)
        cl1_tc = coh.coh_l1(rho_tc)         #;print(cl1_tc) 
        pl1_tc = coh.predict_l1(rho_tc)            #;print(pl1_tc)
        c_p_tc = cl1_tc + pl1_tc            #;print(c_p_tc)
        qc_l1_2q_teo = coh.qcorr_l1(2, 2, proj_psi_5) #;print(qc_l1_2q_teo)
        #soma de coh + predic + qcorr
        soma_C_P_QC_2q_teo = c_p_tc + qc_l1_2q_teo    #;print(soma_C_P_QC_2q_teo)
        #definições para fazer os gráficos
        y4[j] = c_p_c.real ; y5[j] = c_p_tc.real
        coh_2q[j] = cl1_c ; coh_2q_teo[j] = cl1_tc
        previ_2q[j] = pl1_c.real ; previ_2q_teo[j] = pl1_tc.real
        qcorr_2q[j] = qc_l1_2q ; qcorr_2q_teo[j] = qc_l1_2q_teo
        soma_2q[j] = soma_C_P_QC_2q.real ; soma_2q_teo[j] = soma_C_P_QC_2q_teo.real

        #relações Hilbet-Schmidt
        C_hs_2q_teo[j] = coh.coh_hs(rho_tc)
        P_hs_2q_teo[j] = coh.predict_hs_l(rho_tc)
        QC_hs_2q_teo[j] = coh.qcorr_hs(rho_tc)

        #relações Wigner-Yanase        
        C_wy_2q_teo[j] = coh.coh_wy(rho_tc)
        QC_wy_2q_teo[j] = coh.qcorr_wy(rho_tc)

        #relações de entropia realtiva (re)
        C_re_2q_teo[j] = coh.coh_re(rho_tc)         
        #P_vn_2q_teo[j] = coh.predict_vn(rho_tc) #math domain error

        #relações entropias
        S_l_2q_teo[j] = 1 - ent.purity(rho_tc)
        S_vn_2q_teo[j] = ent.von_neumann(rho_tc)
    
    
    x10 = np.zeros(150); y10 = np.zeros(150)
    x11 = np.zeros(150); y11 = np.zeros(150)
    coh_3q = np.zeros(150); coh_3q_teo = np.zeros(150)
    previ_3q = np.zeros(150); previ_3q_teo = np.zeros(150)
    qcorr_3q = np.zeros(150); qcorr_3q_teo = np.zeros(150)
    soma_3q = np.zeros(150); soma_3q_teo = np.zeros(150)
    sj_3q = np.zeros(150)
    C_hs_3q = np.zeros(150); C_hs_3q_teo = np.zeros(150)
    P_hs_3q = np.zeros(150);  P_hs_3q_teo = np.zeros(150)
    C_wy_3q = np.zeros(150);  C_wy_3q_teo = np.zeros(150)
    QC_wy_3q = np.zeros(150); QC_wy_3q_teo = np.zeros(150)
    C_re_3q = np.zeros(150);  C_re_3q_teo = np.zeros(150)
    P_vn_3q = np.zeros(150);  P_vn_3q_teo = np.zeros(150)
    QC_hs_3q= np.zeros(150); QC_hs_3q_teo = np.zeros(150)
    # S entropias
    S_l_3q = np.zeros(150);   S_l_3q_teo = np.zeros(150)
    S_vn_3q = np.zeros(150);  S_vn_3q_teo = np.zeros(150)
    for j in range (0, 150):
        sj_3q[j] = str(j)     #;print(sj_3q)
        x10[j] = 3.9 ; x11[j] = 4.1
        #circuito com 3 qubits yorktown (ibmqx2) 150 circuitos
        path10 = '/home/mauro/Documentos/documents/random_circ/3qubits_ibmqx2/rho_qc/%i.npy' %j
        rho10 = np.load(path10, 'r')        #;print(rho10)
        rhof = pT.pTraceR(4, 2, rho10)      #;print(rhof)
        #relaçoes de norma-l1
        cl1_f = coh.coh_l1(rhof)             #;print(cl1_f)
        pl1_f = coh.predict_l1(rhof)         #;print(pl1_f)
        c_p_f = cl1_f + pl1_f                #;print( c_p_f)
        qc_l1_3q = coh.qcorr_l1(4, 2, rho10) #;print(qc_l1_3q) 
        soma_C_P_QC_3q = c_p_f + qc_l1_3q    #;print(soma_C_P_QC_3q)

        #relações de Hilbet-Schmidt
        C_hs_3q[j] = coh.coh_hs(rhof)
        P_hs_3q[j] = coh.predict_hs_l(rhof)
        QC_hs_3q[j] = coh.qcorr_hs(rhof)

        #relações Wigner-Yanase
        C_wy_3q[j] = coh.coh_wy(rhof)
        QC_wy_3q[j] = coh.qcorr_wy(rhof)

        #relações de entropia realtiva (re)
        C_re_3q[j] = coh.coh_re(rhof)         
        P_vn_3q[j] = coh.predict_vn(rhof)

        #relações entropias
        S_l_3q[j] = 1 - ent.purity(rhof)         
        S_vn_3q[j] = ent.von_neumann(rhof)

        #para calcular o rho teorico 3qubits
        path11 = '/home/mauro/Documentos/documents/random_circ/3qubits_ibmqx2/psi_qc/%i.npy' %j
        psi_11 = np.load(path11, 'r')            #;print(psi_11)
        proj_psi_11 = mf.proj(8, psi_11)         #;print(proj_psi_11)
        rho_tf = pT.pTraceR(4, 2, proj_psi_11)   #;print(rho_tf)
        cl1_tf = coh.coh_l1(rho_tf)              #;print(cl1_tf)
        pl1_tf = coh.predict_l1(rho_tf)          #;print(pl1_tf)
        c_p_tf = cl1_tf + pl1_tf                 #;print(c_p_tf)
        qc_l1_3q_teo = coh.qcorr_l1(4, 2, proj_psi_11) #;print(qc_l1_3q_teo)    
        #soma de coh + predic + qcorr
        soma_C_P_QC_3q_teo = c_p_tf + qc_l1_3q_teo #;print(soma_C_P_QC_3q_teo)        

        #definições para fazer os gráficos
        y10[j] = c_p_f.real ; y11[j] = c_p_tf.real
        coh_3q[j] = cl1_f ; coh_3q_teo[j] = cl1_tf
        previ_3q[j] = pl1_f.real ; previ_3q_teo[j] = pl1_tf.real
        qcorr_3q[j] = qc_l1_3q ; qcorr_3q_teo[j] = qc_l1_3q_teo
        soma_3q[j] = soma_C_P_QC_3q.real ; soma_3q_teo[j] = soma_C_P_QC_3q_teo.real

        #relações Hilbet-Schmidt
        C_hs_3q_teo[j] = coh.coh_hs(rho_tf)
        P_hs_3q_teo[j] = coh.predict_hs_l(rho_tf)
        QC_hs_3q_teo[j] = coh.qcorr_hs(rho_tf)
        
        #relações de Wigner-Yanase
        C_wy_3q_teo[j] = coh.coh_wy(rho_tf)
        QC_wy_3q_teo[j] = coh.qcorr_wy(rho_tf)
        
        #relações de entropia realtiva (re)
        C_re_3q_teo[j] = coh.coh_re(rho_tf)    
        #P_vn_3q_teo[j] = coh.predict_vn(rho_tf) # math domain error

        #relações entropias
        S_l_3q_teo[j] = 1 - ent.purity(rho_tf)         
        S_vn_3q_teo[j] = ent.von_neumann(rho_tf)      
        
    x6 = np.zeros(200); y6 = np.zeros(200)
    x7 = np.zeros(200); y7 = np.zeros(200)
    coh_4q = np.zeros(200); coh_4q_teo = np.zeros(200)
    previ_4q = np.zeros(200); previ_4q_teo = np.zeros(200)
    qcorr_4q = np.zeros(200); qcorr_4q_teo = np.zeros(200)
    soma_4q = np.zeros(200); soma_4q_teo = np.zeros(200)
    sj_4q = np.zeros(200)
    C_hs_4q = np.zeros(200); C_hs_4q_teo = np.zeros(200)
    P_hs_4q = np.zeros(200);  P_hs_4q_teo = np.zeros(200)
    C_wy_4q = np.zeros(200);  C_wy_4q_teo = np.zeros(200)
    QC_wy_4q = np.zeros(200); QC_wy_4q_teo = np.zeros(200)
    C_re_4q = np.zeros(200);  C_re_4q_teo = np.zeros(200)
    P_vn_4q = np.zeros(200);  P_vn_4q_teo = np.zeros(200)
    QC_hs_4q= np.zeros(200); QC_hs_4q_teo = np.zeros(200)
    # S entropias
    S_l_4q = np.zeros(200);   S_l_4q_teo = np.zeros(200)
    S_vn_4q = np.zeros(200);  S_vn_4q_teo = np.zeros(200)
    for j in range(0, 200):
        sj_4q[j] = str(j)     #;print(sj_4q)
        #circuito com 4qubits ibmqx2 200 circuitos
        x6[j] = 7.9; x7[j] = 8.1
        #para calcular o rho experimental de 4qubits do ibmqx2 200 circuitos        
        path6 = '/home/mauro/Documentos/documents/random_circ/4qubits_real_ibmqx2/rho_reduced/%i.npy' %j
        rho6 = np.load(path6, 'r')         #;print(rho6)
        #rhod é o rho reduzido que retorna direto do arquivo da IBM, pq la foi feito tomo em 3 qubits
        #nos outros casos devemos tomar o traço para encontrar o rho reduzido
        rhod = rho6                       #;print(rhod)
        #relação norma-l1
        cl1_d = coh.coh_l1(rhod)          #;print(cl1_d)
        pl1_d = coh.predict_l1(rhod)             #;print(pl1_d)
        c_p_d = cl1_d + pl1_d             #;print(c_p_d)
        dimens_d = rhod.shape[0]          #;print('dimens_d = ', dimens_d)
        #l1-norm quantum correlation
        #adaptado do programa coherence.py, de maneira que aqui não faz o traço parcial,
        #no orginal é rhoA = pTrace.ptraceB(da, db, rhoAB)
        #tendo em vista que o rhoA ja veio reduzido da tomografia do IBMQE, devido a fazer 
        #tomografia de 3 qubits.
        def qcorr_l1_adaptado(da,db,rhoAB):
            rhoA = rhoAB
            qc = 0
            for j in range(0,da-1):
                for k in range(j+1,da):
                    qc += math.sqrt(rhoA[j,j]*rhoA[k,k]) - abs(rhoA[j,k])
            return 2*qc
        qc_l1_4q_adap = qcorr_l1_adaptado(8, 2, rho6) #;print(qc_l1_4q_adap)
        #soma de coh + predic + qcorr
        soma_C_P_QC_4q = c_p_d + qc_l1_4q_adap         #;print(soma_C_P_QC_4q)

        #relações de Hilbet-Schmidt
        C_hs_4q[j] = coh.coh_hs(rhod)
        P_hs_4q[j] = coh.predict_hs_l(rhod)
        QC_hs_4q[j] = coh.qcorr_hs(rhod)

        #relações de Wigner-Yanase
        C_wy_4q[j] = coh.coh_wy(rhod)
        QC_wy_4q[j] = coh.qcorr_wy(rhod)

        #relações de entropia realtiva (re)
        C_re_4q[j] = coh.coh_re(rhod)    
        P_vn_4q[j] = coh.predict_vn(rhod)

        #relações entropias
        S_l_4q[j] = 1 - ent.purity(rhod)         
        S_vn_4q[j] = ent.von_neumann(rhod)      

        #para calcular o rho teorico 4qubits ibmqx2 200 circuitos
        path7 = '/home/mauro/Documentos/documents/random_circ/4qubits_real_ibmqx2/psi_qc/%i.npy' %j
        psi_7 = np.load(path7, 'r')           #;print(psi_7)
        proj_psi_7 = mf.proj(16, psi_7)       #;print(proj_psi_7)
        rho_td = pT.pTraceR(8, 2, proj_psi_7) #;print(rho_td)
        #relação norma-l1
        cl1_td = coh.coh_l1(rho_td)         #;print(cl1_td)
        pl1_td = coh.predict_l1(rho_td)            #;print(pl1_td)
        c_p_td = cl1_td + pl1_td            #;print(c_p_td)
        qc_l1_4q_teo = coh.qcorr_l1(8, 2, proj_psi_7) #;print(qc_l1_4q_teo)        
        #soma de coh + predic + qcorr
        soma_C_P_QC_4q_teo = c_p_td + qc_l1_4q_teo #;print(soma_C_P_QC_4q_teo)
        #definições para fazer os gráficos
        y6[j] = c_p_d.real ; y7[j] = c_p_td.real
        coh_4q[j] = cl1_d; coh_4q_teo[j] = cl1_td
        previ_4q[j] = pl1_d.real; previ_4q_teo[j] = pl1_td.real
        qcorr_4q[j] = qc_l1_4q_adap ; qcorr_4q_teo[j] = qc_l1_4q_teo
        soma_4q[j] = soma_C_P_QC_4q.real; soma_4q_teo[j] = soma_C_P_QC_4q_teo.real

        #relaçoes Hilbet-Schmidt
        C_hs_4q_teo[j] = coh.coh_hs(rho_td)
        P_hs_4q_teo[j] = coh.predict_hs_l(rho_td)
        QC_hs_4q_teo[j] = coh.qcorr_hs(rho_td)
        
        #relação de Wigner-Yanase
        C_wy_4q_teo[j] = coh.coh_wy(rho_td)
        QC_wy_4q_teo[j] = coh.qcorr_wy(rho_td)

        #relações de entropia realtiva (re)
        C_re_4q_teo[j] = coh.coh_re(rho_td)    
        #P_vn_4q_teo[j] = coh.predict_vn(rho_td) # math domain error

        #relaçoes entropias
        S_l_4q_teo[j] = 1 - ent.purity(rho_td)         
        S_vn_4q_teo[j] = ent.von_neumann(rho_td)



    #### linhas para plot parcial das relações de  re (relative entropy).
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o gráfico, nos outros ajusta automaticamente
    plt.xlabel('Dimension')
    '''
    plt.ylabel('$C_{re}$')
    plt.plot(x4, C_re_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, C_re_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, C_re_3q, '.', markersize=2,   color='b')
    plt.plot(x11, C_re_3q_teo, 'x', markersize=2,  color='r')
    plt.plot(x6, C_re_4q, '.', markersize=2,   color='b')
    plt.plot(x7, C_re_4q_teo, 'x', markersize=2,  color='r')
    '''
    '''
    plt.ylabel('$P_{nv}$')
    plt.plot(x4, P_vn_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, P_vn_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, P_vn_3q, '.', markersize=2,  color='b')
    plt.plot(x11, P_vn_3q_teo, 'x', markersize=2, color='r')
    plt.plot(x6, P_vn_4q, '.', markersize=2,   color='b')
    plt.plot(x7, P_vn_4q_teo, 'x', markersize=2,  color='r')    
    '''
    '''
    plt.ylabel('$S_{vn}$')
    plt.plot(x4, S_vn_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, S_vn_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, S_vn_3q, '.', markersize=2,   color='b')
    plt.plot(x11, S_vn_3q_teo, 'x', markersize=2,  color='r')
    plt.plot(x6, S_vn_4q, '.', markersize=2,   color='b')
    plt.plot(x7, S_vn_4q_teo, 'x', markersize=2,  color='r')
    '''
    '''
    plt.ylabel('$C_{re}+P_{vn}$')
    plt.plot(x4, (C_re_2q+P_vn_2q), '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, (C_re_2q_teo+P_vn_2q_teo), 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, (C_re_3q+P_vn_3q), '.', markersize=2,   color='b')
    plt.plot(x11, (C_re_3q_teo+P_vn_3q_teo), 'x', markersize=2,  color='r')
    plt.plot(x6, (C_re_4q+P_vn_4q), '.', markersize=2,   color='b')
    plt.plot(x7, (C_re_4q_teo+P_vn_4q_teo), 'x', markersize=2,  color='r')
    '''
    '''
    plt.ylabel('$C_{re}+P_{vn}+S_{vn}$')
    plt.plot(x4, (C_re_2q+P_vn_2q+S_vn_2q), '.', markersize=2, label='Exp.($C_{re}+P_{vn}+S_{vn}$)',  color='b')
    plt.plot(x5, (C_re_2q_teo+P_vn_2q_teo+S_vn_2q_teo), 'x', markersize=2, label='The.($C_{re}+P_{vn}+S_{vn}$)', color='r')
    plt.plot(x10, (C_re_3q+P_vn_3q+S_vn_3q), '.', markersize=2,   color='b')
    plt.plot(x11, (C_re_3q_teo+P_vn_3q_teo+S_vn_3q_teo), 'x', markersize=2,  color='r')
    plt.plot(x6, (C_re_4q+P_vn_4q+S_vn_4q), '.', markersize=2,   color='b') #color='lawngreen'
    plt.plot(x7, (C_re_4q_teo+P_vn_4q_teo+S_vn_4q_teo), 'x', markersize=2,  color='r') #color='darkturquoise'
    '''
    '''
    plt.xlim((0, 10))
    plt.ylim((0, 3.5))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,math.log(2), marker='_', **marker_style)
    plt.plot(4,math.log(4), marker='_', **marker_style)
    plt.plot(8,math.log(8), marker='_', **marker_style)
    #plt.grid(True)
    plt.legend()
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_hs.eps',
    #                format='eps', dpi=100)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_hs.png',
    #                format='png', dpi=100)
    plt.show()
    '''
    
    '''
    #### linhas para plot parcial das relações de  wy.
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o gráfico, nos outros ajusta automaticamente
    plt.xlabel('Dimension')

    plt.ylabel('$C_{wy}$')
    plt.plot(x4, C_wy_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, C_wy_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, C_wy_3q, '.', markersize=2,   color='b')
    plt.plot(x11, C_wy_3q_teo, 'x', markersize=2,  color='r')
    plt.plot(x6, C_wy_4q, '.', markersize=2,   color='b')
    plt.plot(x7, C_wy_4q_teo, 'x', markersize=2,  color='r')

    plt.ylabel('$P_{hs}$')
    plt.plot(x4, P_hs_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, P_hs_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, P_hs_3q, '.', markersize=2,  color='b')
    plt.plot(x11, P_hs_3q_teo, 'x', markersize=2, color='r')
    plt.plot(x6, P_hs_4q, '.', markersize=2,   color='b')
    plt.plot(x7, P_hs_4q_teo, 'x', markersize=2,  color='r')    
    
    plt.ylabel('$W_{wy}$')
    plt.plot(x4, QC_wy_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, QC_wy_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, QC_wy_3q, '.', markersize=2,   color='b')
    plt.plot(x11, QC_wy_3q_teo, 'x', markersize=2,  color='r')
    plt.plot(x6, QC_wy_4q, '.', markersize=2,   color='b')
    plt.plot(x7, QC_wy_4q_teo, 'x', markersize=2,  color='r')
    
    plt.ylabel('$C_{wy}+P_{hs}$')
    plt.plot(x4, (C_wy_2q+P_hs_2q), '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, (C_wy_2q_teo+P_hs_2q_teo), 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, (C_wy_3q+P_hs_3q), '.', markersize=2,   color='b')
    plt.plot(x11, (C_wy_3q_teo+P_hs_3q_teo), 'x', markersize=2,  color='r')
    plt.plot(x6, (C_wy_4q+P_hs_4q), '.', markersize=2,   color='b')
    plt.plot(x7, (C_wy_4q_teo+P_hs_4q_teo), 'x', markersize=2,  color='r')

    plt.ylabel('$C_{wy}+P_{hs}+W_{wy}$')
    plt.plot(x4, (C_wy_2q+P_hs_2q+QC_wy_2q), '.', markersize=2, label='Experimental',  color='lawngreen')
    plt.plot(x5, (C_wy_2q_teo+P_hs_2q_teo+QC_wy_2q_teo), 'x', markersize=2, label='Theoretical', color='darkturquoise')
    plt.plot(x10, (C_wy_3q+P_hs_3q+QC_wy_3q), '.', markersize=2,   color='lawngreen')
    plt.plot(x11, (C_wy_3q_teo+P_hs_3q_teo+QC_wy_3q_teo), 'x', markersize=2,  color='darkturquoise')
    plt.plot(x6, (C_wy_4q+P_hs_4q+QC_wy_4q), '.', markersize=2,   color='lawngreen')
    plt.plot(x7, (C_wy_4q_teo+P_hs_4q_teo+QC_wy_4q_teo), 'x', markersize=2,  color='darkturquoise')
    
    plt.xlim((0, 10))
    plt.ylim((0, 1.5))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,0.5, marker='_', **marker_style)
    plt.plot(4,0.75, marker='_', **marker_style)
    plt.plot(8,0.875, marker='_', **marker_style)
    #plt.grid(True)
    plt.legend()
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_hs.eps',
    #                format='eps', dpi=100)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_hs.png',
    #                format='png', dpi=100)
    plt.show()
    '''        
    '''
    #### linhas para plot parcial de C_hs.
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o gráfico, nos outros ajusta automaticamente
    plt.xlabel('Dimension')
    plt.ylabel('$C_{hs}$')
    plt.plot(x4, C_hs_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, C_hs_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, C_hs_3q, '.', markersize=2,   color='b')
    plt.plot(x11, C_hs_3q_teo, 'x', markersize=2,  color='r')
    plt.plot(x6, C_hs_4q, '.', markersize=2,   color='b')
    plt.plot(x7, C_hs_4q_teo, 'x', markersize=2,  color='r')
    plt.xlim((0, 10))
    plt.ylim((0, 1.5))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,0.5, marker='_', **marker_style)
    plt.plot(4,0.75, marker='_', **marker_style)
    plt.plot(8,0.875, marker='_', **marker_style)
    #plt.grid(True)
    plt.legend()
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_hs.eps',
    #                format='eps', dpi=100)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_hs.png',
    #                format='png', dpi=100)
    plt.show()
    '''
    
    ### linhas para plot parcial de P_hs.
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o gráfico, nos outros ajusta automaticamente
    plt.xlabel('Dimension')
    plt.ylabel('$P_{hs}$')
    plt.plot(x4, P_hs_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, P_hs_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, P_hs_3q, '.', markersize=2,   color='b')
    plt.plot(x11, P_hs_3q_teo, 'x', markersize=2,  color='r')
    plt.plot(x6, P_hs_4q, '.', markersize=2,   color='b')
    plt.plot(x7, P_hs_4q_teo, 'x', markersize=2,  color='r')    
    plt.xlim((0, 10))
    plt.ylim((0, 1.5))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,0.5, marker='_', **marker_style)
    plt.plot(4,0.75, marker='_', **marker_style)
    plt.plot(8,0.875, marker='_', **marker_style)
    #plt.grid(True)   
    plt.legend()
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/P_hs.eps',
    #                format='eps', dpi=100)    
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/P_hs.png',
    #                format='png', dpi=100)    
    plt.show()
    
    '''
    #### linhas para plot parcial de S_l.
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o gráfico, nos outros ajusta automaticamente
    plt.xlabel('Dimension')
    plt.ylabel('$S_{l}$')
    plt.plot(x4, QC_hs_2q, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, QC_hs_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, QC_hs_3q, '.', markersize=2,  color='b')
    plt.plot(x11, QC_hs_3q_teo, 'x', markersize=2,  color='r')
    plt.plot(x6, QC_hs_4q, '.', markersize=2,  color='b')
    plt.plot(x7, QC_hs_4q_teo, 'x', markersize=2,  color='r')
    plt.xlim((0, 10))
    plt.ylim((0, 1.5))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,0.5, marker='_', **marker_style)
    plt.plot(4,0.75, marker='_', **marker_style)
    plt.plot(8,0.875, marker='_', **marker_style)
    plt.legend()
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/S_l.eps',
    #                format='eps', dpi=100)    
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/S_l.png',
    #                format='png', dpi=100)    
    plt.show()
    
    '''
    ### linhas para plot parcial de C_hs+P_hs .
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o gráfico, nos outros ajusta automaticamente
    plt.xlabel('Dimension')
    plt.ylabel('$C_{hs}+P_{hs}$')
    plt.plot(x4, (C_hs_2q+P_hs_2q), '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, (C_hs_2q_teo+P_hs_2q_teo), 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, (C_hs_3q+P_hs_3q), '.', markersize=2,  color='b')
    plt.plot(x11, (C_hs_3q_teo+P_hs_3q_teo), 'x', markersize=2, color='r')
    plt.plot(x6, (C_hs_4q+P_hs_4q), '.', markersize=2,   color='b')
    plt.plot(x7, (C_hs_4q_teo+P_hs_4q_teo), 'x', markersize=2,  color='r')    
    plt.xlim((0, 10))
    plt.ylim((0, 1.5))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,0.5, marker='_', **marker_style)
    plt.plot(4,0.75, marker='_', **marker_style)
    plt.plot(8,0.875, marker='_', **marker_style)
    plt.legend()
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_P_hs.eps',
    #                format='eps', dpi=100)    
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_P_hs.png',
    #                format='png', dpi=100)    
    #plt.show()
    #print(C_hs_4q+P_hs_4q+S_l_4q)
    #print(C_hs_4q_teo+P_hs_4q_teo+S_l_4q_teo)
    '''
    
    #### linhas para plot parcial de C_hs+P_hs+S_l.
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o gráfico, nos outros ajusta automaticamente
    plt.xlabel('Dimension')
    plt.ylabel('$C_{hs}+P_{hs}+S_{l}$')
    plt.plot(x4, (C_hs_2q + P_hs_2q + S_l_2q), '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, (C_hs_2q_teo + P_hs_2q_teo + S_l_2q_teo), 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, (C_hs_3q + P_hs_3q + S_l_3q), '.', markersize=2,  color='b')
    plt.plot(x11, (C_hs_3q_teo + P_hs_3q_teo + S_l_3q_teo), 'x', markersize=2, color='r')
    plt.plot(x6, (C_hs_4q + P_hs_4q + S_l_4q), '.', markersize=2,  color='b')
    plt.plot(x7, (C_hs_4q_teo + P_hs_4q_teo + S_l_4q_teo), 'x', markersize=2,  color='r')
    plt.xlim((0, 10))
    plt.ylim((0, 1.5))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,0.5, marker='_', **marker_style)
    plt.plot(4,0.75, marker='_', **marker_style)
    plt.plot(8,0.875, marker='_', **marker_style)
    plt.legend()
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_P_S_hs.eps',
    #                format='eps', dpi=100)    
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/plots_hs/C_P_S_hs.png',
    #                format='png', dpi=100)    
    plt.show()
    
    '''
    ###linhas para grafico de C_l1+P_l1
    fig = plt.figure(figsize=(6,4.5), dpi=100)
    plt.xlabel('Dimension')     ;plt.ylabel('$C_{l_{1}}+P_{l_{1}}$')
    plt.plot(x4, y4, '.', markersize=2, label='Experimental',  color='b')
    plt.plot(x5, y5, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, y10, '.', markersize=2,  color='b')
    plt.plot(x11, y11, 'x', markersize=2,  color='r')
    plt.plot(x6, y6, '.', markersize=2,  color='b')
    plt.plot(x7, y7, 'x', markersize=2,  color='r')
    plt.xlim((0, 10))
    plt.ylim((0, 8))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,1, marker='_', **marker_style)
    plt.plot(4,3, marker='_', **marker_style)
    plt.plot(8,7, marker='_', **marker_style)
    #plt.grid(True)   
    plt.legend()
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/c_p_l1.eps',
    #                format='eps', dpi=100)    
    plt.show()
    
    #linhas para fazer os gráficos so de C_l1 separado
    fig = plt.figure(figsize=(6,4.5), dpi=100)
    plt.xlabel('Dimension')     ;plt.ylabel('$C_{l_{1}}$')
    plt.plot(x4, coh_2q, '.', markersize=2, label='Experimental', color='b')
    plt.plot(x5, coh_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, coh_3q, '.', markersize=2, color='b')     #label='3 qubits, exp 150, 4 gates',
    plt.plot(x11, coh_3q_teo, 'x', markersize=2, color='r') #label='3 qubits, teo 150, 4 gates',
    plt.plot(x6, coh_4q, '.', markersize=2, color='b')      #label='4 qubits, exp 200, 4 gates',
    plt.plot(x7, coh_4q_teo, 'x', markersize=2, color='r')  #label='4 qubits, teo 200, 4 gates',
    plt.xlim((0, 10))
    plt.ylim((0, 8))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,1, marker='_', **marker_style)
    plt.plot(4,3, marker='_', **marker_style)
    plt.plot(8,7, marker='_', **marker_style)
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/c_l1.eps',
    #                format='eps', dpi=100)
    plt.show()    
    '''
    '''
    #linhas para fazer o gráfico somente P_l1 separado
    fig = plt.figure(figsize=(6,4.5), dpi=100)
    plt.xlabel('Dimension')     ;plt.ylabel('$P_{l_{1}}$')
    plt.plot(x4, previ_2q, '.', markersize=2, label='Experimental', color='b')
    plt.plot(x5, previ_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, previ_3q, '.', markersize=2, color='b')     #label='3 qubits, exp 150, 4 gates',
    plt.plot(x11, previ_3q_teo, 'x', markersize=2, color='r') #label='3 qubits, teo 150, 4 gates',
    plt.plot(x6, previ_4q, '.', markersize=2, color='b')     #label='4 qubits, exp 200, 4 gates',
    plt.plot(x7, previ_4q_teo, 'x', markersize=2, color='r') #label='4 qubits, teo 200, 4 gates',
    plt.xlim((0, 10))
    plt.ylim((0, 8))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,1, marker='_', **marker_style)
    plt.plot(4,3, marker='_', **marker_style)
    plt.plot(8,7, marker='_', **marker_style)
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/p_l1.eps',
    #                format='eps', dpi=100)
    plt.show()
    '''
    '''
    #linhas para fazer o gráfico somente Quantum correlation (qcorr_l1) separado
    fig = plt.figure(figsize=(6,4.5), dpi=100)
    plt.xlabel('Dimension')     ;plt.ylabel('$W_{l_{1}}$')
    plt.plot(x4, qcorr_2q, '.', markersize=2, label='Experimental', color='b')
    plt.plot(x5, qcorr_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, qcorr_3q, '.', markersize=2,  color='b')     #label='3 qubits, exp 150, 4 gates',
    plt.plot(x11, qcorr_3q_teo, 'x', markersize=2,  color='r') #label='3 qubits, teo 150, 4 gates',
    plt.plot(x6, qcorr_4q, '.', markersize=2,  color='b')      #label='4 qubits, exp 200, 4 gates',
    plt.plot(x7, qcorr_4q_teo, 'x', markersize=2, color='r')   #label='4 qubits, teo 200, 4 gates',
    plt.xlim((0, 10))
    plt.ylim((0, 8))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,1, marker='_', **marker_style)
    plt.plot(4,3, marker='_', **marker_style)
    plt.plot(8,7, marker='_', **marker_style)
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/qcorr_l1.eps',
    #                format='eps', dpi=100)
    plt.show()
    
    #linhas para fazer o gráfico da soma cd C+P+QCcorr
    fig = plt.figure(figsize=(6,4.5), dpi=100)
    plt.xlabel('Dimension')     ;plt.ylabel('$C_{l_{1}}+P_{l_{1}}+W_{l_{1}}$')
    plt.plot(x4, soma_2q, '.', markersize=2, label='Experimental', color='b')
    plt.plot(x5, soma_2q_teo, 'x', markersize=2, label='Theoretical', color='r')
    plt.plot(x10, soma_3q, '.', markersize=2, color='b')    
    plt.plot(x11, soma_3q_teo, 'x', markersize=2, color='r')
    plt.plot(x6, soma_4q, '.', markersize=2, color='b')     
    plt.plot(x7, soma_4q_teo, 'x', markersize=2, color='r')
    plt.xlim((0, 10))
    plt.ylim((0, 8))
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,1, marker='_', **marker_style)
    plt.plot(4,3, marker='_', **marker_style)
    plt.plot(8,7, marker='_', **marker_style)
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/C_P_W_l1.eps',
    #                format='eps', dpi=100)
    plt.show()
    '''
    '''
    #linhas para fazer o gráfico da soma cd C+P+QC em função de j, onde j é o experimento 1,2,3,...
    fig = plt.figure(figsize=(6,4.5), dpi=100)
    plt.xlabel('j experimentos')    ;plt.ylabel('$C_{l_{1}}+P_{l_{1}}+W_{l_{1}}$')
    plt.plot(sj_2q, soma_2q, '.', markersize=2, label='Experimental', color='b')
    plt.plot(sj_2q, soma_2q_teo, linewidth=1, label='Theoretical', color='r')
    plt.plot(sj_3q, soma_3q, '.', markersize=2,  color='b')
    plt.plot(sj_3q, soma_3q_teo, linewidth=1,  color='r')  
    plt.plot(sj_4q, soma_4q, '.', markersize=2, color='b') 
    plt.plot(sj_4q, soma_4q_teo, linewidth=1,  color='r')  
    plt.legend(loc=6)
    plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/C_P_W_l1_func_j.eps',
                    format='eps', dpi=100)
    plt.show()
    
    #linhas para fazer o gráfico da soma cd C+P em função de j, onde j é o experimento 1,2,3,...
    fig = plt.figure(figsize=(6,4.5), dpi=100)
    plt.xlabel('j experimentos')       ;plt.ylabel('$C_{l_{1}}+P_{l_{1}}$')
    plt.plot(sj_2q, y4, '.', markersize=2, label='Experimental', color='b')
    plt.plot(sj_2q, y5, 'x', markersize=2,  color='r')
    plt.plot(sj_3q, y10, '.', markersize=2, label='Experimental', color='green')
    plt.plot(sj_3q, y11, 'x', markersize=2,  color='r')
    plt.plot(sj_4q, y6, '.', markersize=2, label='Experimental', color='cyan')
    plt.plot(sj_4q, y7, 'x', markersize=2, label='Theoretical', color='r')
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/C_P_l1_func_j.eps',
    #                format='eps', dpi=100)
    plt.show()
    
    #linhas para fazer de  Cl1+Pl1 e Cl1+Pl1+Wl1 na mesma figura.
    fig = plt.figure(figsize=(6,4.5), dpi=100)
    plt.xlabel('Dimension')
    #plt.ylabel('$W_{l_{1}}$')
    plt.plot(x4, y4, '.', markersize=2, label='Exp.($C_{l1}+P_{l1}$)',  color='b')
    plt.plot(x5, y5, 'x', markersize=2, label='The.($C_{l1}+P_{l1}$)', color='r')
    plt.plot(x10, y10, '.', markersize=2,  color='b')
    plt.plot(x11, y11, 'x', markersize=2,  color='r')
    plt.plot(x6, y6, '.', markersize=2,  color='b')
    plt.plot(x7, y7, 'x', markersize=2,  color='r')
    plt.xlim((0, 10))
    plt.ylim((0, 8))
    plt.plot(x4, soma_2q, '*', markersize=3, label='Exp.($C_{l1}+P_{l1}+W_{l1})$', color='lawngreen')
    plt.plot(x5, soma_2q_teo, '^', markersize=3, label='The.($C_{l1}+P_{l1}+W_{l1})$', color='darkturquoise')
    plt.plot(x10, soma_3q, '*', markersize=3, color='lawngreen')    
    plt.plot(x11, soma_3q_teo, '^', markersize=3, color='darkturquoise') 
    plt.plot(x6, soma_4q, '*', markersize=3, color='lawngreen')     
    plt.plot(x7, soma_4q_teo, '^', markersize=3, color='darkturquoise') 
    marker_style = dict(linestyle=':', color='0.8', markersize=20,
                    mfc="C0", mec="C0")
    plt.plot(2,1, marker='_', **marker_style)
    plt.plot(4,3, marker='_', **marker_style)
    plt.plot(8,7, marker='_', **marker_style)
    plt.legend(loc=2)
    #plt.savefig('/home/mauro/Documentos/documents/random_circ/plots/C_P_l1_somatotal.eps',
    #                format='eps', dpi=100)
    plt.show()
    '''

coh_predic_random()