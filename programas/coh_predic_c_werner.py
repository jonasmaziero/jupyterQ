import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import coherence as coh
import entropy as ent
import cmath
import math
import mat_func as mf
import pTrace as pT
from math import asin, acos, sqrt, pi

# função para o cálculo da previsibilidade, eq 87, 88 do artigo do Basso_Maziero,
# Quantitative wave-particle duality relations from the density matrix properties
def prev_l1(rho):
    d = rho.shape[0]
    prev = 0.0
    for j in range(0, d-1):
        for k in range(j+1, d):
            prev += math.sqrt((rho[j][j].real)*(rho[k][k].real))
    return d-1-2*prev
### função para o cálculo das correlações quânticas para l1.
### eq. 13 do artigo e adaptado com o fator de multiplicação 2 do github juputerQ/coherence
def qcorr_l1(rho):
    d = rho.shape[0]
    qc_l1 = 0.0
    for j in range(0, d-1):
        for k in range(j+1, d):
            qc_l1 += math.sqrt((rho[j][j].real)*(rho[k][k].real)) - abs(rho[j][k])
    return 2*qc_l1

###abrir arquivos para imprimir dados.
#arquivo1 = open('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/cl1.dat', 'w',)
#arquivo2 = open('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/pl1.dat', 'w',)
#arquivo3 = open('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/soma_c_p.dat', 'w',)
#arq4 = open('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/c_hs.dat', 'w',)
#arq5 = open('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/p_hs.dat', 'w',)
#arq6 = open('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/soma_c_p_hs.dat', 'w',)
#arq7 = open('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/qcl1.dat', 'w',)

# função que chama o arquivo onde está os dados extraídos do ibmqe
# chama as funções para o calcular a coerência e previssibilidade 
# e a soma das duas, tenta fazer o gráfico da C+P em função dos angulos alpha e theta.
def cohe_predic_werner():
    #definição para fazer o grafico em função de alpha e theta
    alpha = np.zeros(11);    theta = np.zeros(11)
    #definição para fazer o grafico em função de X e W
    eixo_x = np.zeros(11);   eixo_y = np.zeros(11)
    #soma de C coerencia e P preditibilidade
    soma_C_P = np.zeros((11,11))
    # C coerencia pra o grafico individual
    C = np.zeros((11,11));    C_hs = np.zeros((11,11));    C_wy = np.zeros((11,11));    C_re = np.zeros((11,11))
    # P preditibilidade pra o grafico individual
    P = np.zeros((11,11));    P_hs = np.zeros((11,11));    P_vn = np.zeros((11,11))
    # QC correlção quântica(W) para o grafico individual
    QC = np.zeros((11,11));    QC_wy = np.zeros((11,11))
    # S entropias para o grafico individual
    S_l = np.zeros((11,11));   S_vn = np.zeros((11,11))

    for k in range(0, 11):
        sk = str(k)                  #;print('k = ', k)
        x = k/10.0                   #;print('x = ', x)
        alpha[k] = 2.0*asin(sqrt(x)) #;print(alpha)
        eixo_x[k]= x                 #;print('eixo_x = ', eixo_x)
        #print()
        for j in range(0, 11):
            sj = str(j)                  #;print('j = ', j)
            w = j/10.0                   #;print('w = ', w)
            theta[j] = acos(-w)          #;print(theta)
            eixo_y[j] = w                #;print('eixo_y = ', eixo_y)
            #path_0 = '/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/simulador/'
            path_0 = '/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/'
            path_rho = path_0 + sk + '/rho_qc/%i.npy' %j
            rho = np.load(path_rho, 'r')         #;print(rho)
            #rho_red é o rho reduzido que retorna da função traço parcial.
            rho_red = pT.pTraceL(2, 2, rho)      #;print(rho_red)
            #relações da norma-l1 (l1)
            cl1 = coh.coh_l1(rho_red)            #;print(cl1)
            pl1 = prev_l1(rho_red)               #;print(pl1)
            c_p = cl1 + pl1                      #;print(c_p)
            qcl1 = qcorr_l1(rho_red)             #;print(qcl1)
            dimens_0 = rho_red.shape[0]          #;print(dimens_0)
            soma_C_P[k,j] = c_p                  #;print(soma_C_P)
            C[k,j] = cl1;  P[k,j] = pl1;  QC[k,j] = qcl1

            #relações de Hibert-Schmidt (hs)
            C_hs[k,j] = coh.coh_hs(rho_red)           #;print(C_hs)
            P_hs[k,j] = coh.predict_hs_l(rho_red)     #;print(P_hs)
            
            ###relações de Wigner-Yanase (wy)
            C_wy[k,j] = coh.coh_wy(rho_red)           #;print(C_wy)
            QC_wy[k,j] = coh.qcorr_wy(rho_red)        #;print(QC_wy)
            
            ###relaçoes de entropia realtiva (re)
            C_re[k,j] = coh.coh_re(rho_red)             #;print(C_re)
            P_vn[k,j] = coh.predict_vn(rho_red)         #;print(P_vn)
            
            #relaçoes entropias
            S_l[k,j] = 1 - ent.purity(rho_red)        #;print(S_l)
            S_vn[k,j] = ent.von_neumann(rho_red)      #;print(S_vn)
    
    ###escreve nos arquivos
    #arquivo1.write(str(C));    arquivo2.write(str(P));    arquivo3.write(str(soma_C_P))
    #arq4.write(str(XXXXXXXXXX)); #arq5.write(str(XXXXXXXXXX));#arq6.write(str(XXXXXXXXXX));
    #arq7.write(str(QC))

    ###salva em .npy ideal para trabalhar com os valores dentro do arquivo.
    #np.save('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/cl1', C)
    #np.save('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/pl1', P)
    #np.save('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/soma_c_p', soma_C_P)
    #np.save('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/qcl1', QC)
    #print(np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/qcl1.npy', 'r'))
    print(math.log(2))
    soma_C_P_teo = np.zeros((11,11))
    # C_teo coerencia pra o grafico individual
    C_teo = np.zeros((11,11)); C_hs_teo = np.zeros((11,11));  C_wy_teo = np.zeros((11,11));   C_re_teo = np.zeros((11,11))
    # P_teo preditibilidade pra o grafico individual
    P_teo = np.zeros((11,11));     P_hs_teo = np.zeros((11,11));    P_vn_teo = np.zeros((11,11))
    # QC_teo correlação quântica teorica
    QC_teo = np.zeros((11,11));    QC_wy_teo = np.zeros((11,11))
    # S entropias
    S_l_teo = np.zeros((11,11));    S_vn_teo = np.zeros((11,11))
    
    for k in range(0, 11):
        sk = str(k)         #; print('k = ', k)
        for j in range(0, 11):
            sj = str(j)     #;print('j = ', j)
            #path_1 = '/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/simulador/'
            path_1 = '/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/'
            path_psi = path_1 + sk + '/psi_qc/%i.npy' %j
            psi = np.load(path_psi, 'r')                #;print(psi)            
            #proj_psi_1 acha a rho atravez do projetor de psi
            proj_psi = mf.proj(4, psi)                  #;print(proj_psi)
            #rho_red_t é o rho reduzido que retorna da função traço parcial.
            rho_red_t = pT.pTraceL(2, 2, proj_psi)      #;print(rho_red_t)
            cl1_t = coh.coh_l1(rho_red_t)               #;print(cl1_t)
            pl1_t = prev_l1(rho_red_t)                  #;print(pl1_t)
            c_p_t = cl1_t + pl1_t                       #;print(c_p_t)
            qcl1_t = qcorr_l1(rho_red_t)                #;print(qcl1_t)
            dimens_t = rho_red_t.shape[0]               #;print(dimens_t)           
            soma_C_P_teo[k,j] = c_p_t                   #;print(soma_C_P_teo)    
            C_teo[k,j] = cl1_t;   P_teo[k,j] = pl1_t;    QC_teo[k,j] = qcl1_t
                        
            #relações de Hibert-Schmidt (hs)
            C_hs_teo[k,j] = coh.coh_hs(rho_red_t)             #;print(C_hs_teo)
            P_hs_teo[k,j] = coh.predict_hs_l(rho_red_t)       #;print(P_hs_teo)
            
            ###relações de Wigner-Yanase (wy)
            C_wy_teo[k,j] = coh.coh_wy(rho_red_t)           #;print(C_wy_teo)
            QC_wy_teo[k,j] = coh.qcorr_wy(rho_red_t)        #;print(QC_wy_teo)
            
            ###relaçoes de entropia realtiva (re)
            C_re_teo[k,j] = coh.coh_re(rho_red_t)           #;print(C_re_teo)
            P_vn_teo[k,j] = coh.predict_vn(rho_red_t)       #;print(P_vn_teo)

            #relaçoes entropias
            S_l_teo[k,j] = 1 - ent.purity(rho_red_t)       #;print(S_l_teo)
            S_vn_teo[k,j] = ent.von_neumann(rho_red_t)     #;print(S_vn_teo)

    ###parte para fazer o Cl1 e o Pl1 analitico
    X = np.arange(0, 1.1, 0.1)      #; print(X)
    W = np.arange(0, 1.1, 0.1)      #; print(W)
    X, W = np.meshgrid(X, W)        #;print(np.meshgrid(X, W))
    Cl1_ana = 2 * W * np.sqrt(X*(1-X))                  #;print(Cl1_ana)
    Pl1_ana = 1 - np.sqrt((1-W+2*W*X)*(1-W+2*W*(1-X)))  #;print(Pl1_ana)
    soma_C_P_ana = Cl1_ana + Pl1_ana                    #;print(soma_C_P_ana)

    '''
    ###linhas para fazer os gráficos para as variaveis entropia realtiva (re).
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o grafico, nos outros ajusta automaticamente
    ax1 = fig.add_subplot(111, projection='3d')                 #para ajustar o grafico
    ax1.set_xlabel('x')
    ax1.set_ylabel('w')
    #basta comentar e descomentar para ir alterando entre um grafico e outro.
    #ax1.set_zlabel('$C_{re}}$')
    #ax1.set_zlabel('$P_{vn}}$')
    #ax1.set_zlabel('$S_{vn}}$')
    #ax1.set_zlabel('$C_{re}+P_{vn}$')
    #ax1.set_zlabel('$C_{re}+P_{vn}+S_{vn}$')
    ###outro tipo de superficie areada
    ### a inversão das posição de X e W esta relacionado com o loop k,j onde faz o cálculo.
    #ax1.plot_wireframe(W, X, C_re_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    #ax1.plot_wireframe(W, X, P_vn_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    #ax1.plot_wireframe(W, X, S_vn_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    ax1.plot_wireframe(W, X, (C_re_teo+P_vn_teo), rstride=0, cstride=1, color = 'r', alpha=0.4)
    ax1.plot_wireframe(W, X, (C_re_teo+P_vn_teo+S_vn_teo), rstride=0, cstride=1, color = 'r', alpha=0.4)
    ###aqui inverter o W pelo X para o gráfico se ajustar.
    #ax1.scatter(W, X, C_re, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, P_vn, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, S_vn, label='Experimental', color = 'b', marker='o', s=2)
    ax1.scatter(W, X, (C_re+P_vn), label='Exp.($C_{wy}+P_{hs}$)', color = 'b', marker='o', s=2)
    ax1.scatter(W, X, (C_re+P_vn+S_vn), label='Exp.($C_{wy}+P_{hs}+W_{wy})$', color = 'darkturquoise', marker='^', s=3)
    ax1.view_init(elev=12, azim=-7)
    ax1.legend()
    plt.show()
    '''
    '''
    ###linhas para fazer os gráficos para as variaveis Wigner-Yanase (wy).
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o grafico, nos outros ajusta automaticamente
    ax1 = fig.add_subplot(111, projection='3d')                 #para ajustar o grafico
    ax1.set_xlabel('x')
    ax1.set_ylabel('w')
    #basta comentar e descomentar para ir alterando entre um grafico e outro.
    #ax1.set_zlabel('$C_{wy}}$')
    #ax1.set_zlabel('$P_{hs}}$')
    #ax1.set_zlabel('$W_{wy}}$')
    #ax1.set_zlabel('$C_{wy}+P_{hs}$')
    #ax1.set_zlabel('$C_{wy}+P_{hs}+W_{wy}$')
    ###outro tipo de superficie areada
    ### a inversão das posição de X e W esta relacionado com o loop k,j onde faz o cálculo.
    #ax1.plot_wireframe(W, X, C_wy_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    #ax1.plot_wireframe(W, X, P_hs_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    #ax1.plot_wireframe(W, X, QC_wy_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    ax1.plot_wireframe(W, X, (C_wy_teo+P_hs_teo), rstride=0, cstride=1, color = 'r', alpha=0.4)
    ax1.plot_wireframe(W, X, (C_wy_teo+P_hs_teo+QC_wy_teo), rstride=0, cstride=1, color = 'r', alpha=0.4)
    ###aqui inverter o W pelo X para o gráfico se ajustar.
    #ax1.scatter(W, X, C_wy, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, P_hs, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, QC_wy, label='Experimental', color = 'b', marker='o', s=2)
    ax1.scatter(W, X, (C_wy+P_hs), label='Exp.($C_{wy}+P_{hs}$)', color = 'b', marker='o', s=2)
    ax1.scatter(W, X, (C_wy+P_hs+QC_wy), label='Exp.($C_{wy}+P_{hs}+W_{wy}$)', color = 'darkturquoise', marker='^', s=3)
    ax1.view_init(elev=12, azim=-7)
    ax1.legend()
    plt.show()
    '''
    '''
    ###linhas para fazer os gráficos para as variaveis hs.
    ###ESTA TUDO CERTO FALTA FAZER OS GRAFICOS COM AS BARRAS DE ERROS PARA A TESE.
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o grafico, nos outros ajusta automaticamente
    ax1 = fig.add_subplot(111, projection='3d')                 #para ajustar o grafico
    ax1.set_xlabel('x')
    ax1.set_ylabel('w')
    #basta comentar e descomentar para ir alterando entre um grafico e outro.
    #ax1.set_zlabel('$C_{hs}}$')
    #ax1.set_zlabel('$P_{hs}}$')
    #ax1.set_zlabel('$S_{l}}$')
    #ax1.set_zlabel('$C_{hs}+P_{hs}+W_{hs}$')
    #ax1.set_zlabel('$C_{hs}+P_{hs}$')
    
    ###outro tipo de superficie areada
    ### a inversão das posição de X e W esta relacionado com o loop k,j onde faz o cálculo.
    #ax1.plot_wireframe(W, X, C_hs_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    #ax1.plot_wireframe(W, X, P_hs_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    #ax1.plot_wireframe(W, X, S_l_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    ax1.plot_wireframe(W, X, (C_hs_teo+P_hs_teo), rstride=0, cstride=1, color = 'r', alpha=0.4)
    ax1.plot_wireframe(W, X, (C_hs_teo+P_hs_teo+S_l_teo), rstride=0, cstride=1, color = 'r', alpha=0.4)

    ###aqui inverter o W pelo X para o gráfico se ajustar.
    #ax1.scatter(W, X, C_hs, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, P_hs, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, S_l, label='Experimental', color = 'b', marker='o', s=2)
    ax1.scatter(W, X, (C_hs+P_hs), label='Exp.($C_{hs}+P_{hs}$)', color = 'b', marker='o', s=2)
    ax1.scatter(W, X, (C_hs+P_hs+S_l), label='Exp.($C_{hs}+P_{hs}+S_{l}$)', color = 'darkturquoise', marker='^', s=3)
    ax1.view_init(elev=12, azim=-7)
    ax1.legend()
    plt.show()
    '''
    '''    
    ###linhas para fazer os gráficos so da coerência (C_{l_{1}}) separado
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o grafico, nos outros ajusta automaticamente
    ax1 = fig.add_subplot(111, projection='3d')                 #para ajustar o grafico
    ax1.set_xlabel('x')
    ax1.set_ylabel('w')
    ax1.set_zlabel('$C_{l_{1}}}$')
    #ax1.plot_surface(X, W, Cl1_ana, rstride=1, cstride=1,  alpha=0.3)
    ###outro tipo de superficie areada
    ax1.plot_wireframe(X, W, Cl1_ana, rstride=1, cstride=0, color = 'r', alpha=0.4)
    ###aqui inverter o W pelo X para o gráfico se ajustar.
    ax1.scatter(W, X, C, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, C_teo, label='Theoretical', color = 'r', marker='x', s=2)    
    ax1.view_init(elev=12, azim=-7)
    ax1.legend()
    #salvar no simualdo
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/simulador/plots/3d_x_w_coh.eps',
    #                format='eps', dpi=100)
    #salavar no real
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/plots/3d_x_w_coh_doteline.eps',
    #                format='eps', dpi=100)
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/plots/3d_x_w_coh_dots.pdf',
    #                format='pdf', dpi=100)
    #ax1.legend()
    plt.show()
    
    #linhas para fazer os gráficos so da preditibilidade (P_{l_{1}}) separado
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o grafico, nos outros ajusta automaticamente
    ax1 = fig.add_subplot(111, projection='3d')                 #para ajustar o grafico
    ax1.set_xlabel('x')
    ax1.set_ylabel('w')
    ax1.set_zlabel('$P_{l_{1}}}$')
    #ax1.plot_surface(X, W, Pl1_ana, rstride=1, cstride=1,  alpha=0.3)
    ###outro tipo de superficie areada
    ax1.plot_wireframe(X, W, Pl1_ana, rstride=1, cstride=0, color = 'r', alpha=0.4)
    ###aqui inverter o W pelo X para o gráfico se ajustar.
    ax1.scatter(W, X, P, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, P_teo, label='Theoretical', color = 'r', marker='x', s=2)
    ax1.view_init(elev=12, azim=-7)
    ax1.legend()
    #salvar no simualdo
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/simulador/plots/3d_x_w_pred.eps',
    #                format='eps', dpi=100)
    #salavar no real
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/plots/3d_x_w_pred_doteline.eps',
    #                format='eps', dpi=100)
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/plots/3d_x_w_pred_dots.pdf',
    #                format='pdf', dpi=100)
    #ax1.legend()
    #plt.show()

    ###linhas para o grafico de Cl1+Pl1 em função de X e W.
    fig = plt.figure(figsize=(6,4.5), dpi=100) #para ajustar o grafico, nos outros ajusta automaticamente
    ax1 = fig.add_subplot(111, projection='3d')                 #para ajustar o grafico
    ax1.set_xlabel('x')
    ax1.set_ylabel('w')
    ax1.set_zlabel('$C_{l_{1}}+P_{l_{1}}$')
    ###parte para fazer o Cl1 e o Pl1 analitico superficie o alpha é a transparencia do gráfico
    ###para superficie
    #ax1.plot_surface(X, W, soma_C_P_ana, rstride=1, cstride=1, alpha=0.3)
    ###outro tipo de superficie areada
    ax1.plot_wireframe(X, W, soma_C_P_ana, rstride=1, cstride=0, color = 'r', alpha=0.4)
    ###para pontos
    #ax1.scatter(X, W, soma_C_P_ana, label='Analytical', color = 'g', marker='^', s=2) #equivale a superficie     
    ###aqui inverter o W pelo X para o gráfico se ajustar.
    ax1.scatter(W, X, soma_C_P, label='Experimental', color = 'b', marker='o', s=2)
    #ax1.scatter(W, X, soma_C_P_teo, label='Theoretical', color = 'r', marker='x', s=2)
    #for j in range(0,11):
    #    ax1.scatter(eixo_x[j], eixo_y, soma_C_P[j], label='Exp', color = 'b', marker='o', s=2)
    #    ax1.scatter(eixo_x[j], eixo_y, soma_C_P_teo[j], label='The', color = 'r', marker='x', s=2)
    ax1.view_init(elev=12, azim=-7)
    ax1.legend()
    #salavar no simulado
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/simulador/plots/3d_x_w_C_P.eps',
    #                format='eps', dpi=100)
    #salavar no real
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/plots/3d_x_w_C_P_doteline.eps',
    #                format='eps', dpi=100)
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/rodada_3/plots/3d_x_w_C_P_dots.pdf',
    #                format='pdf', dpi=100)
    #plt.show()
    '''
    '''
    ###Linhas para colocar o desvio padrão no grafico de cl1.
    fig = plt.figure(figsize=(5.5,4.5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_zlabel('$C_{l_{1}}}$')
    ###leitura dos arquivos
    arq_c1 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/cl1_r1.npy', 'r')
    arq_c2 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/cl1_r2.npy', 'r')
    arq_c3 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/cl1_r3.npy', 'r')
    ###inicialização
    media = np.zeros((11,11))
    media_2 = np.zeros((11,11))
    desvio = np.zeros((11,11))
    #print(media); print(media_2); print(desvio)
    n = 11
    for j in range(0, n):
        for k in range(0, n):
            #print(arq_c1[k,j]); print(arq_c2[k,j]); print(arq_c3[k,j])
            media[k,j] = (arq_c1[k,j] + arq_c2[k,j] + arq_c3[k,j])/3.0
            media_2[k,j] = (arq_c1[k,j]**2 + arq_c2[k,j]**2 + arq_c3[k,j]**2)/3.0
            desvio[k,j] = sqrt(media_2[k,j] - media[k,j]**2)
    #print(media); print(media_2);  print(desvio)
    #print(arq_c1[0,1]); print(arq_c1[0,1]**2)#teste de localização
    ax.plot_wireframe(X, W, Cl1_ana, rstride=1, cstride=0, color = 'r', alpha=0.4)
    ax.scatter(W, X, media, label='Experimental', color = 'b', linestyle="None", marker='o', s=2)
    for i in np.arange(0, len(W)):
        ax.scatter([W[i], W[i]], [X[i], X[i]], [media[i]+desvio[i], media[i]-desvio[i]], color = 'g', marker='_')
    ax.view_init(elev=12, azim=-7)
    ax.legend()
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/cl1_berro.eps',
    #                format='eps', dpi=100)
    #plt.show()
    
    ###Linhas para colocar o desvio padrão no grafico de pl1.
    fig = plt.figure(figsize=(5.5,4.5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_zlabel('$P_{l_{1}}}$')
    #leitura dos arquivos
    arq_p1 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/pl1_r1.npy', 'r')
    arq_p2 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/pl1_r2.npy', 'r')
    arq_p3 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/pl1_r3.npy', 'r')
    #inicialização
    media_p = np.zeros((11,11))
    media_p2 = np.zeros((11,11))
    desvio_p = np.zeros((11,11))
    n = 11
    for j in range(0, n):
        for k in range(0, n):
            #print(arq_p1[k,j]); #print(arq_p2[k,j]); #print(arq_p3[k,j])
            media_p[k,j] = (arq_p1[k,j] + arq_p2[k,j] + arq_p3[k,j])/3.0
            media_p2[k,j] = (arq_p1[k,j]**2 + arq_p2[k,j]**2 + arq_p3[k,j]**2)/3.0
            desvio_p[k,j] = sqrt(media_p2[k,j] - media_p[k,j]**2)
    ax.plot_wireframe(X, W, Pl1_ana, rstride=1, cstride=0, color = 'r', alpha=0.4)
    ax.scatter(W, X, media_p, label='Experimental', color = 'b', linestyle="None", marker='o', s=2)
    for i in np.arange(0, len(W)):
        ax.scatter([W[i], W[i]], [X[i], X[i]], [media_p[i]+desvio_p[i], media_p[i]-desvio_p[i]], color = 'g', marker='_')
    ax.view_init(elev=12, azim=-7)
    ax.legend()
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/pl1_berro.eps',
    #                format='eps', dpi=100)
    #plt.show()
    
    ###Linhas para colocar o desvio padrão no grafico de cl1 + pl1.
    fig = plt.figure(figsize=(5.5,4.5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_zlabel('$C_{l_{1}}+P_{l_{1}}$')
    #leitura dos arquivos
    arq_c_p1 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/soma_c_p_r1.npy', 'r')
    arq_c_p2 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/soma_c_p_r2.npy', 'r')
    arq_c_p3 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/soma_c_p_r3.npy', 'r')
    #inicialização
    media_c_p = np.zeros((11,11))
    media_c_p2 = np.zeros((11,11))
    desvio_c_p = np.zeros((11,11))
    n = 11
    for j in range(0, n):
        for k in range(0, n):
            #print(arq_c_p1[k,j]); #print(arq_c_p2[k,j]); #print(arq_c_p3[k,j])
            media_c_p[k,j] = (arq_c_p1[k,j] + arq_c_p2[k,j] + arq_c_p3[k,j])/3.0
            media_c_p2[k,j] = (arq_c_p1[k,j]**2 + arq_c_p2[k,j]**2 + arq_c_p3[k,j]**2)/3.0
            desvio_c_p[k,j] = sqrt(media_c_p2[k,j] - media_c_p[k,j]**2)
    ax.plot_wireframe(X, W, soma_C_P_ana, rstride=1, cstride=0, color = 'r', alpha=0.4)
    ax.scatter(W, X, media_c_p, label='Experimental', color = 'b', linestyle="None", marker='o', s=2)
    for i in np.arange(0, len(W)):
        ax.scatter([W[i], W[i]], [X[i], X[i]], [media_c_p[i]+desvio_c_p[i], media_c_p[i]-desvio_c_p[i]], color = 'g', marker='_')
    ax.view_init(elev=12, azim=-8)
    ax.legend()
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/cl1_pl1.eps',
    #                format='eps', dpi=100)
    #plt.show()
    
    ###Linhas para colocar o desvio padrão no grafico de W_l1(correlação quântica).
    fig = plt.figure(figsize=(5.5,4.5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_zlabel('$W_{l_{1}}$')    
    #leitura dos arquivos
    arq_qc1 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/qcl1_r1.npy', 'r')
    arq_qc2 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/qcl1_r2.npy', 'r')
    arq_qc3 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/qcl1_r3.npy', 'r')
    #inicialização
    media_qc = np.zeros((11,11))
    media_qc2 = np.zeros((11,11))
    desvio_qc = np.zeros((11,11))
    #print(media);      print(desvio)
    n = 11
    for j in range(0, n):
        for k in range(0, n):
            #print(arq_c_p1[k,j]); #print(arq_c_p2[k,j]); #print(arq_c_p3[k,j])
            media_qc[k,j] = (arq_qc1[k,j] + arq_qc2[k,j] + arq_qc3[k,j])/3.0
            media_qc2[k,j] = (arq_qc1[k,j]**2 + arq_qc2[k,j]**2 + arq_qc3[k,j]**2)/3.0
            desvio_qc[k,j] = sqrt((media_qc2[k,j]-media_qc[k,j]**2)/3.0)
    ax.plot_wireframe(W, X, QC_teo, rstride=0, cstride=1, color = 'r', alpha=0.4)
    ax.scatter(W, X, media_qc, label='Experimental', color = 'b', linestyle="None", marker='o', s=2)
    for i in np.arange(0, len(W)):
        ax.scatter([W[i], W[i]], [X[i], X[i]], [media_qc[i]+desvio_qc[i], media_qc[i]-desvio_qc[i]], color = 'g', marker='_')
    ax.view_init(elev=20, azim=12)
    ax.legend()
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/qcl1_berro.eps',
    #                format='eps', dpi=100)
    #plt.show()
    
    ###Linhas para colocar o desvio padrão no grafico de C+P+W_l1.
    fig = plt.figure(figsize=(5.5,4.5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_zlabel('$C_{l_{1}}+P_{l_{1}}+W_{l_{1}}$')    
    #leitura dos arquivos
    arq_c_p1 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/soma_c_p_r1.npy', 'r')
    arq_c_p2 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/soma_c_p_r2.npy', 'r')
    arq_c_p3 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/soma_c_p_r3.npy', 'r')
    arq_qc1 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/qcl1_r1.npy', 'r')
    arq_qc2 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/qcl1_r2.npy', 'r')
    arq_qc3 = np.load('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/qcl1_r3.npy', 'r')
    #inicialização
    c_p_w1 = np.zeros((11,11)) 
    c_p_w2 = np.zeros((11,11))
    c_p_w3 = np.zeros((11,11))
    media_c_p_w = np.zeros((11,11))
    media_c_p_w2 = np.zeros((11,11))
    desvio_c_p_w = np.zeros((11,11))
    n = 11
    for j in range(0, n):
        for k in range(0, n):
            c_p_w1[k,j] = arq_c_p1[k,j] + arq_qc1[k,j]
            c_p_w2[k,j] = arq_c_p2[k,j] + arq_qc2[k,j]
            c_p_w3[k,j] = arq_c_p3[k,j] + arq_qc3[k,j]
            media_c_p_w[k,j] = (c_p_w1[k,j] + c_p_w2[k,j] + c_p_w3[k,j])/3.0
            media_c_p_w2[k,j] = (c_p_w1[k,j]**2 + c_p_w2[k,j]**2 + c_p_w3[k,j]**2)/3.0
            desvio_c_p_w[k,j] = sqrt(media_c_p_w2[k,j] - media_c_p_w[k,j]**2)
    ax.plot_wireframe(W, X, (soma_C_P_teo + QC_teo), rstride=0, cstride=1, color = 'r', alpha=0.4)
    ax.scatter(W, X, media_c_p_w, label='Experimental', color = 'b', linestyle="None", marker='o', s=2)
    for i in np.arange(0, len(W)):
        ax.scatter([W[i], W[i]], [X[i], X[i]], [media_c_p_w[i]+desvio_c_p_w[i], media_c_p_w[i]-desvio_c_p_w[i]], color = 'g', marker='_')
    ax.view_init(elev=25, azim=-15)
    ax.legend()
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/c_p_w_l1_berro.eps',
    #                format='eps', dpi=100)
    #plt.show()
    
    ###Linhas para colocar dois graficos juntos de cl1 + pl1 e cl1+pl1+Wl1.
    ###Utilizando os valores já calculados acima. para funcionar temos que deixar ativos os outros gráficos.
    fig = plt.figure(figsize=(5.5,4.5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    #ax.set_zlabel('$C_{l_{1}}+P_{l_{1}}$')
    ###aqui plot  cl1 + pl1
    ax.plot_wireframe(X, W, soma_C_P_ana, rstride=1, cstride=0, color = 'r', alpha=0.4)
    ax.scatter(W, X, media_c_p, label='$C_{l1}+P_{l1}$', color = 'b', linestyle="None", marker='o', s=2)
    for i in np.arange(0, len(W)):
        ax.scatter([W[i], W[i]], [X[i], X[i]], [media_c_p[i]+desvio_c_p[i], media_c_p[i]-desvio_c_p[i]], color = 'g', marker='_')
    ###aqui plot  cl1 + pl1 + Wl1
    ax.plot_wireframe(W, X, (soma_C_P_teo + QC_teo), rstride=0, cstride=1, color = 'r', alpha=0.6)
    ax.scatter(W, X, media_c_p_w,  linestyle="None", label='$C_{l1}+P_{l1}+W_{l1}$', color = 'darkturquoise',  marker='^', s=3)
    #for i in np.arange(0, len(W)):
    #    ax.scatter([W[i], W[i]], [X[i], X[i]], [media_c_p_w[i]+desvio_c_p_w[i], media_c_p_w[i]-desvio_c_p_w[i]], color = 'g', marker='_')
    ax.view_init(elev=11, azim=-9)
    ax.legend()
    #plt.savefig('/home/mauro/Dropbox/Doutorado/Artigo_Pesquisa_3/test_comp_werner/real/desviopadrao/cl1_pl1_wl1.eps',
    #                format='eps', dpi=100)
    plt.show()
    '''

cohe_predic_werner()



'''

#aqui fazendo funçoes iguais ao github do coherence.ipynb
def shannon_(d,pv):
    H = 0.0
    for j in range(0,d):
        if pv[j] > 10**-15 and pv[j] < (1-10**-15):
            H -= pv[j]*math.log(pv[j],2)
    return H

import scipy.linalg.lapack as lapak
def von_neumann_(d,rho):
    ev = lapak.zheevd(d,rho)
    return shannon_(d,ev)

def coh_re_(d,rho):
    pv = np.zeros(d,1)
    for j in range(0,d):
        pv[j] = rho[j,j]
    return shannon_(d,pv)- von_neumann_(d,rho)

'''