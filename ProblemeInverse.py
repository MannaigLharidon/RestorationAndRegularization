# -*- coding: utf-8 -*-
"""
Created on %(26/02/2018)s

@author: %(Mannaig L'Haridon)s
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal as ssi
import scipy.fftpack as sfftp

"""
#####################################################################
#                                                                   #
#                       RESTAURATION D'IMAGES                       #
#          TP1: Resolution d'un probleme inverse sur signaux        #
#                                                                   #
#####################################################################
"""


"""                            OBJECTIF DU TP 
Analyser et resoudre deux problemes inverses sur des signaux discretises 
(dimension finie) """

# Chargement des donnees des deux problemes directs
ricker = sio.loadmat('ricker.mat')


"""
////////////////////////////////////////////////////////
//                                                    //
//        PARTIE 1 : PROBLEMES DIRECT ET INVERSE      //
//                                                    //
////////////////////////////////////////////////////////
"""

################## Interpretation du probleme direct ##################

# Chargement des donnees du probleme direct
kramer = sio.loadmat('kramer.mat')

# Affichage des donnees du probleme direct
dimx_K = kramer['dimx']
dimy_K = kramer['dimy']
dima_K = kramer['dima']
A_K = kramer['A']

# Affichage du signal d'entree
x_K = kramer['x']
plt.title('Signal d entree Kramer')
plt.plot(x_K)
plt.figure()

# Calcul et affichage du signal de sortie : signal d'entree par la transformation lineaire A
y_K = np.dot(A_K,x_K)
plt.title('Signal de sortie par transformation lineaire')
plt.plot(y_K)
plt.figure()

# Affichage de l'ondelette a
a_K = kramer['a']

# Calcul et affichage de yc : convolution du signal d'entree par l'ondelette a
yc_K = ssi.convolve(x_K,a_K)
plt.title('Signal de sortie par convolution')
plt.plot(yc_K)
plt.figure()

# Calcul et affichage de la difference entre y et yc
diff_yyc = y_K - yc_K
plt.title('Difference des signaux de sortie')
plt.plot(diff_yyc)
plt.figure()

# Affichage de A
plt.title('Transformation lineaire A')
plt.imshow(A_K)
plt.figure()



################## Etude du caractere bien pose ou ien conditionne du probleme ##################

# Calcul des valeurs singulieres de A
singA = np.linalg.svd(A_K,compute_uv=False)

# Affichage du spectre des valeurs singulieres
plt.title('Valeurs singulieres de A')
plt.plot(singA)
plt.figure()

# Calcul du conditionnement de A
condA = np.linalg.cond(A_K)



################## Problème inverse ##################

# Valeurs d'écarts-types
sigma = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])

# Matrice inverse de A par la pseudo inverse
invA = np.dot(np.linalg.inv(np.dot(np.transpose(A_K),A_K)),np.transpose(A_K))

"""
for s in range(np.size(sigma)):
    
    # Bruiter l'observation y_K avec un bruit gaussien d'ecart type sigma
    gaussien = sigma[s] * np.random.randn(dimy_K,1)
    yb = y_K + gaussien
    
    # Affichage des signaux bruites et non bruites
    plt.plot(y_K)
    plt.plot(yb)
    plt.figure()
    
    # Resolution du probleme inverse
    X_chap = np.dot(invA,yb)
    
    # Affichage des signaux entrees
    plt.plot(x_K)
    plt.plot(X_chap)
    plt.figure()
"""


"""
////////////////////////////////////////////////////////
//                                                    //
//              PARTIE 2 : REGULARISATION             //
//                                                    //
////////////////////////////////////////////////////////
"""

#On travaille maintenant avce les donnees ricker
A_R = ricker['A']
x_R = ricker['x']
dimx_R = ricker['dimx']
dimy_R = ricker['dimy']
dima_R = ricker['dima']
a_R = ricker['a']

################## Regularisation par penalisation sur la norme de la solution, sans bruit ##################

y_R = np.dot(A_R,x_R)
dimA = np.shape(A_R)[0]

# Parametre de regularisation
alpha = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])

for a in range(np.size(alpha)):
    
    # Resolution du probleme inverse avec regularisation
    N = np.dot(np.linalg.inv(np.dot(np.transpose(A_R),A_R)+alpha[a]*np.identity(dimA)),np.transpose(A_R))
    X_chap = np.dot(N,y_R)
    
    # Affichage des entrees
    plt.plot(x_R)
    plt.plot(X_chap)
    plt.figure()
    

################## Regularisation par penalisation sur la norme de la solution, avec bruit ##################






################## Regularisation par Rdige Regression ##################






################## Regularisation par troncature du spectre SVD ##################






################## Regularisation par penalisation sur la norme au carre du gradient de la solution ##################

























