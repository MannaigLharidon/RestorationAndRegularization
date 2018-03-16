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



"""
////////////////////////////////////////////////////////
//                                                    //
//        PARTIE 1 : PROBLEMES DIRECT ET INVERSE      //
//                                                    //
////////////////////////////////////////////////////////
"""

def probleme(data):
       
    """ 1.Interpretation du probleme direct """

    # Dimensions du problème
    dimx = data['dimx']
    dimy = data['dimy']
    dima = data['dima']
    
    # Affichage de la matrice de la transformation linéaire A
    A = data['A']
      
    # Signal d'entrée x
    x = data['x']
    plt.figure()
    plt.title('Signal d entree')
    plt.plot(x)
    plt.show()
    
    # Signal de sortie y : signal d'entree par la transformation lineaire A
    y = np.dot(A,x)
    plt.figure()
    plt.title('Signal de sortie par transformation lineaire')
    plt.plot(y)
    plt.show()
    
    # Ondelette a
    a = data['a']
    plt.figure()
    plt.title('Ondelette')
    plt.plot(y)
    plt.show()
    
    # Signal convolué yc : convolution du signal d'entree par l'ondelette a
    yc = ssi.convolve(x,a)
    plt.figure()
    plt.title('Signal de sortie par convolution')
    plt.plot(yc)
    plt.show()
    
    # Différence des signaux y et yc
    diff_yyc = y - yc
    plt.figure()
    plt.title('Difference des signaux de sortie')
    plt.plot(diff_yyc)
    plt.show()
    
    # Affichage de la matrice de la transformation linéaire A
    plt.figure()
    plt.title('Transformation lineaire A')
    plt.imshow(A)
    plt.show()
    
    
    """ 2.Etude du caractere bien pose ou bien conditionné du probleme """
    
    # Calcul des valeurs singulieres de A
    singA = np.linalg.svd(A,compute_uv=False)
    
    # Affichage du spectre des valeurs singulieres
    plt.figure()
    plt.title('Valeurs singulieres de A')
    plt.plot(singA)
    plt.show()
    
    # Calcul du conditionnement de A
    condA = np.linalg.cond(A)
    
    
    """ 3.Problème inverse """
    
    # Valeurs d'écarts-types
    sigma = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])
    
    # Matrice inverse de A par la pseudo inverse
    invA = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A))
    
    for s in range(np.size(sigma)):
        
        # Signal bruité yb
        gaussien = sigma[s] * np.random.randn(dimy,1)
        yb = y + gaussien
        plt.figure()
        plt.title("y et yb pour sigma={}".format(sigma[s]))
        plt.plot(y)
        plt.plot(yb)
        plt.show()
        
        # Resolution du probleme inverse
        X_chap = np.dot(invA,yb)
        
        # Affichage des signaux entrees
        plt.figure()
        plt.title("x et Xchap pour sigma={}".format(sigma[s]))
        plt.plot(x)
        plt.plot(X_chap)
        plt.show()
        
        
    """ 5.Lien avec la convolution """
    
    # Transformée de Fourier rapide des ondelettes
    fourier = np.fft.fft(a,n=dimx[0][0])
#    modules = 
#    modules = np.sort(modules)[::-1]

    plt.figure()
    plt.plot(fourier)
    
    
    return condA
        


"""
////////////////////////////////////////////////////////
//                                                    //
//              PARTIE 2 : REGULARISATION             //
//                                                    //
////////////////////////////////////////////////////////
"""

##On travaille maintenant avce les donnees ricker
#A_R = ricker['A']
#x_R = ricker['x']
#dimx_R = ricker['dimx']
#dimy_R = ricker['dimy']
#dima_R = ricker['dima']
#a_R = ricker['a']
#
################### Regularisation par penalisation sur la norme de la solution, sans bruit ##################
#
#y_R = np.dot(A_R,x_R)
#dimA = np.shape(A_R)[0]
#
## Parametre de regularisation
#alpha = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])
#
#for a in range(np.size(alpha)):
#    
#    # Resolution du probleme inverse avec regularisation
#    N = np.dot(np.linalg.inv(np.dot(np.transpose(A_R),A_R)+alpha[a]*np.identity(dimA)),np.transpose(A_R))
#    X_chap = np.dot(N,y_R)
#    
#    # Affichage des entrees
#    plt.plot(x_R)
#    plt.plot(X_chap)
#    plt.figure()
#    
#
################### Regularisation par penalisation sur la norme de la solution, avec bruit ##################
#
#
#
#
#
#
################### Regularisation par Rdige Regression ##################
#
#
#
#
#
#
################### Regularisation par troncature du spectre SVD ##################
#
#
#
#
#
#
################### Regularisation par penalisation sur la norme au carre du gradient de la solution ##################
#




if __name__=="__main__":

    # Chargement des donnees du probleme direct
    kramer = sio.loadmat('kramer.mat')
    ricker = sio.loadmat('ricker.mat')
    
    cond_K = probleme(kramer)
    #cond_R = probleme(ricker)

















