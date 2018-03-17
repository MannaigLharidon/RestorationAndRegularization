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
import cmath

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
    dimx, dimy = data['dimx'], data['dimy']

    # Matrice de la transformation linéaire A
    A = data['A']
      
    # Signal d'entrée x
    x = data['x']
#    plt.figure()
#    plt.title('Signal d entree')
#    plt.plot(x)
#    plt.show()
    
    # Signal de sortie y : signal d'entree par la transformation lineaire A
    y = np.dot(A,x)
#    plt.figure()
#    plt.title('Signal de sortie par transformation lineaire')
#    plt.plot(y)
#    plt.show()
    
    # Ondelette a
    a = data['a']
#    plt.figure()
#    plt.title('Ondelette')
#    plt.plot(y)
#    plt.show()
    
    # Signal convolué yc : convolution du signal d'entree par l'ondelette a
    yc = ssi.convolve(x,a)
#    plt.figure()
#    plt.title('Signal de sortie par convolution')
#    plt.plot(yc)
#    plt.show()
    
    # Différence des signaux y et yc
    diff_yyc = y - yc
    plt.figure()
    plt.title('Difference des signaux de sortie')
    plt.plot(diff_yyc)
    plt.show()
    
    # Affichage de la matrice de la transformation linéaire A
#    plt.figure()
#    plt.title('Transformation lineaire A')
#    plt.imshow(A)
#    plt.show()
    
    
    """ 2.Etude du caractere bien pose ou bien conditionné du probleme """
    
    # Calcul des valeurs singulieres de A
    singA = np.linalg.svd(A,compute_uv=False)
    
    # Affichage du spectre des valeurs singulieres
#    plt.figure()
#    plt.title('Valeurs singulieres de A')
#    plt.plot(singA)
#    plt.show()
    
    # Calcul du conditionnement de A
    condA = np.linalg.cond(A)
    
    
    """ 3.Problème inverse """
    
    # Valeurs d'écarts-types
    sigma = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])
    
    # Matrice inverse de A par la pseudo inverse
    invA = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A))
    
    for s in sigma :
        
        # Signal bruité yb
        gaussien = s * np.random.randn(dimy,1)
        yb = y + gaussien
#        plt.figure()
#        plt.title("y et yb pour sigma={}".format(s))
#        plt.plot(y)
#        plt.plot(yb)
#        plt.show()
        
        # Resolution du probleme inverse
        X_chap = np.dot(invA,yb)
        
        # Affichage des signaux entrees
#        plt.figure()
#        plt.title("x et Xchap pour sigma={}".format(s))
#        plt.plot(x)
#        plt.plot(X_chap)
#        plt.show()
        
        
    """ 5.Lien avec la convolution """
    
    # Transformée de Fourier rapide des ondelettes
#    fourier = np.fft.fft(a,n=dimx[0][0])
#    xF,yF = fourier.shape
#    modules = np.zeros((xF,yF))
#    for f in range(xF):
#        for g in range(yF):
#            modules[f][g] = cmath.polar(fourier[f][g])[0]
#    modules = np.sort(modules)[::-1]
#
#    plt.figure()
#    plt.title('Modules de fft(a) et valeurs singulières de A')
#    plt.plot(singA[1],'r',label='valeurs singulieres')
#    plt.plot(modules,'b',label='modules')
#    plt.legend()
#    plt.show()    
    
    return condA
        


"""
////////////////////////////////////////////////////////
//                                                    //
//              PARTIE 2 : REGULARISATION             //
//                                                    //
////////////////////////////////////////////////////////
"""



""" 1-2.Regularisation par penalisation sur la norme de la solution """

def penNorme(data,mode):
    """
    Régularisation par pénalisation sur la norme de la solution
    --> Trouver la valeur max de alpha qui permet d'avoir un bon résultat
    
    ENTREE :
        - data : données issues d'un fichier .mat
        - mode : 'with_noise' = avec bruit ; 'without_noise' = sans bruit 
    """
    A, x = data['A'], data['x']
    dimA, dimy = np.shape(A)[0], data['dimy']
    alpha = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])
    
    y = np.dot(A,x)
    if mode=="with_noise":
        for a in alpha:
            yb = y + a * np.random.randn(dimy,1)
        y = yb
    else:
        y = y

    for a in alpha:
        # Resolution 
        N = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)+a*np.identity(dimA)),np.transpose(A))
        X_chap = np.dot(N,y)
        # Affichage des entrees
        plt.plot(x,color='r',label="signal entree")
        plt.plot(X_chap,color'b',label="signal estime")
        plt.legend()
    plt.show()



################## Regularisation par Ridge Regression ##################

def ridge(data,mode):
    """
    Régularisation par Ridge regression
    --> Trouver la valeur de alpha qui semble optimale
    
    ENTREE :
        - data : données issues d'un fichier .mat
        - mode : 'with_noise' = avec bruit ; 'without_noise' = sans bruit 
    """
    A, x = data['A'], data['x']
    y = np.dot(A,x)
    alpha = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])
    if mode=="with_noise":
        for a in alpha:
            yb = y + a * np.random.randn(dimy,1)
        y = yb
    else:
        y = y
    
    for a in alpha:
        N = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)+a*np.diag(np.diag(np.dot(np.transpose(A),A)))),np.transpose(A))
        X_chap = np.dot(N,y)
        # Affichage des entrees
        plt.plot(x,color='r',label="signal entree")
        plt.plot(X_chap,color'b',label="signal estime")
        plt.legend()
    plt.show()

    




################## Regularisation par troncature du spectre SVD ##################

def troncSVD(data,mode):
    """
    Régularisation par tronxature du spectre SVD
    --> Trouver la valeur de alpha qui semble optimale
    
    ENTREE :
        - data : données issues d'un fichier .mat
        - mode : 'with_noise' = avec bruit ; 'without_noise' = sans bruit 
    """
    A, x = data['A'], data['x']
    y = np.dot(A,x)
    alpha = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])
    if mode=="with_noise":
        for a in alpha:
            yb = y + a * np.random.randn(dimy,1)
        y = yb
    else:
        y = y
    
    for a in alpha:
        """ Partie a modifier !
#        sing=np.linalg.svd(data["A"])
#        for i in range(len(sing[1])):
#            if sing[1][i]<alpha:
#                sing[1][i]=0
#        xi=np.dot(np.dot(np.dot(sing[2],np.nan_to_num(np.diag(1/sing[1]))),sing[0].T),yb)
#
#        N = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)+a*np.diag(np.diag(np.dot(np.transpose(A),A)))),np.transpose(A))
#        X_chap = np.dot(N,y)
        """
        # Affichage des entrees
        plt.plot(x,color='r',label="signal entree")
        plt.plot(X_chap,color'b',label="signal estime")
        plt.legend()
    plt.show()




################## Regularisation par penalisation sur la norme au carre du gradient de la solution ##################

def penNorme_carreGrad(data,mode):
    """
    Régularisation par tronxature du spectre SVD
    --> Trouver la valeur de alpha qui semble optimale
    
    ENTREE :
        - data : données issues d'un fichier .mat
        - mode : 'with_noise' = avec bruit ; 'without_noise' = sans bruit 
    """
    A, x = data['A'], data['x']
    y = np.dot(A,x)
    alpha = np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,1,10])
    if mode=="with_noise":
        for a in alpha:
            yb = y + a * np.random.randn(dimy,1)
        y = yb
    else:
        y = y
    
    for a in alpha:
        """ Partie a modifier !
#        sing=np.linalg.svd(data["A"])
#        for i in range(len(sing[1])):
#            if sing[1][i]<alpha:
#                sing[1][i]=0
#        xi=np.dot(np.dot(np.dot(sing[2],np.nan_to_num(np.diag(1/sing[1]))),sing[0].T),yb)
#
#        N = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)+a*np.diag(np.diag(np.dot(np.transpose(A),A)))),np.transpose(A))
#        X_chap = np.dot(N,y)
        """
        # Affichage des entrees
        plt.plot(x,color='r',label="signal entree")
        plt.plot(X_chap,color'b',label="signal estime")
        plt.legend()
    plt.show()



if __name__=="__main__":

    # Chargement des donnees du probleme direct
    kramer = sio.loadmat('kramer.mat')
    ricker = sio.loadmat('ricker.mat')
    
    # Partie 1 
    #cond_K = probleme(kramer)
    #cond_R = probleme(ricker)

    # Partie 2
    
    """ 1.Regularisation par pénalisation sur la norme de la solution, sans bruit """
    penNorme(kramer,mode='without_noise')

    """ 2.Regularisation par pénalisation sur la norme de la solution, avec bruit """
    penNorme(ricker,mode='with_noise')

    """ 3.Régularisation par ridge degression """
    ridge(ricker,mode='with_noise')













