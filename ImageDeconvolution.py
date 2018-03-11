# -*- coding: utf-8 -*-
"""
Created on Friday Mars 9 2018

@author: Mannaig L'Haridon
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.signal as ssi
import scipy.fftpack as sfft
import cv2


"""
#############################################################################
#                                                                           #
#                   RESTAURATION ET REGULARISATION D'IMAGE                  #
#                       TP2 : DECONVOLUTION D'UNE IMAGE                     #
#                                                                           #
#############################################################################

"""

def normFiltre(filtre):
    """
    Determine le nombre de pixels appartenant a un cercle puis normalise
    But : normaliser le filtre
    """
    cpt = 0
    taille = filtre.shape
    for l in range(taille[0]):
        for c in range(taille[1]):
            if filtre[l][c] == 1.0 :
                cpt += 1
    filtre /= cpt
    return filtre
                

def cFiltre(taille1,taille2,rayon):
    """
    Cree un filtre normalise en forme de disque
    """
    filtre = np.zeros((taille1,taille2))
    centre1, centre2 = int(abs(taille1/2)), int(abs(taille2/2))
    filtre = cv2.circle(filtre,(centre1,centre2),rayon,1,-1)
    filtre = normFiltre(filtre)    
    return filtre
    
    
def dct(filtre):
    """
    Calcul de la transformée en cosinus discrète
    """
    taille = filtre.shape
    t1 = int(abs(taille[0]/2))
    t2 = int(abs(taille[1]/2))
    filtre2 = np.zeros((t1,t2))
    for l in range(t1):
        for c in range(t2):
            filtre2[l][c] = filtre[l][c]
    return filtre2




if __name__ == "__main__":

    plt.figure(1)
    I = io.imread('lena.png')
    I = np.float32(I)
    taille = I.shape
    Inorm = I/256
    plt.title('Lena')
    plt.imshow(Inorm,cmap='gray')

    
    """
    ////////////////////////////////////////////////////////
    //                                                    //
    //    PARTIE 1 : PROBLEMES DIRECT ET EFFET DE BORD    //
    //                                                    //
    ////////////////////////////////////////////////////////
    """
    
    ########## Filtrage de l'image par convolutin ##########
    plt.figure(2)
    a = cFiltre(11,11,5)
    plt.title('Filtre a normalisé en forme de disque')
    plt.imshow(a)

    plt.figure(3)
    #Gestion des effets de bord : boundary = "wrap","symm",...
    yc = ssi.convolve2d(Inorm,a,mode="same")
    plt.title('Lena convoluée avec le disque a')
    plt.imshow(yc,cmap='gray')
    
    
    ########## Fonction de transfert optique t ##########
    plt.figure(4)
    grille = cFiltre(taille[0],taille[1],5)
    t = np.fft.fft2(grille)
    plt.title('Transformée de Fourier du filtre')
    plt.imshow(abs(t))


    ########## Filtrage avec periodisation de l'image ##########
    plt.figure(5)
    x = np.fft.fft2(Inorm)
    plt.title('Transformée de Fourier de Lena')
    plt.imshow(abs(x))

    plt.figure(6)
    y = x*t
    yf = np.fft.ifftshift(abs(np.fft.ifft2(y)))
    plt.title('Lena filtrée')
    plt.imshow(yf,cmap="gray")

    plt.figure(7)    
    dy = yf - yc
    plt.title('Fourier vs convolution')
    plt.imshow(abs(dy),cmap='gray')
    plt.colorbar()
    
    
    ########## Filtrage avec symétrisation de l'image ##########
    plt.figure(8)
    grille2 = cFiltre(2*taille[0],2*taille[1],5)
    t_grille2 = np.fft.fft2(grille2)
    plt.title('Fonction de transfert optique n°2')
    plt.imshow(abs(t_grille2))
    
    dct_res = dct(x)
    """ A FINIR !!!!!!! """

    
    
    """
    ////////////////////////////////////////////////////////
    //                                                    //
    //            PARTIE 2 : PROBLEME INVERSE             //
    //                                                    //
    ////////////////////////////////////////////////////////
    """
    
    ########## Filtrage inverse ##########
    plt.figure(9)
    tyf = np.fft.fft2(yf)
    divF = np.divide(tyf,t)
    invF = np.fft.ifft2(divF)
    plt.imshow(np.fft.ifftshift(abs(invF)),cmap='gray')
        
    # Faire les tests sur les autres cas :)
    
    
    ########## Régularisation du filtre inverse fft ##########
    
    
    
    ########## Régularisation du filtrage inverse dct ##########
    
    
    
    ########## Effet du bruit Gaussien ##########
    
    
    
    ########## Effet du bruit Poivre et Sel ##########
    
    
    
    




