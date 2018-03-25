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
    for l in range(filtre.shape[0]):
        for c in range(filtre.shape[1]):
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
    
    
def mDCT(filtre):
    """
    Calcul de la transformée en cosinus discrète
    """
    t1 = int(abs(filtre.shape[0]/2))
    t2 = int(abs(filtre.shape[1]/2))
    filtre2 = np.zeros((t1,t2))
    for l in range(t1):
        for c in range(t2):
            filtre2[l][c] = filtre[l][c]
    return filtre2


def filtInv(img_bruitee,filtre,name=""):
    plt.figure()
    t_img = np.fft.fft2(img_bruitee)
    t = np.fft.fft2(filtre)
    divF = np.divide(t_img,t)
    invF = np.fft.ifft2(divF)
    plt.title("Filtrage inverse sur {}".format(name))
    plt.imshow(np.fft.ifftshift(abs(invF)),cmap='gray')




if __name__ == "__main__":

    plt.figure()
    I = io.imread('lena.png')
    I = np.float32(I)
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
    plt.figure()
    a = cFiltre(11,11,5)
    plt.title('Filtre a normalisé en forme de disque')
    plt.imshow(a)

    plt.figure()
    #Gestion des effets de bord : boundary = "fill","wrap","symm"
    yc = ssi.convolve2d(Inorm,a,mode="same")
    plt.title('Lena convoluée avec le disque a')
    plt.imshow(yc,cmap='gray')
    yc1 = ssi.convolve2d(Inorm,a,mode="same",boundary="wrap")
    yc2 = ssi.convolve2d(Inorm,a,mode="same",boundary="symm")
    
    
    ########## Fonction de transfert optique t ##########
    plt.figure()
    grille = cFiltre(I.shape[0],I.shape[1],5)
    t = np.fft.fft2(grille)
    plt.title('Transformée de Fourier du filtre')
    plt.imshow(abs(t))


    ########## Filtrage avec periodisation de l'image ##########
    plt.figure()
    x = np.fft.fft2(Inorm)
    plt.title('Transformée de Fourier de Lena')
    plt.imshow(abs(x))

    plt.figure()
    y = x*t
    yf = np.fft.ifftshift(abs(np.fft.ifft2(y)))
    plt.title('Lena filtrée')
    plt.imshow(yf,cmap="gray")

    plt.figure()    
    dy = yf - yc1
    plt.title('Fourier vs convolution')
    plt.imshow(abs(dy),cmap='gray')
    plt.colorbar()
    
    
    ########## Filtrage avec symétrisation de l'image ##########
    plt.figure()
    grille2 = cFiltre(2*I.shape[0],2*I.shape[1],5)
    t2 = np.fft.fft2(grille2)
    plt.title('Fonction de transfert optique n°2')
    plt.imshow(abs(t2))
    
    plt.figure() 
    dct_t2 = mDCT(t2)
    dct_x = cv2.dct(x)
    y2 = dct_x * dct_t2
    yf2 = abs(cv2.idct(y2))
    plt.title('Lena filtrée avec dct')
    plt.imshow(yf2,cmap="gray")
    
    plt.figure()    
    dy2 = yf2 - yc2
    plt.title('dct vs convolution with symmetry')
    plt.imshow(abs(dy2),cmap='gray')
    plt.colorbar()
    

    
    
    """
    ////////////////////////////////////////////////////////
    //                                                    //
    //            PARTIE 2 : PROBLEME INVERSE             //
    //                                                    //
    ////////////////////////////////////////////////////////
    """
    
    ########## Filtrage inverse ##########
    # Cas yc : filtrage de l'image par convolution
    filtInv(yc,a,"yc")
    # Cas yf : filtrage avec périodisation de l'image
    filtInv(yf,grille,"yf")    
    # Cas yf2 : filtrage avec symmétrisation de l'image
    filtInv(yf2,grille2,"yf2")
        
    
    ########## Régularisation du filtre inverse fft ##########
    
    
    
    ########## Régularisation du filtrage inverse dct ##########
    
    
    
    ########## Effet du bruit Gaussien ##########
    
    
    
    ########## Effet du bruit Poivre et Sel ##########
    
    
    
    





