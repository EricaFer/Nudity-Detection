import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.io import imread, imshow

def read_image(url):
    imagem = cv2.cvtColor(imread(url), cv2.COLOR_RGB2HSV)

    return imagem

def apply_mask(imagem):
    x = imagem.shape[0]
    y = imagem.shape[1]

    mask = np.zeros((x+1,y+1))

    for i in range(0,x):
        for j in range(0,y):

            h,s,v = imagem[i,j,:]

            if (0 <= h <= 20) and (25 <= s <= 180):

                mask[i,j] = 1

    return mask,x,y

def opening_closing(mask):
    kernel = np.ones((18,18),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing

def label_image(closing,x,y):
    label_image = label(closing)
    
    flat_label_image = label_image.flatten()

    # Número de regiões
    num_regioes = max(flat_label_image)

    regioes_dict = {}

    # não leva em conta os que tem indice 0
    for i in range(1,int(num_regioes)+1):

        regioes_dict[str(i)] = len(np.where(flat_label_image == i)[0])

    lista_chaves = list(regioes_dict.keys())
    lista_valores = list(regioes_dict.values())
    
    position = lista_valores.index(max(lista_valores))
    maior_regiao_indice = lista_chaves[position]

    area_imagem = x*y
    area_regiao = regioes_dict[maior_regiao_indice]

    porcentagem_ocupada = (area_regiao/area_imagem)*100

    return num_regioes,porcentagem_ocupada