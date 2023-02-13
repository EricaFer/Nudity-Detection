from utils.preprocessing import *
from tqdm import tqdm
import pandas as pd

def get_nudez(dataset,nsfw_urls):
    index = 0
    while dataset.shape[0] < 500:

        url = nsfw_urls[index]

        try:
            imagem = read_image(url)
            mask,x,y = apply_mask(imagem)
            closing = opening_closing(mask)
            num_regioes,porcentagem_ocupada = label_image(closing,x,y)

            nova_linha = pd.Series(
                {
                    'area_ocupada': porcentagem_ocupada, 
                    'numero_regioes': num_regioes,
                    'label':1
                }    
            )

            dataset = pd.concat(
                [
                    dataset,
                    nova_linha.to_frame().T,
                ],
                ignore_index=True
                )

        except Exception as e:
            print(e)
        finally:
            index += 1
            
    dataset.to_csv('./data/resultado.csv')
    return dataset

def get_vestidas(dataset):
    for index in tqdm(range(1,501),desc='Progress'):
        try:

            file_index = str(index)

            while len(file_index) < 4:
                file_index = '0' + file_index 

            imagem = read_image(f'./images/imagens_vestidas/img_{file_index}.png')
            mask,x,y = apply_mask(imagem)
            closing = opening_closing(mask)
            num_regioes,porcentagem_ocupada = label_image(closing,x,y)

            nova_linha = pd.Series(
                {
                    'area_ocupada': porcentagem_ocupada, 
                    'numero_regioes': num_regioes,
                    'label':0
                }    
            )

            dataset = pd.concat(
                [
                    dataset,
                    nova_linha.to_frame().T,
                ],
                ignore_index=True
                )

        except Exception as e:
            print(e)

    dataset.to_csv('./data/resultado.csv')
    return dataset