from utils.imports  import *
from utils.modelo import *
from utils.preprocessing import *

import pandas as pd

# Pega o dataset
nsfw_urls = []
with open('./images/urls_nsfw.txt','r') as f:
    nsfw_urls = [line.rstrip() for line in f]

dataset = pd.DataFrame(columns=['area_ocupada','numero_regioes','label'])

dataset =  get_nudez(dataset,nsfw_urls)
dataset = get_vestidas(dataset)

dataset_clean = dataset[dataset.numero_regioes<=50]

# Gera modelo
X_train, X_test, Y_train,Y_test,X,Y = prepara_fit(dataset)
best_k,pred = pega_melhor_k(X_train, X_test, Y_train,Y_test)
X_test = processa_resultados(X_test,Y_test,pred)

#Plota gráficos
gera_matrix_confusao(Y_test,pred)
plot_TRUE(X_test)
plot_FALSE(X_test)
classify_and_plot(X_train, Y_train, Y,best_k)