import numpy as np
import pandas as pd

# Geração de gráficos
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from matplotlib.colors import ListedColormap

# Machine Learning
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.model_selection  import train_test_split

def prepara_fit(dataset):
    
    X = dataset.drop(columns = ['label'], axis = 1)
    Y = dataset.label

    X_train, X_test, Y_train,Y_test = train_test_split(X,Y,
                                                        test_size = 0.25,
                                                        stratify= Y,
                                                        random_state = 0)

    return X_train, X_test, Y_train,Y_test,X,Y

def pega_melhor_k(X_train, X_test, Y_train,Y_test):
    f1_list=[]
    k_list=[]

    for k in range(1,40):
        clf=KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
        clf.fit(X_train,Y_train)
        pred=clf.predict(X_test)
        f=f1_score(Y_test,pred,average='macro')
        f1_list.append(f)
        k_list.append(k)

    best_f1_score=max(f1_list)
    best_k=k_list[f1_list.index(best_f1_score)]        
    print("Optimum K value =",best_k," with F1-Score =",best_f1_score)

    clf=KNeighborsClassifier(n_neighbors=best_k,n_jobs=-1)
    clf.fit(X_train,Y_train)
    pred=clf.predict(X_test)
    f=f1_score(Y_test,pred,average='macro')
    f1_list.append(f)
    k_list.append(k)

    return best_k,pred

def processa_resultados(X_test,Y_test,pred):
    def resultados(pred,label):
        if pred == 1:
            if pred == label:
                return 'True Positive'
            else:
                return 'False Positive'
        else:
            if pred == label:
                return 'True Negative'
            else:
                return 'False Negative'

    X_test['pred'] = pred
    X_test['label'] = Y_test
    X_test['class'] = X_test.apply(
        lambda x: resultados(x.pred,x.label),axis = 1
    )

    return X_test

def gera_matrix_confusao(Y_test,pred):
    cf_matrix = confusion_matrix(Y_test,pred)
    fig = sns.heatmap(
        cf_matrix/np.sum(cf_matrix), 
        annot=True, 
        fmt='.2%', 
        cmap='Blues'
    )

    plt.savefig('graficos/cf.eps')

def plot_TRUE(X_test):

    fig = plt.figure(figsize = (10,8))

    sns.set_theme(style = 'whitegrid')
    sns.set_context(context='paper',font_scale=2)

    sns.scatterplot(
        data=X_test[X_test['class'].isin(['True Positive','True Negative'])], 
        x="area_ocupada", 
        y="numero_regioes", 
        hue="class",
        s = 150,
        palette = {
            'True Positive':'#8ecae6',
            'True Negative':'#e63946',
        }
    )

    plt.xlabel(
        'Área ocupada (%)', 
        )
    plt.ylabel(
        'Número de regiões', 
        )

    plt.legend(
        bbox_to_anchor=(0.45,1.09, 0.52, 0), 
        ncol=2,
        frameon=False
        )

    plt.xlim(left = -0.5)
    plt.ylim(bottom= 0)

    plt.savefig(
        f'./graficos/true.eps',
        dpi=1200,
        bbox_inches='tight'
    )

def plot_FALSE(X_test):
    fig = plt.figure(figsize = (10,8))

    sns.set_theme(style = 'whitegrid')
    sns.set_context(context='paper',font_scale=2)

    sns.scatterplot(
        data=X_test[X_test['class'].isin(['False Positive','False Negative'])], 
        x="area_ocupada", 
        y="numero_regioes", 
        hue="class",
        s = 150,
        palette = {
            'False Positive':'#8ecae6',
            'False Negative':'#e63946',
        }
    )

    plt.xlabel(
        'Área ocupada (%)', 
        )
    plt.ylabel(
        'Número de regiões', 
        )

    plt.legend(
        bbox_to_anchor=(0.45,1.09, 0.52, 0), 
        ncol=2,
        frameon=False
        )

    plt.xlim(left = -0.5)
    plt.ylim(bottom= 0)

    plt.savefig(
        f'./graficos/false.eps',
        dpi=1200,
        bbox_inches='tight'
    )

def classify_and_plot(X_train, Y_train, Y,best_k):

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold  = ListedColormap(['#FF0000', '#0000FF'])

    rcParams['figure.figsize'] = 5, 5

    h = 0.2

    clf = neighbors.KNeighborsClassifier(n_neighbors = best_k)
    clf.fit(X_train, Y_train)

    X = np.array(X)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)   
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Resultado KNN(k = 21)")
    fig.savefig('./graficos/knn.png')