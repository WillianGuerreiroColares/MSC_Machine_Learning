################### PPGEE 2021/01 ####################
############## Aprendizado de máquina ################
# Trabalho 2: Classificador probabilístico Naive Bayes
# Alunos: Carlos Henrique
#         Willian Guerreiro Colares

#Import de bibliotecas para manipulação e visualização de dados
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Função para cálculo de densidade de probabilidade Gaussiana
#Parâmetros: x->feature, u-> média, sigma->desvio padrão
def gauss_distribution(x,u,sigma):
    p_x_w = math.exp(-0.5*pow(x-u,2)/pow(sigma,2))/(sigma*math.sqrt(2*math.pi))
    return p_x_w

#Função para cálculo de média
#Aux é uma matriz Mx4
def average(aux):
    soma = 0
    for i in range(len(aux)):
        soma = soma + aux[i]
    return soma/len(aux)

#Função para cálculo de desvio padrão
#Aux é uma matriz Mx4
def standard_dev(aux):
    soma = 0
    std_dev = 0
    for i in range(len(aux)):
        soma = soma + pow(aux[i] - average(aux),2)
    std_dev = soma/len(aux)
    for i in range(4):
        std_dev[i] = math.sqrt(std_dev[i])
    return std_dev

#Passo 1: Leitura do Dataset
dataset = pd.read_excel('iris_dataset.xlsx')

#Obtenção do número de observações N=150
N = dataset.shape[0]

#Passo 2: Tratamento dos dados
dataset['species'].replace('setosa', 1, inplace=True)
dataset['species'].replace('versicolor', 2, inplace=True)
dataset['species'].replace('virginica', 3, inplace=True)

graph_labels = ['setosa','versicolor','virginica']

#Passo 3: Obtenção das características, matriz de características e labels
X_meas1 = np.array((dataset['meas1'])).reshape((-1,1))
X_meas2 = np.array((dataset['meas2'])).reshape((-1,1))
X_meas3 = np.array((dataset['meas3'])).reshape((-1,1))
X_meas4 = np.array((dataset['meas4'])).reshape((-1,1))
#Matriz de características
X = np.concatenate([X_meas1,X_meas2,X_meas3,X_meas4],axis = 1)
#Labels
Y = np.array((dataset['species'])).reshape((-1,1))

#Passo 4: Segmentação da matriz de características em 3 classes W1, W2 e W3 (50 elementos cada)
#Classe 1 : 1º ao 50º elemento
w1 = X[:50]
#Classe 2 : 51º até o primeiro dos 50 últimos
w2 = X[50:-50]
#Classe 3: Os 50 últimos elementos
w3 = X[-50:]

#Número de pastas
K = 5

acuracia_media = 0

for p in range(K):
    print('Pasta k=',p+1)
    # [0;c[   [c;d[    [d;N[      sendo  [c;d[ o subset de teste   e  d-c = testing_size = 10
    c = 10*p
    d = 10*(p+1)

    #Obtenção dos conjuntos de teste
    w1_testing = w1[c:d]
    w2_testing = w2[c:d]
    w3_testing = w3[c:d]
    #Obtenção dos conjuntos de treinamento
    w1_trainning = np.concatenate((w1[0:c],w1[d:50]),axis=0)
    w2_trainning = np.concatenate((w2[0:c],w2[d:50]),axis=0)
    w3_trainning = np.concatenate((w3[0:c],w3[d:50]),axis=0)
    
    #Etapa de treinamento (obtencao de medias e desvios padroes)
    w1_u = average(w1_trainning)
    w1_sigma = standard_dev(w1_trainning) 
    #print('Media (Classe 1)')
    print(w1_u)
    #print('Desvio padrão (classe 1)')
    print(w1_sigma)
    print(' ')
    w2_u = average(w2_trainning)
    w2_sigma = standard_dev(w2_trainning)
    #print('Media (Classe 2)')
    print(w2_u)
    #print('Desvio padrão (classe 2)')
    print(w2_sigma)
    print(' ')
    w3_u = average(w3_trainning)
    w3_sigma = standard_dev(w3_trainning)
    #print('Media (Classe 3)')
    print(w3_u)
    #print('Desvio padrão (classe 3)')
    print(w3_sigma)
    print(' ')

    #Declaração de matrizes 10x3 
    w1_test = np.ones((10,3))
    w2_test = np.ones((10,3))
    w3_test = np.ones((10,3))

    for i in range(10):
        for j in range(4):
            #Obtenção de p(w1|x), p(w2|x) e p(w3|x) no conjunto de teste W1
            w1_test[i][0] = w1_test[i][0] * gauss_distribution(w1_testing[i][j],w1_u[j],w1_sigma[j])
            w1_test[i][1] = w1_test[i][1] * gauss_distribution(w1_testing[i][j],w2_u[j],w2_sigma[j])
            w1_test[i][2] = w1_test[i][2] * gauss_distribution(w1_testing[i][j],w3_u[j],w3_sigma[j])

            #Obtenção de p(w1|x), p(w2|x) e p(w3|x) no conhunto de teste W2
            w2_test[i][0] = w2_test[i][0] * gauss_distribution(w2_testing[i][j],w1_u[j],w1_sigma[j])
            w2_test[i][1] = w2_test[i][1] * gauss_distribution(w2_testing[i][j],w2_u[j],w2_sigma[j])
            w2_test[i][2] = w2_test[i][2] * gauss_distribution(w2_testing[i][j],w3_u[j],w3_sigma[j])

            #Obtenção de p(w1|x), p(w2|x) e p(w3|x) no conhunto de teste W3
            w3_test[i][0] = w3_test[i][0] * gauss_distribution(w3_testing[i][j],w1_u[j],w1_sigma[j])
            w3_test[i][1] = w3_test[i][1] * gauss_distribution(w3_testing[i][j],w2_u[j],w2_sigma[j])
            w3_test[i][2] = w3_test[i][2] * gauss_distribution(w3_testing[i][j],w3_u[j],w3_sigma[j])

    #Obtenção de p(w1|x), p(w2|x) e p(w3|x) dos conjuntos W1,W2 e W3
    #Probabilidade a priori = 1/3
    w1_test = w1_test/3
    w2_test = w2_test/3
    w3_test = w3_test/3

    #Variáveis para armazenar os casos de acerto na classificação
    predicted_w1 = 0
    predicted_w2 = 0
    predicted_w3 = 0

    confusion_matrix = np.zeros((3,3))

    #Contabilização de p(w1|x) > p(w2|x) e p(w1|x) > p(w3|x)
    for element in w1_test:
        i = np.where(element == max(element))
        confusion_matrix[0][i]+=1
        if i[0][0] == 0:
            predicted_w1 += 1
            
    #Contabilização de p(w2|x) > p(w1|x) e p(w2|x) > p(w3|x)
    for element in w2_test:
        i = np.where(element == max(element))
        confusion_matrix[1][i]+=1
        if i[0][0] == 1:
            predicted_w2 += 1

    #Contabilização de p(w3|x) > p(w2|x) e p(w3|x) > p(w1|x)
    for element in w3_test:
        i = np.where(element == max(element))
        confusion_matrix[2][i]+=1
        if i[0][0] == 2:
            predicted_w3 += 1

    print(confusion_matrix)
    window_name = 'Pasta ' + str(p+1)
    fig = plt.figure(window_name)
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix,cmap = 'Blues')
    for (c,r),label in np.ndenumerate(confusion_matrix):
        ax.text(r,c,int(label),ha='center',va='center')
    ax.set_xticklabels(['']+graph_labels)
    ax.set_yticklabels(['']+graph_labels)
    fig.colorbar(cax)
    plt.show()

    #Classificações corretas das classes 1,2 e 3
    print(predicted_w1,predicted_w2,predicted_w3)
    #Cálculo da acurária para os 30 elementos do conjunto de testes
    acuracia = 100*(predicted_w1+predicted_w2+predicted_w3)/30
    print('Acuracia ',acuracia)
    #Acumulo da acuracia para cada pasta
    acuracia_media+=acuracia
#Cálculo da acurácia média
acuracia_media = acuracia_media/5
print('Acuracia media ',acuracia_media)

if False:
    f1, (ax1,ax2) = plt.subplots(1,2)
    f1.canvas.set_window_title('Conjunto de dados reais')

    #Measure1, Measure2 e Species
    scatter = ax1.scatter(X_meas1, X_meas2, c = Y)
    ax1.set_xlabel(dataset.columns[0])
    ax1.set_ylabel(dataset.columns[1])
    handles, labels = scatter.legend_elements()
    labels = ['setosa','versicolor','virginica']
    ax1.legend(handles, labels, loc='best')

    #Measure1, Measure3 e Species
    scatter = ax2.scatter(X_meas1, X_meas3, c = Y)
    ax2.set_xlabel(dataset.columns[0])
    ax2.set_ylabel(dataset.columns[2])
    handles, labels = scatter.legend_elements()
    labels = ['setosa','versicolor','virginica']
    ax2.legend(handles, labels, loc='best')


    f2, (ax3,ax4) = plt.subplots(1,2)
    f2.canvas.set_window_title('Conjunto de dados reais')

    #Measure1, Measure4 e Species
    scatter = ax3.scatter(X_meas1, X_meas4, c = Y)
    ax3.set_xlabel(dataset.columns[0])
    ax3.set_ylabel(dataset.columns[3])
    handles, labels = scatter.legend_elements()
    labels = ['setosa','versicolor','virginica']
    ax3.legend(handles, labels, loc='best')

    #Measure2, Measure3 e Species
    scatter = ax4.scatter(X_meas2, X_meas3, c = Y)
    ax4.set_xlabel(dataset.columns[1])
    ax4.set_ylabel(dataset.columns[2])
    handles, labels = scatter.legend_elements()
    labels = ['setosa','versicolor','virginica']
    ax4.legend(handles, labels, loc='best')

    f3, (ax5,ax6) = plt.subplots(1,2)
    f3.canvas.set_window_title('Conjunto de dados reais')

    #Measure2, Measure4 e Species
    scatter = ax5.scatter(X_meas2, X_meas4, c = Y)
    ax5.set_xlabel(dataset.columns[1])
    ax5.set_ylabel(dataset.columns[3])
    handles, labels = scatter.legend_elements()
    labels = ['setosa','versicolor','virginica']
    ax5.legend(handles, labels, loc='best')

    #Measure3, Measure4 e Species
    scatter = ax6.scatter(X_meas3, X_meas4, c = Y)
    ax6.set_xlabel(dataset.columns[2])
    ax6.set_ylabel(dataset.columns[3])
    handles, labels = scatter.legend_elements()
    labels = ['setosa','versicolor','virginica']
    ax6.legend(handles, labels, loc='best')


    plt.show()