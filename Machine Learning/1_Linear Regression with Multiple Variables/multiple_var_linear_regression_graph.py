################### PPGEE 2021/01 ####################
############## Aprendizado de máquina ################
# Trabalho 1: Regressão linear com múltiplas variáveis
# Alunos: Davi Cauassa Leão
#         Willian Guerreiro Colares

#Import de bibliotecas para manipulação e visualização de dados
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#Declaração dos Arrays que irão armazenar os Erros Médios Quadráticos e 
#coeficientes de correlação de Pearson para cada pasta
MSE = np.empty(5, dtype=float) 
corr = np.empty(5, dtype=float) 

#Leitura do Dataset
dataset = pd.read_excel('hospital.xls')

#Obtenção do número de observações
N = dataset.shape[0]

#Preparação de dados e ajuste de variáveis não categóricas 
#Male -> 1 Female ->0
#Smoker True/False -> 1/0
dataset['Sex'].replace('Female', 0, inplace=True)
dataset['Sex'].replace('Male', 1, inplace=True)
dataset['Smoker'] = dataset['Smoker'] * 1

#Passo 1: Determinação da matriz de padrões X
X_sex = np.array((dataset['Sex'])).reshape((-1,1))
X_age = np.array((dataset['Age'])).reshape((-1,1))
X_weight = np.array((dataset['Weight'])).reshape((-1,1))
X_smoker = np.array((dataset['Smoker'])).reshape((-1,1))
X_ones = np.ones(N,dtype=int).reshape(-1,1)

#Obtenção da matriz de padrões a partir da concatenação dos vetores-coluna de características
X = np.concatenate([X_sex,X_age,X_weight,X_smoker,X_ones],axis = 1)
print(X)
#Número de pastas
K = 5

#Tamanho do subset de treinamento é igual ao número de observações dividido pelo número de pastas
testing_size = round(N/5)
#print(testing_size)

#Obtenção de valores a serem preditos (Y_treino == coluna BloodPressure_1)
Y = np.array((dataset['BloodPressure_1'])).reshape((-1,1))

#Realização de treinamento e predição a k = 5 pastas
for i in range(K):
    #Intervalos de segmentação do dataset
    # [0;c[   [c;d[    [d;N[      sendo  [c;d[ o subset de teste   e  d-c = testing_size = 20
    c = testing_size*i
    d = testing_size*i+testing_size
    #Obtenção dos índices do conjunto de testes
    k_subset = np.arange(c, d, 1)
    #Obtenção dos índices do conjunto de treinamento a partir da união dos intervalos restantes
    trainning_subset = np.append(np.arange(0, c, 1),np.arange(d,N,1))

    #Obtenção dos dados de entrada X(treino e teste) e dados de saída(treino e valor esperado)
    x_treino, x_teste = X[trainning_subset,:], X[k_subset,:]
    y_treino, y_esperado = Y[trainning_subset,:], Y[k_subset,:]

    print('Fold: ',i+1)

    #Obtenção da pseudo inversa   PI = inv(𝑋'.𝑋).𝑋'  onde 𝑋' é a matriz transposta de X
    X_plus = (np.linalg.inv((x_treino.T).dot(x_treino))).dot(x_treino.T)
    #Obtenção dos pesos W = (PI).y
    w = X_plus.dot(y_treino)
    print('Vetor de pesos: ')
    print(w)

    #Etapa de predição

    #Inicialização do vetor com os dados de predições iguais a 0
    y_predito = np.zeros(testing_size,dtype=int).reshape(-1,1)
    print('\t  X\t\tY_predito  Y_esperado')
    for j in range(testing_size):
        y_predito[j] = (w.T).dot(x_teste[j])
        print(x_teste[j],y_predito[j],y_esperado[j])
    
    #Erro médio quadrático
    soma = 0
    for j in range(testing_size):
        soma = soma + pow(y_predito[j]-y_esperado[j],2)
    MSE[i] = soma/testing_size
    #print(MSE[i])

    #Correlação de pearson
    covariancia = 0
    variancia_predito = 0
    variancia_esperado = 0

    #Cálculo da média dos vetores de predição e valor esperado
    mean_predito = np.average(y_predito)
    mean_esperado = np.average(y_esperado)  

    for j in range(testing_size):
        covariancia = covariancia + (y_predito[j] - mean_predito)*(y_esperado[j] - mean_esperado)
        variancia_predito = variancia_predito + pow(y_predito[j] - mean_predito,2)
        variancia_esperado = variancia_esperado + pow(y_esperado[j] - mean_esperado,2)
    corr[i] = covariancia / ((math.sqrt(variancia_predito)) * (math.sqrt(variancia_esperado)))
    
    #Visualização dos dados para o k-folder corrente
    #0-sex 1-Age 2-Weight 3-Smoker
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    figure_name = 'Pasta k = ' + str(i+1)
    f.canvas.set_window_title(figure_name)

    #Idade, pressão arterial e fumante/não-fumante
    scatter = ax1.scatter(x_teste.T[1], y_predito, c=x_teste.T[3])
    ax1.set_xlabel(dataset.columns[3])
    ax1.set_ylabel(dataset.columns[6])
    handles, labels = scatter.legend_elements()
    labels = ['Não Fumante','Fumante']
    ax1.legend(handles, labels, loc='best')
    
    #Peso, pressão arterial e fumante/não-fumante
    scatter = ax2.scatter(x_teste.T[2], y_predito, c=x_teste.T[3])
    ax2.set_xlabel(dataset.columns[4])
    ax2.set_ylabel(dataset.columns[6])
    handles, labels = scatter.legend_elements()
    labels = ['Não Fumante','Fumante']
    ax2.legend(handles, labels, loc='best')

    #Genero, pressão arterial e fumante/não-fumante
    scatter = ax3.scatter(x_teste.T[0], y_predito, c=x_teste.T[3])
    ax3.set_xlabel(dataset.columns[2])
    ax3.set_ylabel(dataset.columns[6])
    handles, labels = scatter.legend_elements()
    labels = ['Não Fumante','Fumante']
    ax3.legend(handles, labels, loc='best')
    
print('MSE: ',MSE)
print('Valor médio do erro médio quadrático ',np.average(MSE))
print('Correlacoes: ',corr)
print('Valor médio do coeficiente de Pearson ',np.average(corr))

#Gráfico com os valores reais que relacionam idade, pressão máxima e condição de fumante ou não

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
f.canvas.set_window_title('Conjunto de dados reais')

#Idade, pressão arterial e fumante/não-fumante
scatter = ax1.scatter(X_age, Y, c=X_smoker)
ax1.set_xlabel(dataset.columns[3])
ax1.set_ylabel(dataset.columns[6])
handles, labels = scatter.legend_elements()
labels = ['Não Fumante','Fumante']
ax1.legend(handles, labels, loc='best')

#Peso, pressão arterial e fumante/não-fumante
scatter = ax2.scatter(X_weight, Y, c=X_smoker)
ax2.set_xlabel(dataset.columns[4])
ax2.set_ylabel(dataset.columns[6])
handles, labels = scatter.legend_elements()
labels = ['Não Fumante','Fumante']
ax2.legend(handles, labels, loc='best')

#Gênero, pressão arterial e fumante/não-fumante
scatter = ax3.scatter(X_sex, Y, c=X_smoker)
ax3.set_xlabel(dataset.columns[2])
ax3.set_ylabel(dataset.columns[6])
handles, labels = scatter.legend_elements()
labels = ['Não Fumante','Fumante']
ax3.legend(handles, labels, loc='best')

plt.show()