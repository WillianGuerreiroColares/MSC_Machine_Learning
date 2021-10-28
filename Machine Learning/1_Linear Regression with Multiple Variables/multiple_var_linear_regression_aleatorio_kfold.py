import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

MSE = np.empty(5, dtype=float) 
corr = np.empty(5, dtype=float) 

dataset = pd.read_excel('hospital.xls')
#print(dataset)
#Data preparation 
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
X_ones = np.ones(100,dtype=int).reshape(-1,1)

X = np.concatenate([X_sex,X_age,X_weight,X_smoker,X_ones],axis = 1)
#print(X)

#Obtenção de valores a serem preditos (Y_treino)
Y = np.array((dataset['BloodPressure_1'])).reshape((-1,1))

#Realização de treinamento e predição a 5 pastas

for i in range(5):
    indices = np.random.permutation(X.shape[0])
    x_treino_id, x_teste_id = indices[:80],indices[80:]
    x_treino, x_teste = X[x_treino_id,:], X[x_teste_id,:]
    y_treino, y_esperado = Y[x_treino_id,:], Y[x_teste_id,:]
    print('Fold: ',i+1)
    #print(np.concatenate([x_treino,y_treino],axis = 1))
    #Obtenção da pseudo inversa
    X_plus = (np.linalg.inv((x_treino.T).dot(x_treino))).dot(x_treino.T)
    #Obtenção dos pesos W
    w = X_plus.dot(y_treino)
    #print(X)
    #print(X.T)
    #print(w)

    #Predição
    y_predito = np.zeros(20,dtype=int).reshape(-1,1)
    print('\t  X\t\tY_predito  Y_esperado')
    for j in range(20):
        y_predito[j] = (w.T).dot(x_teste[j])
        print(x_teste[j],y_predito[j],y_esperado[j])
    
    #Erro médio quadrático
    soma = 0
    for j in range(20):
        soma = soma + pow(y_predito[j]-y_esperado[j],2)
    MSE[i] = soma/20
    #print(MSE[i])

    #Correlação de pearson
    covariancia = 0
    variancia_predito = 0
    variancia_esperado = 0
    mean_predito = np.average(y_predito)
    mean_esperado = np.average(y_esperado)
    for j in range(20):
        covariancia = covariancia + (y_predito[j] - mean_predito)*(y_esperado[j] - mean_esperado)
        variancia_predito = variancia_predito + pow(y_predito[j] - mean_predito,2)
        variancia_esperado = variancia_esperado + pow(y_esperado[j] - mean_esperado,2)
    corr[i] = covariancia / ((math.sqrt(variancia_predito)) * (math.sqrt(variancia_esperado)))
    #print(corr[i])
    

print('MSE: ',MSE)
print('Média de 5 Folds ',np.average(MSE))
print('Correlacoes: ',corr)
print('Média de 5 Folds ',np.average(corr))