################### PPGEE 2021/01 ####################
############## Aprendizado de máquina ################
# Trabalho 5: Regressão com Random Forest
# Alunos: Carlos Henrique
#         Willian Guerreiro Colares

#Import de bibliotecas para manipulação e visualização de dados
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime
import matplotlib.pyplot as plt
import time


#Função de avaliação do modelo já treinado
def evaluate(model, test_features, test_labels):
    #Erro absoluto médio
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    #Acurácia
    accuracy = 100 - mape
    #Erro médio quadrático
    N = len(predictions)
    Sum = 0
    for i,element in enumerate(predictions):
        Sum = Sum + math.pow((element - test_labels[i]),2)
    mse = Sum / N

    print('Model Performance')
    print('Average Error: {:0.4f} degrees'.format(np.mean(errors)))
    print('MAPE: {:0.2f}%'.format(mape))
    print('Accuracy = {:0.2f}%'.format(accuracy))
    print('MSE :{:0.2f}'.format(mse))
    
#Leitura dos dados
features = pd.read_excel('temps_extended.xlsx')
#Exibição das 5 primeiras observações
print(features.tail(5))

#Variáveis a serem utilizadas:  year, ws_1, temp2,temp1,average

#TODO: Visualização dos dados

#Labels são os valores que serão preditos (saída)
labels = np.array(features['actual'])

#Labels retirados de features, bem como os dados que não serão utilizados
features = features.drop('actual',axis = 1)
features = features.drop('month',axis = 1)
features = features.drop('day',axis = 1)
features = features.drop('weekday',axis = 1)
features = features.drop('friend',axis = 1)
features = features.drop('prcp_1',axis = 1)
features = features.drop('snwd_1',axis = 1)

print(features.head(5))

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

#Armazenamento dos nomes das características para uso futuro
feature_list = list(features.columns)

features = np.array(features)
#Criação do conjunto de treino e teste
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

################## TREINAMENTO ##############################
#Fixed params
#bootstrap = True, max_depth = 100, max_features = sqrt, min_samples_leaf = 4
#min_samples_split = 12, n_estimators = 300

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(bootstrap = True,
                           max_depth = 100,
                           max_features = 'sqrt',
                           min_samples_leaf= 4,
                           min_samples_split= 12,
                           n_estimators = 300, 
                           random_state = 42, verbose= 0)

# Train the model on training data

rf.fit(train_features, train_labels)

evaluate(rf, test_features, test_labels)

