################### PPGEE 2021/01 ####################
############## Aprendizado de máquina ################
# Trabalho 5: Regressão com Random Forest
# Alunos: Carlos Henrique
#         Willian Guerreiro Colares

# Import de bibliotecas para manipulação e visualização de dados
import time
import math
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

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
    print('MSE :{:0.2f}%'.format(mse))

# Leitura dos dados
features = pd.read_excel('temps_extended.xlsx')

# Exibição das 5 primeiras observações
print(features.head(5))

# Variáveis a serem utilizadas:  year, ws_1, temp2,temp1,average

# Labels são os valores que serão preditos (saída)
labels = np.array(features['actual'])

# Labels retirados de features, bem como os dados que não serão utilizados
features = features.drop('actual', axis=1)
features = features.drop('month', axis=1)
features = features.drop('day', axis=1)
features = features.drop('weekday', axis=1)
features = features.drop('friend', axis=1)
features = features.drop('prcp_1', axis=1)
features = features.drop('snwd_1', axis=1)

# Armazenamento dos nomes das características para uso futuro
feature_list = list(features.columns)

features = np.array(features)

# Criação do conjunto de treino e teste
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)


################## TREINAMENTO ##############################

# Number of trees in random forest
n_estimators = np.arange(200,2200,200)  #start = range de 200 a 2200(aberto), com saltos de 200
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = list(np.arange(10,110,10))
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 12]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 50 different combinations, and use all available cores



rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 50, scoring='neg_mean_absolute_error',
                              cv = None, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
t_1 = time.time()
rf_random.fit(train_features, train_labels)
t_2 = time.time()

best_random = rf_random.best_estimator_
print('Best parameters Ramdom search: ',best_random.get_params())
print('Tunning time: ',t_2 - t_1,' seconds')
evaluate(best_random, test_features, test_labels)
print("\n")
parameters = best_random.get_params()
opt_max_depth = parameters['max_depth']
opt_min_samples_leaf = parameters['min_samples_leaf']
opt_min_samples_split = parameters['min_samples_split']
opt_n_estimator = parameters['n_estimators']

#Grid search 1
while True:
    Input = input("Enter para o grid 1")
    if opt_max_depth == 10:
        opt_max_depth = 20
    elif opt_max_depth == 100:
        opt_max_depth = 90

    if opt_n_estimator == 200:
        opt_n_estimator = 400
    elif opt_n_estimator == 2000:
        opt_n_estimator = 1800

    if opt_min_samples_leaf == 1 or opt_min_samples_leaf == 4:
        opt_min_samples_leaf = 2

    if opt_min_samples_split == 2 or opt_min_samples_split == 10:
        opt_min_samples_split = 5

    param_grid = {
        'bootstrap': [True],
        'max_depth': [opt_max_depth-10, opt_max_depth, opt_max_depth+10],
        'max_features': ['sqrt'],
        'min_samples_leaf': [opt_min_samples_leaf-1,opt_min_samples_leaf,opt_min_samples_leaf+2],
        'min_samples_split': [opt_min_samples_split-3,opt_min_samples_split,opt_min_samples_split+5],
        'n_estimators': [opt_n_estimator-200, opt_n_estimator, opt_n_estimator+200]
    }


    # Create a based model
    #rf = RandomForestRegressor()

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                            scoring = 'neg_mean_absolute_error', cv = None,
                            n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    t_1 = time.time()
    grid_search.fit(train_features, train_labels)
    t_2 = time.time()
    best_grid = grid_search.best_estimator_

    print('Best parameters Grid search 1: ')
    print(best_grid.get_params())
    print('Tunning time: ',t_2 - t_1,' seconds')
    #evaluate(best_grid, test_features, test_labels)
    evaluate(best_grid, test_features, test_labels)

    parameters = best_grid.get_params()
    opt_max_depth = parameters['max_depth']
    opt_min_samples_leaf = parameters['min_samples_leaf']
    opt_min_samples_split = parameters['min_samples_split']
    opt_n_estimator = parameters['n_estimators']

    print("\n")



