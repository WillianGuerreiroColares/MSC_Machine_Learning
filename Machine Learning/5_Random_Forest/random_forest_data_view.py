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
visualization_data = features
#Exibição das 5 primeiras observações
print(features.head(5))
#Variáveis a serem utilizadas:  year, ws_1, temp2,temp1,average

#TODO: Visualização dos dados

#Labels são os valores que serão preditos (saída)
labels = np.array(features['actual'])


#Labels retirados de features, bem como os dados que não serão utilizados

features = features.drop('actual',axis = 1)
#features = features.drop('month',axis = 1)
#features = features.drop('day',axis = 1)
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


######################## TESTE ##############################

# Use the forest's predict method on the test data

'''
#Alguns dados de teste isolados
custom_tf1 = [[2012, 12.3, 63, 61, 59.1]]   #ROW 656
custom_tf2 = [[2013, 4.25, 79, 82, 76.7]]   #ROW 932
custom_tf3 = [[2014, 7.61, 81, 87, 76.9]]   #ROW 1319
custom_tf4 = [[2015, 10.74, 52, 52, 51.8]]  #ROW 1523
'''

#Predição sobre todo o conjunto de teste
predictions = rf.predict(test_features)
'''
#Predição sobre as 4 observações citadas acima
prediction1 = rf.predict(custom_tf1)
prediction2 = rf.predict(custom_tf2)
prediction3 = rf.predict(custom_tf3)
prediction4 = rf.predict(custom_tf4)
print(prediction1,prediction2,prediction3,prediction4)
'''
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#TODO RMS ERROR
#############################################################

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

evaluate(rf, test_features, test_labels)

################## Análises ##################################

#Análise de importância de características

#Obtenção da importância de cada característica
importances = list(rf.feature_importances_)
print(importances)

#Obtenção de lista na forma: [(nome_da_caracteristica,importancia)]
feature_importance = [(feature, round(importance, 2)) for feature, importance
in zip(feature_list, importances)]

print(feature_importance)
 
#Análise através da visualização dos dados
'''
features.insert(2, "Team", "Any")
features.insert(2, "Team", "Any")
features.insert(2, "Team", "Any")
'''


# Dates of training values
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'Valor real')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'Predição')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Data'); plt.ylabel('Temperatura máxima (ºF)'); plt.title('Valores reais e preditos')

plt.show()
