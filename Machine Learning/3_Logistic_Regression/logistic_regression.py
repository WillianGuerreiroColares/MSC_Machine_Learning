import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# learning rate scheduler
def schedule(epoch, lr, logs = {}):
    if epoch >= 600:
        return 0.0001
    return 0.001

#Leitura do banco de dados e variáveis alvo
data = pd.read_excel('ovarianInputs.xlsx',header = None)
target_data = pd.read_excel('ovarianTargets.xls',header = None)

#Definição de Labels do gráfico da matriz de confusão
graph_labels = ['Câncer','Não - Câncer']

#Data
x = np.array(data).reshape((-1,100))
ones_columm = np.ones(216,dtype=float).reshape(-1,1)
x = np.concatenate([x,ones_columm],axis = 1)

#Targets
y = np.array(target_data[0]).reshape((-1,1))

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)
#skf = StratifiedKFold(n_splits=5)

# K-fold Cross Validation model evaluation
fold_no = 1

accuracies = list()
sensibilities = list()
specificities = list()

for train, test in kfold.split(x, y):
    X_train, X_test, y_train, y_test = x[train],x[test],y[train],y[test]
    
    N, D = X_train.shape   
    #escalonamento [0,1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(X_train)
    print(X_test)
    #input('break')

    #Definição da primeira camada de neurônio
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #utilizar a função de perda logística
    model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    
    # Train the model
    r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, callbacks = [scheduler])

    aux = model.layers[0].get_weights()
    #Pesos
    print(aux[0])

    # Evaluate the model - evaluate() returns loss and accuracy
    print("Train score:", model.evaluate(X_train, y_train))
    print("Test score:", model.evaluate(X_test, y_test))

    print('Making predictions')
    count = 0
    sensibility = 0
    specificity = 0
    P = model.predict(X_test)
    P = np.round(P).flatten()
    for j,element in enumerate(P):
        if element == y_test[j]:
            count = count + 1
            if element == 0:
                specificity = specificity + 1
            else:
                sensibility = sensibility + 1
    
    true_quantity = np.count_nonzero(y_test == 1)
    false_quantity = np.count_nonzero(y_test == 0)
   
    acc = round(count/len(P),4)
    sens = round(sensibility/true_quantity,4)
    spec = round(specificity/false_quantity,4)
    # Calculate the accuracy, compare it to evaluate() output
    print("Manually calculated accuracy:", acc)
    print("Manually calculated sensibility:", sens)
    print("Manually calculated specificity:", spec)
    print("Pacientes com cancer: (y_true) ",true_quantity)
    print("Pacientes normais: (y_true) ",false_quantity)

    #testing = np.concatenate([y_test,np.array(P).reshape((-1,1))],axis = 1)
    #print(testing)

    accuracies.append(acc)
    sensibilities.append(sens)
    specificities.append(spec)

    
    #Bloco de codigo para estilizacao e visualizacao da matriz de confusao
    window_name = 'Pasta ' + str(fold_no) + ' Matriz de confusão'
    fig = plt.figure(window_name)
    matrix = confusion_matrix(P,y_test)
    #print(matrix)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix,cmap = 'Blues')
    for (m,n),label in np.ndenumerate(matrix):
        ax.text(n,m,int(label),ha='center',va='center')
    ax.set_xticklabels(['']+graph_labels)
    ax.set_yticklabels(['']+graph_labels)
    fig.colorbar(cax)

    # Plot what's returned by model.fit()
    window_name = 'Pasta ' + str(fold_no) + ' Perdas'
    fig = plt.figure(window_name)
    plt.plot(r.history['loss'], label='Perda (Treino)')
    plt.plot(r.history['val_loss'], label='Perda (Teste)')
    plt.legend()

    # Plot the accuracy 
    window_name = 'Pasta ' + str(fold_no) + ' Acurácias'
    fig = plt.figure(window_name)
    plt.plot(r.history['accuracy'], label='Acurácia (Treino)')
    plt.plot(r.history['val_accuracy'], label='Acurácia (Teste)')
    plt.legend()

    #plt.show()

    fold_no = fold_no + 1

print('Acurácias: ',accuracies)
print('Sensibilidades: ',sensibilities)
print('ESpecificidades: ',specificities)
print('Media: ',np.mean(accuracies))
print('Maxima: ',np.max(accuracies))

accuracies.append(np.mean(accuracies))
sensibilities.append(np.mean(sensibilities))
specificities.append(np.mean(specificities))

metricas = [accuracies, sensibilities ,specificities]
print(metricas)

labels = ['Pasta 1', 'Pasta 2', 'Pasta 3', 'Pasta 4', 'Pasta 5', 'Mean']

sns.set_theme(style= 'whitegrid')

x = np.arange(len(labels))  # the label locations
width = 0.25# the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracies, width, label='Acurácia')
rects2 = ax.bar(x + width/2, sensibilities, width, label='Sensibilidade')
rects3 = ax.bar(x + 3*width/2, specificities, width, label='Especificidade')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc = 'lower right')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()


plt.show()


