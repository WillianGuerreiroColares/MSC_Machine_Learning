import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier

# Create function returning a compiled network
def create_network():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(100,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                loss='mean_squared_error',
                metrics=['accuracy'])
    return model

#Leitura do banco de dados e variáveis alvo
data = pd.read_excel('ovarianInputs.xlsx',header = None)
target_data = pd.read_excel('ovarianTargets.xls',header = None)

graph_labels = ['Câncer','Não-câncer']

#Data
x = np.array(data).reshape((-1,100))

#Targets
y = np.array(target_data[0]).reshape((-1,1))



# Define the K-fold Cross Validator
#kfold = KFold(n_splits=5, shuffle=False)
skf = StratifiedKFold(n_splits=5)

# K-fold Cross Validation model evaluation
fold_no = 1


neural_network = KerasClassifier(build_fn=create_network, 
                                 epochs=100, 
                                 batch_size=100, 
                                 verbose=0)

print(cross_val_score(neural_network, x, y, cv=5))


'''
for train, test in skf.split(x, y):
    X_train, X_test, y_train, y_test = x[train],x[test],y[train],y[test]
    N, D = X_train.shape

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
              loss='mean_squared_error',
              metrics=['accuracy'])

    # Train the model
    r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

    # Evaluate the model - evaluate() returns loss and accuracy
    print("Train score:", model.evaluate(X_train, y_train))
    print("Test score:", model.evaluate(X_test, y_test))

    print('Making predictions')
    P = model.predict(X_test)
    P = np.round(P).flatten()
    print(P)

    # Calculate the accuracy, compare it to evaluate() output
    print("Manually calculated accuracy:", np.mean(P == y_test))
    print("Evaluate output:", model.evaluate(X_test, y_test))

    #Bloco de codigo para estilizacao e visualizacao da matriz de confusao
    window_name = 'Pasta ' + str(fold_no)
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
    plt.figure()
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()

    # Plot the accuracy 
    plt.figure()
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()

    plt.show()

    fold_no = fold_no + 1
    
'''