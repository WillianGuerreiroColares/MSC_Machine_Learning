import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

'''from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split'''


# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

def visualize(data):

    #print(data, data.shape)
    xdata = data[:151]
    ydata = data[151:302]
    zdata = data[302:]


    t = np.arange(0, 151, 1)

    fig = plt.figure()

    mag1 = np.sqrt(np.square(list(xdata[0])) + np.square(list(ydata[0])) + np.square(list(zdata[0])))
    plt.subplot(3,3,1)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Levantando (cadeira)')
    plt.plot(t,xdata[0],t,ydata[0],t,zdata[0])

    mag2 = np.sqrt(np.square(list(xdata[42])) + np.square(list(ydata[42])) + np.square(list(zdata[42])))
    plt.subplot(3,3,2)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Levantando (cama)')
    plt.plot(t,xdata[42],t,ydata[42],t,zdata[42])

    plt.subplot(3,3,3)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Caminhando')
    plt.plot(t,xdata[88],t,ydata[88],t,zdata[88])

    plt.subplot(3,3,4)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Correndo')
    plt.plot(t,xdata[461],t,ydata[461],t,zdata[461])

    plt.subplot(3,3,5)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Descendo escada')
    plt.plot(t,xdata[877],t,ydata[877],t,zdata[877])

    plt.subplot(3,3,6)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Pulando')
    plt.plot(t,xdata[1135],t,ydata[1135],t,zdata[1135])

    plt.subplot(3,3,7)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Subindo escada')
    plt.plot(t,xdata[1299],t,ydata[1299],t,zdata[1299])

    plt.subplot(3,3,8)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Deitando na cama')
    plt.plot(t,xdata[1607],t,ydata[1607],t,zdata[1607])

    plt.subplot(3,3,9)
    plt.xlabel('Tempo')
    plt.ylabel('Magnitudes')
    plt.title('Sentando na cadeira')
    plt.plot(t,xdata[1669],t,ydata[1669],t,zdata[1669])

    '''axs[1].plot(t,xdata[0])
    axs[2].plot(t,xdata[0])
    axs[3].plot(t,xdata[0])
    axs[4].plot(t,xdata[0])
    axs[5].plot(t,xdata[0])
    axs[6].plot(t,xdata[0])
    axs[7].plot(t,xdata[0])
    axs[8].plot(t,xdata[0])

    fig.tight_layout()'''
    #fig.legend(['x ', 'y', 'z'])

    plt.show()

def Classes_index(labels, classes):
    num_classes = len(classes)
    occur_list = []
    for i in range(num_classes):
        occur_list.append(np.where(np.array(labels) == i+1)[1])
    return occur_list

def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 60, 64
    print(trainX.shape)
    print(trainy.shape)
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(200, input_shape=(n_timesteps,n_features)))
    #model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(trainX, trainy, validation_data = (testX, testy), epochs=epochs, batch_size=batch_size, verbose=verbose)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

# fix random seed for reproducibility
np.random.seed(7)

#Define classes
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#Load data and labels
data = pd.read_excel('dados.xlsx', header=None)
labels = pd.read_excel('labels.xlsx', header=None)

#Splitting data according to ratio 0.85
classes_index = Classes_index(labels, classes)

visualize(data)

print(data, data.shape, type(data))

x = np.array(data)
print(x)
x = np.transpose(x)

y = np.array(labels)
y = np.transpose(y)

'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

#print('org',X_train, X_train.shape, len(X_train))
# reshape input to be [samples, time steps, features]   [151,453,3]
trainX = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(trainX, trainX.shape, len(trainX))
print(testX, testX.shape, len(testX))

analytics_classes = []
graph_labels = []
for i in range(9):
    qty = np.where(y_train == i + 1)[1]
    graph_labels.append(len(qty))
    analytics_classes.append(100*len(qty)/len(y_train))

#print(analytics_classes, np.sum(analytics_classes))
#print(graph_labels)

n = np.arange(1, 10, 1)
fig, ax = plt.subplots(1)
hbars = ax.bar(n, analytics_classes)
ax.set_xticks(classes)
ax.bar_label(hbars, labels = graph_labels)
plt.show()

#Preprocessing and encoding
y_train = y_train - 1
y_test = y_test - 1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Model 3D Input
#score = evaluate_model(trainX, y_train, testX, y_test)

# run an experiment
scores = list()
for r in range(10):
    score = evaluate_model(trainX, y_train, testX, y_test)
    score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)
# summarize results
summarize_results(scores)'''