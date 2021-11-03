import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import seaborn as sns

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

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    ax1.plot(t, xdata[0], t, ydata[0], t, zdata[0])
    ax2.plot(t, xdata[1], t, ydata[1], t, zdata[1])
    ax3.plot(t, xdata[2], t, ydata[2], t, zdata[2])
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
    n_timesteps, n_features, n_outputs = trainX.shape[2], trainX.shape[1], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(250, input_shape=(n_features, n_timesteps)))
    #model.add(Dropout(0.5))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(trainX, trainy, validation_data = (testX, testy), epochs=epochs, batch_size=batch_size, verbose=verbose)

    '''# list all data in history
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
    plt.show()'''

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, model

# fix random seed for reproducibility
np.random.seed(7)

#Define classes
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#Test size
test_size = 0.85

#Load data and labels
data = pd.read_excel('dados.xlsx', header=None)
#print(data, data.shape, type(data))
#visualize(data)

xdata = np.array(data[:151])
ydata = np.array(data[151:302])
zdata = np.array(data[302:])

xdata = np.transpose(xdata)
ydata = np.transpose(ydata)
zdata = np.transpose(zdata)

X3D = np.zeros((1724,3,151))

for i in range(1724):
    mov_block = np.vstack((xdata[i], ydata[i], zdata[i]))
    X3D[i] = mov_block

labels = pd.read_excel('labels.xlsx', header=None)

#Splitting data according to ratio 0.85

y = np.array(labels)
y = np.transpose(y)

X_train, X_test, y_train, y_test = train_test_split(X3D, y, test_size=0.15)


print(X_train, X_train.shape, len(X_train))
print(X_test, X_test.shape, len(X_test))

analytics_classes = []
graph_labels = []
for i in range(9):
    qty = np.where(y_train == i + 1)[1]
    graph_labels.append(len(qty))
    analytics_classes.append(100*len(qty)/len(y_train))

#print(analytics_classes, np.sum(analytics_classes))
#print(graph_labels)

'''n = np.arange(1, 10, 1)
fig, ax = plt.subplots(1)
hbars = ax.bar(n, analytics_classes)
ax.set_xticks(classes)
ax.bar_label(hbars, labels = graph_labels)
plt.show()'''

#Preprocessing and encoding
y_train = y_train - 1
y_test = y_test - 1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Model 3D Input
#score = evaluate_model(trainX, y_train, testX, y_test)

# run an experiment
scores = list()
model_list = list()
for r in range(10):
    score, model = evaluate_model(X_train, y_train, X_test, y_test)
    model_list.append(model)
    score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)
# summarize results
summarize_results(scores)


best_model_index = np.argmax(scores)
print('Better model: ', best_model_index + 1)

best_model = model_list[best_model_index]

predicted = best_model.predict(X_test)

P = np.round(predicted)

CM = confusion_matrix(y_test.argmax(axis=1), P.argmax(axis=1))

print(CM)

con_mat_norm = np.around(CM.astype('float') / CM.sum(axis=1)[:, np.newaxis], decimals=2)


con_mat_df = pd.DataFrame(CM,
                     index = classes,
                     columns = classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

best_model.save("model.h5")

