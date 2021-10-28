import cv2
import time
import pandas as pd
import numpy as np
from twoDLDA_BI import reduce
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn as sns


# learning rate scheduler
def schedule(epoch, lr, logs = {}):
    if epoch >= 600:
        return 0.0001
    return 0.001

#Número de classes
N = 40
N_i = 10

#Lista para armazenar todas as faces
faces = list()

#Leitura das imagens
for classes in range(N):
    for element in range(N_i):
        path = 'ORL\s'+str(classes+1)+'\\'+str(element+1)+'.pgm'
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        '''
        cv2.imshow('image', image)
        cv2.waitKey(0)
        '''
#Redução de dimensionalidade (obtenção de u e v)
U,V = reduce('ORL')



#Teste da redução de dimensionalidade
for elements in faces: 
    reduced = np.dot(np.transpose(V),elements)
    reduced = np.dot(reduced,U)
    #cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)     
    maxV = np.max(reduced)
    minV = np.min(reduced)
    result = cv2.normalize(reduced,None,0,1,cv2.NORM_MINMAX)
    '''
    cv2.imshow('image', result)
    cv2.waitKey(0)
    '''

#Data
x = np.zeros(shape=(400,100),dtype = float)
ones_columm = np.ones(400,dtype=float).reshape(-1,1)


#Targets
y = np.zeros(400,dtype=float).reshape(-1,1)


for i,elements in enumerate(faces):
    reduced = np.dot(np.transpose(V),elements)
    reduced = np.dot(reduced,U)
    faces_vectorizes = reduced.reshape((-1,1))
    faces_vectorizes = np.transpose(faces_vectorizes)
    x[i] = faces_vectorizes
    y[i] = int(i/10)+1
 

# convert integers to dummy variables (i.e. one hot encoded)
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)
y = dummy_y

print(y)

#x = np.concatenate([x,ones_columm],axis = 1)


accuracies = list()

X_train = list()
X_test = list()
y_train = list()
y_test = list()

N = len(x)

for i in range(N+1):
    if i % 10 == 0 and i!=0:
        X_train.append(x[i-10:i-3])
        X_test.append(x[i-3:i])
        y_train.append(y[i-10:i-3])
        y_test.append(y[i-3:i])
        #print(i)
        #print('Interval Treino: ',i-10,'-',i-3)
        #print('Interval Teste: ',i-3,'-',i)

X_train = np.array(X_train).reshape(280,100)
y_train = np.array(y_train).reshape(280,40)

X_test = np.array(X_test).reshape(120,100)
y_test = np.array(y_test).reshape(120,40)

N,D = X_train.shape

#escalonamento [0,1]
'''
norm = np.linalg.norm(X_train)
X_train = X_train/norm

norm = np.linalg.norm(X_test)
X_test = X_test/norm
'''

'''
print(X_train)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

scaler = MinMaxScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)
'''

#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#print('Xteste',X_test)
'''
print(X_train)
print(np.shape(X_train))
print(X_test)
print(np.shape(X_test))
print(y_train)
print(np.shape(y_train))
print(y_test)
print(np.shape(y_test))
input('break')
'''

#Definição da primeira camada de neurônio
'''
model = tf.keras.models.Sequential([
tf.keras.layers.Input(shape=(D,)),
tf.keras.layers.Dense(1, activation='relu')
])
'''
model = tf.keras.models.Sequential([
tf.keras.layers.Input(shape=(D,)),
tf.keras.layers.Dense(40, activation='relu'),
tf.keras.layers.Dense(40, activation='softmax')
])

#utilizar a função de perda logística
#trocar optimizer para sgd
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, callbacks = [scheduler])

aux = model.layers[0].get_weights()

#Pesos
#print(aux[0])

# Evaluate the model - evaluate() returns loss and accuracy
print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))

print('Making predictions')
count = 0

print(X_test.shape)
P = model.predict(X_test)
P = np.round(P).flatten()

print(P.shape)

print(np.shape(X_test))
print(np.shape(P))
print(np.shape(y_test))
P = P.reshape(120,40)

for j,element in enumerate(P):
    #print('element',element)
    #print('teste',y_test[j])
    if (element == y_test[j]).all():
        count = count + 1

true_quantity = np.count_nonzero(y_test == 1)
false_quantity = np.count_nonzero(y_test == 0)

acc = round(count/len(P),4)

# Calculate the accuracy, compare it to evaluate() output
print("Manually calculated accuracy:", acc)

#testing = np.concatenate([y_test,np.array(P).reshape((-1,1))],axis = 1)
#print(testing)

accuracies.append(acc)

'''
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
'''

# Plot what's returned by model.fit()
window_name = 'Perdas'
fig = plt.figure(window_name)
plt.plot(r.history['loss'], label='Perda (Treino)')
plt.plot(r.history['val_loss'], label='Perda (Teste)')
plt.legend()

# Plot the accuracy 
window_name = 'Acurácias'
fig = plt.figure(window_name)
plt.plot(r.history['accuracy'], label='Acurácia (Treino)')
plt.plot(r.history['val_accuracy'], label='Acurácia (Teste)')
plt.legend()

#plt.show()

print('Acurácias: ',accuracies)

plt.show()

count = 0 


for i,elements in enumerate(faces):
    reduced = np.dot(np.transpose(V),elements)
    reduced = np.dot(reduced,U)
    faces_vectorizes = reduced.reshape((-1,1))
    faces_vectorizes = np.transpose(faces_vectorizes)
    #print(faces_vectorizes)
    #scaler = MinMaxScaler()
    #scaler.fit(faces_vectorizes)
    #faces_vectorizes = scaler.transform(faces_vectorizes)

    predicao = model.predict(faces_vectorizes)
    #print(predicao)

    try:
        indice = np.where(predicao == 1)
        if (int(i/10)) == indice[1]:
            count = count + 1
        #print(indice[1])
        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN) 
        cv2.imshow('image',faces[i])
        cv2.namedWindow("predicao", cv2.WND_PROP_FULLSCREEN) 
        cv2.imshow('predicao',faces[int(10*indice[1])])
        cv2.waitKey(0)
    except:
        print('Not recognized')
        pass

print(100*count/len(faces))
