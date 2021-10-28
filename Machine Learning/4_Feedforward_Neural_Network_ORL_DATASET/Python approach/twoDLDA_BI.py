import numpy as np
import cv2

def reduce(Path):
    N = 400
    c = 40
    N_i = int(N/c)
    #Cálculo da matriz de covariância entre-classes

    #Passo 1: Obtenção das médias de cada classe e média global

    #Média geral
    A = np.zeros(shape=(112,92),dtype = float)
    #A = im2double(zeros(112,92))
    #Média por classe
    #Aux = im2double(zeros(112,92))
    Aux = np.zeros(shape=(112,92),dtype = float)

    A_i = list()

    for folder in range(c):
        folder_name = 's' + str(folder+1)
        for pic in range(N_i):
            path = Path + '\s' + str(folder+1)+'\\'+str(pic+1)+'.pgm'
            image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            A = A + image
            Aux = Aux + image
        Aux = Aux/10
        A_i.append(Aux)
        Aux = np.zeros(shape=(112,92),dtype = float)

    A = A / N

    #Passo 2: Redução horizontal e vertical

    #Passo 3: Obtenção de S_b (H e V)
    S_b_H = np.zeros(shape=(92,92),dtype = float)
    S_b_V = np.zeros(shape=(112,112),dtype = float)

    U = A_i[0] - A
    for i in range(c):
        S_b_H = S_b_H + 10*(np.dot((np.transpose(A_i[i] - A)),(A_i[i] - A)))
        S_b_V = S_b_V + 10*(np.dot((A_i[i] - A),np.transpose((A_i[i] - A))))
    
    S_b_H = S_b_H / N
    S_b_V = S_b_V / N

    #Passo 4: Obtenção de S_w (H e V)

    S_w_H = np.zeros(shape=(92,92),dtype = float)
    S_w_V = np.zeros(shape=(112,112),dtype = float)
    for i in range(c):
        folder_name = 's' + str(i+1)
        for j in range(N_i):
            path = Path + '\s' + str(i+1)+'\\'+str(j+1)+'.pgm'
            image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            S_w_H = S_w_H + np.dot(np.transpose(image - A_i[i]),(image - A_i[i]))
            S_w_V = S_w_V + np.dot(image - A_i[i],np.transpose(image - A_i[i]))

    S_w_H = S_w_H / N
    S_w_V = S_w_V / N

    #Passo 5: Determinação das matrizes U E V

    #Passo 5.1: Multiplicação entre as matrizes de covariância
    product1 = np.dot(np.linalg.inv(S_w_H),S_b_H)
    product2 = np.dot(np.linalg.inv(S_w_V),S_b_V)

    #Obtenção dos 10 maiores autovalores e seus respectivos autovetores
    
    #Matriz U
    [autovalores1,autovetores1] = np.linalg.eig(product1)
    indices_ordered = np.flip(np.argsort(autovalores1))
    U = autovetores1[:,indices_ordered]
    U = U[:,0:10]
  
    #Matriz V
    [autovalores2,autovetores2] = np.linalg.eig(product2)
    indices_ordered = np.flip(np.argsort(autovalores2))
    V = autovetores2[:,indices_ordered]
    V = V[:,0:10]

    
    return U,V
