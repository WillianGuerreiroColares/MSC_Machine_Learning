% Projeto 3: Aprendizado de Máquina
% Implementação de uma rede recorrente LSTM para classificação de atividade humana utilizando sensores acelerômetros
% Alunos: Carlso Fonseca e Willian Guerreiro

% Load Sequence Data
% Load the human activity recognition data. The data contains seven time series of sensor data obtained from a smartphone worn on the body. Each sequence has three features and varies in length. The three features correspond to the accelerometer readings in three different directions.
%load HumanActivityTrain

load data.mat
X
XTrain = X';

load labels.mat
Y
YTrain = categorical(Y');

TrainSet = 0.85;
TestSet  = 0.15;
    
X3D = cell(uint16(1724*TrainSet),1);
Y3D = zeros(uint16(1724*TrainSet),1);

X3D_Teste = cell(uint16(1724*TestSet),1);
Y3D = zeros(uint16(1724*TestSet),1);



shuffled = randperm(1724);

for i = 1:uint16(1724*TrainSet)
    X3D(i) = {[XTrain(shuffled(i), 1:151); XTrain(shuffled(i), 152:302); XTrain(shuffled(i), 303:453)]};        
end
Y3D = YTrain(shuffled(1:uint16(1724*TrainSet)));

for i = uint16(1724*TrainSet+1):1724
    X3D_Teste(i - uint16(1724*TrainSet)) = {[XTrain(shuffled(i), 1:151); XTrain(shuffled(i), 152:302); XTrain(shuffled(i), 303:453)]};    
end
Y3D_Teste = YTrain(shuffled(uint16(1724*TrainSet+1):1724));
      
% Define LSTM Network Architecture
% Define the LSTM network architecture. Specify the input to be sequences of size 3 (the number of features of the input data). Specify an LSTM layer with 200 hidden units, and output the full sequence. Finally, specify five classes by including a fully connected layer of size 5, followed by a softmax layer and a classification layer.
 
 numFeatures = 3;
 numHiddenUnits = 250;
 numClasses = 9;
 
 layers = [ ...
     sequenceInputLayer(numFeatures)
     lstmLayer(numHiddenUnits,'OutputMode','last')
     fullyConnectedLayer(numClasses)
     softmaxLayer
     classificationLayer];
 
 %Specify the training options. Set the solver to 'adam'. Train for 60 epochs. To prevent the gradients from exploding, set the gradient threshold to 2.
 
 options = trainingOptions('adam', ...
     'MaxEpochs',30, ...
     'GradientThreshold',2, ...
     'Verbose',0, ...
     'ExecutionEnvironment','gpu',...
     'Plots','training-progress');
 
%Train the LSTM network with the specified training options using trainNetwork. Each mini-batch contains the whole training set, so the plot is updated once per epoch. The sequences are very long, so it might take some time to process each mini-batch and update the plot.
 
net = trainNetwork(X3D,Y3D,layers,options);

%Test LSTM Network  
% Classify the test data using classify. 
 
YPred = classify(net,X3D_Teste);

% Calculate the accuracy of the predictions. 
acc = sum(YPred == Y3D_Teste)./numel(Y3D_Teste)
 
C = confusionmat(Y3D_Teste, YPred)

confusionchart(C)

