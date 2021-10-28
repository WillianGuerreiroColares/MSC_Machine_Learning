

% Carregamento do Dataset
imds = imageDatastore('ORL','IncludeSubfolders',true,'LabelSource','foldernames');

numTrainFiles = 8;
%numTestFiles = 2;

[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');

%[imdsTest, imdsValidation] = splitEachLabel(imdsValidation, numTestFiles, 'randomize');

%{
for i=1:100
    img = readimage(imds,i);
    imshow(img);
end
%}


layers = [
imageInputLayer([112 92 1],"Name","imageinput")

convolution2dLayer([5 5],32,"Name","conv1","Padding","same")
batchNormalizationLayer("Name","batchnorm1")
reluLayer("Name","relu1")
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer([5 5],32,"Name","conv2","Padding","same")
batchNormalizationLayer("Name","batchnorm2")
reluLayer("Name","relu2")
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer([5 5],32,"Name","conv3","Padding","same")
batchNormalizationLayer("Name","batchnorm3")
reluLayer("Name","relu3")
maxPooling2dLayer(2,'Stride',2)

fullyConnectedLayer(40,"Name","fc")  %40 Numero de classes
softmaxLayer("Name","softmax")
classificationLayer("Name","classoutput")];

options = trainingOptions('adam', 'ValidationData',imdsValidation, ...
                          'ValidationFrequency',30,'Verbose',true, ...
                          'Plots','training-progress');

op = input('Train? (y/n)','s');

if strcmp(op,'y') > 0
    net = trainNetwork(imdsTrain,layers,options);
else
    net = net_best;
end

%Função para visualizar a rede:
analyzeNetwork(net)

YPred = classify(net,imdsValidation);
YTest = imdsValidation.Labels;
accuracy = sum(YPred == YTest)/numel(YTest);


%{
for i=1:15
    img = readimage(imdsValidation,i);
    label = classify(net,img);
    scoreMap = gradCAM(net,img,label);
    
    figure
    imshow(img)
    hold on
    imagesc(scoreMap,'AlphaData',0.5)
    colormap jet
end

%}

%Lendo a primeira imagem do conjunto de validação
img_teste = readimage(imdsValidation,1);
%Obtém-se as saídas após passada pela camada conv1
act1 = activations(net,img_teste,'conv1');
%Obtém-se as dimensões da estrutura 112x92x32x1 [H x W x Kernels x ColorChannels]
sz = size(act1);
%Configura a estrutura no formato 112x92x1x32
%São 32 Kernels 112x92 com 1 canal de cores
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
%Exibição dos 32 Kernels em uma única grade
I = imtile(mat2gray(act1),'GridSize',[4 8]);
imshow(I);

individuo1 = readimage(imdsValidation,10);
individuo2 = readimage(imdsValidation,30);

label1 = classify(net,individuo1);

label2 = classify(net,individuo2);

scoreMap1 = gradCAM(net,individuo1,label1);

scoreMap2 = gradCAM(net,individuo2,label2);

figure
imshow(individuo1)
hold on
imagesc(scoreMap1,'AlphaData',0.5)
colormap jet

figure
imshow(individuo2)
hold on
imagesc(scoreMap2,'AlphaData',0.5)
colormap jet

C = confusionmat(YTest,YPred);
figure
confusionchart(C)

%Obtenção de pesos e polarizações da camada N

W = net.Layers(2).Weights; %Weights of Convolution layer
B = net.Layers(2).Bias; %Biases of Convolution layer
