

% Carregamento do Dataset
imds = imageDatastore('ORL','IncludeSubfolders',true,'LabelSource','foldernames');

numTrainFiles = 7;
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
convolution2dLayer([5 5],32,"Name","conv","Padding","same")
batchNormalizationLayer("Name","batchnorm")
reluLayer("Name","relu")
dropoutLayer("Name","dropout")
maxPooling2dLayer(2,'Stride',2)
fullyConnectedLayer(40,"Name","fc")  %40 Numero de classes
softmaxLayer("Name","softmax")
classificationLayer("Name","classoutput")];

options = trainingOptions('adam', 'ValidationData',imdsValidation, ...
                          'ValidationFrequency',30,'Verbose',true, ...
                          'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YTest = imdsValidation.Labels;
accuracy = sum(YPred == YTest)/numel(YTest);
