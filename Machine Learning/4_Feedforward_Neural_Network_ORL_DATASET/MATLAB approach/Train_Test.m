function [Train Test] = Train_Test(DATA,No_SampleClass,No_TrainSamples,No_TestSamples)
[m n TotalSamples]=size(DATA);

%----------- Conjunto de Treinamento  --------------
d1 = 1; d2 = No_TrainSamples;
t1 = 1; t2 = No_TrainSamples;

for i = 1: No_TrainSamples*(TotalSamples/No_SampleClass);
Train(:,:,d1:d2) = DATA(:,:,t1:t2);
d1 = d1+No_TrainSamples; 
d2 = d2+No_TrainSamples;
t1 = t1+No_SampleClass; 
t2 = t2+No_SampleClass;

if (t1 > TotalSamples)
    break;
end
end
%----------- Conjunto de Teste --------------
c1 = 1; c2 = No_TestSamples;
e1 = No_TrainSamples+1; e2 = No_TestSamples+No_TrainSamples;

for i = 1:No_TestSamples*(TotalSamples/No_SampleClass);
    Test(:,:,c1:c2) = DATA(:,:,e1:e2);
    c1 = c1+No_TestSamples; 
    c2 = c2+No_TestSamples;
    e1 = e1+No_SampleClass; 
    e2 = e2+No_SampleClass;
    
    if (e1 > TotalSamples)
        break;
    end
end