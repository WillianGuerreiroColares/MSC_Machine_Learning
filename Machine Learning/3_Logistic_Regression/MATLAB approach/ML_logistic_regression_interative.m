% Projeto 3: Aprendizado de Máquina
% Implementação de um Classificador por Regressão Logística
% Com cálculo da Função de Custo e Gradiente descendente 
% Alunos: Carlso Fonseca e William Guerreiro

X = xlsread("ovarianInputs.xlsx");
R = xlsread("ovarianTargets.xls");
Y = R(:,1);
m = length(Y);
[r,c] = size(X);
idx = randperm(m);
y = Y;
p = 0.8;  % Para k-fold 05 pastas: 80% treinamento, 20% teste

xtrain = X(idx(1:round(p*m)),:);     % x_treino randomico
ytrain = y(idx(1:round(p*m)),:);     % y_treino correspondente
xtest  = X(idx(round(p*m)+1:m),:); % x_teste = complemento 
ytest  = y(idx(round(p*m)+1:m),:); % y_teste correspondente 
Num_Pastas= 5;
Acuracia_Media = 0;

for Pasta = 1:Num_Pastas

% Atualiza w pelo gradiente descendente
alpha = 0.001;
w = double(randperm(c+1)/100);
w = -w';  % vetor de pesos iniciais 

mtrain  = length(ytrain);
xtrain2 = [ones(mtrain,1) xtrain];
Acuracia_TrainingSet = 0.0;

cont  = 1;
epoch = 1;

while (epoch <= 1200)  % Criterio de parada do algoritmo
     
    if epoch >= 600     % critério de revaliação do alpha
        alpha = 0.0001; % aumenta a precisão em 10x após 600 épocas
    end
    
    ztrain  = xtrain2*w;                % w'x
    htrain = 1.0./(1.0 + exp(-ztrain)); % Sigmoide
    
    for j = 1:c+1         % Calculo do Gradiente Descendente
        somatoria = 0.0;
        for i = 1:mtrain
            somatoria = somatoria + alpha*double((htrain(i) - ytrain(i))*xtrain2(i,j));
        end    
            w(j) = w(j) - somatoria;           
    end    
    
    Loss = 0.0;
    
    for i = 1:mtrain     % Calculo da Função de Custo
        Loss = Loss + ytrain(i)*log(htrain(i)) + (1-ytrain(i))*log(1-htrain(i));
    end
    Loss = Loss * (-1.0/mtrain);
    perda(cont) = Loss;
    cont = cont + 1;
    
    % Define o treshold de > 0.5 para classificar pacientes com cancer
     
    ytrainpred = double(htrain > 0.5);
    Acuracia_TrainingSet = mean(double(ytrainpred == ytrain))*100;
    epoch = epoch + 1;
end

fprintf('Pasta No: %d',Pasta);
Acuracia_TrainingSet
figure
plot(perda);
histogram(htrain,10); 

mtest  = length(ytest);
xtest2 = [ones(mtest,1) xtest];
ztest  = xtest2*w;

% Avaliação do Modelo encontrado (w) no conjunto de testes
htest = 1.0./(1.0 + exp(-ztest));
ytestpred = double(htest > 0.5);
ytestpred_sem = double(htest <= 0.5);
Acuracia_TestingSet = mean(double(ytestpred == ytest))*100
histogram(htest,10); 
Acuracia_Media = Acuracia_Media + Acuracia_TestingSet;

if Pasta == 1   % guarda a pasta com melhor desempenho
    max = Acuracia_TestingSet
    wMax = w;
    PastaMax = Pasta;
else 
    if Acuracia_TestingSet > max
        max = Acuracia_TestingSet;
        wMax = w;
        PastaMax = Pasta;
    end
end


% Matriz de Confusão
true_cancer      = sum(double(ytest == 1))
true_no_cancer    = sum(double(ytest == 0))
predicted_cancer = sum(double(ytestpred == 1)) 
true_positive    = sum(double(ytest == 1) .* double(ytestpred == 1))
true_negative    = sum(double(ytest == 0) .* double(ytestpred_sem == 1))

Precision = 100.0*true_positive/predicted_cancer
Recall = 100.0*true_positive/true_cancer
Specificidade = 100.0*true_negative/(true_no_cancer)


%F1 Score

F1 = 2*Precision*Recall/(Precision + Recall)

perda = 1;
end

% Apresentação dos Resultados e gravação dos pesos da pasta com melhor
% desempenho no conjunto de testes

PastaMax
Acuracia_Media = Acuracia_Media/5
max
xlswrite('w-cf-5fold.xls',wMax);