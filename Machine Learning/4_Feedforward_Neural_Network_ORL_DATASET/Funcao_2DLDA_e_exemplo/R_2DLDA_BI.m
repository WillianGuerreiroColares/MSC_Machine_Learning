function [U,V] = R_2DLDA_BI(path)
%2DLDA Summary of this function goes here
%   Detailed explanation goes here
%Obtenção de informações sobre o dataset

N = 400;
c = 40;
N_i = N/c;

%Cálculo da matriz de covariância entre-classes

%Passo 1: Obtenção das médias de cada classe e média global

%Média geral
A = im2double(zeros(112,92));
%Média por classe
Aux = im2double(zeros(112,92));
A_i = {};

for folder=1:c
    folder_name = strcat('s',num2str(folder));
    for pic=1:10
        image_name = strcat(num2str(pic),'.pgm');
        image = imread(strcat(path,'\',folder_name,'\',image_name));
        A = A + double(image);
        Aux = Aux + double(image);
    end
    Aux = Aux/10;
    A_i{folder} = Aux;
    Aux = im2double(zeros(112,92));
end

A = A / N;

%Passo 2: Redução horizontal e vertical

%Passo 3: Obtenção de S_b (H e V)
S_b_H = im2double(zeros(92,92));
S_b_V = im2double(zeros(112,112));
for i=1:c
    S_b_H = S_b_H + N_i*((A_i{i} - A)')*(A_i{i} - A);
    S_b_V = S_b_V + N_i*(A_i{i} - A)*(A_i{i} - A)';
end
S_b_H = S_b_H / N;
S_b_V = S_b_V / N;
%Passo 4: Obtenção de S_w (H e V)

S_w_H = im2double(zeros(92,92));
S_w_V = im2double(zeros(112,112));
for i=1:c
    folder_name = strcat('s',num2str(i));
    for j=1:10
        image_name = strcat(num2str(j),'.pgm');
        image = imread(strcat(path,'\',folder_name,'\',image_name));
        image = double(image);
        S_w_H = S_w_H + ((image - A_i{i})')*(image - A_i{i});
        S_w_V = S_w_V + (image - A_i{i})*(image - A_i{i})';
    end
end

S_w_H = S_w_H / N;
S_w_V = S_w_V / N;

%Passo 5: Determinação das matrizes U E V

%Passo 5.1: Multiplicação entre as matrizes de covariância
product1 = inv(S_w_H)*S_b_H;
product2 = inv(S_w_V)*S_b_V;

%Obtenção dos 10 maiores autovalores e seus respectivos autovetores

%Matriz U
[autovetores1,autovalores1] = eig(product1);
autovetores1
[autovalores_sorted1,indices1] = sort(diag(autovalores1),'descend');
U = autovetores1(:,indices1);
U = U(:,1:10);

%Matriz V
[autovetores2,autovalores2] = eig(product2);
[autovalores_sorted2,indices2] = sort(diag(autovalores2),'descend');
V = autovetores2(:,indices2);
V = V(:,1:10);

end

