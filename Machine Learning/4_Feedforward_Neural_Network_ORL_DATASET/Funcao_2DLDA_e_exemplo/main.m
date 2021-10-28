clear all; 
close all;
clc;

[U, V] = R_2DLDA_BI('ORL');

%------------------------------%

Threshold = 5;
%Passo 6: Obtenção das imagens reduzidas

test_samples = {};
scores ={};

for i=1:40
    test_samples{i} = imread(strcat('ORL\s',num2str(i),'\','1.pgm'));
end

%40 indivíduos
for k=1:40
    individuo = test_samples{k};
    individuo_r = (V')*double(individuo)*U;
    for i=1:40
        folder_name = strcat('s',num2str(i));
        for j=1:10
            image_name = strcat(num2str(j),'.pgm');
            image = imread(strcat('ORL\',folder_name,'\',image_name));
            image_r = (V')*double(image)*U;
            
            diff = individuo_r - image_r;
            diff_r = reshape(diff,1,[]);
            
            subplot(2,4,1);
            imshow(image);
            subplot(2,4,2);
            imshow(image_r,[],'InitialMagnification','fit');
            subplot(2,4,3);
            imshow(individuo);
            subplot(2,4,4);
            imshow(individuo_r,[],'InitialMagnification','fit');
            subplot(2,4,5:8);
            plot(diff_r);
            title(num2str(mean(diff_r)));
    
            pause(1);
            
            
        end
    end
end



