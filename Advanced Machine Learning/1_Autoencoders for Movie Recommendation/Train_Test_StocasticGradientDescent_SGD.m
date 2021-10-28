for i=1:1
    close all;
    clear all;
    clc

    disp('Loading dataset')
    Mov = readtable("Movies.xlsx");
    T = xlsread("Ratings.xlsx");


    NoUsers = 610;
    NoHiddenLayers = 100;
    NoMovies = 9742;
    NoRatings = 100836;


    R = zeros(NoUsers,NoMovies);
    R_train = zeros(NoUsers,NoMovies);
    R_teste = zeros(NoUsers,NoMovies);
    
   
    
    U = ones(NoUsers,  NoHiddenLayers);  % U:  n×k input-to-hidden matrix
    V = ones(NoMovies, NoHiddenLayers);  % Vt: kxd hidden-output matrix 

    % R é a matriz com a tripla: usuários, filmes, ratings
    for j = 1:NoRatings
        R(T(j,1),T(j,2)) = T(j,4);
    end

    for i=1:NoUsers
        for j=1:NoMovies
            if j > 7742
                R_teste(i,j) = R(i,j);
            else
                R_train(i,j) = R(i,j);
            end
        end
    end
            

    U = -0.25 + 0.5*rand(size(U));
    V = -0.25 + 0.5*rand(size(V));

    %U = rand(size(U));
    %V = rand(size(V));


    U_plus = U;
    V_plus = V;


    alpha = 0.002;
    lambda = 0.001;


    e = R_train;
    frobenius_norm = 200;
    frobenius_array = [frobenius_norm];

    mse_value = 200;
    mse_array =[mse_value];

    mse_teste_value = 200;
    mse_teste_array = [mse_teste_value];

    h1 = plot(frobenius_array,'YDataSource','frobenius_array','DisplayName','Frobenius norm');
    ylim([0 200]);
    xlabel('Epochs');
    hold on;
    h2 = plot(mse_array,'YDataSource','mse_array','DisplayName','10*MSE (treino)');
    ylim([0 200]);
    legend;
    h3 = plot(mse_teste_array,'YDataSource','mse_teste_array','DisplayName','10*MSE (teste)');
    ylim([0 200]);
    legend;
    
    disp('Starting ...');
    tic;
    epochs = 0;
    while (mse_value > 0.1)
        epochs = epochs + 1;
        i_shuffled = randperm(NoUsers);
        for i = 1:NoUsers
            j_shuffled = randperm(NoMovies);
            for j = 1: NoMovies
                soma = 0.0;
                somal = 0.0;
                if R(i_shuffled(i), j_shuffled(j)) ~= 0

                    for q = 1:NoHiddenLayers
                        soma = soma + U(i_shuffled(i), q)*V(j_shuffled(j), q);
                        somal = somal + lambda*U(i_shuffled(i), q)^2 + lambda*V(j_shuffled(j), q)^2;
                    end

                    e(i_shuffled(i), j_shuffled(j)) = (R(i_shuffled(i), j_shuffled(j)) - soma) + somal;
                    %e(i_shuffled(i), j_shuffled(j)) = R(i_shuffled(i), j_shuffled(j)) - soma;

                    U_plus(i_shuffled(i),:) = U(i_shuffled(i),:)*(1-2*alpha*lambda) + 2*alpha*e(i_shuffled(i),j_shuffled(j))*V(j_shuffled(j),:);
                    V_plus(j_shuffled(j),:) = V(j_shuffled(j),:)*(1-2*alpha*lambda) + 2*alpha*e(i_shuffled(i),j_shuffled(j))*U(i_shuffled(i),:);
                    U(i_shuffled(i),:) = U_plus(i_shuffled(i),:);
                    V(j_shuffled(j),:) = V_plus(j_shuffled(j),:);

                end

            end
        end

        %%%%%%%%%%%%%%%%%% Cálculo de M e plot de gráficos %%%%%%%%%%%%%%%%%%%
        M = U * V';
        M_teste = U * V';

        for i = 1:NoUsers
            for j = 1: NoMovies            
                if R_train(i,j) == 0
                    M(i,j) = 0;
                end
            end
        end
        
        for i = 1:NoUsers
            for j = 1: NoMovies            
                if R_teste(i,j) == 0
                    M_teste(i,j) = 0;
                end
            end
        end

        mse_value = sum(sum((M  - R_train).^2))/nnz(R_train);
        mse_teste = sum(sum((M_teste - R_teste).^2))/nnz(R_teste);
        
        mse_array = [mse_array ; 10*mse_value];
        disp(['mse ',num2str(mse_value)])
        
        mse_teste_array = [mse_teste_array ; 10*mse_teste];
        disp(['mse teste ',num2str(mse_teste)])

        frobenius_norm = norm(e,'fro');
        frobenius_array = [frobenius_array ; frobenius_norm];
        disp(['Frobenius norm ',num2str(frobenius_norm), newline]);

        refreshdata;
        drawnow;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end


    %%% 2a. Parte: Recomendação de Filmes para usuários selecionados:

    R_base = U * V';
    Tabela_Rec = zeros(8,6); 
    Tabela_Rank = zeros(8,6); 
    Users = [40, 92, 123, 245, 312, 460, 514, 590];

    for u = 1:8

     contador = 0;
     for j = 1:NoMovies 
        if R_train(Users(u),j) > 0
            R_base(Users(u),j) = 0; % despreza as avaliações originais dos filmes feitas pelo usuário "u"
            contador = contador + 1;
        end
     end 

     [Rec, I] = sort(R_base(Users(u),:), 'descend');

     for col = 1:5
        Tabela_Rec(u, col)  = I(col); 
        Tabela_Rank(u, col) = R_base(Users(u),I(col));
     end


        Tabela_Rec(u, 6)  =  sum((M(Users(u), :) - R_train(Users(u), :)).^2)/contador;

    end

    %%% 

    elapsed_time = toc;
    disp(['Elapsed Time: ',num2str(elapsed_time)]);
    disp(['Epochs : ',num2str(epochs)]);

    tabela_movies = cell(8,6,1);
    for i=1:8
        for j=1:6
            if j~= 6
                tabela_movies{i,j} = table2array(Mov(Tabela_Rec(i,j),3));
            else
                tabela_movies{i,j} = Tabela_Rec(i,j);
            end
        end
    end


    movies_scores = zeros(8,5);

    for i=1:8
        for j=1:5
            movie_index = Tabela_Rec(i,j);
            movies_scores(i,j) = sum(R(:,movie_index));
        end
    end

    path = strcat(datestr(datetime(),'mmmm-dd-yyyy HH-MM-SS'),'.xls');
    path = strcat('results2/','tabela_rec_',path);
    xlswrite(path,Tabela_Rec);
    
    path = strcat(datestr(datetime(),'mmmm-dd-yyyy HH-MM-SS'),'.xls');
    path = strcat('results3/','movie_scores_',path);
    xlswrite(path,movies_scores);
end