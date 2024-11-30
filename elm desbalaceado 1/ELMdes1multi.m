function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,fmeasure,test_gmean] = ELMdes1multi(TTRAI, PTRAI, TTEST, PTEST, Elm_Type, NumberofHiddenNeurons, ActivationFunction,C)
% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
% train_data=load(TrainingData_File);%SE CARGAN DATOS
T=TTRAI;% SE SEPARAN LAS SALIDAS
P=PTRAI;% SE SEPARAN LAS ENTRADAS
% clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
% test_data=load(TestingData_File);%LO MISMO QUE PARA EL ENTRENAMIENTO
TV.T=TTEST;
TV.P=PTEST;
% clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);%NUMERO DE DATOS DE ENTRADA PARA EL ENTRAMIENTO, CORRIDAS
NumberofTestingData=size(TV.P,2);%NUMERO DE DATOS DE ENTRADA PARA EL TESTEO, CORRIDAS
NumberofInputNeurons=size(P,1);%NUMERO DE NEURONAS DE LA ENTRADA, ENTRADAS POR CORRIDA
[x,posx1] = find(T==1);
[x,posx2] = find(T==2);

clear x;
W = zeros(length(T),length(T));

for i = 1:length(T)
    if T(i) == 1
        W(i,i)= 1/length(posx1);
    elseif T(i)==2
        W(i,i)= 1/length(posx2);   
    end 
    
end

if Elm_Type~=REGRESSION%ENTRA SI NO ES CONTINUO, Q HACE?
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 8:(NumberofTrainingData+NumberofTestingData) %%cambios de 2 a 8
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
       
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T;

end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases
 start_time_train=cputime;%COMIENZA RELOJ DE ENTRENAMIENTO

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;%MATRIZ DE DATOS ALEATORIOS PARA LOS PESOS, CONSTANTE POR CORRIDAS
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1)*2-1;%VECTOR DE DATOS ALEATORIOS PARA LOS BIAS, CONSTANTE POR CORRIDAS
tempH=InputWeight*P;%PESOS POR ENTRADAS, LOS PESOS SON CONSTANTES EN CADA CORRIDA
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);%SE EXPADEN LOS BIAS DE ACUERDO A LOS DATOS DE ENTRADA Y NUMERO DE CAPAS OCULTAS
%   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;%SALIDA DE LA CAPA OCULTA CONSIDERANDO FUNCION LINEAL

%%%%%%%%%%% Calculate hidden neuron output matrix H
%SE APLICA FUNCION NO LINEAL A LA SALIDA TEMPH
% switch lower(ActivationFunction)
%     case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));%
%     case {'coseno'}
%         %%%%%%%% Sine
%         H = cos(tempH); %   
%     case {'hardlim'}
%         %%%%%%%% Hard Limit
%         H = double(hardlim(tempH));
%     case {'tribas'}
%         %%%%%%%% Triangular basis function
%         H = tribas(tempH);
%     case {'radbas'}
%         %%%%%%%% Radial basis function
%         H = radbas(tempH);
%         %%%%%%%% More activation functions can be added here  
%     case {'hiperbolica'}
%         %%%%%%%% Sine
%         H = (1 - exp(-tempH)) ./ (1 + exp(-tempH));% 
% end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%CACULO  DE LOS PESOS DESPUES DE LA CAPA OCULTA

% OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
if (NumberofTrainingData>NumberofHiddenNeurons)
% n = NumberofHiddenNeurons;
OutputWeight=((H*W*H'+speye(NumberofHiddenNeurons)/C)\(H*W*T')); %LxL
else
% n = size(T,2);
OutputWeight=H*((W*H'*H+speye(NumberofTrainingData)/C)\(W*T')); %NxN
end
%OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
%implementation; one can set regularizaiton factor C properly in classification applications 
%OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2 //refer to 2012 IEEE TSMC-B paper
%implementation; one can set regularizaiton factor C properly in classification applications

%If you use faster methods or kernel method, PLEASE CITE in your paper properly: 

%Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010. 

 end_time_train=cputime;%FIN DEL TIEMPO DE ENTRAMIENTO? EL ENTRAMIENTO CONSISTE EN SOLO CALCULAR B?
 TrainingTime=1*(end_time_train-start_time_train) ;       %   Calculate CPU time (seconds) spent for training ELM
%TrainingTime=0;

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))     ;          %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
% % switch lower(ActivationFunction)
%     case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
%     case {'coseno'}
%         %%%%%%%% Sine
%         H_test = cos(tempH_test);        
%     case {'hardlim'}
%         %%%%%%%% Hard Limit
%         H_test = double(hardlim(tempH_test));        
%     case {'tribas'}
%         %%%%%%%% Triangular basis function
%         H_test = tribas(tempH_test);        
%     case {'radbas'}
%         %%%%%%%% Radial basis function
%         H_test = radbas(tempH_test);        
%         %%%%%%%% More activation functions can be added here  
%     case {'hiperbolica'}
%         %%%%%%%% Sigmoid 
%         H_test = (1 - exp(-tempH_test)) ./ (1 + exp(-tempH_test));
% end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=1*(end_time_test-start_time_test) ;          %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
% TestingTime=0;
plotconfusion(TV.T,TY);
title('');
ylabel('Clase de salida');
xlabel('Clase destino');
if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY)) ;           %   Calculate testing accuracy (RMSE) for regression case
end
if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
      MissClassificationRate_Training=0;
      MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
  TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
%  TrainingAccuracy=0;
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  ;
end

%    TP=0;
%     TN=0;
%     FP=0;
%     FN=0;
%     
%     for i = 1 : size(T, 2)
%         [x, label_index_expected]=max(T(:,i));
%         [x, label_index_actual]=max(Y(:,i));
%         if label_index_expected == 1 & label_index_actual ==1
%             TP = TP + 1;
%         elseif label_index_expected == 1 & label_index_actual ==2
%             FN = FN + 1;
%         elseif label_index_expected == 2 & label_index_actual ==1
%             FP = FP + 1;
%         elseif label_index_expected == 2 & label_index_actual ==2
%             TN = TN + 1;
%         end
%         %if label_index_actual~=label_index_expected
%         %    MissClassificationRate_Training=MissClassificationRate_Training+1;
%         %end
%     end
%     %TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)  ;
%     train_sensitivity = TP/(TP+FN);
%     train_specificity = TN/(TN+FP);
%     train_gmean = sqrt(train_sensitivity * train_specificity);
    
%     TP=0;
% %     TN=0;
%     FP=0;
%     FN=0;
%     for i = 1 : size(TV.T, 2)
%         [x, label_index_expected]=max(TV.T(:,i));
%         [x, label_index_actual]=max(TY(:,i));
%         %if label_index_actual~=label_index_expected
%         %%    MissClassificationRate_Testing=MissClassificationRate_Testing+1;
%         %end
%         if label_index_expected == 1 & label_index_actual ==1
%             TP = TP + 1;
%         elseif label_index_expected == 1 & (label_index_actual ==2 || label_index_actual ==3)
%             FN = FN + 1;
%         elseif (label_index_actual ==2 || label_index_actual ==3) & label_index_actual ==1
%             FP = FP + 1;
% %         elseif label_index_expected == 2 & label_index_actual ==2
% %             TN = TN + 1;
%         end
%     end
%     %TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  ;
% %     test_sensitivity = TP/(TP+FN);
% %     test_specificity = TN/(TN+FP);
%     precision=TP/(TP+FP);
%     recall=TP/(TP+FN);
%     fmeasure1=(2*(precision*recall))/(precision+recall);
%     
%         TP=0;
% %     TN=0;
%     FP=0;
%     FN=0;
%     for i = 1 : size(TV.T, 2)
%         [x, label_index_expected]=max(TV.T(:,i));
%         [x, label_index_actual]=max(TY(:,i));
%         %if label_index_actual~=label_index_expected
%         %%    MissClassificationRate_Testing=MissClassificationRate_Testing+1;
%         %end
%         if label_index_expected == 2 & label_index_actual ==2
%             TP = TP + 1;
%         elseif label_index_expected == 2 & (label_index_actual ==1 || label_index_actual ==3)
%             FN = FN + 1;
%         elseif (label_index_actual ==1 || label_index_actual ==3) & label_index_actual ==2
%             FP = FP + 1;
% %         elseif label_index_expected == 2 & label_index_actual ==2
% %             TN = TN + 1;
%         end
%     end
%     %TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  ;
% %     test_sensitivity = TP/(TP+FN);
% %     test_specificity = TN/(TN+FP);
%     precision=TP/(TP+FP);
%     recall=TP/(TP+FN);
%     fmeasure2=(2*(precision*recall))/(precision+recall);
%     
%         TP=0;
% %     TN=0;
%     FP=0;
%     FN=0;
%     for i = 1 : size(TV.T, 2)
%         [x, label_index_expected]=max(TV.T(:,i));
%         [x, label_index_actual]=max(TY(:,i));
%         %if label_index_actual~=label_index_expected
%         %%    MissClassificationRate_Testing=MissClassificationRate_Testing+1;
%         %end
%         if label_index_expected == 3 & label_index_actual ==3
%             TP = TP + 1;
%         elseif label_index_expected == 3 & (label_index_actual ==1 || label_index_actual ==2)
%             FN = FN + 1;
%         elseif (label_index_actual ==2 || label_index_actual ==1) & label_index_actual ==3
%             FP = FP + 1;
% %         elseif label_index_expected == 2 & label_index_actual ==2
% %             TN = TN + 1;
%         end
%     end
%     %TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  ;
% %     test_sensitivity = TP/(TP+FN);
% %     test_specificity = TN/(TN+FP);
%     precision=TP/(TP+FP);
%     recall=TP/(TP+FN);
%     fmeasure3=(2*(precision*recall))/(precision+recall);
%     fmeasure=(fmeasure1+fmeasure2+fmeasure3)/3;
    fmeasure=0;
%     test_gmean =1;% sqrt(test_sensitivity * test_specificity);
    
%     
%     for i = 1 : size(TV.T, 2)
%         [x, label_index_expected]=max(TV.T(:,i));
%         [x, label_index_actual]=max(TY(:,i));
%         if label_index_actual~=label_index_expected
%             MissClassificationRate_Testing=MissClassificationRate_Testing+1;
%         end
%     end
%     TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  

MissClassificationRate_Testing1=0;
MissClassificationRate_Testing2=0;


% para sacar la media geometrica
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if (label_index_actual~=label_index_expected) && (label_index_expected==1)
            MissClassificationRate_Testing1=MissClassificationRate_Testing1+1;
        end
    end
        for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if (label_index_actual~=label_index_expected) && (label_index_expected==2)
            MissClassificationRate_Testing2=MissClassificationRate_Testing2+1;
        end
    end
       
   
    
[a,tx]=max(TV.T(:,:));
q=[ 1-MissClassificationRate_Testing1/length(tx(find(tx==1))) ...
    1-MissClassificationRate_Testing1/length(tx(find(tx==2))) ...
    ];
% q
% q(find(q==0))=1/length(tx(find(tx==4)));
% if sum(isnan(q))==1
%  ValidationAccuracy=(prod(q(1,2:5)))^(1/4);
% else
test_gmean=prod(q)^(1/2);%% para el tipo de clases
end