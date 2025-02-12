clear
load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\fecha1\datos_mezclados.mat')
T = readtable('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\fecha1\etiquetas_mezcladas.csv');
T2 = table2array(T);
%%
clear
load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\cargadas\datos_700.mat')
T = readtable('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\cargadas\etiquetas_700.csv');
T2 = table2array(T);
%%
%222
% Parámetros de validación cruzada y ELM
K = 5; % Número de particiones para K-Fold
c = cvpartition(T2, 'KFold', K);
neuronas = round(linspace(1,round((1-1/K)*length(T2)),50));
 % Número de neuronas ocultas en la ELM
PR = 2.^(-16:4:16); % Valor de regularización para ELM regularizado
%%
% Inicializar matrices para almacenar el rendimiento
TestingAccuracyv2 = zeros(length(PR), length(neuronas), K);
TrainingAccuracyv2 = zeros(length(PR), length(neuronas), K);
TrainingTimev2 = zeros(length(neuronas), K);
TestingTimev2 = zeros(length(neuronas), K);

% Realizar validación cruzada
for jjj = 1:length(PR)
    for jj = 1:length(neuronas)
        for j = 1:c.NumTestSets
            % Índices de entrenamiento y prueba para la partición actual
            idxTrain = training(c, j);
            idxTest = test(c, j);

            % Extraer conjuntos de entrenamiento y prueba
            entrenamientoe = datos_mezclados(:, idxTrain); % Entradas de entrenamiento
            entrenamientos = T2(idxTrain)'; % Salidas de entrenamiento
            testeoe = datos_mezclados(:, idxTest); % Entradas de prueba
            testeos = T2(idxTest)'; % Salidas de prueba

            % Normalización de entrenamiento y prueba en el rango [-1, 1]
            entrenamientoe = 2 * (entrenamientoe / max(max(entrenamientoe))) - 1;
            testeoe = 2 * (testeoe / max(max(testeoe))) - 1;

            % Guardar en variables temporales para la llamada a ELM
            TTRAI = entrenamientos;
            PTRAI = entrenamientoe;
            TTEST = testeos;
            PTEST = testeoe;

            % Entrenamiento y prueba de la ELM regularizada para la partición actual
            [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,fmeasure,test_gmean] = ...
                ELMdes1(TTRAI, PTRAI, TTEST, PTEST, 1, neuronas(jj), 'sig', PR(jjj));
            
            % Almacenar los resultados de precisión en matrices 3D
            TestingAccuracyv2(jjj, jj, j) = TestingAccuracy;
            TrainingAccuracyv2(jjj, jj, j) = TrainingAccuracy;
            TrainingTimev2(jjj,jj, j) = TrainingTime;
            TestingTimev2(jjj,jj, j) = TestingTime;
            % test_gmeanv2(jjj,jj,j)=test_gmean;
        end
    end
end
%%
% Cargar conjuntos de entrenamiento y testeo preprocesados
load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\entrenamiento_2021\cargadas\datos_entrenamiento.mat', 'datos_mezclados');
datos_entrenamiento = datos_mezclados; % Renombrar a datos_entrenamiento
clear datos_mezclados; 

load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\cargadas\datos_testeo.mat', 'datos_mezclados');
datos_testeo = datos_mezclados; % Renombrar a datos_testeo
clear datos_mezclados; 

% Cargar etiquetas correspondientes
T_entrenamiento = readtable('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\entrenamiento_2021\cargadas\etiquetas_entrenamiento.csv');
T_testeo = readtable('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\cargadas\etiquetas_testeo.csv');

% Convertir tablas de etiquetas a arrays
T2_entrenamiento = table2array(T_entrenamiento);
T2_testeo = table2array(T_testeo);
%%
%esteeeeee essss
TTRAI=T2_entrenamiento'+1;
PTRAI= datos_entrenamiento;
TTEST=T2_testeo'+1;
PTEST= datos_testeo;

N=2300%round(linspace(10,2500,30));
PR = 2.^-6%(-20:4:20);
%%
%Normalizacion de -1 a 1
PTRAI=(PTRAI/max(max(PTRAI)))*2-1;
PTEST=(PTEST/max(max(PTEST)))*2-1;

%%
% Variables para almacenar resultados
numValidaciones = 10;
testingAccuracies = zeros(numValidaciones, 1);
for jjj = 1:length(PR)
for a= 1:length(N)
for i = 1:numValidaciones
     %disp(i);

[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,test_gmean] = ...
        ELMdes1(TTRAI, PTRAI, TTEST, PTEST, 1, N(a), 'sig', PR(jjj));
    

    % Almacenamiento del resultado de TestingAccuracy
    % TrainingAccuracyX(a,i)=TrainingAccuracy;
    TestingAccuracyX(jjj,a,i) = TestingAccuracy;
    test_gmeanX(jjj,a,i)=test_gmean;
    TrainingTimeX(jjj,a,i)=TrainingTime;
end
end
end
%%
% MTrainingAccuracyX=mean(TrainingAccuracyX,2)
TrainingTimeX(jjj,a,i)=TrainingTime;
MTrainingTimeX=mean(mean(TrainingTimeX,3));
std(MTrainingTimeX)
MTestingAccuracyX = mean(mean(TestingAccuracyX,3));
Mtest_gmeanX = mean(mean(test_gmeanX,3));
MTrainingTimeX=mean(mean(TrainingTimeX,3));
des_estar_accuracy=std(MTestingAccuracyX);
des_estar_gmean=std(Mtest_gmeanX);

%%
%4GRACIFOS
% Calcular promedios de rendimiento a lo largo de las particiones K
[NO, pr] = meshgrid(N, PR);

% subplot(1, 2, 1);
contourf(NO, log2(pr),MTestingAccuracyX, 15);
c = colorbar('southoutside');
c.Label.String = 'Exactitud,Exac';
colormap summer
grid on; grid minor;
xlabel('Neuronas ocultas, N');
ylabel('Parametro de regularización, 2^C');
title('a) Exactitud');
clim([0.4 1])

% subplot(1, 2, 2);
% contourf(NO, log2(pr), mean(TrainingAccuracyv2,3), 15);
% c = colorbar('southoutside');
% c.Label.String = 'Exactitud';
% colormap summer
% grid on; grid minor;
% xlabel('Neuronas ocultas, N'); ylabel('Parametro de regularization, 2^C')
% title('Exactitud en testeo');
% figure;
% plot(neuronas, mean(mean(TrainingTimev2, 3)), '-o', 'DisplayName', 'Tiempo Entrenamiento');
% hold on;
% plot(neuronas, mean(mean(TestingTimev2, 3)), '-s', 'DisplayName', 'Tiempo Testeo');
% xlabel('Número de Neuronas Ocultas');
% ylabel('Tiempo (segundos)');
% title('Tiempo de Entrenamiento y testeo');
% legend('show');
% grid on;

figure;
contourf(NO,log2(pr),Mtest_gmeanX,15);
c = colorbar('southoutside');
c.Label.String = 'Media geométrica, Media-G';grid on; grid minor;
colormap summer
xlabel('Neuronas ocultas, N'); ylabel('Parametro de regularización, 2^C')
title('b) Media geométrica');
clim([0 0.6])
