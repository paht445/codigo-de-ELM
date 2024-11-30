clear
load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\fecha1\datos_mezclados.mat')
T = readtable('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\fecha1\etiquetas_mezcladas.csv');
T2 = table2array(T);
%%
clear
%111
load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\cargadas\datos_700.mat')
T = readtable('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\cargadas\etiquetas_700.csv');
T2 = table2array(T);

%%
%2021 y 2022
% Cargar conjuntos de entrenamiento y testeo preprocesados
load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\entrenamiento_2021\cargadas\datos_entrenamiento.mat', 'datos_mezclados');
datos_entrenamiento = datos_mezclados; % Renombrar a datos_entrenamiento
clear datos_mezclados; % Limpiar para evitar conflictos

load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\cargadas\datos_testeo.mat', 'datos_mezclados');
datos_testeo = datos_mezclados; % Renombrar a datos_testeo
clear datos_mezclados; % Limpiar para evitar conflictos

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

N=1200;%round(linspace(10,2500,30));
PR = 2.^-6%(-20:4:20);
%%
%Normalizacion de -1 a 1
PTRAI=(PTRAI/max(max(PTRAI)))*2-1;
PTEST=(PTEST/max(max(PTEST)))*2-1;
%%
%esteeeeee essss

% Variables para almacenar resultados
numValidaciones = 10;
testingAccuracies = zeros(numValidaciones, 1);
for jjj = 1:length(PR)
for a= 1:length(N)
for i = 1:numValidaciones
     %disp(i);

[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,test_gmean] = ...
                ELMrw2(TTRAI, PTRAI, TTEST, PTEST, 1, N(a), 'sig', PR(jjj));
    

    % Almacenamiento del resultado de TestingAccuracy
    % TrainingAccuracyX(a,i)=TrainingAccuracy;
    TestingAccuracyX(jjj,a,i) = TestingAccuracy;
    test_gmeanX(jjj,a,i)=test_gmean;
    TrainingTimeX(jjj,a,i)=TrainingTime;
end
end
end
%%
TrainingTime
% MTrainingAccuracyX=mean(TrainingAccuracyX,2)
MTestingAccuracyX = mean(mean(TestingAccuracyX,3));
Mtest_gmeanX = mean(mean(test_gmeanX,3));
MTrainingTimeX =mean(mean(TrainingTime,3))
std(MTrainingTimeX)
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