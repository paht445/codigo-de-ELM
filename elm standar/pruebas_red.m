
%%
clear
load('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\fecha1\datos_mezclados.mat')
T = readtable('C:\Users\paht4\Desktop\base de datos\modulo\imagenes\fecha1\etiquetas_mezcladas.csv');
T2 = table2array(T);
%%
%eberia servir dice 111111

clear
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
% esteeeee esssz
P= datos_entrenamiento;
PP= datos_testeo;
T=T2_entrenamiento'+1;
TT=T2_testeo'+1;
N=1000;%round(linspace(10,(2500),30));
%%
P=(P/max(max(P)))*2-1;
PP=(PP/max(max(PP)))*2-1;
%Normalizacion

%%
% Variables para almacenar resultados
numValidaciones = 10;
testingAccuracies = zeros(numValidaciones, 1);

for a= 1:length(N)
for i = 1:numValidaciones
    % disp(i);

 [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy, test_gmean] = ...
    elm_mc(T,P,TT,PP, 1,N(a) , 'sig');
    
    % Almacenamiento del resultado de TestingAccuracy
    % TrainingAccuracyX(a,i)=TrainingAccuracy;
    TestingAccuracyX(a,i) = TestingAccuracy;
    test_gmeanX(a,i)=test_gmean;
    TrainingTimeX(a,i)=TrainingTime;
end
end

%%
TrainingTimeX(a,i)=TrainingTime;
MTrainingTimeX=mean(TrainingTimeX,2);
std(MTrainingTimeX);
% MTrainingAccuracyX=mean(TrainingAccuracyX,2)
MTrainingTimeX=mean(TrainingTimeX,2);
MTestingAccuracyX= mean(TestingAccuracyX,2);
Mtest_gmeanX=mean(test_gmeanX,2);
% % Cálculo del promedio y desviación estándar
% promedioTestingAccuracy = mean(testingAccuracies);
% desviacionEstandarTestingAccuracy = std(testingAccuracies);
% 
% % Mostrar resultados
% fprintf('Promedio de TestingAccuracy: %f\n', promedioTestingAccuracy);
% fprintf('Desviación estándar de TestingAccuracy: %f\n', desviacionEstandarTestingAccuracy);


%%
meanTestingAccuracy = mean(mean(TestingAccuracyMC, 2))
meanTrainingAccuracy = mean(mean(TrainingAccuracyMC, 2))
meanTrainingTime = mean(mean(TrainingTimeMC, 2))
meanTestingTime = mean(mean(TestingTimeMC, 2))
meanTestgmean= mean(mean(TestgmeanMC,2))
%%
std(MTestingAccuracyX)

std(Mtest_gmeanX)


% mean(TrainingTimem,2) %tiempo de entrenamiento 
% std(TrainingTimem)

%%
%GRAFICOS
% figure;
% semilogy(1: length(neuronas),TestingAccuracyv2,'r');
% figure;
% semilogy(1: c.NumTestSets,TrainingAccuracyv2,'g');
% Graficar exactitud de entrenamiento y testeo en función de las neuronas ocultas
figure;
%plot(neuronas, meanTrainingAccuracy, '-o', 'DisplayName', 'Exactitud Entrenamiento');
hold on;
plot(N, MTestingAccuracyX, '-s');
xlabel('Neuronas Ocultas, N');
ylabel('Exactitud, Exac');
title('a) Exactitud');

grid on;
axis([0 2500 0.4 1 ])
% % Graficar tiempo de entrenamiento y testeo en función de las neuronas ocultas
% figure;
% plot(neuronas, meanTrainingTime, '-o', 'DisplayName', 'Tiempo Entrenamiento');
% hold on;
% plot(neuronas, meanTestingTime, '-s', 'DisplayName', 'Tiempo Testeo');
% xlabel('Número de Neuronas Ocultas');
% ylabel('Tiempo (segundos)');
% title('Tiempo de Entrenamiento y testeo');
% legend('show');
% grid on;

figure;
plot(N, Mtest_gmeanX, '-o');hold on;grid on; grid minor;
xlabel('Neuronas Ocultas, N');
ylabel('Media geométrica, G-mean');

title('b) Media geométrica ')
axis([0 2500 0 0.6])

