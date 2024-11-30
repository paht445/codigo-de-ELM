% Preprocesamiento de imágenes
infected_folder = 'C:\Users\paht4\Desktop\base de datos\modulo\03_11_2021\Ground_RGB_Photos\enfermas';  % Carpeta con imágenes de clase positiva (por ejemplo, infectadas)
healthy_folder = 'C:\Users\paht4\Desktop\base de datos\modulo\03_11_2021\Ground_RGB_Photos\Healthy';        % Carpeta con imágenes de clase negativa (por ejemplo, sanas)
image_size = [64, 64];  % Tamaño al que se redimensionarán las imágenes

% Leer imágenes infectadas
infected_images = dir(fullfile(infected_folder, '*.jpg'));
num_infected = numel(infected_images);
infected_data = zeros(num_infected, prod(image_size)); % Matriz para almacenar las imágenes como vectores

for i = 1:num_infected
    img = imread(fullfile(infected_folder, infected_images(i).name));
    img = imresize(img, image_size);  % Redimensionar a image_size
    if size(img, 3) == 3
        img = rgb2gray(img);  % Convertir a escala de grises si es RGB
    end
    img = double(img) / 255;  % Normalizar a [0, 1]
    infected_data(i, :) = img(:)';  % Convertir a vector fila y almacenar
end

% Leer imágenes sanas
healthy_images = dir(fullfile(healthy_folder, '*.jpg'));
num_healthy = numel(healthy_images);
healthy_data = zeros(num_healthy, prod(image_size));

for i = 1:num_healthy
    img = imread(fullfile(healthy_folder, healthy_images(i).name));
    img = imresize(img, image_size);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = double(img) / 255;
    healthy_data(i, :) = img(:)';
end

% Crear etiquetas: 1 para infectado, 0 para sano
labels_infected = ones(num_infected, 1);
labels_healthy = zeros(num_healthy, 1);

% Combinar datos y etiquetas
data = [infected_data; healthy_data];
labels = [labels_infected; labels_healthy];

% Guardar los datos preprocesados en un archivo .mat
save('preprocessed_images.mat', 'data', 'labels');
disp('Preprocesamiento completo. Datos guardados en preprocessed_images.mat');
%%
% Cargar datos preprocesados
load('preprocessed_images.mat', 'data', 'labels');

% Dividir los datos en entrenamiento y prueba con validación cruzada
c = cvpartition(labels, 'KFold', 5);  % Validación cruzada K-fold
NO = round(linspace(1, 0.8 * length(labels), 30));  % Configuración de neuronas ocultas

% Configuración de la ELM
ActivationFunctions = {'sig', 'sin', 'hardlim'};  % Funciones de activación a probar
results = struct();  % Estructura para almacenar los resultados

% Bucle de validación cruzada
for jj = 1:length(NO)  % Iterar sobre diferentes números de neuronas ocultas
    for af_idx = 1:length(ActivationFunctions)  % Iterar sobre funciones de activación
        af = ActivationFunctions{af_idx};
        
        TrainingTimeAll = [];
        TestingTimeAll = [];
        TrainingAccuracyAll = [];
        TestingAccuracyAll = [];
        
        for fold = 1:c.NumTestSets  % Recorrer particiones de validación cruzada
            idxTrain = training(c, fold);
            idxTest = test(c, fold);
            
            train_images = data(idxTrain, :);
            train_labels = labels(idxTrain);
            test_images = data(idxTest, :);
            test_labels = labels(idxTest);
            
            % Normalización entre -1 y 1
            train_images = 2 * (train_images / max(max(train_images))) - 1;
            test_images = 2 * (test_images / max(max(test_images))) - 1;
            
            % Entrenar ELM
            [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ...
                ELM(train_labels', train_images', test_labels', test_images', 1, NO(jj), af);
            
            % Guardar resultados
            TrainingTimeAll = [TrainingTimeAll; TrainingTime];
            TestingTimeAll = [TestingTimeAll; TestingTime];
            TrainingAccuracyAll = [TrainingAccuracyAll; TrainingAccuracy];
            TestingAccuracyAll = [TestingAccuracyAll; TestingAccuracy];
        end
        
        % Almacenar resultados promedio para cada combinación de NO y función de activación
        results(jj).(af).TrainingTime = mean(TrainingTimeAll);
        results(jj).(af).TestingTime = mean(TestingTimeAll);
        results(jj).(af).TrainingAccuracy = mean(TrainingAccuracyAll);
        results(jj).(af).TestingAccuracy = mean(TestingAccuracyAll);
    end
end

% Graficar resultados de precisión y tiempos
figure;
hold on;
for af_idx = 1:length(ActivationFunctions)
    af = ActivationFunctions{af_idx};
    TrainingAcc = arrayfun(@(x) x.(af).TrainingAccuracy, results);
    TestingAcc = arrayfun(@(x) x.(af).TestingAccuracy, results);
    plot(NO, TrainingAcc, 'DisplayName', ['Entrenamiento - ' af]);
    plot(NO, TestingAcc, '--', 'DisplayName', ['Testeo - ' af]);
end
xlabel('Cantidad de neuronas ocultas, N');
ylabel('Precisión');
legend('show');
title('Desempeño de la ELM con diferentes funciones de activación');
hold off;

% Graficar tiempos de entrenamiento y prueba
figure;
hold on;
for af_idx = 1:length(ActivationFunctions)
    af = ActivationFunctions{af_idx};
    TrainingTime = arrayfun(@(x) x.(af).TrainingTime, results);
    TestingTime = arrayfun(@(x) x.(af).TestingTime, results);
    plot(NO, TrainingTime, 'DisplayName', ['Tiempo de Entrenamiento - ' af]);
    plot(NO, TestingTime, '--', 'DisplayName', ['Tiempo de Prueba - ' af]);
end
xlabel('Cantidad de neuronas ocultas, N');
ylabel('Tiempo (s)');
legend('show');
title('Tiempos de la ELM con diferentes funciones de activación');
hold off;
